/*
 * HNSW Baseline — 朴素实现，用于消融实验对照
 * 与 V4 优化版的主要区别：
 *   1. 距离计算：纯标量 float32 循环（无 NEON Intrinsic）
 *   2. 存储格式：float32（无 FP16 半精度压缩）
 *   3. 并发锁：std::mutex（无汇编 yield 自旋锁）
 *   4. 随机数：std::mt19937（无 SplitMix64）
 *   5. 无 pipeline prefetch
 *   6. 无 alignas(64) 缓存行对齐
 *   7. 无 thread_local 静态缓冲，使用 unordered_set 做 visited
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <queue>
#include <cmath>
#include <unordered_set>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <random>

using namespace std;
namespace py = pybind11;

// [基线] Node: 无 alignas(64)，float32 存储，std::mutex
struct Node
{
    mutex mtx_;
    int id;
    int level;
    vector<float> data;
    vector<vector<Node *>> neighbors;

    Node(int id, const vector<float> &data, int level, int M)
        : id(id), level(level), data(data)
    {
        neighbors.resize(level + 1);
        for (int i = 0; i <= level; ++i)
        {
            size_t capacity = (i == 0) ? (2 * M + 1) : (M + 1);
            neighbors[i].reserve(capacity);
        }
    }
};

class HNSW
{
public:
    int M_;
    int ef_construction_;
    int max_layers_;
    size_t dim_;

    vector<shared_ptr<Node>> nodes_;
    mutex nodes_mutex_;

    atomic<Node *> enter_point_{nullptr};
    atomic<int> current_max_level_{-1};

    HNSW(int M = 16, int ef_construction = 200, size_t dim = 128, size_t expected_elements = 100000)
        : M_(M), ef_construction_(ef_construction), max_layers_(16), dim_(dim)
    {
        nodes_.reserve(expected_elements);
    }

    /* [基线] 纯标量距离计算：无 NEON，无 FMA，无循环展开 */
    float distance(const vector<float> &v1, const vector<float> &v2)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < v1.size(); ++i)
        {
            float diff = v1[i] - v2[i];
            sum += diff * diff;
        }
        return sum;
    }

    /* [基线] search_layer: 无 prefetch, 无 thread_local, unordered_set visited */
    vector<pair<float, Node *>> search_layer(
        Node *ep, const vector<float> &query, int ef, int layer_level)
    {
        using Pair = pair<float, Node *>;
        unordered_set<int> visited;

        auto cmp_min = [](const Pair &a, const Pair &b)
        { return a.first > b.first; };
        auto cmp_max = [](const Pair &a, const Pair &b)
        { return a.first < b.first; };

        vector<Pair> candidates, results;

        float d = distance(ep->data, query);
        candidates.push_back({d, ep});
        push_heap(candidates.begin(), candidates.end(), cmp_min);
        results.push_back({d, ep});
        push_heap(results.begin(), results.end(), cmp_max);
        visited.insert(ep->id);

        while (!candidates.empty())
        {
            if (candidates.front().first > results.front().first)
                break;

            pop_heap(candidates.begin(), candidates.end(), cmp_min);
            Node *c = candidates.back().second;
            candidates.pop_back();

            vector<Node *> nbrs;
            {
                lock_guard<mutex> lock(c->mtx_);
                nbrs = c->neighbors[layer_level];
            }

            for (auto *neighbor : nbrs)
            {
                if (visited.count(neighbor->id))
                    continue;
                visited.insert(neighbor->id);

                float dn = distance(neighbor->data, query);
                if (dn < results.front().first || results.size() < (size_t)ef)
                {
                    candidates.push_back({dn, neighbor});
                    push_heap(candidates.begin(), candidates.end(), cmp_min);
                    results.push_back({dn, neighbor});
                    push_heap(results.begin(), results.end(), cmp_max);
                    if (results.size() > (size_t)ef)
                    {
                        pop_heap(results.begin(), results.end(), cmp_max);
                        results.pop_back();
                    }
                }
            }
        }

        sort(results.begin(), results.end(), [](const Pair &a, const Pair &b)
             { return a.first < b.first; });
        return results;
    }

    int get_random_level()
    {
        /* [基线] mt19937: 2496 字节状态 */
        static thread_local mt19937 rng(hash<thread::id>{}(this_thread::get_id()));
        static thread_local uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng);
        if (r == 0.0)
            r = 1e-10;
        int level = static_cast<int>(-log(r) * (1.0 / log((double)M_)));
        return min(level, max_layers_);
    }

    void insert_thread_safe(const vector<float> &vec, int id)
    {
        int level = get_random_level();
        /* [基线] 直接存 float32，无 FP16 转换 */
        auto new_node = make_shared<Node>(id, vec, level, M_);

        {
            lock_guard<mutex> lock(nodes_mutex_);
            nodes_.push_back(new_node);
        }

        Node *curr_ep = enter_point_.load(memory_order_acquire);
        int curr_max_lvl = current_max_level_.load(memory_order_acquire);

        if (curr_ep == nullptr)
        {
            Node *expected_ep = nullptr;
            if (enter_point_.compare_exchange_strong(expected_ep, new_node.get(),
                                                     memory_order_release, memory_order_relaxed))
            {
                current_max_level_.store(level, memory_order_release);
                return;
            }
            curr_ep = enter_point_.load(memory_order_acquire);
            curr_max_lvl = current_max_level_.load(memory_order_acquire);
        }

        float d_min = distance(curr_ep->data, vec);
        for (int lc = curr_max_lvl; lc > level; lc--)
        {
            bool changed = true;
            while (changed)
            {
                changed = false;
                if (curr_ep->level < lc)
                    break;
                vector<Node *> nc;
                {
                    lock_guard<mutex> lock(curr_ep->mtx_);
                    nc = curr_ep->neighbors[lc];
                }
                for (auto *nb : nc)
                {
                    if (nb->level < lc)
                        continue;
                    float d = distance(nb->data, vec);
                    if (d < d_min)
                    {
                        d_min = d;
                        curr_ep = nb;
                        changed = true;
                    }
                }
            }
        }

        for (int lc = min(curr_max_lvl, level); lc >= 0; lc--)
        {
            auto search_res = search_layer(curr_ep, vec, ef_construction_, lc);
            int max_conn = (lc == 0) ? (M_ * 2) : M_;
            int count = 0;
            for (size_t i = 0; i < search_res.size() && count < max_conn; i++)
            {
                Node *neighbor = search_res[i].second;
                {
                    lock_guard<mutex> lock(neighbor->mtx_);
                    neighbor->neighbors[lc].push_back(new_node.get());
                    if (neighbor->neighbors[lc].size() > (size_t)max_conn)
                    {
                        float max_d = -1.0f;
                        int worst = -1;
                        for (size_t j = 0; j < neighbor->neighbors[lc].size(); ++j)
                        {
                            float d = distance(neighbor->data, neighbor->neighbors[lc][j]->data);
                            if (d > max_d)
                            {
                                max_d = d;
                                worst = j;
                            }
                        }
                        if (worst != -1)
                        {
                            neighbor->neighbors[lc][worst] = neighbor->neighbors[lc].back();
                            neighbor->neighbors[lc].pop_back();
                        }
                    }
                }
                {
                    lock_guard<mutex> lock(new_node->mtx_);
                    new_node->neighbors[lc].push_back(neighbor);
                }
                count++;
            }
            if (!search_res.empty())
                curr_ep = search_res[0].second;
        }

        while (true)
        {
            int old_level = current_max_level_.load(memory_order_acquire);
            if (level <= old_level)
                break;
            if (current_max_level_.compare_exchange_weak(old_level, level,
                                                         memory_order_release, memory_order_relaxed))
            {
                enter_point_.store(new_node.get(), memory_order_release);
                break;
            }
        }
    }

    void parallel_insert(const vector<vector<float>> &vectors, int num_threads)
    {
        int total = vectors.size();
        auto worker = [&](int s, int e)
        {
            for (int i = s; i < e; ++i)
                this->insert_thread_safe(vectors[i], i);
        };
        vector<thread> threads;
        int chunk = total / num_threads;
        for (int i = 0; i < num_threads; ++i)
        {
            int s = i * chunk;
            int e = (i == num_threads - 1) ? total : (i + 1) * chunk;
            threads.emplace_back(worker, s, e);
        }
        for (auto &t : threads)
            t.join();
    }

    vector<int> search_knn(const vector<float> &query, int k, int ef_search = 0)
    {
        vector<int> ids;
        Node *curr_ep = enter_point_.load(memory_order_acquire);
        if (!curr_ep)
            return ids;

        if (ef_search <= 0)
            ef_search = max(k, 100);
        ef_search = max(ef_search, k);

        int curr_max_lvl = current_max_level_.load(memory_order_acquire);
        float d_min = distance(curr_ep->data, query);
        for (int lc = curr_max_lvl; lc > 0; lc--)
        {
            bool changed = true;
            while (changed)
            {
                changed = false;
                if (curr_ep->level < lc)
                    break;
                vector<Node *> nc;
                {
                    lock_guard<mutex> lock(curr_ep->mtx_);
                    nc = curr_ep->neighbors[lc];
                }
                for (auto *nb : nc)
                {
                    if (nb->level < lc)
                        continue;
                    float d = distance(nb->data, query);
                    if (d < d_min)
                    {
                        d_min = d;
                        curr_ep = nb;
                        changed = true;
                    }
                }
            }
        }

        auto cands = search_layer(curr_ep, query, ef_search, 0);
        for (int i = 0; i < min((int)cands.size(), k); ++i)
            ids.push_back(cands[i].second->id);
        return ids;
    }
};

vector<vector<float>> load_fvecs(const string &filepath)
{
    vector<vector<float>> data;
    ifstream file(filepath, ios::binary);
    if (!file.is_open())
        throw runtime_error("Could not open: " + filepath);
    int dim;
    while (file.read(reinterpret_cast<char *>(&dim), sizeof(int)))
    {
        vector<float> v(dim);
        file.read(reinterpret_cast<char *>(v.data()), dim * sizeof(float));
        data.push_back(v);
    }
    return data;
}

vector<vector<int>> load_ivecs(const string &filepath)
{
    vector<vector<int>> data;
    ifstream file(filepath, ios::binary);
    if (!file.is_open())
        throw runtime_error("Could not open: " + filepath);
    int dim;
    while (file.read(reinterpret_cast<char *>(&dim), sizeof(int)))
    {
        vector<int> v(dim);
        file.read(reinterpret_cast<char *>(v.data()), dim * sizeof(int));
        data.push_back(v);
    }
    return data;
}

PYBIND11_MODULE(hnsw_lib, m)
{
    m.doc() = "HNSW Baseline: scalar float32, std::mutex, mt19937, no NEON, no prefetch";
    m.def("load_fvecs", &load_fvecs);
    m.def("load_ivecs", &load_ivecs);
    py::class_<HNSW>(m, "HNSW")
        .def(py::init<int, int, size_t, size_t>(),
             py::arg("M") = 16, py::arg("ef_construction") = 200,
             py::arg("dim") = 128, py::arg("expected_elements") = 100000)
        .def("insert", &HNSW::insert_thread_safe)
        .def("parallel_insert", &HNSW::parallel_insert)
        .def("search", &HNSW::search_knn,
             py::arg("query"), py::arg("k"), py::arg("ef_search") = 0);
}
