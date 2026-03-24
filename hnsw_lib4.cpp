/*
 * HNSW V4 final — V1 原版 + 两处精准手术
 *
 * 手术 1: search_knn 接受 ef_search 参数（修复 recall 不随 search_k 变化的 bug）
 * 手术 2: search_layer 流水线预取（遍历邻居时提前 prefetch 下一批向量）
 *
 * 其他所有代码与 V1 完全一致，一行不动。
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <arm_neon.h>
#include <arm_fp16.h>
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

using namespace std;
namespace py = pybind11;

class SplitMix64
{
    uint64_t state_;
public:
    explicit SplitMix64(uint64_t seed) : state_(seed) {}
    uint64_t next()
    {
        uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double next_double()
    {
        return (next() >> 11) * (1.0 / (1ULL << 53));
    }
};

using Float16Vec = vector<__fp16>;

struct alignas(64) Node
{
    atomic_flag lock_ = ATOMIC_FLAG_INIT;
    int id;
    int level;
    Float16Vec data;
    vector<vector<Node *>> neighbors;

    Node(int id, const Float16Vec &data, int level, int M)
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

class SpinLockGuard
{
    atomic_flag &flag_;
public:
    SpinLockGuard(atomic_flag &flag) : flag_(flag)
    {
        while (flag_.test_and_set(memory_order_acquire))
        {
#if defined(__aarch64__) || defined(__arm__)
            __asm__ volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
            __asm__ volatile("pause" ::: "memory");
#endif
        }
    }
    ~SpinLockGuard()
    {
        flag_.clear(memory_order_release);
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
    atomic_flag nodes_lock_ = ATOMIC_FLAG_INIT;

    struct alignas(64) AlignedAtomicNodePtr
    {
        atomic<Node *> value{nullptr};
    };
    struct alignas(64) AlignedAtomicInt
    {
        atomic<int> value{-1};
    };

    AlignedAtomicNodePtr enter_point_;
    AlignedAtomicInt current_max_level_;

    HNSW(int M = 16, int ef_construction = 200, size_t dim = 128, size_t expected_elements = 100000)
        : M_(M), ef_construction_(ef_construction), max_layers_(16), dim_(dim)
    {
        nodes_.reserve(expected_elements);
    }

    // ---- 距离计算: 与 V1 完全一致 ----
    float distance(const Float16Vec &v1, const Float16Vec &v2)
    {
        const size_t dim = dim_;
        const __fp16 *p1 = v1.data();
        const __fp16 *p2 = v2.data();

        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);

        size_t i = 0;
        for (; i + 32 <= dim; i += 32)
        {
            float16x8_t a0 = vld1q_f16(p1 + i);
            float16x8_t b0 = vld1q_f16(p2 + i);
            float16x8_t diff0_f16 = vsubq_f16(a0, b0);
            float32x4_t diff0_lo = vcvt_f32_f16(vget_low_f16(diff0_f16));
            float32x4_t diff0_hi = vcvt_f32_f16(vget_high_f16(diff0_f16));
            sum0 = vfmaq_f32(sum0, diff0_lo, diff0_lo);
            sum0 = vfmaq_f32(sum0, diff0_hi, diff0_hi);

            float16x8_t a1 = vld1q_f16(p1 + i + 8);
            float16x8_t b1 = vld1q_f16(p2 + i + 8);
            float16x8_t diff1_f16 = vsubq_f16(a1, b1);
            float32x4_t diff1_lo = vcvt_f32_f16(vget_low_f16(diff1_f16));
            float32x4_t diff1_hi = vcvt_f32_f16(vget_high_f16(diff1_f16));
            sum1 = vfmaq_f32(sum1, diff1_lo, diff1_lo);
            sum1 = vfmaq_f32(sum1, diff1_hi, diff1_hi);

            float16x8_t a2 = vld1q_f16(p1 + i + 16);
            float16x8_t b2 = vld1q_f16(p2 + i + 16);
            float16x8_t diff2_f16 = vsubq_f16(a2, b2);
            float32x4_t diff2_lo = vcvt_f32_f16(vget_low_f16(diff2_f16));
            float32x4_t diff2_hi = vcvt_f32_f16(vget_high_f16(diff2_f16));
            sum2 = vfmaq_f32(sum2, diff2_lo, diff2_lo);
            sum2 = vfmaq_f32(sum2, diff2_hi, diff2_hi);

            float16x8_t a3 = vld1q_f16(p1 + i + 24);
            float16x8_t b3 = vld1q_f16(p2 + i + 24);
            float16x8_t diff3_f16 = vsubq_f16(a3, b3);
            float32x4_t diff3_lo = vcvt_f32_f16(vget_low_f16(diff3_f16));
            float32x4_t diff3_hi = vcvt_f32_f16(vget_high_f16(diff3_f16));
            sum3 = vfmaq_f32(sum3, diff3_lo, diff3_lo);
            sum3 = vfmaq_f32(sum3, diff3_hi, diff3_hi);
        }

        for (; i + 8 <= dim; i += 8)
        {
            float16x8_t a = vld1q_f16(p1 + i);
            float16x8_t b = vld1q_f16(p2 + i);
            float16x8_t diff_f16 = vsubq_f16(a, b);
            float32x4_t diff_lo = vcvt_f32_f16(vget_low_f16(diff_f16));
            float32x4_t diff_hi = vcvt_f32_f16(vget_high_f16(diff_f16));
            sum0 = vfmaq_f32(sum0, diff_lo, diff_lo);
            sum0 = vfmaq_f32(sum0, diff_hi, diff_hi);
        }

        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);
        float result = vaddvq_f32(sum0);

        for (; i < dim; ++i)
        {
            float diff = static_cast<float>(p1[i]) - static_cast<float>(p2[i]);
            result += diff * diff;
        }

        return result;
    }

    float distance(const Float16Vec &v1, const vector<float> &query)
    {
        const size_t dim = dim_;
        const __fp16 *p1 = v1.data();
        const float *p2 = query.data();

        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);

        size_t i = 0;
        for (; i + 8 <= dim; i += 8)
        {
            float16x8_t a_f16 = vld1q_f16(p1 + i);
            float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a_f16));
            float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a_f16));
            float32x4_t b_lo = vld1q_f32(p2 + i);
            float32x4_t b_hi = vld1q_f32(p2 + i + 4);
            float32x4_t diff_lo = vsubq_f32(a_lo, b_lo);
            float32x4_t diff_hi = vsubq_f32(a_hi, b_hi);
            sum0 = vfmaq_f32(sum0, diff_lo, diff_lo);
            sum1 = vfmaq_f32(sum1, diff_hi, diff_hi);
        }

        sum0 = vaddq_f32(sum0, sum1);
        float result = vaddvq_f32(sum0);

        for (; i < dim; ++i)
        {
            float diff = static_cast<float>(p1[i]) - p2[i];
            result += diff * diff;
        }

        return result;
    }

    /*
     * [手术2] 预取辅助 — 提前把下一个邻居的向量数据拉进 L1
     */
    void prefetch_node_data(Node *node) const
    {
        if (!node) return;
        const char *p = reinterpret_cast<const char *>(node->data.data());
        __builtin_prefetch(p, 0, 3);
        __builtin_prefetch(p + 64, 0, 3);
        if (dim_ > 128)
        {
            __builtin_prefetch(p + 128, 0, 3);
            __builtin_prefetch(p + 192, 0, 3);
        }
        if (dim_ > 256)
        {
            for (int o = 256; o < 512; o += 64)
                __builtin_prefetch(p + o, 0, 3);
        }
        if (dim_ > 512)
        {
            for (int o = 512; o < 1920; o += 64)
                __builtin_prefetch(p + o, 0, 3);
        }
    }

    /*
     * search_layer — V1 原版逻辑 + [手术2] 流水线预取
     */
    vector<pair<float, Node *>> search_layer(
        Node *ep, const vector<float> &query, int ef, int layer_level)
    {
        using Pair = pair<float, Node *>;

        static thread_local vector<unsigned int> visited_array;
        static thread_local unsigned int visit_version = 0;
        static thread_local vector<Pair> candidates;
        static thread_local vector<Pair> results;
        static thread_local vector<Node *> local_neighbors_buf;

        visit_version++;
        if (visit_version == 0)
        {
            fill(visited_array.begin(), visited_array.end(), 0);
            visit_version = 1;
        }

        auto cmp_min_heap = [](const Pair &a, const Pair &b)
        { return a.first > b.first; };
        auto cmp_max_heap = [](const Pair &a, const Pair &b)
        { return a.first < b.first; };

        candidates.clear();
        results.clear();
        local_neighbors_buf.reserve(128);

        float d = distance(ep->data, query);
        candidates.push_back({d, ep});
        push_heap(candidates.begin(), candidates.end(), cmp_min_heap);
        results.push_back({d, ep});
        push_heap(results.begin(), results.end(), cmp_max_heap);

        if (ep->id >= (int)visited_array.size())
            visited_array.resize(ep->id + 10000, 0);
        visited_array[ep->id] = visit_version;

        while (!candidates.empty())
        {
            float closest_candidate_dist = candidates.front().first;
            float furthest_result_dist = results.front().first;
            if (closest_candidate_dist > furthest_result_dist)
                break;

            pop_heap(candidates.begin(), candidates.end(), cmp_min_heap);
            Node *c = candidates.back().second;
            candidates.pop_back();

            {
                SpinLockGuard lock(c->lock_);
                local_neighbors_buf.assign(c->neighbors[layer_level].begin(), c->neighbors[layer_level].end());
            }

            // [手术2] 流水线预取: 先预取前几个未访问邻居
            const int PA = 3;
            int pf = 0;
            for (size_t j = 0; j < local_neighbors_buf.size() && pf < PA; ++j)
            {
                Node *n = local_neighbors_buf[j];
                if (n->id >= (int)visited_array.size())
                    visited_array.resize(n->id + 10000, 0);
                if (visited_array[n->id] != visit_version)
                {
                    prefetch_node_data(n);
                    pf++;
                }
            }

            for (size_t j = 0; j < local_neighbors_buf.size(); ++j)
            {
                // [手术2] 处理第 j 个时，预取第 j+PA 个
                if (j + PA < local_neighbors_buf.size())
                {
                    Node *future = local_neighbors_buf[j + PA];
                    if (future->id >= (int)visited_array.size())
                        visited_array.resize(future->id + 10000, 0);
                    if (visited_array[future->id] != visit_version)
                        prefetch_node_data(future);
                }

                Node *neighbor = local_neighbors_buf[j];
                if (neighbor->id >= (int)visited_array.size())
                    visited_array.resize(neighbor->id + 10000, 0);
                if (visited_array[neighbor->id] != visit_version)
                {
                    visited_array[neighbor->id] = visit_version;
                    float dist_neighbor = distance(neighbor->data, query);

                    if (dist_neighbor < results.front().first || results.size() < (size_t)ef)
                    {
                        candidates.push_back({dist_neighbor, neighbor});
                        push_heap(candidates.begin(), candidates.end(), cmp_min_heap);

                        results.push_back({dist_neighbor, neighbor});
                        push_heap(results.begin(), results.end(), cmp_max_heap);

                        if (results.size() > (size_t)ef)
                        {
                            pop_heap(results.begin(), results.end(), cmp_max_heap);
                            results.pop_back();
                        }
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
        static thread_local SplitMix64 rng(hash<thread::id>{}(this_thread::get_id()));
        double r = rng.next_double();
        if (r == 0.0)
            r = 1e-10;
        int level = static_cast<int>(-log(r) * (1.0 / log((double)M_)));
        return min(level, max_layers_);
    }

    // ---- 插入: 与 V1 完全一致 ----
    void insert_thread_safe(const vector<float> &vec, int id)
    {
        int level = get_random_level();

        Float16Vec fp16_vec(vec.size());
        for (size_t i = 0; i < vec.size(); ++i)
        {
            fp16_vec[i] = static_cast<__fp16>(vec[i]);
        }

        auto new_node = make_shared<Node>(id, fp16_vec, level, M_);

        {
            SpinLockGuard lock(nodes_lock_);
            nodes_.push_back(new_node);
        }

        Node *curr_ep = enter_point_.value.load(memory_order_acquire);
        int curr_max_lvl = current_max_level_.value.load(memory_order_acquire);

        if (curr_ep == nullptr)
        {
            Node *expected_ep = nullptr;
            if (enter_point_.value.compare_exchange_strong(expected_ep, new_node.get(), memory_order_release, memory_order_relaxed))
            {
                current_max_level_.value.store(level, memory_order_release);
                return;
            }
            else
            {
                curr_ep = enter_point_.value.load(memory_order_acquire);
                curr_max_lvl = current_max_level_.value.load(memory_order_acquire);
            }
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

                vector<Node *> neighbors_copy;
                {
                    SpinLockGuard lock(curr_ep->lock_);
                    neighbors_copy = curr_ep->neighbors[lc];
                }

                for (Node *neighbor : neighbors_copy)
                {
                    if (neighbor->level < lc)
                        continue;

                    float d = distance(neighbor->data, vec);
                    if (d < d_min)
                    {
                        d_min = d;
                        curr_ep = neighbor;
                        changed = true;
                    }
                }
            }
        }

        for (int lc = min(curr_max_lvl, level); lc >= 0; lc--)
        {
            auto search_res = search_layer(curr_ep, vec, ef_construction_, lc);

            int max_connections = (lc == 0) ? (M_ * 2) : M_;

            int count = 0;
            for (size_t i = 0; i < search_res.size() && count < max_connections; i++)
            {
                Node *neighbor = search_res[i].second;

                {
                    SpinLockGuard lock(neighbor->lock_);
                    neighbor->neighbors[lc].push_back(new_node.get());

                    size_t layer_max = (size_t)max_connections;
                    if (neighbor->neighbors[lc].size() > layer_max)
                    {
                        float max_d = -1.0f;
                        int worst_idx = -1;
                        for (size_t j = 0; j < neighbor->neighbors[lc].size(); ++j)
                        {
                            float d = distance(neighbor->data, neighbor->neighbors[lc][j]->data);
                            if (d > max_d)
                            {
                                max_d = d;
                                worst_idx = j;
                            }
                        }
                        if (worst_idx != -1)
                        {
                            neighbor->neighbors[lc][worst_idx] = neighbor->neighbors[lc].back();
                            neighbor->neighbors[lc].pop_back();
                        }
                    }
                }
                {
                    SpinLockGuard lock(new_node->lock_);
                    new_node->neighbors[lc].push_back(neighbor);
                }
                count++;
            }
            if (!search_res.empty())
                curr_ep = search_res[0].second;
        }

        while (true)
        {
            int old_level = current_max_level_.value.load(memory_order_acquire);
            if (level <= old_level)
                break;
            if (current_max_level_.value.compare_exchange_weak(old_level, level, memory_order_release, memory_order_relaxed))
            {
                enter_point_.value.store(new_node.get(), memory_order_release);
                break;
            }
        }
    }

    // ---- 并行插入: V1 原版 ----
    void parallel_insert(const vector<vector<float>> &vectors, int num_threads)
    {
        int total = vectors.size();

        auto worker = [&](int start_idx, int end_idx)
        {
            for (int i = start_idx; i < end_idx; ++i)
            {
                this->insert_thread_safe(vectors[i], i);
            }
        };

        vector<thread> threads;
        int chunk_size = total / num_threads;

        for (int i = 0; i < num_threads; ++i)
        {
            int start = i * chunk_size;
            int end = (i == num_threads - 1) ? total : (i + 1) * chunk_size;
            threads.emplace_back(worker, start, end);
        }

        for (auto &t : threads)
            t.join();
    }

    /*
     * [手术1] search_knn — ef_search 参数正确暴露
     *
     * V1 bug: ef_search 硬编码为 max(k, 100)，search_k 从未生效
     * 修复: 接受第三个参数 ef_search，默认 0 时回退到 max(k, 100)
     *       → 完全兼容现有测试脚本
     */
    vector<int> search_knn(const vector<float> &query, int k, int ef_search = 0)
    {
        vector<int> result_ids;
        Node *curr_ep = enter_point_.value.load(memory_order_acquire);
        if (curr_ep == nullptr)
            return result_ids;

        // [手术1] 使用传入的 ef_search，默认回退到 V1 行为
        if (ef_search <= 0)
            ef_search = max(k, 100);
        ef_search = max(ef_search, k);

        int curr_max_lvl = current_max_level_.value.load(memory_order_acquire);
        float d_min = distance(curr_ep->data, query);
        for (int lc = curr_max_lvl; lc > 0; lc--)
        {
            bool changed = true;
            while (changed)
            {
                changed = false;
                if (curr_ep->level < lc)
                    break;

                vector<Node *> neighbors_copy;
                {
                    SpinLockGuard lock(curr_ep->lock_);
                    neighbors_copy = curr_ep->neighbors[lc];
                }
                for (Node *neighbor : neighbors_copy)
                {
                    if (neighbor->level < lc)
                        continue;

                    float d = distance(neighbor->data, query);
                    if (d < d_min)
                    {
                        d_min = d;
                        curr_ep = neighbor;
                        changed = true;
                    }
                }
            }
        }

        auto candidates = search_layer(curr_ep, query, ef_search, 0);

        for (int i = 0; i < min((int)candidates.size(), k); ++i)
        {
            result_ids.push_back(candidates[i].second->id);
        }
        return result_ids;
    }
};

vector<vector<float>> load_fvecs(const string &filepath)
{
    vector<vector<float>> data;
    ifstream file(filepath, ios::binary);
    if (!file.is_open())
        throw runtime_error("Could not open file: " + filepath);
    int dimension;
    while (file.read(reinterpret_cast<char *>(&dimension), sizeof(int)))
    {
        vector<float> vector_data(dimension);
        file.read(reinterpret_cast<char *>(vector_data.data()), dimension * sizeof(float));
        data.push_back(vector_data);
    }
    return data;
}

vector<vector<int>> load_ivecs(const string &filepath)
{
    vector<vector<int>> data;
    ifstream file(filepath, ios::binary);
    if (!file.is_open())
        throw runtime_error("Could not open file: " + filepath);
    int dimension;
    while (file.read(reinterpret_cast<char *>(&dimension), sizeof(int)))
    {
        vector<int> vector_data(dimension);
        file.read(reinterpret_cast<char *>(vector_data.data()), dimension * sizeof(int));
        data.push_back(vector_data);
    }
    return data;
}

PYBIND11_MODULE(hnsw_lib, m)
{
    m.doc() = "HNSW V4 final: V1 + Fixed ef_search + Pipeline Prefetch";
    m.def("load_fvecs", &load_fvecs);
    m.def("load_ivecs", &load_ivecs);

    py::class_<HNSW>(m, "HNSW")
        .def(py::init<int, int, size_t, size_t>(),
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("dim") = 128,
             py::arg("expected_elements") = 100000)
        .def("insert", &HNSW::insert_thread_safe, "Thread-safe Insert")
        .def("parallel_insert", &HNSW::parallel_insert, "Parallel Batch Insert")
        .def("search", &HNSW::search_knn,
             py::arg("query"),
             py::arg("k"),
             py::arg("ef_search") = 0);
}
