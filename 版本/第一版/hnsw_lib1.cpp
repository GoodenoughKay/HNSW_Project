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

// SplitMix64 随机数生成器，用于采样节点层级
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

    // 生成 [0.0, 1.0) 的浮点数
    double next_double()
    {
        return (next() >> 11) * (1.0 / (1ULL << 53));
    }
};

// 用 FP16 存储向量，内存减半，SIMD 一次可以处理更多维度
using Float16Vec = vector<__fp16>;

struct alignas(64) Node // 64 字节对齐，避免多线程时 false sharing
{
    atomic_flag lock_ = ATOMIC_FLAG_INIT;
    int id;
    int level;
    Float16Vec data; // FP16 格式存储

    vector<vector<Node *>> neighbors; // 每层的邻居列表

    // 第 0 层最大连接数是 2*M，高层是 M，按论文的设定
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

// 自旋锁，用 RAII 管理加解锁
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
    int M_;               // 每层最大连接数
    int ef_construction_; // 构建时的候选集大小
    int max_layers_;      // 最大层数
    size_t dim_;          // 向量维度

    vector<shared_ptr<Node>> nodes_;
    atomic_flag nodes_lock_ = ATOMIC_FLAG_INIT;

    // 用 alignas 把这两个原子变量隔离到不同缓存行，防止相互干扰
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

    // FP16-FP16 欧氏距离，NEON SIMD 4路展开
    float distance(const Float16Vec &v1, const Float16Vec &v2)
    {
        const size_t dim = dim_;
        const __fp16 *p1 = v1.data();
        const __fp16 *p2 = v2.data();

        // 用 FP32 累加，防止 FP16 精度不够溢出
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

    // 查询时节点是 FP16，查询向量是 FP32，单独处理
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

    // 在某一层做贪心搜索，返回最近的 ef 个节点
    vector<pair<float, Node *>> search_layer(
        Node *ep, const vector<float> &query, int ef, int layer_level)
    {
        using Pair = pair<float, Node *>;

        // 用线程本地变量复用内存，避免每次搜索都重新分配
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

        // candidates 是最小堆，results 是最大堆
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
            // 候选集里最近的节点比结果集里最远的还要远，说明没有继续搜索的必要了
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

            for (Node *neighbor : local_neighbors_buf)
            {
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
            r = 1e-10; // 避免 log(0)
        int level = static_cast<int>(-log(r) * (1.0 / log((double)M_)));
        return min(level, max_layers_);
    }

    void insert_thread_safe(const vector<float> &vec, int id)
    {
        int level = get_random_level();

        // 输入是 FP32，转成 FP16 再存
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
            // 第一个节点，CAS 设置入口点
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

        // 在高于新节点层级的层上做贪心搜索，找到合适的插入入口
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

        // 从新节点所在层往下，每层搜索并建立双向连接
        for (int lc = min(curr_max_lvl, level); lc >= 0; lc--)
        {
            auto search_res = search_layer(curr_ep, vec, ef_construction_, lc);

            // 第 0 层最大连接数是 2*M，高层是 M
            int max_connections = (lc == 0) ? (M_ * 2) : M_;

            int count = 0;
            for (size_t i = 0; i < search_res.size() && count < max_connections; i++)
            {
                Node *neighbor = search_res[i].second;

                {
                    SpinLockGuard lock(neighbor->lock_);
                    neighbor->neighbors[lc].push_back(new_node.get());

                    // 超出上限就删掉距离最远的那条边
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

        // 如果新节点层级更高，更新全局入口点
        while (true)
        {
            int old_level = current_max_level_.value.load(memory_order_acquire);
            if (level <= old_level)
                break;
            if (current_max_level_.value.compare_exchange_weak(old_level, level,
                                                               memory_order_release, memory_order_relaxed))
            {
                enter_point_.value.store(new_node.get(), memory_order_release);
                break;
            }
        }
    }

    // 多线程并行插入，按线程数平均分配任务
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

    vector<int> search_knn(const vector<float> &query, int k)
    {
        vector<int> result_ids;
        Node *curr_ep = enter_point_.value.load(memory_order_acquire);
        if (curr_ep == nullptr)
            return result_ids;

        int curr_max_lvl = current_max_level_.value.load(memory_order_acquire);
        float d_min = distance(curr_ep->data, query);

        // 高层贪心搜索，找到第 0 层的入口
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

        int ef_search = max(k, 100);
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
    m.doc() = "HNSW C++ Library with FP16 SIMD, Lock-Free CAS & M3-Optimized Cache Layout";
    m.def("load_fvecs", &load_fvecs);
    m.def("load_ivecs", &load_ivecs);

    py::class_<HNSW>(m, "HNSW")
        .def(py::init<int, int, size_t, size_t>(),
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("dim") = 128,
             py::arg("expected_elements") = 100000)
        .def("insert", &HNSW::insert_thread_safe, "Thread-safe Insert with FP16 conversion")
        .def("parallel_insert", &HNSW::parallel_insert, "Parallel Batch Insert")
        .def("search", &HNSW::search_knn, "Search Top-K with FP16-FP32 mixed precision");
}
