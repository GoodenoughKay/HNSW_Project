import sys
import time
import os
import csv
import numpy as np

# 确保能加载到编译好的 C++ 动态库
sys.path.append('./build')
try:
    import hnsw_lib
except ImportError as e:
    print(f"Error loading hnsw_lib: {e}")
    sys.exit(1)

# GIST数据集路径
BASE_PATH = os.path.join(os.path.dirname(__file__), "data/gist/gist_base.fvecs")
if not os.path.exists(BASE_PATH):
    print(f"Error: Dataset not found at {BASE_PATH}")
    sys.exit(1)

# 测试规模与线程配置
SIZES = [10000, 100000, 500000, 1000000]
MAX_SIZE = 1000000
THREADS = [1, 4, 8]

# [修复] k 固定为 10，ef_search 控制搜索深度
RECALL_K = 10
EF_SEARCH_LIST = [10, 20, 50, 100, 200, 500]

# 采样查询数量
EVAL_QUERY_COUNT = 1000


def precompute_ground_truth(queries, base_vectors, k=10):
    """
    预计算暴力搜索 ground truth（只做一次，不随 ef_search 重复计算）
    使用 numpy 批量计算加速
    """
    print(f"  Precomputing brute-force ground truth for {len(queries)} queries (k={k})...")
    t0 = time.perf_counter()

    gt = []
    for i, qvec in enumerate(queries):
        # 向量化距离计算
        distances = np.sum((base_vectors - qvec) ** 2, axis=1)
        gt_ids = np.argsort(distances)[:k]
        gt.append(gt_ids.tolist())

        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (len(queries) - i - 1)
            print(f"    {i+1}/{len(queries)} done, ETA: {eta:.1f}s")

    t1 = time.perf_counter()
    print(f"  Ground truth computed in {t1-t0:.1f}s")
    return gt


def evaluate_recall_and_latency(index, queries, ground_truth, k, ef_search):
    """
    固定 k，用 ef_search 控制搜索深度。
    """
    total_hits = 0
    total_gt = 0
    total_search_time = 0.0

    for qvec, gt_ids in zip(queries, ground_truth):
        q_start = time.perf_counter()
        pred_ids = index.search(qvec, k, ef_search)
        q_end = time.perf_counter()
        total_search_time += (q_end - q_start)

        pred_set = set(pred_ids)
        hits = sum(1 for gid in gt_ids[:k] if gid in pred_set)
        total_hits += hits
        total_gt += len(gt_ids[:k])

    num_queries = len(queries)
    if total_gt == 0:
        recall = 0.0
    else:
        recall = total_hits / total_gt

    if num_queries == 0 or total_search_time <= 0:
        avg_query_ms = 0.0
        query_qps = 0.0
    else:
        avg_query_ms = (total_search_time / num_queries) * 1000.0
        query_qps = num_queries / total_search_time

    return recall, avg_query_ms, query_qps


def save_csv(rows, out_path):
    header = [
        "size", "dim", "threads", "ef_search",
        "build_time_s", "build_qps", "speedup",
        "recall", "query_latency_ms", "query_qps",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print(f"--- [HNSW Performance Benchmark - GIST] ---")
    print(f"Loading GIST dataset from {BASE_PATH}...")

    all_vectors = hnsw_lib.load_fvecs(BASE_PATH)[:MAX_SIZE]
    all_vectors = [np.array(v, dtype=np.float32) for v in all_vectors]

    total_vectors = len(all_vectors)
    dim = len(all_vectors[0])
    print(f"Loaded {total_vectors} vectors with dimension {dim}")

    M = 16
    ef_construction = 200
    csv_rows = []

    print("-" * 140)
    print(f"{'Size (N)':<10} | {'Threads':<8} | {'ef_search':<10} | {'Build(s)':<10} | {'Speedup':<9} | {'Build QPS':<10} | {'Recall@10':<10} | {'Query Lat(ms)':<14} | {'Query QPS':<10}")
    print("-" * 140)

    for size in SIZES:
        if size > total_vectors:
            print(f"Skipping size {size} (exceeds dataset size {total_vectors})")
            continue

        data_subset = all_vectors[:size]
        base_np = np.array(data_subset, dtype=np.float32)

        # 随机选择查询向量
        np.random.seed(42)  # 固定种子，结果可复现
        query_indices = np.random.choice(size, min(EVAL_QUERY_COUNT, size), replace=False)
        queries = [data_subset[i] for i in query_indices]

        # 预计算 ground truth（只做一次）
        ground_truth = precompute_ground_truth(queries, base_np, k=RECALL_K)

        size_times = {}

        for num_threads in THREADS:
            # 创建索引
            index = hnsw_lib.HNSW(M=M, ef_construction=ef_construction, dim=dim, expected_elements=size)

            build_start = time.perf_counter()
            if num_threads == 1:
                for i in range(size):
                    index.insert(data_subset[i], i)
            else:
                index.parallel_insert(data_subset, num_threads)
            build_end = time.perf_counter()

            build_time = build_end - build_start
            build_qps = size / build_time
            size_times[num_threads] = build_time
            baseline = size_times[THREADS[0]]
            speedup = baseline / build_time

            # 测试不同 ef_search
            for ef_search in EF_SEARCH_LIST:
                recall, avg_q_ms, q_qps = evaluate_recall_and_latency(
                    index, queries, ground_truth,
                    k=RECALL_K, ef_search=ef_search,
                )

                csv_rows.append({
                    "size": size, "dim": dim, "threads": num_threads,
                    "ef_search": ef_search,
                    "build_time_s": build_time, "build_qps": build_qps,
                    "speedup": speedup,
                    "recall": recall,
                    "query_latency_ms": avg_q_ms, "query_qps": q_qps,
                })

                print(f"{size:<10} | {num_threads:<8} | {ef_search:<10} | {build_time:<10.2f} | {speedup:<9.2f} | {build_qps:<10.0f} | {recall:<10.4f} | {avg_q_ms:<14.4f} | {q_qps:<10.1f}")

        print("-" * 140)

    csv_output = os.path.join(os.path.dirname(__file__), "recall_vs_qps_data_gist.csv")
    save_csv(csv_rows, csv_output)
    print(f"\nAll done. Data saved to {csv_output}")


if __name__ == "__main__":
    main()
