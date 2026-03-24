import sys
import time
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# 确保能加载到编译好的 C++ 动态库
sys.path.append('./build')
try:
    import hnsw_lib
except ImportError as e:
    print(f"Error loading hnsw_lib: {e}")
    sys.exit(1)

# 数据集路径
BASE_PATH = os.path.join(os.path.dirname(__file__), "data/sift/sift_base.fvecs")
QUERY_PATH = os.path.join(os.path.dirname(__file__), "data/sift/sift_query.fvecs")
GT_PATH = os.path.join(os.path.dirname(__file__), "data/sift/sift_groundtruth.ivecs")
if not os.path.exists(BASE_PATH):
    print(f"Error: Dataset not found at {BASE_PATH}")
    sys.exit(1)
if not os.path.exists(QUERY_PATH):
    print(f"Error: Query set not found at {QUERY_PATH}")
    sys.exit(1)
if not os.path.exists(GT_PATH):
    print(f"Error: Ground truth not found at {GT_PATH}")
    sys.exit(1)

# 测试规模与线程配置
SIZES = [10000, 100000, 500000, 1000000]
MAX_SIZE = 1000000
THREADS = [1, 4, 8]

# [修复] k 固定为 10（返回 top-10），search_k 控制 ef_search（搜索深度）
RECALL_K = 10
EF_SEARCH_LIST = [10, 20, 50, 100, 200, 500]

EVAL_QUERY_LIMIT = 10000


def evaluate_search_quality_and_latency(index, queries, ground_truth, size, k, ef_search):
    """
    固定返回 top-k，用 ef_search 控制搜索深度。
    Recall@K 随 ef_search 增大而单调上升。
    """
    total_hits = 0
    total_gt = 0
    total_search_time = 0.0
    valid_query_count = 0

    for qvec, gt_ids in zip(queries, ground_truth):
        q_start = time.perf_counter()
        # [修复] 传入 ef_search 作为第三个参数
        pred_ids = index.search(qvec, k, ef_search)
        q_end = time.perf_counter()
        total_search_time += (q_end - q_start)
        valid_query_count += 1

        pred_set = set(pred_ids)

        # 子索引规模为 size 时，只统计真实存在于索引内的真值点
        valid_gt = [gid for gid in gt_ids[:k] if gid < size]
        if not valid_gt:
            continue

        total_hits += sum(1 for gid in valid_gt if gid in pred_set)
        total_gt += len(valid_gt)

    if total_gt == 0:
        recall = 0.0
    else:
        recall = total_hits / total_gt

    if valid_query_count == 0 or total_search_time <= 0:
        avg_query_ms = 0.0
        query_qps = 0.0
    else:
        avg_query_ms = (total_search_time / valid_query_count) * 1000.0
        query_qps = valid_query_count / total_search_time

    return recall, avg_query_ms, query_qps


def save_recall_qps_csv(rows, out_path):
    header = [
        "size",
        "threads",
        "ef_search",
        "build_time_s",
        "build_qps",
        "speedup",
        "recall",
        "query_latency_ms",
        "query_qps",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


print(f"--- [HNSW Performance Benchmark - SIFT] Loading {MAX_SIZE} vectors ---")
try:
    all_vectors = hnsw_lib.load_fvecs(BASE_PATH)[:MAX_SIZE]
    all_queries = hnsw_lib.load_fvecs(QUERY_PATH)
    all_ground_truth = hnsw_lib.load_ivecs(GT_PATH)
except Exception as e:
    print(f"Data load failed: {e}")
    sys.exit(1)

if EVAL_QUERY_LIMIT < 0:
    eval_count = min(len(all_queries), len(all_ground_truth))
else:
    eval_count = min(EVAL_QUERY_LIMIT, len(all_queries), len(all_ground_truth))
eval_queries = all_queries[:eval_count]
eval_ground_truth = all_ground_truth[:eval_count]

results = {}
csv_rows = []

print("-" * 140)
print(f"{'Size (N)':<10} | {'Threads':<8} | {'ef_search':<10} | {'Build(s)':<10} | {'Speedup':<9} | {'Build QPS':<10} | {'Recall@10':<10} | {'Query Lat(ms)':<14} | {'Query QPS':<10}")
print("-" * 140)

for size in SIZES:
    subset_vectors = all_vectors[:size]
    size_times = {}
    size_curves = {}

    for t_num in THREADS:
        index = hnsw_lib.HNSW(M=16, ef_construction=200)

        t0 = time.perf_counter()
        if t_num == 1:
            for i in range(size):
                index.insert(subset_vectors[i], i)
        else:
            index.parallel_insert(subset_vectors, t_num)
        t1 = time.perf_counter()

        cost = t1 - t0
        size_times[t_num] = cost
        baseline_t1 = size_times[THREADS[0]]
        speedup = baseline_t1 / cost
        qps = size / cost

        curve_points = []
        for ef_search in EF_SEARCH_LIST:
            # [修复] k 固定为 RECALL_K=10，ef_search 控制搜索深度
            recall, avg_q_ms, q_qps = evaluate_search_quality_and_latency(
                index,
                eval_queries,
                eval_ground_truth,
                size=size,
                k=RECALL_K,
                ef_search=ef_search,
            )
            point = {
                "ef_search": ef_search,
                "recall": recall,
                "query_lat_ms": avg_q_ms,
                "query_qps": q_qps,
            }
            curve_points.append(point)

            csv_rows.append(
                {
                    "size": size,
                    "threads": t_num,
                    "ef_search": ef_search,
                    "build_time_s": cost,
                    "build_qps": qps,
                    "speedup": speedup,
                    "recall": recall,
                    "query_latency_ms": avg_q_ms,
                    "query_qps": q_qps,
                }
            )
            print(f"{size:<10} | {t_num:<8} | {ef_search:<10} | {cost:<10.4f} | {speedup:<9.2f} | {qps:<10.0f} | {recall:<10.4f} | {avg_q_ms:<14.4f} | {q_qps:<10.1f}")

        size_curves[t_num] = curve_points

    results[size] = {
        "times": size_times,
        "curves": size_curves,
    }
    print("-" * 140)

csv_output = os.path.join(os.path.dirname(__file__), "recall_vs_qps_data_sift.csv")
save_recall_qps_csv(csv_rows, csv_output)

print("\nBenchmarking Complete. Generating Recall vs QPS charts...")

for size, metrics in results.items():
    plt.figure(figsize=(8, 5))
    for t_num in THREADS:
        points = metrics["curves"][t_num]
        x_qps = [p["query_qps"] for p in points]
        y_recall = [p["recall"] for p in points]

        plt.plot(
            x_qps,
            y_recall,
            marker="o",
            linewidth=2,
            label=f"Threads={t_num}",
        )
        for p in points:
            plt.annotate(
                f"ef={p['ef_search']}",
                (p["query_qps"], p["recall"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    plt.title(f"SIFT Recall@10 vs QPS (N={size})", fontsize=14, fontweight="bold")
    plt.xlabel("Query Throughput (QPS)", fontsize=12)
    plt.ylabel("Recall@10", fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")
    plt.savefig(f"chart_sift_recall_vs_qps_{size}.png", dpi=300, bbox_inches="tight")
    plt.close()

print("All done. Data saved to recall_vs_qps_data_sift.csv")
