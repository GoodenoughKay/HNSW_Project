import sys
import time
import os
import csv

BASE_PATH = os.path.join(os.path.dirname(__file__), "data/sift/sift_base.fvecs")
QUERY_PATH = os.path.join(os.path.dirname(__file__), "data/sift/sift_query.fvecs")
GT_PATH = os.path.join(os.path.dirname(__file__), "data/sift/sift_groundtruth.ivecs")

SIZES = [10000, 100000]
THREADS = [1, 8]
EF_SEARCH_LIST = [10, 100]
RECALL_K = 10
EVAL_QUERY_LIMIT = 1000

def load_module(so_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("hnsw_lib", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def evaluate(index, queries, ground_truth, size, k, ef_search):
    total_hits, total_gt, total_time, count = 0, 0, 0.0, 0
    for qvec, gt_ids in zip(queries, ground_truth):
        t0 = time.perf_counter()
        pred_ids = index.search(qvec, k, ef_search)
        total_time += time.perf_counter() - t0
        count += 1
        pred_set = set(pred_ids)
        valid_gt = [gid for gid in gt_ids[:k] if gid < size]
        if valid_gt:
            total_hits += sum(1 for gid in valid_gt if gid in pred_set)
            total_gt += len(valid_gt)
    recall = total_hits / total_gt if total_gt > 0 else 0.0
    avg_ms = (total_time / count) * 1000 if count > 0 else 0
    qps = count / total_time if total_time > 0 else 0
    return recall, avg_ms, qps

def run_benchmark(hnsw_mod, label, all_vectors, eval_queries, eval_gt):
    results = []
    for size in SIZES:
        subset = all_vectors[:size]
        for t_num in THREADS:
            index = hnsw_mod.HNSW(M=16, ef_construction=200)
            t0 = time.perf_counter()
            if t_num == 1:
                for i in range(size):
                    index.insert(subset[i], i)
            else:
                index.parallel_insert(subset, t_num)
            build_time = time.perf_counter() - t0

            for ef in EF_SEARCH_LIST:
                recall, avg_ms, qps = evaluate(index, eval_queries, eval_gt, size, RECALL_K, ef)
                row = {'version': label, 'size': size, 'threads': t_num, 'ef_search': ef,
                       'build_time_s': round(build_time, 4), 'build_qps': round(size/build_time),
                       'recall': round(recall, 4), 'query_qps': round(qps), 'query_lat_ms': round(avg_ms, 4)}
                results.append(row)
                print(f"  [{label:>12}] N={size:>7} t={t_num} ef={ef:>3} "
                      f"build={build_time:>8.2f}s recall={recall:.4f} QPS={qps:>8.0f}")
    return results

def main():
    print("=" * 70)
    print("HNSW Ablation Study: Baseline vs V4 Optimized")
    print("=" * 70)

    baseline_so = os.path.join(os.path.dirname(__file__), "hnsw_baseline.so")
    v4_so = os.path.join(os.path.dirname(__file__), "build", "hnsw_lib.so")

    # 用 V4 模块加载数据
    v4_mod = load_module(v4_so)
    print(f"\nLoading data...")
    all_vectors = v4_mod.load_fvecs(BASE_PATH)[:max(SIZES)]
    all_queries = v4_mod.load_fvecs(QUERY_PATH)[:EVAL_QUERY_LIMIT]
    all_gt = v4_mod.load_ivecs(GT_PATH)[:EVAL_QUERY_LIMIT]
    print(f"Loaded {len(all_vectors)} vectors, {len(all_queries)} queries")

    baseline_mod = load_module(baseline_so)

    print(f"\n{'='*70}")
    print("Phase 1: BASELINE (scalar float32, std::mutex, no NEON)")
    print(f"{'='*70}")
    bl_results = run_benchmark(baseline_mod, "baseline", all_vectors, all_queries, all_gt)

    print(f"\n{'='*70}")
    print("Phase 2: V4 OPTIMIZED (FP16 NEON, spinlock, prefetch)")
    print(f"{'='*70}")
    v4_results = run_benchmark(v4_mod, "v4_optimized", all_vectors, all_queries, all_gt)

    # 保存 CSV
    all_results = bl_results + v4_results
    csv_path = os.path.join(os.path.dirname(__file__), "ablation_results.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_results[0].keys())
        w.writeheader()
        w.writerows(all_results)

    # 打印对比
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<14} {'Config':<22} {'Baseline':>12} {'V4 Opt':>12} {'Speedup':>10}")
    print("-" * 70)
    for size in SIZES:
        for t in THREADS:
            for ef in EF_SEARCH_LIST:
                bl = [r for r in bl_results if r['size']==size and r['threads']==t and r['ef_search']==ef]
                v4 = [r for r in v4_results if r['size']==size and r['threads']==t and r['ef_search']==ef]
                if bl and v4:
                    bl, v4 = bl[0], v4[0]
                    tag = f"N={size//1000}K t={t} ef={ef}"
                    if ef == EF_SEARCH_LIST[0]:
                        sp = bl['build_time_s'] / v4['build_time_s'] if v4['build_time_s'] > 0 else 0
                        print(f"  {'Build':<12} {tag:<22} {bl['build_time_s']:>10.2f}s {v4['build_time_s']:>10.2f}s {sp:>9.1f}x")
                    sp = v4['query_qps'] / bl['query_qps'] if bl['query_qps'] > 0 else 0
                    print(f"  {'Query QPS':<12} {tag:<22} {bl['query_qps']:>12.0f} {v4['query_qps']:>12.0f} {sp:>9.1f}x")
                    print(f"  {'Recall':<12} {tag:<22} {bl['recall']:>12.4f} {v4['recall']:>12.4f}")
        print()

    print(f"CSV saved to {csv_path}")

if __name__ == "__main__":
    main()