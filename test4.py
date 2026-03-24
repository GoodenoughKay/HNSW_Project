import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Rendering setup for macOS with English labels
# ==========================================
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300  
plt.rcParams['savefig.dpi'] = 300

sns.set_theme(style="whitegrid", font="DejaVu Sans")

# Load benchmark data up to 1M vectors
df = pd.read_csv('recall_vs_qps_data.csv')

# Build-stage metrics are independent of search_k; use search_k=1 rows as baseline
df_build = df[df['search_k'] == 1].copy()
sizes = ['10k', '100k', '500k', '1M']

# ==========================================
# Figure 4-1: Build time (log scale)
# ==========================================
def plot_fig_4_1():
    plt.figure(figsize=(8, 6))
    t1 = df_build[df_build['threads'] == 1]['build_time_s'].values
    t4 = df_build[df_build['threads'] == 4]['build_time_s'].values
    t8 = df_build[df_build['threads'] == 8]['build_time_s'].values

    plt.plot(sizes, t1, marker='o', color='#95a5a6', linestyle='--', label='1 Thread', linewidth=2)
    plt.plot(sizes, t4, marker='s', color='#3498db', label='4 Threads', linewidth=2.5)
    plt.plot(sizes, t8, marker='D', color='#e74c3c', label='8 Threads', linewidth=2.5)

    plt.yscale('log')
    plt.title('Figure 4-1: Build Time Scalability Across Thread Counts (10k - 1M)', fontsize=14, pad=15)
    plt.xlabel('Dataset Size (Vectors)', fontsize=12)
    plt.ylabel('Build Time (s, Log Scale)', fontsize=12)

    for i, txt in enumerate(t1): plt.annotate(f"{txt:.1f}s", (sizes[i], t1[i]), xytext=(0, 10), textcoords="offset points", ha='center', color='#7f8c8d')
    for i, txt in enumerate(t4): plt.annotate(f"{txt:.1f}s", (sizes[i], t4[i]), xytext=(0, 10), textcoords="offset points", ha='center', color='#2980b9')
    for i, txt in enumerate(t8): plt.annotate(f"{txt:.1f}s", (sizes[i], t8[i]), xytext=(0, -15), textcoords="offset points", ha='center', color='#c0392b')

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('fig_4_1_build_time.png')
    print("Figure 4-1 generated: fig_4_1_build_time.png")

# ==========================================
# Figure 4-2: Multithread speedup bar chart
# ==========================================
def plot_fig_4_2():
    plt.figure(figsize=(8, 6))
    speedups = df_build[df_build['threads'] == 8]['speedup'].values
    
    bars = plt.bar(sizes, speedups, color='#8e44ad', width=0.5, alpha=0.85)
    plt.title('Figure 4-2: Multithread Speedup (8 Threads vs 1 Thread)', fontsize=14, pad=15)
    plt.xlabel('Dataset Size (Vectors)', fontsize=12)
    plt.ylabel('Speedup Ratio', fontsize=12)
    plt.ylim(0, 5) 
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'{yval:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold', color='white', bbox=dict(facecolor='#8e44ad', edgecolor='none', boxstyle='round,pad=0.2'))
                 
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('fig_4_2_speedup.png')
    print("Figure 4-2 generated: fig_4_2_speedup.png")

# ==========================================
# Figure 4-3: Build throughput trend
# ==========================================
def plot_fig_4_3():
    plt.figure(figsize=(8, 6))
    q1 = df_build[df_build['threads'] == 1]['build_qps'].values
    q8 = df_build[df_build['threads'] == 8]['build_qps'].values

    plt.plot(sizes, q1, marker='o', color='#7f8c8d', label='1-Thread QPS', linewidth=2)
    plt.plot(sizes, q8, marker='D', color='#e67e22', label='8-Thread QPS', linewidth=2.5)

    plt.title('Figure 4-3: Build Throughput Degradation vs Dataset Size', fontsize=14, pad=15)
    plt.xlabel('Dataset Size (Vectors)', fontsize=12)
    plt.ylabel('Build Throughput (QPS)', fontsize=12)

    for i, txt in enumerate(q1): plt.annotate(f"{int(txt)}", (sizes[i], q1[i]), xytext=(0, -15), textcoords="offset points", ha='center', color='#7f8c8d')
    for i, txt in enumerate(q8): plt.annotate(f"{int(txt)}", (sizes[i], q8[i]), xytext=(0, 10), textcoords="offset points", ha='center', color='#d35400', fontweight='bold')

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('fig_4_3_build_qps.png')
    print("Figure 4-3 generated: fig_4_3_build_qps.png")

# ==========================================
# Figure 4-4: Recall vs QPS Pareto frontier (1M)
# ==========================================
def plot_fig_4_4():
    df_pareto = df[(df['size'] == 1000000) & (df['threads'] == 8)].copy()
    
    plt.figure(figsize=(8, 6))
    plt.plot(df_pareto['recall'], df_pareto['query_qps'], marker='o', markersize=8, linewidth=2.5, color='#e74c3c')
    
    for i, row in df_pareto.iterrows():
        plt.annotate(f"k={int(row['search_k'])}", (row['recall'], row['query_qps']), textcoords="offset points", xytext=(10, 5), ha='left', fontsize=10)

    plt.title('Figure 4-4: Recall vs QPS Pareto Frontier at N=1,000,000', fontsize=14, pad=15)
    plt.xlabel('Recall@10', fontsize=12)
    plt.ylabel('Query Throughput (QPS)', fontsize=12)
    plt.tight_layout()
    plt.savefig('fig_4_4_pareto.png')
    print("Figure 4-4 generated: fig_4_4_pareto.png")

# ==========================================
# Figure 4-5: Query latency scaling trend
# ==========================================
def plot_fig_4_5():
    df_latency = df[(df['threads'] == 8) & (df['search_k'] == 10)].copy()
    
    plt.figure(figsize=(8, 6))
    plt.plot(df_latency['size'], df_latency['query_latency_ms'], marker='s', markersize=8, linewidth=2.5, color='#2980b9')
    
    plt.xscale('log')
    plt.title('Figure 4-5: Query Latency Scaling with Dataset Size $N$ (search_k=10)', fontsize=14, pad=15)
    plt.xlabel('Dataset Size (Log Scale)', fontsize=12)
    plt.ylabel('Average Query Latency (ms)', fontsize=12)
    
    plt.xticks([10000, 100000, 500000, 1000000], ['10k', '100k', '500k', '1M'])
    
    for i, row in df_latency.iterrows():
        plt.annotate(f"{row['query_latency_ms']:.3f} ms", (row['size'], row['query_latency_ms']), textcoords="offset points", xytext=(-15, 10), ha='center', fontsize=10, color='#2c3e50')

    plt.tight_layout()
    plt.savefig('fig_4_5_query_latency.png')
    print("Figure 4-5 generated: fig_4_5_query_latency.png")

if __name__ == '__main__':
    plot_fig_4_1()
    plot_fig_4_2()
    plot_fig_4_3()
    plot_fig_4_4()
    plot_fig_4_5()
    print("\nAll 5 figures have been generated successfully.")