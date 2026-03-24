import matplotlib.pyplot as plt
import numpy as np

# 直接使用你终端跑出的最新真实数据，无需重新运行 C++ 测试
SIZES = [10000, 50000, 100000, 200000]
# 格式: [单线程耗时, 4线程耗时, 8线程耗时]
TIMES = {
    10000:  [2.0341,   0.6762,  0.5737],
    50000:  [23.3102,  7.2089,  6.1289],
    100000: [66.3088,  20.7349, 16.3897],
    200000: [177.4831, 61.6412, 48.5504]
}

# 提取各维度数据
t1_list = [TIMES[s][0] for s in SIZES]
t4_list = [TIMES[s][1] for s in SIZES]
t8_list = [TIMES[s][2] for s in SIZES]

size_labels = ['10k', '50k', '100k', '200k']
x_pos = np.arange(len(SIZES))

plt.style.use('seaborn-v0_8-whitegrid') # 使用高大上的学术网格风格

# =====================================================================
# 图 1：对数坐标尺度的可扩展性折线图 (复刻你的图2)
# =====================================================================
plt.figure(figsize=(10, 6))
plt.plot(x_pos, t1_list, marker='o', markersize=8, color='#A0A0A0', linestyle='--', linewidth=2, label='Single Thread')
plt.plot(x_pos, t4_list, marker='s', markersize=8, color='#3498DB', linestyle='-', linewidth=2.5, label='4-Thread Parallel')
plt.plot(x_pos, t8_list, marker='D', markersize=8, color='#E74C3C', linestyle='-', linewidth=2.5, label='8-Thread Parallel')

plt.yscale('log') # 设置对数坐标！非常适合展示量级差距
plt.title('Scalability Analysis: 1 vs 4 vs 8 Threads (10k-200k)', fontsize=14, fontweight='bold')
plt.xlabel('Dataset Size (Number of Vectors)', fontsize=12, fontweight='bold')
plt.ylabel('Construction Time (s) - Log Scale', fontsize=12, fontweight='bold')
plt.xticks(x_pos, size_labels, fontsize=11)

# 添加数值标签
for i in range(len(SIZES)):
    plt.annotate(f'{t1_list[i]:.1f}s', (x_pos[i], t1_list[i]), textcoords="offset points", xytext=(0,10), ha='center', color='#7F8C8D', fontweight='bold')
    plt.annotate(f'{t4_list[i]:.1f}s', (x_pos[i], t4_list[i]), textcoords="offset points", xytext=(0,10), ha='center', color='#2980B9', fontweight='bold')
    plt.annotate(f'{t8_list[i]:.1f}s', (x_pos[i], t8_list[i]), textcoords="offset points", xytext=(0,-15), ha='center', color='#C0392B', fontweight='bold')

plt.legend(loc='upper left', fontsize=11)
plt.savefig('final_scalability_log_scale.png', dpi=300, bbox_inches='tight')
plt.close()


# =====================================================================
# 图 2：并发效率一致性柱状图 (复刻你的图1)
# =====================================================================
speedups_8t = [t1_list[i] / t8_list[i] for i in range(len(SIZES))]
avg_speedup = sum(speedups_8t) / len(speedups_8t)

plt.figure(figsize=(9, 5.5))
bars = plt.bar(x_pos, speedups_8t, color='#9B59B6', width=0.5) # 使用高雅的紫色
plt.axhline(y=avg_speedup, color='#7F8C8D', linestyle='--', linewidth=2, label=f'Avg Speedup: {avg_speedup:.2f}x')

plt.title('Parallel Efficiency Consistency (8-Thread Speedup)', fontsize=14, fontweight='bold')
plt.xlabel('Dataset Size', fontsize=12, fontweight='bold')
plt.ylabel('Speedup Ratio', fontsize=12, fontweight='bold')
plt.xticks(x_pos, size_labels, fontsize=11)
plt.ylim(0, max(speedups_8t) * 1.3) # 留出顶部空间

# 在柱子上打标签 (紫色背景，白字)
for bar, sp in zip(bars, speedups_8t):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{sp:.2f}x', 
             ha='center', va='bottom', color='white', fontweight='bold',
             bbox=dict(facecolor='#8E44AD', edgecolor='none', boxstyle='round,pad=0.3'))

plt.legend(loc='upper right', fontsize=11)
plt.savefig('final_speedup_consistency.png', dpi=300, bbox_inches='tight')
plt.close()


# =====================================================================
# 图 3：额外加餐 - QPS衰减证明图 (用于论证 Cache Miss 和 复杂度)
# =====================================================================
qps_1t = [SIZES[i] / t1_list[i] for i in range(len(SIZES))]
qps_8t = [SIZES[i] / t8_list[i] for i in range(len(SIZES))]

plt.figure(figsize=(9, 5.5))
plt.plot(x_pos, qps_1t, marker='o', markersize=8, color='#7F8C8D', linestyle='-', linewidth=2, label='Single Thread QPS')
plt.plot(x_pos, qps_8t, marker='D', markersize=8, color='#E67E22', linestyle='-', linewidth=2.5, label='8-Thread QPS')

plt.title('System Throughput Degradation Analysis (Cache Miss Impact)', fontsize=14, fontweight='bold')
plt.xlabel('Dataset Size', fontsize=12, fontweight='bold')
plt.ylabel('Throughput (Queries Per Second)', fontsize=12, fontweight='bold')
plt.xticks(x_pos, size_labels, fontsize=11)

for i in range(len(SIZES)):
    plt.annotate(f'{int(qps_8t[i])}', (x_pos[i], qps_8t[i]), textcoords="offset points", xytext=(0,10), ha='center', color='#D35400', fontweight='bold')

plt.legend(loc='upper right', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('final_qps_degradation.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ 成功生成 3 张综合分析图！请查看当前目录：")
print("1. final_scalability_log_scale.png (对数折线图)")
print("2. final_speedup_consistency.png (紫色加速比柱状图)")
print("3. final_qps_degradation.png (橙色QPS衰减图 - 绝佳的论点配图)")
