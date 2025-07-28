import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def auto_text(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, f'{height:.1f}',
                ha='center', va='bottom', fontsize=12)

config = {
    "font.family": 'Times New Roman',
    "font.size": 18,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
plt.rcParams.update(config)

bar_width = 0.22

with open('./test/ex2/total.csv', 'r') as f:
    totallines = [float(line.strip()) for line in f.readlines()]
total_time = lines[0:3]
transfer_time = []
gpu_time = []
cpu_time = []

with open('./test/ex2/nopipeline.csv', 'r') as f:
    for line in f:
        a, b, c = map(float, line.strip().split(','))
        transfer_time.append(a)
        gpu_time.append(b)
        cpu_time.append(c)

# 原始柱子高度
total_bar = [1, 1, 1]
# node2vec 拆分为三部分
n2v_cpu = [t / c for c, t in zip(cpu_time, total_time)]
n2v_gpu = [t / g for g, t in zip(gpu_time, total_time)]
n2v_transfer = [t / s for s, t in zip(transfer_time, total_time)]

dataset = ["TW", "FR", "UK"]
x = np.arange(len(dataset))
x2 = x - 0.5 * bar_width
x1 = x + 0.5 * bar_width

fig, ax = plt.subplots(figsize=(6, 4.5))

lns1 = ax.bar(x2, total_bar, color='#C00000', width=bar_width,
              label="Total", edgecolor='black')

cpu_bar = ax.bar(x1, n2v_cpu, width=bar_width, color='#4672C4', label="CPU", edgecolor='black')
gpu_bar = ax.bar(x1, n2v_gpu, bottom=n2v_cpu, width=bar_width, color='#ED7D31', label="GPU", edgecolor='black')
transfer_bar = ax.bar(x1, n2v_transfer, bottom=np.array(n2v_cpu)+np.array(n2v_gpu),
                      width=bar_width, edgecolor='black', color='#70AD47', label='Transfer')




# 轴标签与网格
ax.set_xticks(x)
ax.set_xticklabels(dataset)
# ax.set_xlabel("Dataset")
ax.set_ylabel("Normalized Runtime")
ax.set_ylim(0, 2.1)
ax.grid(True, axis='y', linestyle=':', alpha=0.7)

# 图例
ax.legend(handles=[lns1, cpu_bar, gpu_bar, transfer_bar],
          loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4,
          fontsize=15, columnspacing=0.6, handletextpad=0.6, labelspacing=0.15)

plt.tight_layout()
plt.savefig("4-pipeline-n2v.svg", dpi=300, bbox_inches="tight")
plt.show()
