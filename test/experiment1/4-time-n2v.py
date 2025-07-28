import matplotlib.pyplot as plt
import numpy as np

# 字体配置
config = {
    "font.family": 'Times New Roman',
    "font.size": 18,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
plt.rcParams.update(config)

# 柱宽
bar_width = 0.22

# 数据
with open('./baseline.csv', 'r') as f:
    baselines = [float(line.strip()) for line in f.readlines()]
with open('./result.csv', 'r') as f:
    lines = [float(line.strip()) for line in f.readlines()]

# 原始时间数据
cggraph_time = baselines[0:5]
lighttraffic_time = baselines[5:10]
sowalker_time = baselines[10:15]
gowalker_time = lines[0:5]

# 转换为倍数（cggraph自身保持1）
cggraph_bar = [1] * 5
lighttraffic_bar = [lt / cg for cg, lt in zip(cggraph_time,lighttraffic_time)] 
sowalker_bar = [so / cg for cg, so in zip(cggraph_time,sowalker_time)]
cggraph_time=[113.095,890.95,563.595,2140.9,7576.467222]
gowalker_bar = [cg / go for cg, go in zip(cggraph_time,gowalker_time)]
dataset = ["TW", "FR", "UK", "YH", "K30"]

# 横坐标起点
x = np.arange(len(dataset))
x1 = x - 1.5 * bar_width
x2 = x - 0.5 * bar_width
x3 = x + 0.5 * bar_width
x4 = x + 1.5 * bar_width

fig, axes1 = plt.subplots(figsize=(8, 5))

# 柱状图绘制
lns1 = axes1.bar(x1, cggraph_bar, color='#ED7D31', width=bar_width,
                 label=r"$\mathrm{CGgraph}$", edgecolor='black')
lns2 = axes1.bar(x2, lighttraffic_bar, color='#4672C4', width=bar_width,
                 label=r"$\mathrm{LightTraffic}$", edgecolor='black')
lns3 = axes1.bar(x3, sowalker_bar, color='#70AD47', width=bar_width,
                 label=r"$\mathrm{SOWalker}$", edgecolor='black')
lns4 = axes1.bar(x4, gowalker_bar, color='#C00000', width=bar_width,
                 label=r"$\mathrm{GOWalker}$", edgecolor='black')

# x轴标签设置（居中）
axes1.set_xticks(x)
axes1.set_xticklabels(dataset, rotation=0)

# y轴格式设置
# axes1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
axes1.set_ylabel("Normalized Runtime")
axes1.grid(True, axis='y', linestyle=':')
axes1.set_ylim(0, 3)

# 图例设置
axes1.legend(handles=[lns1, lns2, lns3, lns4],
             prop={'size': 15},
             ncol=4,
             bbox_to_anchor=(0.5, 0.84),
             loc=9,
             borderaxespad=-2.4,
             columnspacing=0.5,
             handletextpad=0.4)

# 保存与显示
plt.savefig("4-time-n2v.svg", dpi=300, bbox_inches="tight")
plt.show()
