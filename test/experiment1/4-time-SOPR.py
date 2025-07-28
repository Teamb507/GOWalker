import matplotlib.pyplot as plt
import numpy as np

config = {
    "font.family": 'Times New Roman',
    "font.size": 18,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
plt.rcParams.update(config)

bar_width = 0.22

with open('./test/experiment1/result.csv', 'r') as f:
    lines = [float(line.strip()) for line in f.readlines()]

cggraph_time = [311.169,47.4922,117.947,535.213,3000]
gowalker_time = lines[0:5]

cggraph_bar = [1, 1, 1, 1, 1]
lighttraffic_bar = [2.078424464,
                    2.058777947,
                    1.488168415,
                    2.37,
                    1.73]
sowalker_bar = [2.329923611,
                2.028975284,
                2.9391252,
                2.24,
                1.53]
gowalker_bar = [ cg /go for go, cg in zip(gowalker_time, cggraph_time)]
dataset = ["TW", "FR", "UK", "YH", "K30"]

x = np.arange(len(dataset))
x1 = x - 1.5 * bar_width
x2 = x - 0.5 * bar_width
x3 = x + 0.5 * bar_width
x4 = x + 1.5 * bar_width

fig, axes1 = plt.subplots(figsize=(8, 5))

lns1 = axes1.bar(x1, cggraph_bar, color='#ED7D31', width=bar_width,
                 label=r"$\mathrm{CGgraph}$", edgecolor='black')
lns2 = axes1.bar(x2, lighttraffic_bar, color='#4672C4', width=bar_width,
                 label=r"$\mathrm{LightTraffic}$", edgecolor='black')
lns3 = axes1.bar(x3, sowalker_bar, color='#70AD47', width=bar_width,
                 label=r"$\mathrm{SOWalker}$", edgecolor='black')
lns4 = axes1.bar(x4, gowalker_bar, color='#C00000', width=bar_width,
                 label=r"$\mathrm{GOWalker}$", edgecolor='black')

axes1.set_xticks(x)
axes1.set_xticklabels(dataset, rotation=0)

axes1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
axes1.set_ylabel("Normalized Runtime")
axes1.grid(True, axis='y', linestyle=':')
axes1.set_ylim(0, 3.5)

axes1.legend(handles=[lns1, lns2, lns3, lns4],
             prop={'size': 15},
             ncol=4,
             bbox_to_anchor=(0.5, 0.84),
             loc=9,
             borderaxespad=-2.4,
             columnspacing=0.5,
             handletextpad=0.4)

plt.savefig("4-time-pr.svg", dpi=300, bbox_inches="tight")
plt.show()
