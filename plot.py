import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42
import seaborn as sns
sns.set(style='whitegrid')

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 5))

barwidth = 0.2
r1 = np.arange(5) * 1.
r2 = [x + barwidth for x in r1]
r3 = [x + barwidth*2 for x in r1]
rs = [r1,r2,r3]

results = np.load(f"summary.npy", allow_pickle=True).item()

methods = ["top", "tocp", "greedy"]

colors = [
    "tab:red",
    "tab:green",
    "tab:blue",
]

fs=12
## plot cost ##
for t, H in enumerate([2,4,6,8,10]):
    for i, m in enumerate(methods):
        cost_mean = results[H][m]['cost_mean']
        axs[0].bar(rs[i][t], cost_mean, width=barwidth, color=colors[i], edgecolor='black', capsize=5, alpha=1., linewidth=2.5)
axs[0].set_yticks([0, 2, 4, 6])
axs[0].set_ylabel("Avg. Cost", fontsize=fs)
axs[0].set_ylim(0, 6)

## plot time ##
for t, H in enumerate([2,4,6,8,10]):
    for i, m in enumerate(methods):
        t_mean = results[H][m]['t_mean'] / 60
        axs[1].bar(rs[i][t], t_mean, width=barwidth, color=colors[i], edgecolor='black', capsize=5, alpha=1., linewidth=2.5)
axs[1].set_yticks([0, 10, 20, 30, 40, 50])
axs[1].set_ylabel("Avg. Time (min)", fontsize=fs)
axs[1].set_ylim(0, 50)

## plot invalid cases ##
for t, H in enumerate([2,4,6,8,10]):
    for i, m in enumerate(methods):
        n_invalid = results[H][m]['n_unfinished']
        axs[2].bar(rs[i][t], n_invalid, width=barwidth, color=colors[i], edgecolor='black', capsize=5, alpha=1., linewidth=2.5)
axs[2].set_yticks([0, 20, 40, 60])
axs[2].set_ylabel("Num. Unfinished", fontsize=fs)
axs[2].set_ylim(0, 60)

plt.xticks(r2, ["H=2", "H=4", "H=6", "H=8", "H=10"], fontsize=fs)
fig.tight_layout()

plt.savefig("exp.pdf")
