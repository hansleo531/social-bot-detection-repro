# -*- coding: utf-8 -*-
"""跨数据集实验（C_15 → T_20）曲线图"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

path = r"D:\adabot\cross_dataset.log"
with open(path, "rb") as f:
    head = f.read(4)
enc = "utf-16" if head[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"

for name in ["Microsoft YaHei", "SimHei"]:
    try:
        font_manager.findfont(name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [name]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

runs, cur = [], None
for l in open(path, encoding=enc, errors="ignore"):
    if "Exp" in l and "Source Dataset" in l:
        cur = []
        runs.append(cur)
        continue
    m = re.search(r"Test Accuracy: ([\d.]+)%, F1-Score: ([\d.]+)%, Recall: ([\d.]+)%, Precision: ([\d.]+)%", l)
    if m and cur is not None:
        cur.append(tuple(float(x) for x in m.groups()))

iters = [1000 * (i + 1) for i in range(len(runs[0]))]
mean = [tuple(sum(v) / len(v) for v in zip(*c)) for c in zip(*runs)]
acc = [m[0] for m in mean]
f1 = [m[1] for m in mean]
rc = [m[2] for m in mean]
ps = [m[3] for m in mean]

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), dpi=200)

ax = axes[0]
ax.plot(iters, acc, marker="o", ms=4, lw=1.8, color="#C0392B", label="准确率")
ax.plot(iters, f1, marker="s", ms=4, lw=1.8, color="#2E75B6", label="F1")
ax.axhline(65.4, ls="--", lw=1.0, color="#C0392B", alpha=0.5)
ax.axhline(74.2, ls="--", lw=1.0, color="#2E75B6", alpha=0.5)
ax.text(1050, 65.9, "论文 65.4", fontsize=7.5, color="#C0392B")
ax.text(1050, 74.7, "论文 74.2", fontsize=7.5, color="#2E75B6")
ax.set_title("准确率与 F1（5 次均值）", fontsize=10)
ax.set_xlabel("迭代次数", fontsize=9)
ax.set_ylabel("百分比 (%)", fontsize=9)
ax.set_ylim(50, 80)
ax.set_xticks(iters)
ax.tick_params(labelsize=8)
ax.grid(True, ls=":", lw=0.6, color="#BBBBBB")
ax.legend(fontsize=8, frameon=False, loc="lower right")

ax = axes[1]
ax.plot(iters, rc, marker="^", ms=4, lw=1.8, color="#548235", label="召回率")
ax.plot(iters, ps, marker="v", ms=4, lw=1.8, color="#BF8F00", label="精确率")
ax.axhline(93.5, ls="--", lw=1.0, color="#548235", alpha=0.5)
ax.axhline(61.5, ls="--", lw=1.0, color="#BF8F00", alpha=0.5)
ax.text(1050, 94.2, "论文 93.5", fontsize=7.5, color="#548235")
ax.text(1050, 62.2, "论文 61.5", fontsize=7.5, color="#BF8F00")
ax.set_title("召回率与精确率（5 次均值）", fontsize=10)
ax.set_xlabel("迭代次数", fontsize=9)
ax.set_ylabel("百分比 (%)", fontsize=9)
ax.set_ylim(50, 100)
ax.set_xticks(iters)
ax.tick_params(labelsize=8)
ax.grid(True, ls=":", lw=0.6, color="#BBBBBB")
ax.legend(fontsize=8, frameon=False, loc="lower right")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout()
out = r"D:\adabot\fig_cross_dataset.png"
fig.savefig(out)
print("saved", out)
