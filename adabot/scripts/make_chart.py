# -*- coding: utf-8 -*-
"""从 full_run_com5to6.log 解析 5 次重复的测试曲线并绘图"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

LOG = r"D:\adabot\full_run_com5to6.log"
OUT = r"D:\adabot\fig_curve.png"

# 中文字体
for name in ["Microsoft YaHei", "SimHei", "DengXian"]:
    try:
        font_manager.findfont(name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [name]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

curves = []
cur = None
pat_exp = re.compile(r"\^+\s*EXP(\d+)\s*\^+")
pat_test = re.compile(r"Test Accuracy:\s*([\d.]+)%")

with open(LOG, "rb") as fh:
    head = fh.read(4)
enc = "utf-16" if head[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"

for line in open(LOG, encoding=enc):
    m = pat_exp.search(line)
    if m:
        cur = []
        curves.append(cur)
        continue
    if cur is not None:
        t = pat_test.search(line)
        if t:
            cur.append(float(t.group(1)))

iters = [500 * (i + 1) for i in range(len(curves[0]))]
print("experiments:", len(curves), "points:", [len(c) for c in curves])

fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
colors = ["#8FB3D9", "#7EA6D2", "#6D99CB", "#5C8CC4", "#4B7FBD"]
for i, c in enumerate(curves):
    ax.plot(iters, c, marker="o", ms=3.5, lw=1.2, alpha=0.9,
            color=colors[i % len(colors)], label=f"第 {i + 1} 次")

mean = [sum(v) / len(v) for v in zip(*curves)]
ax.plot(iters, mean, color="#C0392B", lw=2.2, marker="s", ms=4,
        label="5 次均值")

ax.set_xlabel("迭代次数", fontsize=9)
ax.set_ylabel("目标域（com6）准确率 (%)", fontsize=9)
ax.set_xticks(iters)
ax.tick_params(labelsize=8)
ax.set_ylim(40, 80)
ax.grid(True, ls=":", lw=0.6, color="#BBBBBB")
ax.legend(fontsize=8, ncol=3, frameon=False, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT)
print("saved", OUT)
