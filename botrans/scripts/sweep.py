# -*- coding: utf-8 -*-
"""BotTrans 全目标域扫描：10 个目标域，各自用其余 9 个域作为源域"""
import glob
import os
import re
import subprocess
import time

PY = r"D:\adabot\venv\Scripts\python.exe"
ROOT = r"D:\botrans"
EPOCH = 50
cs_path = os.path.join(ROOT, "sweep_summary.csv")

rows = []
with open(cs_path, "w", encoding="utf-8") as f:
    f.write("target,best_f1,best_auc,minutes\n")

for t in range(10):
    srcs = ",".join(str(i) for i in range(10) if i != t)
    logf = os.path.join(ROOT, f"sweep_tgt{t}.log")
    print(f"[start] target={t} src={srcs}", flush=True)
    t0 = time.time()
    with open(logf, "w", encoding="utf-8", errors="ignore") as f:
        subprocess.run(
            [PY, "train.py", "--src_name", srcs, "--tgt_name", str(t),
             "--cuda", "0", "--epoch", str(EPOCH), "--repeat", "1"],
            cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
        )
    dur = time.time() - t0

    best_f1 = best_auc = float("nan")
    logs = sorted(glob.glob(os.path.join(ROOT, "log", "*.log")), key=os.path.getmtime)
    if logs:
        txt = open(logs[-1], encoding="utf-8", errors="ignore").read()
        f1s = [float(x) for x in re.findall(r"Target:\s*f1-score:\s*([\d.]+)", txt)]
        aucs = [float(x) for x in re.findall(r"Target:\s*f1-score:\s*[\d.]+\s*roc-auc:\s*([\d.]+)", txt)]
        if f1s:
            best_f1 = max(f1s)
        if aucs:
            best_auc = max(aucs)

    rows.append((t, best_f1, best_auc, dur / 60))
    print(f"[done] target={t} best_f1={best_f1:.4f} best_auc={best_auc:.4f} {dur/60:.1f}min", flush=True)
    with open(cs_path, "w", encoding="utf-8") as f:
        f.write("target,best_f1,best_auc,minutes\n")
        for r in rows:
            f.write("%d,%.4f,%.4f,%.1f\n" % r)

print("ALL_DONE", flush=True)
