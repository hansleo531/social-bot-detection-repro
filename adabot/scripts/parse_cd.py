import re

path = r"D:\adabot\cross_dataset.log"
with open(path, "rb") as f:
    head = f.read(4)
enc = "utf-16" if head[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"

lines = open(path, encoding=enc, errors="ignore").read().splitlines()
runs, cur = [], None
for l in lines:
    if "Exp" in l and "Source Dataset" in l:
        cur = []
        runs.append(cur)
        continue
    m = re.search(r"Test Accuracy: ([\d.]+)%, F1-Score: ([\d.]+)%, Recall: ([\d.]+)%, Precision: ([\d.]+)%", l)
    if m and cur is not None:
        cur.append(tuple(float(x) for x in m.groups()))

print("encoding:", enc, "runs:", len(runs))
for i, c in enumerate(runs):
    fin = c[-1]
    peak = max(c, key=lambda x: x[0])
    print(f"EXP{i}  final acc={fin[0]:.2f} f1={fin[1]:.2f} rc={fin[2]:.2f} ps={fin[3]:.2f} | peak acc={peak[0]:.2f} @iter{(c.index(peak)+1)*1000}")
print("acc curve:", [[round(x[0], 2) for x in c] for c in runs])
