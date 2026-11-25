# -*- coding: utf-8 -*-
import torch

for d in [0, 6, 3, 2]:
    p = rf"D:\botrans\data\{d}\processed\data.pt"
    try:
        data, slices = torch.load(p, weights_only=False)
        y = data.y
        n = data.num_nodes
        e = data.num_edges
        feat = tuple(data.x.shape) if "x" in data else None
        print(f"domain {d}: nodes={n} edges={e} x={feat} pos={int(y.sum())} pos_ratio={float(y.float().mean()):.3f}")
        print("   keys:", list(data.keys()))
    except Exception as ex:
        print("domain", d, "ERR", type(ex).__name__, ex)
