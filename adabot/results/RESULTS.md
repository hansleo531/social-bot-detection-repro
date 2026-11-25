# AdaBot 复现记录：com5 → com6

## 环境
- GPU: RTX 5060 Ti 8GB (sm_120), driver 610.62
- Python 3.10.11 + venv
- torch 2.8.0+cu128, torch_geometric 2.x, pyg-lib 0.6.0+pt28cu128, scikit-learn
- 数据: 仓库自带 `Pre_Data/T_22_com/`（未额外下载 TwiBot-22）

## 路径
所有文件位于 `D:\adabot`（2026-11-17 从 C 盘迁移并重建 venv，C 盘已释放）。

```
D:\adabot\
├── venv\                  # Python 3.10 + torch 2.8.0+cu128 + pyg 2.8.1 + pyg-lib 0.6.0
├── repo\                  # AdaBot 源码 + Pre_Data
├── run_com5to6.py         # 单对社区启动脚本
├── full_run_com5to6.log   # 5 次重复完整日志
└── RESULTS.md
```

## 运行命令
```
cd D:\adabot
.\venv\Scripts\python.exe -u run_com5to6.py --data_root_path .\repo\Pre_Data\T_22_com/ \
    --iterations 5000 --exp_times 5 --test_report 500
```
（`run_com5to6.py` 由 `repo/AdaBot-cross-community.py` 复制，仅改两处：`--coms` 默认 `[[5,6]]`、`--exp_all` 默认 `False`）

## 结果（目标域 com6，test 792 用户 / train 5542 用户）

| 重复 | Accuracy | F1 |
|---|---|---|
| EXP0 | 74.37% | 0.6823 |
| EXP1 | 73.99% | 0.6645 |
| EXP2 | 74.24% | 0.6699 |
| EXP3 | 73.74% | 0.6719 |
| EXP4 | 74.12% | 0.6656 |
| **均值±标准差** | **74.09 ± 0.22** | **0.6708 ± 0.0064** |

## 与论文对比
论文表 5-4（AdaBot，B 组，5→6）：**74.8 ± 0.4**。
复现 74.09 ± 0.22，低 0.7 个百分点，标准差更小。属正常波动（脚本未固定随机种子，torch 版本亦与作者不同）。

## 耗时
- 每个 run（5000 iter）：约 3 分 40 秒
- 5 次重复合计：18.5 分钟
- 单 iter：约 0.044 秒

## 备注
- 新版 PyG 的 `NeighborLoader` 需要 `pyg-lib` 或 `torch-sparse`，否则报 `ImportError: requires either 'pyg-lib' or 'torch-sparse'`。
- `--data_root_path` 默认相对当前工作目录，脚本与 `Pre_Data` 不在同一层时要显式传。
- `RandomNodeSplit` 在 `exp_times` 循环外，5 次重复共用同一划分，std 只反映初始化/采样方差。
- 完整日志：`full_run_com5to6.log`
