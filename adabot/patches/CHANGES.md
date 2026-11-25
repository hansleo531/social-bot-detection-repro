# 相对原仓库的改动说明（AdaBot）

原仓库：<https://github.com/Bryson-Arrot/AdaBot>（ICMR 2025）

## 改动范围

**未修改任何模型代码**。原仓库只有两个入口脚本：

- `AdaBot-cross-community.py`（社区间迁移）
- `AdaBot-cross-dataset.py`（跨数据集迁移）

复现时另存了一份 `run_com5to6.py`，仅改配置，不动算法：

| 参数 | 原值 | 复现值 | 说明 |
|---|---|---|---|
| `coms` | 全组合 | `[[5, 6]]` | 只跑社区 5 → 社区 6 |
| `exp_all` | `True` | `False` | 不做 40 组合全扫描 |

```python
# run_com5to6.py 中的关键改动
coms = [[5, 6]]
exp_all = False
```

## 数据预处理

原仓库自带 `Pre_Data/`（TwiBot-22 十个社区、Cresci-15、TwiBot-20 的图对象），
无需额外处理，直接把目录放到仓库根目录即可。

## 运行时注意

1. `--data_root_path` 相对**当前工作目录**解析，需在仓库根目录执行；
2. 需要 `pyg-lib`（`NeighborLoader` 依赖），版本必须与 torch 匹配
   （本仓库实验为 `pyg-lib 0.6.0+pt28cu128`）；
3. RTX 50 系（sm_120）必须用 cu128 及以上的 PyTorch。

## 复现结果

| 设置 | 准确率 | F1 | 原文 | 差距 |
|---|---|---|---|---|
| 社区间 5→6（5 次重复） | 74.09 ± 0.22 | 0.671 | 74.8 ± 0.4 / 0.684 | −0.71 / −1.3pp |
| 跨数据集 Cresci-15→TwiBot-20 | 63.96 / 73.80 / 93.81 / 60.82 | — | 65.4 / 74.2 / 93.5 / 61.5 | ≤1.5pp |
