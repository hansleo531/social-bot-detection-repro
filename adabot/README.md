# AdaBot 复现（ICMR 2025）

## 原仓库
<https://github.com/Bryson-Arrot/AdaBot>｜论文 DOI: 10.1145/3731715.3733268

## 本目录内容
- `scripts/run_com5to6.py` —— 社区间设置（5→6）的驱动脚本，由原仓库 `AdaBot-cross-community.py` 改配置得到
- `scripts/make_chart.py` / `make_chart_cd.py` —— 结果曲线绘制
- `scripts/parse_cd.py` —— 跨数据集日志解析
- `results/RESULTS.md` —— 复现结果汇总
- `results/fig_curve.png`、`fig_cross_dataset.png` —— 论文配图
- `results/*.log` —— 原始运行日志

---

## 一、输入数据的形式与规模

原仓库自带 `Pre_Data/`，三个子目录对应三种数据设置，均为 **PyTorch 张量文件（.pt）**，无需额外预处理。

### 1.1 目录与文件结构

```
Pre_Data/
├── T_22_com/          # TwiBot-22 的 10 个社区（社区间迁移实验）
│   ├── com{i}_edge_index.pt   (2, E)      int64    有向边
│   ├── com{i}_edge_type.pt    (E,)        int64    边类型，共 13 种
│   ├── com{i}_label.pt        (N,)        int64    0=人类 1=机器人
│   ├── com{i}_meta.pt         (N, 8)      float32  元数据特征
│   ├── com{i}_text.pt         (N, 768)    float32  RoBERTa 句向量
│   └── com{i}_meta_mean/std.pt (8,)      float32  元数据标准化参数
├── C_15/              # Cresci-2015（跨数据集实验的源域）
│   ├── egde_index.pt  (2, E)   int64   （原仓库拼写如此）
│   ├── egde_type.pt   (E,)     int64
│   ├── label.pt       (N,)     int64
│   ├── meta.pt        (N, 6)   float32
│   ├── text.pt        (N, 768) float32
│   └── id.pt
└── T_20/              # TwiBot-20（跨数据集实验的目标域）
    ├── 同 C_15，另加 train_mask.pt / test_mask.pt
```

### 1.2 规模

**TwiBot-22 十个社区（`T_22_com/`）**

| 社区 | 节点数 | 边数 | 机器人 | 机器人占比 |
|---|---:|---:|---:|---:|
| com0 | 9,000 | 190 | 4,158 | 46.2% |
| com1 | 8,809 | 491 | 4,000 | 45.4% |
| com2 | 9,262 | 474 | 4,379 | 47.3% |
| com3 | 9,043 | 401 | 4,195 | 46.4% |
| com4 | 9,085 | 349 | 4,228 | 46.5% |
| com5 | 7,725 | 2,247 | 3,186 | 41.2% |
| com6 | 7,918 | 5,533 | 3,381 | 42.7% |
| com7 | 7,794 | 1,998 | 3,254 | 41.8% |
| com8 | 7,742 | 1,917 | 3,184 | 41.1% |
| com9 | 7,685 | 2,313 | 3,144 | 40.9% |
| **合计** | **84,063** | **15,913** | **37,109** | **44.1%** |

> 说明：com0–com4 的边数明显少于 com5–com9（190–491 vs 1,998–5,533），这是原始数据文件的实际内容，可能与抽样方式有关；做图方法对比时需注意这一差异。

**跨数据集（`C_15/`、`T_20/`）**

| 数据集 | 节点数 | 边数 | 机器人 | 机器人占比 | 元数据维度 |
|---|---:|---:|---:|---:|---:|
| Cresci-2015 | 5,301 | 14,220 | 3,351 | 63.2% | 6 |
| TwiBot-20 | 11,826 | 16,908 | 6,589 | 55.7% | 6 |

### 1.3 一次运行的输入

`run_com5to6.py` 读取 `com5`（源域，有标签）与 `com6`（目标域，训练时不使用标签），
每个社区加载上述 7 个张量文件，合计约 23 MB（主要是 `text.pt` 的 768 维句向量）。

---

## 二、评估指标

| 指标 | 实现 | 说明 |
|---|---|---|
| **Accuracy** | `sklearn.metrics.accuracy_score` | 主指标，论文报告的是百分制（%） |
| **F1** | `sklearn.metrics.f1_score`（默认 binary，正类 = 机器人） | 主指标 |
| **Precision** | `precision_score` | 仅在跨数据集设置中报告 |
| **Recall** | `recall_score` | 仅在跨数据集设置中报告 |

- **社区间设置**（`AdaBot-cross-community.py`）：报告 Accuracy 与 F1；
- **跨数据集设置**（`AdaBot-cross-dataset.py`）：报告 Accuracy、F1、Recall、Precision；
- **重复策略**：每组实验重复 5 次，取均值 ± 标准差（`run_com5to6.py` 中 `repeat=5`）；
- 原论文对多组设置做 Student t 检验，本复现只报均值与标准差。

---

## 三、运行方式

把本目录的 `scripts/` 内容放到原仓库根目录（与 `Pre_Data/` 同级），然后：

```bash
# 社区间：TwiBot-22 社区 5 训练、社区 6 测试，5 次重复
python -u run_com5to6.py

# 跨数据集：Cresci-15 训练、TwiBot-20 测试
python AdaBot-cross-dataset.py --dataset1 cresci-15 --dataset2 twibot-20 --data_root_path Pre_Data
```

> 建议加 `-u`，并把输出同时写入日志：
> `python -u run_com5to6.py 2>&1 | Tee-Object -FilePath run.log`

## 四、关键配置（`run_com5to6.py`）

| 参数 | 取值 | 说明 |
|---|---|---|
| `coms` | `[[5, 6]]` | 源社区 5，目标社区 6 |
| `exp_all` | `False` | 不做全组合扫描 |
| `repeat` | `5` | 5 次重复，报均值 ± 标准差 |
| `epochs` | 200 | 与原文一致 |

## 五、踩过的坑

1. **`NeighborLoader` 依赖 `pyg-lib`**：不装会直接报错，且必须是 `0.6.0+pt28cu128` 匹配版本。
2. **`--data_root_path` 相对 CWD 解析**：必须在仓库根目录执行，否则找不到 `Pre_Data/`。
3. **RTX 50 系需要 cu128**：cu121 及以下没有 sm_120 内核，会报 `no kernel image is available`。
4. **输出缓冲**：长时间运行时建议 `python -u`，否则日志不落盘。

## 六、结果

| 设置 | 准确率 | F1 | 原文 |
|---|---|---|---|
| 社区间 5→6 | 74.09 ± 0.22 | 0.671 | 74.8 ± 0.4 / 0.684 |
| 跨数据集 | 63.96 / 73.80 / 93.81 / 60.82 | — | 65.4 / 74.2 / 93.5 / 61.5 |

（跨数据集四项依次为 Accuracy / F1 / Recall / Precision，均为百分制）
