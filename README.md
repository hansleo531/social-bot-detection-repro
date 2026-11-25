# 社交机器人检测跨域方法复现：AdaBot & BotTrans

本仓库记录两篇跨域社交机器人检测论文的**复现脚本、实验配置与结果**。
原始模型代码来自作者开源仓库，本仓库只包含：复现所需的驱动脚本、数据预处理/解析脚本、改动说明与实验结果。

| 论文 | 会议 | 复现设置 | 主要结果 | 状态 |
|---|---|---|---|---|
| **AdaBot**: Adaptive Social Bot Detection through Bridging the Feature Bias Between Source and Target Users | ICMR 2025 | 社区间 5→6（5 次重复）；跨数据集 Cresci-15→TwiBot-20 | 准确率 **74.09 ± 0.22**（原文 74.8 ± 0.4）；跨数据集四项指标与原文差距 ≤1.5pp | ✅ 复现成立 |
| **BotTrans**: A Multi-source Graph Domain Adaptation Approach for Social Bot Detection | ECML-PKDD 2025 | 10 个目标域 × 其余 9 域作源，50 epoch | 平均 **F1 0.723 / AUC 0.779**；9 域正常（AUC 0.74–0.91），域 6 失败（0.538） | ✅ 基本成立（域 6 为数据问题） |

> 注意：两个仓库的**数据均不随本仓库分发**（体积与版权原因），获取方式见第 3 节。

---

## 1. 环境

| 项目 | 版本 |
|---|---|
| 系统 | Windows 11 |
| Python | 3.10.11（venv） |
| PyTorch | 2.8.0+cu128 |
| PyTorch Geometric | 2.8.1 |
| pyg-lib | 0.6.0+pt28cu128（AdaBot 的 NeighborLoader 依赖） |
| GPU | NVIDIA RTX 5060 Ti 8 GB（驱动 610.62） |

安装（国内建议用清华镜像）：

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric==2.8.1
pip install pyg-lib==0.6.0 -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install -r requirements.txt
```

> RTX 50 系（sm_120）必须用 cu128 及以上版本的 PyTorch，否则会报 `no kernel image is available`。

---

## 2. 目录结构

```
.
├── adabot/
│   ├── scripts/        # 复现驱动脚本（run_com5to6.py 等）
│   ├── patches/        # 相对原仓库的改动说明
│   └── results/        # 结果、日志、曲线图
├── botrans/
│   ├── scripts/        # 全量扫描、结果解析、案例分析脚本
│   ├── patches/        # 相对原仓库的改动说明
│   └── results/        # 10 域结果 CSV、柱状图、日志
├── requirements.txt
└── README.md
```

---

## 3. 数据获取（**不随仓库分发**）

| 数据集 | 规模 | 获取方式 |
|---|---|---|
| TwiBot-22 十社区（AdaBot 主实验） | 84,063 节点 / 15,913 边 | 原作者仓库 `Pre_Data/T_22_com/` |
| Cresci-2015（跨数据集源域） | 5,301 节点 / 14,220 边 | 原作者仓库 `Pre_Data/C_15/` |
| TwiBot-20（跨数据集目标域） | 11,826 节点 / 16,908 边 | 原作者仓库 `Pre_Data/T_20/` |
| BotTrans 10 域 | 44,534 节点 / 238,955 边 | 原作者仓库 `data.zip` + `graphlet.zip` |

- AdaBot：<https://github.com/Bryson-Arrot/AdaBot>（`Pre_Data/` 约 580 MB）
- BotTrans：<https://github.com/Skyorca/BotTrans>（`data.zip` 约 50 MB）

拿到数据后，把本仓库的脚本放到对应位置即可运行（见各子目录 README）。

---

## 4. 输入数据的形式与规模

### 4.1 AdaBot：`.pt` 张量，直接可加载

每个社区/数据集是一组 PyTorch 张量，无需预处理：

| 文件 | 形状 | 说明 |
|---|---|---|
| `*_edge_index.pt` | (2, E) | 有向边，int64 |
| `*_edge_type.pt` | (E,) | 边类型，TwiBot-22 共 13 种 |
| `*_label.pt` | (N,) | 0=人类，1=机器人 |
| `*_meta.pt` | (N, 8) 或 (N, 6) | 元数据特征（TwiBot-22 为 8 维，Cresci/TwiBot-20 为 6 维） |
| `*_text.pt` | (N, 768) | RoBERTa 句向量 |

| 数据集 | 节点数 | 边数 | 机器人占比 |
|---|---:|---:|---:|
| TwiBot-22 十社区合计 | 84,063 | 15,913 | 44.1% |
| Cresci-2015 | 5,301 | 14,220 | 63.2% |
| TwiBot-20 | 11,826 | 16,908 | 55.7% |

### 4.2 BotTrans：CSV + 预拼装图对象

原始 CSV 每域 4 个文件（`nodes` / `raw_feat` 11 列 / `edges_new` / `graphlet`），
预处理后得到 PyG 图对象 `data.pt`：

| 字段 | 形状 | 说明 |
|---|---|---|
| `x` | (N, 10) | 节点数值特征 |
| `edge_index` | (2, E) | 有向关注边 |
| `x_graphlet` | (N, 73) | 结构特征，用于跨域建图的边权 |
| `y` | (N,) | 0/1 标签 |

10 个域合计 **44,534 节点、238,955 边**，人机比例约 1:1（47.2% 为机器人）。
单域规模 4,201–4,875 节点。详见 `botrans/README.md` 的分域表。

---

## 5. 评估指标

| 论文 | 指标 | 说明 |
|---|---|---|
| **AdaBot** | Accuracy、F1 | 社区间设置报这两个；5 次重复取均值 ± 标准差 |
| **AdaBot** | Accuracy、F1、Recall、Precision | 跨数据集设置全报（论文 Table 3 口径） |
| **BotTrans** | F1、ROC-AUC、AP | 主指标为 F1 与 AUC；AP 为辅助 |

实现均调用 `sklearn.metrics`，正类为“机器人”，阈值 0.5。统计口径与注意事项见各子目录 README。

---

## 6. 结果摘要

### 6.1 AdaBot

| 设置 | 准确率 | F1 | 原文 | 差距 |
|---|---|---|---|---|
| 社区间 5→6（5 次） | 74.09 ± 0.22 | 0.671 | 74.8 ± 0.4 / 0.684 | −0.71 / −1.3pp |
| 跨数据集 Cresci-15→TwiBot-20 | 63.96 / 73.80 / 93.81 / 60.82 | — | 65.4 / 74.2 / 93.5 / 61.5 | ≤1.5pp |

（跨数据集四项依次为 Accuracy / F1 / Recall / Precision）

### 6.2 BotTrans（10 个目标域，各用其余 9 域作源）

| 目标域 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 均值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 | 0.821 | 0.762 | 0.726 | 0.720 | 0.750 | 0.712 | 0.664 | 0.690 | 0.699 | 0.686 | **0.723** |
| AUC | 0.911 | 0.834 | 0.800 | 0.788 | 0.850 | 0.776 | **0.538** | 0.776 | 0.786 | 0.735 | **0.779** |

原文 9 源设置：F1 0.800 / AUC 0.852。差距主要来自训练轮数（本实验 50 epoch，原文 1000）。

---

## 7. 已知问题与发现

1. **BotTrans 域 6 稳定失败**：四次独立实验（换源域、加到 300 epoch）AUC 均落在 0.50–0.58。
   进一步分析发现该域仅用元数据的逻辑回归 5 折 AUC 只有 0.507（域 0 为 0.866），
   即目标域本身缺乏可分信号，不是训练不足。
2. **F1 会奖励退化模型**：BotTrans 按目标域 F1 选 checkpoint，在类别均衡的低可分域上，
   “全判一类”即可把 F1 刷到 0.66。300 epoch 实验最终选中的是第 4 轮模型（AUC 0.496），
   而真实峰值在第 218 轮（0.575）。低可分域建议用 AUC 选模型。
3. **AdaBot 的 `--data_root_path` 是相对当前工作目录解析的**，需在仓库根目录运行。

---

## 8. 声明

- 模型与数据集版权归原作者；本仓库仅包含复现脚本与结果，改动清单见各 `patches/` 目录。
- 若用于论文引用，请引用原论文，而非本仓库。
