# BotTrans 复现（ECML-PKDD 2025）

## 原仓库
<https://github.com/Skyorca/BotTrans>｜论文 DOI: 10.1007/978-3-032-06066-2_14

## 本目录内容
- `scripts/sweep.py` —— 全量扫描驱动：10 个目标域，每个用其余 9 域作源
- `scripts/check_data.py` —— 数据完整性检查
- `scripts/explore_domains.py` / `separability.py` —— 域画像与元数据可分性分析
- `scripts/make_cases2.py` / `enrich_cases2.py` —— 典型案例筛选与派生指标
- `results/sweep_summary.csv` —— 10 个目标域的最终结果
- `results/sweep_chart.png` —— F1/AUC 柱状图
- `results/domain_dist.png` —— 两个域的粉丝分布对比
- `results/cases_*.csv` / `profile.csv` —— 案例分析数据
- `results/*.log` —— 各目标域运行日志

---

## 一、输入数据的形式与规模

### 1.1 目录结构

```
data/
└── <域号>/                  # 0–9，共 10 个域
    ├── raw/                 # 原始 CSV（可直接用 Excel 打开）
    │   ├── group<i>_nodes.csv       # id,label        账号 ID 与标签
    │   ├── group<i>_raw_feat.csv    # 11 列账号属性
    │   ├── group<i>_edges_new.csv   # source,target   有向关注边
    │   └── group<i>_graphlet.csv    # 预计算的 graphlet 特征
    └── processed/
        └── data.pt          # 预先拼装好的 PyG 图对象
```

### 1.2 各字段含义

**`group<i>_raw_feat.csv`（11 列）**

| 列名 | 类型 | 说明 |
|---|---|---|
| `id` | string | 账号 ID，形如 `u1234567890` |
| `followers_count` | int | 粉丝数 |
| `following_count` | int | 关注数 |
| `active_days` | int | 活跃天数（少数样本为负值，属原始数据异常） |
| `name_len` | int | 昵称长度 |
| `uname_len` | int | 用户名长度 |
| `tweet_count` | int | 推文数 |
| `listed_count` | int | 被加入列表的次数 |
| `protected` | 0/1 | 是否受保护 |
| `verified` | 0/1 | 是否认证 |
| `picture` | 0/1 | 是否使用默认头像 |

**`processed/data.pt` 中的图对象**

| 字段 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `x` | (N, 10) | float32 | 节点特征（由 raw_feat 的 10 个数值列构成） |
| `edge_index` | (2, E) | int64 | 有向关注边 |
| `x_graphlet` | (N, 73) | float32 | graphlet 结构特征，用于跨域建图的边权计算 |
| `y` | (N,) | int64 | 0=人类 1=机器人 |
| `num_nodes` | int | — | 节点数 |

### 1.3 规模（10 个域）

| 域 | 节点数 | 边数 | 机器人 | 机器人占比 |
|---|---:|---:|---:|---:|
| 0 | 4,440 | 22,223 | 2,035 | 45.8% |
| 1 | 4,201 | 21,229 | 1,999 | 47.6% |
| 2 | 4,295 | 21,650 | 2,033 | 47.3% |
| 3 | 4,313 | 21,951 | 2,000 | 46.4% |
| 4 | 4,225 | 21,213 | 1,963 | 46.5% |
| 5 | 4,513 | 24,628 | 2,108 | 46.7% |
| 6 | 4,875 | 31,732 | 2,420 | 49.6% |
| 7 | 4,486 | 24,514 | 2,124 | 47.3% |
| 8 | 4,614 | 24,580 | 2,190 | 47.5% |
| 9 | 4,572 | 25,235 | 2,151 | 47.0% |
| **合计** | **44,534** | **238,955** | **21,023** | **47.2%** |

> 人机比例接近 1:1 是论文刻意采样的结果；跨域建图时，每个源节点会到**其他源域**中找 Top-K（默认 5）结构最相似的节点连边，因此一次训练实际处理的图规模约为「目标域 N + 其余 9 域 N」。

---

## 二、评估指标

| 指标 | 实现 | 说明 |
|---|---|---|
| **F1** | `sklearn.metrics.f1_score`（binary，正类 = 机器人） | 主指标，论文 Table 1 以 F1 为主 |
| **ROC-AUC** | `sklearn.metrics.roc_auc_score` | 主指标，与 F1 并列报告 |
| **AP** | `sklearn.metrics.average_precision_score` | 辅助指标（正类稀疏时比 AUC 更敏感） |

**统计口径**：

- 每个目标域训练 50 epoch（原文 1000），取**目标域 F1 最高的那个 epoch**作为最好结果；
- 同一 epoch 的 AUC、AP 一并记录（即 `sweep_summary.csv` 中的 `best_f1` / `best_auc`）；
- 10 个目标域各跑一次，报均值与逐域结果；`repeat` 参数可控制重复次数（本复现为 1）。

**⚠️ 已知陷阱：按 F1 选模型在低可分域上会选中退化模型**

在类别均衡（约 1:1）的数据上，“全部判为机器人”即可把 F1 刷到 0.66 左右。
域 6 的 300 epoch 实验中，最终选中的是第 4 轮模型（F1 0.664，**AUC 仅 0.496**），
而训练中 AUC 的真实峰值在第 218 轮（0.575）。
因此在这一类域上，**建议改成按验证集 AUC 选择 checkpoint**，或同时报告 AUC 峰值。

---

## 三、运行方式

```bash
# 单个目标域
python -u train.py --src_name 0,1,2,3,4,5,7,8,9 --tgt_name 6 --cuda 0 --epoch 50 --repeat 1

# 全量扫描（10 个目标域，约 9 小时 / 单卡）
python sweep.py
```

## 四、关键设置

| 参数 | 取值 |
|---|---|
| 目标域 / 源域 | 10 个目标 × 其余 9 个源 |
| epoch | 50（原文 1000） |
| 跨域邻居 K | 5 |
| 边特征 | graphlet |
| batch size | 64（按源数均分） |
| 设备 | 单卡 RTX 5060 Ti 8GB，约 52 分钟/域 |

## 五、对原仓库的改动

见 `patches/CHANGES.md`。核心只有一处：`dataloader.py` 中 `torch.load` 需要 `weights_only=False`（PyTorch 2.6+ 默认改为 True，加载 PyG 图对象会报 `UnpicklingError`）。

## 六、结果（50 epoch，9 源）

| 目标域 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 均值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 | 0.821 | 0.762 | 0.726 | 0.720 | 0.750 | 0.712 | 0.664 | 0.690 | 0.699 | 0.686 | 0.723 |
| AUC | 0.911 | 0.834 | 0.800 | 0.788 | 0.850 | 0.776 | 0.538 | 0.776 | 0.786 | 0.735 | 0.779 |

原文 9 源：F1 0.800 / AUC 0.852。

## 七、域 6 的情况

| 实验 | 设定 | AUC |
|---|---|---|
| 初测 | 4 源（2,3,4,5），100 epoch | 0.539 |
| 换源 | 4 源（0,1,7,8），30 epoch | 0.540 |
| 全源 | 9 源，50 epoch | 0.538 |
| 加轮数 | 9 源，300 epoch | 峰值 0.575；按 F1 选出的模型 0.496 |

只用一个逻辑回归（10 个数值特征、5 折）分人机：域 0 为 0.866，域 6 为 0.507。
说明该目标域本身缺乏可分信号，不是训练不足。
