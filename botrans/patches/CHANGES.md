# 相对原仓库的改动说明（BotTrans）

原仓库：<https://github.com/Skyorca/BotTrans>（未改动模型结构、损失函数与训练逻辑）

## 1. `dataloader.py`：`torch.load` 增加 `weights_only=False`

**原因**：PyTorch 2.6 起 `torch.load` 默认 `weights_only=True`，加载 PyG 的 `Data` 对象会抛
`UnpicklingError: Weights only load failed`。

**改动**：

```python
# 修改前
data = torch.load(path)

# 修改后
data = torch.load(path, weights_only=False)
```

## 2. 数据目录整理

原仓库 `data.zip` 解压后，各域目录直接位于根目录（且部分域多一层嵌套），
而代码期望 `data/<域号>/raw/group<域号>_*.csv`。整理命令（PowerShell）：

```powershell
# 把解压出的 0..9 目录移动到 data/ 下
foreach ($i in 0..9) { Move-Item "0$i" "data\$i" -ErrorAction SilentlyContinue }
```

## 3. 新增脚本（不属于原仓库）

| 脚本 | 作用 |
|---|---|
| `sweep.py` | 依次跑 10 个目标域，结果追加到 `sweep_summary.csv`，可断点续跑 |
| `check_data.py` | 校验每个域的 raw/processed 是否齐全 |
| `explore_domains.py` | 统计各域规模、人机比、粉丝/推文分布 |
| `separability.py` | 用逻辑回归测各域元数据可分性（5 折 AUC） |
| `make_cases2.py` | 按模式挑选典型案例账号 |
| `enrich_cases2.py` | 计算派生指标（注册时间、日均发帖、入/出度、邻居 bot 占比） |

## 4. 未改动但要留意

`train.py` 默认按**目标域 F1** 保存 best checkpoint。在类别均衡、低可分的域上，
“全判一类”即可获得较高 F1，导致选到退化模型（域 6 的 300 epoch 实验中，最终模型 AUC 仅 0.496，
而训练峰值出现在第 218 轮，AUC 0.575）。如需严谨复现，建议改为按验证集 AUC 选择。
