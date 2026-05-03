# 受控退化实验 — 广义性能倒挂阈值定理 (GPIT) 验证

验证 SUPRA 论文提出的"广义性能倒挂阈值定理"在物理层面的正确性。

---

## 理论背景

GPIT 定理指出：GNN 的拓扑聚合面临双重压迫：

| 维度 | 符号 | 含义 |
|------|------|------|
| 特征维度 | $\sigma_\epsilon^2$ | 大模型特征越来越强（内在噪声减小），GNN 引入的结构污染会超越平滑收益 |
| 拓扑维度 | $\beta$ | 图同质性下降时，性能倒挂阈值 $\tau(\beta) \to \infty$，使 GNN 在异质图上纯粹有害 |

---

## 实验 A：特征维度 — 受控语义退化

**目的**：验证随着特征信噪比降低，强制拓扑聚合的危害会超越其收益。

### 噪声注入公式

$$X_{\text{noisy}} = X + \alpha \cdot \sigma(X) \cdot \mathcal{N}(0, 1)$$

其中 $\alpha$ 为噪声强度，$\sigma(X)$ 为每维特征的标准差。

### 噪声比例

```
noise_ratios = [0.0, 0.1, 0.3, 0.5, 0.8, 1.2, 2.0]
```

### 预期现象

- **Pure MLP**（无拓扑）：随噪声增大快速下降
- **MMGCN**（强制拓扑）：下降较慢，noise_ratio 较低时反超 MLP
- **SUPRA**（解耦+辅助损失）：全局保持最高或平齐

---

## 实验 B：拓扑维度 — 受控同质性退化

**目的**：验证随着同质性破坏，强制聚合的 GNN 会断崖式下跌。

### 边重连逻辑

对于每个重连比例 $p$：
1. 随机选取图中比例为 $p$ 的边断开
2. 在随机节点对之间重连（避免自环和重复边）
3. 保持总边数不变

### 重连比例

```
rewire_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
```

### 预期现象

- **Pure MLP**：不受边结构影响，水平直线
- **MMGCN**：随着 $p$ 增大出现断崖式下跌
- **SUPRA**：辅助损失兜底，紧咬 MLP 水平线

---

## 对比模型

| 模型 | 调用方式 | 描述 |
|------|----------|------|
| **Pure MLP** | `Early_GNN --backend mlp` | 无图结构，纯 MLP on concat features |
| **MMGCN** | `Late_GNN --model_name GCN` | 强制拓扑，多模态 GNN 融合 |
| **SUPRA** | `SUPRA --aux_weight 0.7 --mlp_variant ablate` | 三通道解耦 + 辅助损失 |

---

## 运行命令

### 快速测试（Toys 数据集）

```bash
python tools/run_degradation_experiments.py \
    --data_name Toys \
    --text_feature /path/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /path/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /path/ToysGraph.pt \
    --save_dir Results/degradation \
    --n_runs 3 --n_epochs 300 \
    --gpu 0
```

### 完整实验（Movies 数据集）

```bash
python tools/run_degradation_experiments.py \
    --data_name Movies \
    --text_feature /path/Movies_roberta_base_512_mean.npy \
    --visual_feature /path/Movies_openai_clip-vit-large-patch14.npy \
    --graph_path /path/MoviesGraph.pt \
    --embed_dim 256 --n_layers 3 --lr 0.001 --dropout 0.3 \
    --aux_weight 0.7 \
    --save_dir Results/degradation \
    --n_runs 3 --n_epochs 300 \
    --gpu 0
```

### Reddit-M 数据集

```bash
python tools/run_degradation_experiments.py \
    --data_name Reddit-M \
    --text_feature /path/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /path/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /path/RedditMGraph.pt \
    --embed_dim 256 --n_layers 3 --lr 0.0005 --dropout 0.3 \
    --aux_weight 0.7 \
    --save_dir Results/degradation \
    --n_ratios 5 --n_epochs 500 \
    --n_runs 3 \
    --gpu 0
```

### 自定义噪声/重连比例

```bash
python tools/run_degradation_experiments.py \
    ... \
    --noise_ratios "0.0,0.2,0.5,1.0,2.0" \
    --rewire_ratios "0.0,0.1,0.3,0.5,0.7,1.0"
```

---

## 输出文件

```
Results/degradation/
├── <data_name>_degradation.pdf     # Publication-ready 图表
├── noise_degradation_results.csv     # 噪声实验数值结果
└── rewire_degradation_results.csv   # 重连实验数值结果
```

CSV 格式：
```csv
ratio,model,mean_acc,std_acc
0.0,early_mlp,0.823456,0.012345
0.0,late_gnn,0.845678,0.009876
0.0,supra,0.867890,0.008765
...
```

---

## 图表说明

- **左图**（$\sigma^2_\epsilon$）：特征维度退化，X 轴为噪声强度 $\alpha$
- **右图**（$\beta$）：拓扑维度退化，X 轴为边重连比例 $p$
- **Y 轴**：Accuracy（越高越好）
- **误差带**：`plt.fill_between`，$\pm 1$ std，透明度 15%
- **三条线**：Pure MLP（橙）、MMGCN（蓝）、SUPRA（绿）

---

## 代码架构

```
GNN/Utils/graph_degradation.py     # 工具函数
├── inject_feature_noise()        # 特征加噪
└── rewire_edges()                # 边重连

tools/run_degradation_experiments.py  # 主实验脚本
├── build_early_mlp()           # Pure MLP（backend=mlp）
├── build_late_gnn()             # MMGCN（LateFusionMAG）
├── build_supra()                # SUPRA Full
├── run_noise_experiment()       # 实验 A
├── run_rewire_experiment()      # 实验 B
└── plot_degradation()           # Publication-ready 绘图
```

---

## 超参数一致性

所有三个模型共用以下超参数（来自 `add_common_args`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `embed_dim` | 256 | 嵌入维度 |
| `n_hidden` | 256 | 隐层维度 |
| `n_layers` | 3 | GNN 层数 |
| `dropout` | 0.3 | Dropout 比例 |
| `lr` | 0.001 | 学习率（Movies/Grocery） |
| `wd` | 0.0001 | Weight decay |
| `label_smoothing` | 0.1 | 标签平滑 |
| `early_stop_patience` | 50 | 早停耐心值 |
| `eval_steps` | 1 | 每 epoch 评估 |
| `n_runs` | 3 | 每次条件运行次数 |
| `base_seed` | 42 | 随机种子基准 |

---

## 关键设计决策

1. **加噪公式**：`X + α·σ(X)·N(0,1)` — 每维独立缩放，保持相对噪声水平与数据集特征分布一致
2. **边重连**：re-wire 而非 add/remove，保持总边数近似恒定
3. **Pure MLP 对照组**：`Early_GNN --backend mlp` 真正不走任何图结构（`graph` 参数在前向传播中完全未使用）
4. **Seed 控制**：退化操作（加噪/重连）和模型训练分别使用独立 seed，保证可复现性
