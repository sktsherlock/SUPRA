# 受控退化实验 — 广义性能倒挂阈值定理 (GPIT) 验证

验证 SUPRA 论文提出的"广义性能倒挂阈值定理"在物理层面的正确性。

---

## 理论背景

GPIT 定理指出：GNN 的拓扑聚合面临**双重压迫**：

| 维度 | 符号 | 含义 |
|------|------|------|
| 特征维度 | $\sigma_\epsilon^2$ | 大模型特征越来越强（内在噪声减小），GNN 引入的结构污染会超越平滑收益 |

---

## 实验：特征维度 — 受控语义退化

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
- **SUPRA**（解耦，无辅助损失强化）：全局保持最高或平齐

---

## 对比模型

| 模型 | 调用方式 | 描述 |
|------|----------|------|
| **Pure MLP** | `Early_GNN --backend mlp` | 无图结构，纯 MLP on concat features |
| **MMGCN** | `Late_GNN --model_name GCN` | 强制拓扑，多模态 GNN 融合 |
| **SUPRA** | `SUPRA --aux_weight 0.0 --mlp_variant ablate` | 三通道解耦（无辅助损失强化） |

---

## 运行命令

### 一步跑完所有数据集（推荐）

一条命令跑完 Reddit-M（正文图）+ Movies/Grocery/Toys（附录图）：

```bash
python tools/run_degradation_experiments.py \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --embed_dim 256 \
    --n_layers 3 \
    --lr 0.0005 \
    --wd 0.0001 \
    --dropout 0.3 \
    --n_runs 3 \
    --n_epochs 300 \
    --save_dir Results/degradation \
    --gpu 0 \
    --appendix_datasets \
        "Movies:/mnt/input/MAGB_Dataset/Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy:/mnt/input/MAGB_Dataset/Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy:/mnt/input/MAGB_Dataset/Movies/MoviesGraph.pt:0.001:3,\
Grocery:/mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy:/mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy:/mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt:0.001:3,\
Toys:/mnt/input/MAGB_Dataset/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy:/mnt/input/MAGB_Dataset/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy:/mnt/input/MAGB_Dataset/Toys/ToysGraph.pt:0.0005:2"
```

**输出**：
- `degradation_2x2.pdf` — **正文图**（2×2 四数据集并排：Reddit-M / Movies / Grocery / Toys）

### 仅跑单个数据集

```bash
python tools/run_degradation_experiments.py \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --embed_dim 256 --n_layers 3 --lr 0.0005 \
    --dropout 0.3 --n_runs 3 --n_epochs 300 \
    --save_dir Results/degradation --gpu 0
```

### 自定义噪声比例

```bash
python tools/run_degradation_experiments.py \
    ... \
    --noise_ratios "0.0,0.2,0.5,1.0,2.0"
```

---

## 输出文件

```
Results/degradation/
├── degradation_2x2.pdf             # 正文图（2×2 四数据集并排）
├── Reddit-M_noise_degradation_results.csv  # 各 dataset 的数值结果
├── Movies_noise_degradation_results.csv
├── Grocery_noise_degradation_results.csv
└── Toys_noise_degradation_results.csv
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

### 正文图（2×2 四数据集并排）
- 四个数据集（Reddit-M / Movies / Grocery / Toys）以 2×2 网格展示于正文中
- X轴：**左 = 高噪声（最差特征）→ 右 = 原始（最佳特征）**，清晰呈现"特征质量提升时 Pure MLP / SUPRA 受益更大"这一趋势
- Y轴：Accuracy
- 三条曲线：Pure MLP（橙）、MMGCN（蓝）、SUPRA（绿）
- 误差带：`plt.fill_between`，$\pm 1$ std，透明度 15%
- 统一 X 轴标签，Y 轴标签仅最左子图显示

---

## 代码架构

```
GNN/Utils/graph_degradation.py     # 工具函数
├── inject_feature_noise()        # 特征加噪

tools/run_degradation_experiments.py  # 主实验脚本
├── build_early_mlp()           # Pure MLP
├── build_late_gnn()             # MMGCN（LateFusionMAG）
├── build_supra()                # SUPRA（aux_weight=0.0）
├── run_noise_experiment()       # 噪声实验
├── plot_degradation()           # 正文单图
└── plot_multi_dataset()         # 附录 1×N 并排图
```

---

## 超参数一致性

三个模型共用以下超参数，数据集间仅 lr 和 n_layers 不同：

| 参数 | Toys | Movies | Grocery | Reddit-M | 说明 |
|------|------|--------|---------|----------|------|
| `embed_dim` | 256 | 256 | 256 | 256 | 嵌入维度 |
| `n_layers` | 2 | 3 | 3 | 3 | GNN 层数 |
| `lr` | 0.0005 | 0.001 | 0.001 | 0.0005 | 学习率 |
| `wd` | 0.0001 | 0.0001 | 0.0001 | 0.0001 | Weight decay |
| `dropout` | 0.3 | 0.3 | 0.3 | 0.3 | Dropout 比例 |
| `aux_weight` | 0.0 | 0.0 | 0.0 | 0.0 | SUPRA 辅助损失权重（实验固定为 0） |
| `n_epochs` | 300 | 300 | 300 | 300 | 最大训练轮数 |
| `n_runs` | 3 | 3 | 3 | 3 | 每次条件运行次数 |
| `base_seed` | 42 | 42 | 42 | 42 | 随机种子基准 |

---

## 关键设计决策

1. **加噪公式**：`X + α·σ(X)·N(0,1)` — 每维独立缩放，保持相对噪声水平与数据集特征分布一致
2. **aux_weight=0**：SUPRA 三通道共享 `logits_final` loss 回传，不额外强化 Ut/Uv 通道，验证架构本身的拓扑解耦鲁棒性
3. **Pure MLP 对照组**：`Early_GNN --backend mlp` 真正不走任何图结构（`graph` 参数在前向传播中完全未使用）
4. **Seed 控制**：退化操作（加噪）和模型训练分别使用独立 seed，保证可复现性
