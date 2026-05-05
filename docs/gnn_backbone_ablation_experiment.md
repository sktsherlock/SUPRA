# GNN Backbone 消融实验 — SUPRA 架构通用性验证

验证 SUPRA 架构的 C 通道 GNN 层可替换为其他主流 GNN 算子（GAT / SAGE / JKNet），且性能具有可比性。

---

## 实验动机

SUPRA 的核心创新在于**三通道解耦架构**（C/Ut/Uv），其中 C 通道负责跨模态共享语义的图传播。C 通道的 GNN 算子采用模块化设计，理论上可替换为任意 GNN 变体。

本实验验证：
1. GAT（注意力机制）、SAGE（邻居聚合）、JKNet（跳连聚合）均可无缝替换 GCN
2. 不同 GNN 算子在 SUPRA 架构下均能保持精度优势
3. `aux_weight`（辅助损失权重）对不同 GNN 算子的影响一致

---

## 对比模型

| 模型 | GNN 算子 | 关键参数 | 备注 |
|------|----------|----------|------|
| **SUPRA-GCN** | GCN（对称归一化） | 3层, n_hidden=256 | 基线（已在正文） |
| **SUPRA-GAT** | GAT（多头注意力） | 3层, n_heads=4, attn_drop=0.0 | 注意力机制 |
| **SUPRA-SAGE** | GraphSAGE（邻居聚合） | 3层, aggregator=mean | 归纳式学习 |
| **SUPRA-JKNet** | JKNet（跳连聚合） | 3层, aggr=concat | 多尺度表征 |

---

## 超参数配置

各 GNN 算子的最优参数来自已验证的基线配置（如 `efficiency_analysis_commands.md`）：

| 参数 | GCN | GAT | SAGE | JKNet |
|------|-----|-----|------|-------|
| `n_layers` | 3 | 3 | 3 | 3 |
| `n_hidden` / `embed_dim` | 256 | 256 | 256 | 256 |
| `lr` | 0.0005 | 0.001 | 0.0005 | 0.0005 |
| `wd` | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| `dropout` | 0.3 | 0.3 | 0.3 | 0.3 |
| 特殊参数 | — | `--n_heads=4 --attn_drop=0.0 --edge_drop=0.0` | `--aggregator=mean` | `--jknet_aggr=concat` |

所有模型共同超参数：
- `mlp_variant=ablate`（无 MLP 投影，纯 concat 进 GNN）
- `aux_weight` ∈ `{0.0, 0.5}`（分别对应消融版和完整版）
- `n_epochs=1000`，`early_stop_patience=30`
- `train_ratio=0.6`，`val_ratio=0.2`，`label_smoothing=0.1`
- `selfloop=True`，`undirected=True`
- `n_runs=3`，`seed=42`

---

## 数据集

| 数据集 | 路径 | lr | n_layers | 特征 |
|--------|------|-----|----------|------|
| **Reddit-M** | `/mnt/input/MAGB_Dataset/Reddit-M/` | 0.0005 | 3 | Llama 3.2 11B Vision |
| **Movies** | `/mnt/input/MAGB_Dataset/Movies/` | 0.001 | 3 | Llama 3.2 11B Vision |
| **Grocery** | `/mnt/input/MAGB_Dataset/Grocery/` | 0.001 | 3 | Llama 3.2 11B Vision |
| **Toys** | `/mnt/input/MAGB_Dataset/Toys/` | 0.0005 | 2 | Llama 3.2 11B Vision |

---

## 实验配置矩阵

每个数据集 × 每个 GNN 算子 × 每个 aux_weight = 1 条命令，共 **4 × 4 × 2 = 32 条命令**。

### aux_weight = 0.0（消融版，验证架构本身）

#### SUPRA-GAT（aux=0.0）

```bash
# Reddit-M
python -m GNN.SUPRA \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --model_name GAT \
    --embed_dim 256 --n-layers 3 --n-hidden 256 \
    --dropout 0.3 --lr 0.001 --wd 0.0001 \
    --n_heads 4 --attn_drop 0.0 --edge_drop 0.0 \
    --aux_weight 0.0 --mlp_variant ablate \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/supra_gat_reddit_m_aux0.csv \
    --disable_wandb --gpu 0

# Movies
python -m GNN.SUPRA \
    --data_name Movies \
    --text_feature /mnt/input/MAGB_Dataset/Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Movies/MoviesGraph.pt \
    --model_name GAT --embed_dim 256 --n-layers 3 --n-hidden 256 \
    --dropout 0.3 --lr 0.001 --wd 0.0001 \
    --n_heads 4 --attn_drop 0.0 --edge_drop 0.0 \
    --aux_weight 0.0 --mlp_variant ablate \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/supra_gat_movies_aux0.csv \
    --disable_wandb --gpu 0

# Grocery
python -m GNN.SUPRA \
    --data_name Grocery \
    --text_feature /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --model_name GAT --embed_dim 256 --n-layers 3 --n-hidden 256 \
    --dropout 0.3 --lr 0.001 --wd 0.0001 \
    --n_heads 4 --attn_drop 0.0 --edge_drop 0.0 \
    --aux_weight 0.0 --mlp_variant ablate \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/supra_gat_grocery_aux0.csv \
    --disable_wandb --gpu 0

# Toys
python -m GNN.SUPRA \
    --data_name Toys \
    --text_feature /mnt/input/MAGB_Dataset/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Toys/ToysGraph.pt \
    --model_name GAT --embed_dim 256 --n-layers 2 --n-hidden 256 \
    --dropout 0.3 --lr 0.001 --wd 0.0001 \
    --n_heads 4 --attn_drop 0.0 --edge_drop 0.0 \
    --aux_weight 0.0 --mlp_variant ablate \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/supra_gat_toys_aux0.csv \
    --disable_wandb --gpu 0
```

#### SUPRA-SAGE（aux=0.0）

```bash
# Reddit-M
python -m GNN.SUPRA \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --model_name SAGE --embed_dim 256 --n-layers 3 --n-hidden 256 \
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
    --aggregator mean \
    --aux_weight 0.0 --mlp_variant ablate \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/supra_sage_reddit_m_aux0.csv \
    --disable_wandb --gpu 0

# Movies
python -m GNN.SUPRA \
    --data_name Movies \
    --text_feature /mnt/input/MAGB_Dataset/Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Movies/MoviesGraph.pt \
    --model_name SAGE --embed_dim 256 --n-layers 3 --n-hidden 256 \
    --dropout 0.3 --lr 0.001 --wd 0.0001 \
    --aggregator mean \
    --aux_weight 0.0 --mlp_variant ablate \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/supra_sage_movies_aux0.csv \
    --disable_wandb --gpu 0

# Grocery
python -m GNN.SUPRA \
    --data_name Grocery \
    --text_feature /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --model_name SAGE --embed_dim 256 --n-layers 3 --n-hidden 256 \
    --dropout 0.3 --lr 0.001 --wd 0.0001 \
    --aggregator mean \
    --aux_weight 0.0 --mlp_variant ablate \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/supra_sage_grocery_aux0.csv \
    --disable_wandb --gpu 0

# Toys
python -m GNN.SUPRA \
    --data_name Toys \
    --text_feature /mnt/input/MAGB_Dataset/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Toys/ToysGraph.pt \
    --model_name SAGE --embed_dim 256 --n-layers 2 --n-hidden 256 \
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
    --aggregator mean \
    --aux_weight 0.0 --mlp_variant ablate \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/supra_sage_toys_aux0.csv \
    --disable_wandb --gpu 0
```

#### SUPRA-JKNet（aux=0.0）

```bash
# Reddit-M
python -m GNN.SUPRA \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --model_name JKNet --embed_dim 256 --n-layers 3 --n-hidden 256 \
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
    --jknet_aggr concat \
    --aux_weight 0.0 --mlp_variant ablate \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/supra_jknet_reddit_m_aux0.csv \
    --disable_wandb --gpu 0

# Movies
python -m GNN.SUPRA \
    --data_name Movies \
    --text_feature /mnt/input/MAGB_Dataset/Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Movies/MoviesGraph.pt \
    --model_name JKNet --embed_dim 256 --n-layers 3 --n-hidden 256 \
    --dropout 0.3 --lr 0.001 --wd 0.0001 \
    --jknet_aggr concat \
    --aux_weight 0.0 --mlp_variant ablate \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/supra_jknet_movies_aux0.csv \
    --disable_wandb --gpu 0

# Grocery
python -m GNN.SUPRA \
    --data_name Grocery \
    --text_feature /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --model_name JKNet --embed_dim 256 --n-layers 3 --n-hidden 256 \
    --dropout 0.3 --lr 0.001 --wd 0.0001 \
    --jknet_aggr concat \
    --aux_weight 0.0 --mlp_variant ablate \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/supra_jknet_grocery_aux0.csv \
    --disable_wandb --gpu 0

# Toys
python -m GNN.SUPRA \
    --data_name Toys \
    --text_feature /mnt/input/MAGB_Dataset/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Toys/ToysGraph.pt \
    --model_name JKNet --embed_dim 256 --n-layers 2 --n-hidden 256 \
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
    --jknet_aggr concat \
    --aux_weight 0.0 --mlp_variant ablate \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/supra_jknet_toys_aux0.csv \
    --disable_wandb --gpu 0
```

### aux_weight = 0.5（完整版）

将上述所有命令中的 `--aux_weight 0.0` 替换为 `--aux_weight 0.5`，并更新输出 CSV 文件名中的 `aux0` → `aux0.5`。

示例（SUPRA-GAT，Reddit-M，aux=0.5）：

```bash
python -m GNN.SUPRA \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --model_name GAT \
    --embed_dim 256 --n-layers 3 --n-hidden 256 \
    --dropout 0.3 --lr 0.001 --wd 0.0001 \
    --n_heads 4 --attn_drop 0.0 --edge_drop 0.0 \
    --aux_weight 0.5 --mlp_variant ablate \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/supra_gat_reddit_m_aux0.5.csv \
    --disable_wandb --gpu 0
```

---

## F1 指标

F1 指标通过 `--metric f1_macro` 或 `--metric f1_weighted` 参数指定。需对每个 CSV 配置单独运行一次：

```bash
# 在上述所有命令中添加：--metric f1_macro
# 输出文件名改为 *_f1macro.csv
```

---

## 输出文件

```
Results/ablation/
├── supra_gat_reddit_m_aux0.csv
├── supra_gat_reddit_m_aux0.5.csv
├── supra_gat_movies_aux0.csv
├── supra_gat_movies_aux0.5.csv
├── supra_gat_grocery_aux0.csv
├── supra_gat_grocery_aux0.5.csv
├── supra_gat_toys_aux0.csv
├── supra_gat_toys_aux0.5.csv
├── supra_sage_reddit_m_aux0.csv
├── supra_sage_reddit_m_aux0.5.csv
├── supra_sage_movies_aux0.csv
├── supra_sage_movies_aux0.5.csv
├── supra_sage_grocery_aux0.csv
├── supra_sage_grocery_aux0.5.csv
├── supra_sage_toys_aux0.csv
├── supra_sage_toys_aux0.5.csv
├── supra_jknet_reddit_m_aux0.csv
├── supra_jknet_reddit_m_aux0.5.csv
├── supra_jknet_movies_aux0.csv
├── supra_jknet_movies_aux0.5.csv
├── supra_jknet_grocery_aux0.csv
├── supra_jknet_grocery_aux0.5.csv
├── supra_jknet_toys_aux0.csv
└── supra_jknet_toys_aux0.5.csv
```

---

## 代码架构

```
GNN/SUPRA.py                    # SUPRA 模型定义
├── _build_gnn_backbone()       # GNN 算子工厂（GCN/SAGE/GAT/RevGAT/JKNet）
├── _make_mp_layers()           # C 通道 GNN 层堆叠
└── forward_multiple()          # 三通道前向传播

GNN/Library/
├── GCN.py                      # GCN 实现
├── GAT.py                      # GAT 实现（多头注意力）
├── GraphSAGE.py                # SAGE 实现
└── JKNet.py                    # JKNet 实现（concat/max/last 聚合）
```

---

## 预期现象

1. **不同 GNN 算子性能接近**：SUPRA-GCN / GAT / SAGE / JKNet 在同一数据集上的 Accuracy 差异应在 1-2% 以内，证明 SUPRA 架构对 GNN 算子选择不敏感
2. **aux_weight=0.5 一致正增益**：在所有 GNN 算子下，aux_weight=0.5 相较于 aux_weight=0.0 均应有提升，证明辅助损失的梯度复活效应具有通用性
3. **附录图**：以热力图或分组柱状图展示 4 数据集 × 4 GNN 算子 × 2 aux_weight 的 Accuracy 矩阵
