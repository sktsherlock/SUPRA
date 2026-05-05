# GNN Backbone 消融实验 — SUPRA 架构通用性验证

验证**不同的 GNN 算子**（GAT / SAGE / JKNet）可替换 SUPRA 的默认 GCN，且在**任意 GNN 算子**下 SUPRA 架构均优于普通早期融合基线（Early_GNN）。

---

## 实验设计

### 两类架构 × 三种 GNN 算子

| | Early_GNN（基线） | SUPRA |
|---|---|---|
| 特征融合 | `concat(text, visual)` → GNN | 三通道解耦（C/Ut/Uv） |
| 图传播 | 所有模态共享结构，同质化聚合 | C 通道图传播，Ut/Uv 保留独特语义 |
| GNN 算子 | GAT / SAGE / JKNet | GAT / SAGE / JKNet |

### 实验分组

| 组 | 架构 | GNN 算子 | aux_weight | 数据集 | 命令数 |
|----|------|---------|-----------|--------|-------|
| 1 | Early_GNN | GAT / SAGE / JKNet | N/A | 4 | 12 |
| 2 | SUPRA | GAT / SAGE / JKNet | 0.0 / 0.5 | 4 | 24 |
| 合计 | | | | | **36** |

**已有数据**（可直接引用）：
- Early_GNN-GCN（Reddit-M, Movies, Grocery, Toys）：见 `efficiency_analysis_commands.md`
- SUPRA-GCN（aux=0.1）：见 `efficiency_analysis_commands.md`

---

## 共同超参数

所有模型共享（确保对比公平）：

| 参数 | 值 |
|------|-----|
| `embed_dim` / `n_hidden` | 256 |
| `n_layers` | 3（Toys 用 2） |
| `dropout` | 0.3 |
| `wd` | 0.0001 |
| `early_stop_patience` | 30 |
| `n_epochs` | 1000 |
| `train_ratio` | 0.6 |
| `val_ratio` | 0.2 |
| `label_smoothing` | 0.1 |
| `undirected` | True |
| `selfloop` | True |
| `n_runs` | 3 |
| `seed` | 42 |

数据集专用 lr：

| 数据集 | lr |
|--------|-----|
| Reddit-M | 0.0005 |
| Movies | 0.001 |
| Grocery | 0.001 |
| Toys | 0.0005 |

---

## 数据集路径

```
Reddit-M:
  /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy
  /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy
  /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt

Movies:
  /mnt/input/MAGB_Dataset/Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy
  /mnt/input/MAGB_Dataset/Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy
  /mnt/input/MAGB_Dataset/Movies/MoviesGraph.pt

Grocery:
  /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy
  /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy
  /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt

Toys:
  /mnt/input/MAGB_Dataset/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy
  /mnt/input/MAGB_Dataset/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy
  /mnt/input/MAGB_Dataset/Toys/ToysGraph.pt
```

---

## 组 1：Early_GNN（普通多模态 GNN 基线）

调用方式：`--backend gnn --early_no_encoder True`

### Early_GNN-GAT

```bash
# Reddit-M
python -m GNN.Baselines.Early_GNN \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --backend gnn --model_name GAT --early_no_encoder True \
    --n-hidden 256 --n-layers 3 --dropout 0.3 \
    --lr 0.0005 --wd 0.0001 \
    --n_heads 4 --attn_drop 0.0 --edge_drop 0.0 \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/early_gnn_gat_reddit_m.csv \
    --disable_wandb --gpu 0

# Movies
python -m GNN.Baselines.Early_GNN \
    --data_name Movies \
    --text_feature /mnt/input/MAGB_Dataset/Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Movies/MoviesGraph.pt \
    --backend gnn --model_name GAT --early_no_encoder True \
    --n-hidden 256 --n-layers 3 --dropout 0.3 \
    --lr 0.001 --wd 0.0001 \
    --n_heads 4 --attn_drop 0.0 --edge_drop 0.0 \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/early_gnn_gat_movies.csv \
    --disable_wandb --gpu 0

# Grocery
python -m GNN.Baselines.Early_GNN \
    --data_name Grocery \
    --text_feature /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --backend gnn --model_name GAT --early_no_encoder True \
    --n-hidden 256 --n-layers 3 --dropout 0.3 \
    --lr 0.001 --wd 0.0001 \
    --n_heads 4 --attn_drop 0.0 --edge_drop 0.0 \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/early_nnn_gat_grocery.csv \
    --disable_wandb --gpu 0

# Toys
python -m GNN.Baselines.Early_GNN \
    --data_name Toys \
    --text_feature /mnt/input/MAGB_Dataset/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Toys/ToysGraph.pt \
    --backend gnn --model_name GAT --early_no_encoder True \
    --n-hidden 256 --n-layers 2 --dropout 0.3 \
    --lr 0.0005 --wd 0.0001 \
    --n_heads 4 --attn_drop 0.0 --edge_drop 0.0 \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/early_gnn_gat_toys.csv \
    --disable_wandb --gpu 0
```

### Early_GNN-SAGE

```bash
# Reddit-M
python -m GNN.Baselines.Early_GNN \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --backend gnn --model_name SAGE --early_no_encoder True \
    --n-hidden 256 --n-layers 3 --dropout 0.3 \
    --lr 0.0005 --wd 0.0001 \
    --aggregator mean \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/early_gnn_sage_reddit_m.csv \
    --disable_wandb --gpu 0

# Movies
python -m GNN.Baselines.Early_GNN \
    --data_name Movies \
    --text_feature /mnt/input/MAGB_Dataset/Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Movies/MoviesGraph.pt \
    --backend gnn --model_name SAGE --early_no_encoder True \
    --n-hidden 256 --n-layers 3 --dropout 0.3 \
    --lr 0.001 --wd 0.0001 \
    --aggregator mean \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/early_gnn_sage_movies.csv \
    --disable_wandb --gpu 0

# Grocery
python -m GNN.Baselines.Early_GNN \
    --data_name Grocery \
    --text_feature /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --backend gnn --model_name SAGE --early_no_encoder True \
    --n-hidden 256 --n-layers 3 --dropout 0.3 \
    --lr 0.001 --wd 0.0001 \
    --aggregator mean \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/early_gnn_sage_grocery.csv \
    --disable_wandb --gpu 0

# Toys
python -m GNN.Baselines.Early_GNN \
    --data_name Toys \
    --text_feature /mnt/input/MAGB_Dataset/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Toys/ToysGraph.pt \
    --backend gnn --model_name SAGE --early_no_encoder True \
    --n-hidden 256 --n-layers 2 --dropout 0.3 \
    --lr 0.0005 --wd 0.0001 \
    --aggregator mean \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/early_gnn_sage_toys.csv \
    --disable_wandb --gpu 0
```

### Early_GNN-JKNet

```bash
# Reddit-M
python -m GNN.Baselines.Early_GNN \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --backend gnn --model_name JKNet --early_no_encoder True \
    --n-hidden 256 --n-layers 3 --dropout 0.3 \
    --lr 0.0005 --wd 0.0001 \
    --jknet_aggr concat \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/early_gnn_jknet_reddit_m.csv \
    --disable_wandb --gpu 0

# Movies
python -m GNN.Baselines.Early_GNN \
    --data_name Movies \
    --text_feature /mnt/input/MAGB_Dataset/Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Movies/MoviesGraph.pt \
    --backend gnn --model_name JKNet --early_no_encoder True \
    --n-hidden 256 --n-layers 3 --dropout 0.3 \
    --lr 0.001 --wd 0.0001 \
    --jknet_aggr concat \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/early_gnn_jknet_movies.csv \
    --disable_wandb --gpu 0

# Grocery
python -m GNN.Baselines.Early_GNN \
    --data_name Grocery \
    --text_feature /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --backend gnn --model_name JKNet --early_no_encoder True \
    --n-hidden 256 --n-layers 3 --dropout 0.3 \
    --lr 0.001 --wd 0.0001 \
    --jknet_aggr concat \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/early_gnn_jknet_grocery.csv \
    --disable_wandb --gpu 0

# Toys
python -m GNN.Baselines.Early_GNN \
    --data_name Toys \
    --text_feature /mnt/input/MAGB_Dataset/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Toys/ToysGraph.pt \
    --backend gnn --model_name JKNet --early_no_encoder True \
    --n-hidden 256 --n-layers 2 --dropout 0.3 \
    --lr 0.0005 --wd 0.0001 \
    --jknet_aggr concat \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/early_gnn_jknet_toys.csv \
    --disable_wandb --gpu 0
```

---

## 组 2：SUPRA（不同 GNN 算子，aux=0.0 和 aux=0.5）

调用方式：`python -m GNN.SUPRA --model_name <GNN> --mlp_variant ablate --aux_weight <X>`

每个 GNN 算子 × 每个数据集 × aux_weight ∈ {0.0, 0.5} = 3 × 4 × 2 = 24 条命令。

**特殊 GNN 参数**（与 Early_GNN 相同）：

| GNN | 额外参数 |
|-----|---------|
| GAT | `--n_heads 4 --attn_drop 0.0 --edge_drop 0.0` |
| SAGE | `--aggregator mean` |
| JKNet | `--jknet_aggr concat` |

### SUPRA-GAT

```bash
# ============ aux=0.0 ============
# Reddit-M
python -m GNN.SUPRA \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --model_name GAT --embed_dim 256 --n-layers 3 --n-hidden 256 \
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
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
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
    --n_heads 4 --attn_drop 0.0 --edge_drop 0.0 \
    --aux_weight 0.0 --mlp_variant ablate \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/supra_gat_toys_aux0.csv \
    --disable_wandb --gpu 0

# ============ aux=0.5 ============
# 将 --aux_weight 0.0 改为 --aux_weight 0.5，文件名改为 *_aux0.5.csv
# Reddit-M
python -m GNN.SUPRA \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --model_name GAT --embed_dim 256 --n-layers 3 --n-hidden 256 \
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
    --n_heads 4 --attn_drop 0.0 --edge_drop 0.0 \
    --aux_weight 0.5 --mlp_variant ablate \
    --n-runs 3 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/ablation/supra_gat_reddit_m_aux0.5.csv \
    --disable_wandb --gpu 0

# Movies / Grocery / Toys（aux=0.5）: 同上改 --aux_weight 0.5
```

### SUPRA-SAGE

```bash
# ============ aux=0.0 ============
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

# Movies / Grocery / Toys（aux=0.0）: 同上格式
# ============ aux=0.5 ============
# 将 --aux_weight 0.0 改为 --aux_weight 0.5，文件名改为 *_aux0.5.csv
```

### SUPRA-JKNet

```bash
# ============ aux=0.0 ============
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

# Movies / Grocery / Toys（aux=0.0）: 同上格式
# ============ aux=0.5 ============
# 将 --aux_weight 0.0 改为 --aux_weight 0.5，文件名改为 *_aux0.5.csv
```

---

## F1 指标

在所有 accuracy 命令中加入 `--metric f1_macro`，输出文件名改为 `*_f1macro.csv`。

---

## 输出文件汇总

```
Results/ablation/
# Early_GNN 基线
├── early_gnn_gat_reddit_m.csv
├── early_gnn_gat_movies.csv
├── early_gnn_gat_grocery.csv
├── early_gnn_gat_toys.csv
├── early_gnn_sage_reddit_m.csv
├── early_gnn_sage_movies.csv
├── early_gnn_sage_grocery.csv
├── early_gnn_sage_toys.csv
├── early_gnn_jknet_reddit_m.csv
├── early_gnn_jknet_movies.csv
├── early_gnn_jknet_grocery.csv
├── early_gnn_jknet_toys.csv
# SUPRA (aux=0.0)
├── supra_gat_reddit_m_aux0.csv
├── supra_gat_movies_aux0.csv
├── supra_gat_grocery_aux0.csv
├── supra_gat_toys_aux0.csv
├── supra_sage_reddit_m_aux0.csv
├── supra_sage_movies_aux0.csv
├── supra_sage_grocery_aux0.csv
├── supra_sage_toys_aux0.csv
├── supra_jknet_reddit_m_aux0.csv
├── supra_jknet_movies_aux0.csv
├── supra_jknet_grocery_aux0.csv
├── supra_jknet_toys_aux0.csv
# SUPRA (aux=0.5)
├── supra_gat_reddit_m_aux0.5.csv
├── supra_gat_movies_aux0.5.csv
├── supra_gat_grocery_aux0.5.csv
├── supra_gat_toys_aux0.5.csv
├── supra_sage_reddit_m_aux0.5.csv
├── supra_sage_movies_aux0.5.csv
├── supra_sage_grocery_aux0.5.csv
├── supra_sage_toys_aux0.5.csv
├── supra_jknet_reddit_m_aux0.5.csv
├── supra_jknet_movies_aux0.5.csv
├── supra_jknet_grocery_aux0.5.csv
└── supra_jknet_toys_aux0.5.csv
```

---

## 预期现象

1. **SUPRA 全面优于 Early_GNN**：在任意 GNN 算子（GAT/SAGE/JKNet）下，SUPRA(aux=0.0) 的 Accuracy 均高于对应的 Early_GNN
2. **aux_weight=0.5 一致正增益**：在所有 GNN 算子下，aux_weight=0.5 相较于 aux_weight=0.0 有提升
3. **GCN vs GAT vs SAGE vs JKNet**：同一架构下不同 GNN 算子性能差异应在 1-2% 以内，证明 SUPRA 架构优势与具体 GNN 算子选择无关

---

## 代码架构

```
GNN/SUPRA.py                    # SUPRA 模型，三通道解耦架构
├── _build_gnn_backbone()       # GNN 算子工厂（GCN/SAGE/GAT/JKNet）
└── forward_multiple()           # 三通道前向传播

GNN/Baselines/Early_GNN.py      # Early_GNN 基线（concat → GNN → classifier）
└── _build_gnn_backbone()       # GNN 算子工厂（支持全部 8 种算子）
```
