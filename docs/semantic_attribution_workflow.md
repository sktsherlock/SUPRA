# Semantic Attribution Workflow

语义归因分析工作流，用于展示 SUPRA 的三通道设计如何保留模态独特语义。

**数据集**: Reddit-M、Grocery
**特征组**: Llama
**数据路径**: `/mnt/input/MAGB_Dataset`

---

## 一、工作流概述

```
Stage 1: 训练各模型  ─────────────────────────────────────────
  Text MLP        → save → text_mlp_test_pred.pt
  Image MLP       → save → image_mlp_test_pred.pt
  Late_GNN-GCN    → save → late_gnn_gcn_test_pred.pt
  Late_GNN-GAT    → save → late_gnn_gat_test_pred.pt
  Early_GNN-GCN   → save → early_gnn_gcn_test_pred.pt
  SUPRA           → save → supra_test_pred.pt
  NTSFormer      → save → ntsformer_test_pred.pt
  MIG_GT         → save → mig_gt_test_pred.pt

Stage 2: 语义归因分析  ─────────────────────────────────────
  python -m GNN.Utils.semantic_attribution \
      --data_name Reddit-M \
      --pred_dir Results/attribution/Reddit-M/ \
      --result_csv Results/attribution/Reddit-M/summary.csv \
      --save_plot Results/attribution/Reddit-M/attribution.png
```

---

## 二、Stage 1 — 训练 & 导出预测

所有模型统一使用原始训练脚本，通过 `--export_predictions` 参数在训练结束后自动导出测试集预测文件。

> **重要**：所有模型使用相同的 `--n-runs 1 --seed 42`，保证测试集划分完全一致，预测才可比较。

### 2.1 基础模型（Text MLP / Image MLP / Early_GNN / Late_GNN）

使用 `Early_GNN` 脚本跑单模态 MLP 和早期/晚期融合基线：

```bash
# Text MLP
python -m GNN.Baselines.Early_GNN \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --backend mlp --single_modality text \
    --n-layers 2 \
    --n-runs 1 --seed 42 \
    --result_csv Results/attribution/Reddit-M/text_mlp_best.csv \
    --export_predictions Results/attribution/Reddit-M/text_mlp_test_pred.pt \
    --disable_wandb --gpu 0

# Image MLP
python -m GNN.Baselines.Early_GNN \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --backend mlp --single_modality visual \
    --n-layers 3 \
    --n-runs 1 --seed 42 \
    --result_csv Results/attribution/Reddit-M/image_mlp_best.csv \
    --export_predictions Results/attribution/Reddit-M/image_mlp_test_pred.pt \
    --disable_wandb --gpu 0

# Late_GNN-GCN（无编码器基线，raw concat → 各自 GNN）
python -m GNN.Baselines.Late_GNN \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --model_name GCN \
    --n-layers 2 \
    --late_no_encoder true \
    --n-runs 1 --seed 42 \
    --result_csv Results/attribution/Reddit-M/late_gnn_gcn_best.csv \
    --export_predictions Results/attribution/Reddit-M/late_gnn_gcn_test_pred.pt \
    --disable_wandb --gpu 0

# Late_GNN-GAT
python -m GNN.Baselines.Late_GNN \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --model_name GAT \
    --n-layers 2 \
    --late_no_encoder true \
    --n-runs 1 --seed 42 \
    --result_csv Results/attribution/Reddit-M/late_gnn_gat_best.csv \
    --export_predictions Results/attribution/Reddit-M/late_gnn_gat_test_pred.pt \
    --disable_wandb --gpu 0

# Early_GNN-GCN（无编码器基线，raw concat → GNN）
python -m GNN.Baselines.Early_GNN \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --backend gnn --model_name GCN --early_no_encoder true \
    --n-layers 2 --lr 0.001 \
    --n-runs 1 --seed 42 \
    --result_csv Results/attribution/Reddit-M/early_gnn_gcn_best.csv \
    --export_predictions Results/attribution/Reddit-M/early_gnn_gcn_test_pred.pt \
    --disable_wandb --gpu 0
```

### 2.2 SUPRA / NTSFormer / MIG_GT

使用各自主训练脚本：

```bash
# SUPRA
python -m GNN.SUPRA \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --embed_dim 256 --n-layers 4 --n_hidden 256 \
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
    --aux_weight 0.1 --mlp_variant ablate \
    --n-runs 1 --seed 42 \
    --result_csv Results/attribution/Reddit-M/supra_best.csv \
    --export_predictions Results/attribution/Reddit-M/supra_test_pred.pt \
    --disable_wandb --gpu 0

# NTSFormer（仅 nts_sign_k=1 是 sweep 最优，其余用代码默认值）
python -m GNN.Baselines.NTSFormer \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --n_hidden 256 --n-layers 1 \
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
    --nts_sign_k 1 \
    --n-runs 1 --seed 42 \
    --result_csv Results/attribution/Reddit-M/ntsformer_best.csv \
    --export_predictions Results/attribution/Reddit-M/ntsformer_test_pred.pt \
    --disable_wandb --gpu 0

# MIG_GT
python -m GNN.Baselines.MIG_GT \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --n_hidden 256 --n-layers 1 \
    --dropout 0.3 --lr 0.001 --wd 0.0001 \
    --k_t 3 --k_v 2 \
    --mgdcf_alpha 0.1 --mgdcf_beta 0.9 \
    --num_samples 10 --tur_weight 1.0 \
    --n-runs 1 --seed 42 \
    --result_csv Results/attribution/Reddit-M/mig_gt_best.csv \
    --export_predictions Results/attribution/Reddit-M/mig_gt_test_pred.pt \
    --disable_wandb --gpu 0
```

---

## 三、Stage 2 — 语义归因可视化

所有预测文件齐全后，一键运行归因分析：

```bash
# Reddit-M
python -m GNN.Utils.semantic_attribution \
    --data_name Reddit-M \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --pred_dir Results/attribution/Reddit-M/ \
    --result_csv Results/attribution/Reddit-M/summary.csv \
    --save_plot Results/attribution/Reddit-M/attribution.png

# Grocery
python -m GNN.Utils.semantic_attribution \
    --data_name Grocery \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --pred_dir Results/attribution/Grocery/ \
    --result_csv Results/attribution/Grocery/summary.csv \
    --save_plot Results/attribution/Grocery/attribution.png
```

---

## 四、模型说明

| 模型 | 架构 | 堆叠图中的角色 |
|------|------|-------------|
| Text MLP | 单模态MLP（仅文本特征） | 提供 T-Unique 上界 |
| Image MLP | 单模态MLP（仅视觉特征） | 提供 V-Unique 上界 |
| Late_GNN-GCN | 双独立GNN → 融合 | 晚期融合基线 |
| Late_GNN-GAT | 双独立GAT → 融合 | 晚期融合基线 |
| Early_GNN-GCN | concat → GNN | 早期融合基线（≈ MMGCN） |
| SUPRA | 三通道（C/Ut/Uv） | 我们的方法 |
| NTSFormer | SIGN + Transformer | 多模态图Transformer基线 |
| MIG_GT | MGDCF + Transformer | 多模态图Transformer基线 |

---

## 五、预测文件命名规范

放在 `--pred_dir` 目录下：

| 文件名 | 对应模型 |
|--------|---------|
| `text_mlp_test_pred.pt` | Text MLP |
| `image_mlp_test_pred.pt` | Image MLP |
| `late_gnn_gcn_test_pred.pt` | Late_GNN-GCN |
| `late_gnn_gat_test_pred.pt` | Late_GNN-GAT |
| `early_gnn_gcn_test_pred.pt` | Early_GNN-GCN |
| `supra_test_pred.pt` | SUPRA |
| `ntsformer_test_pred.pt` | NTSFormer |
| `mig_gt_test_pred.pt` | MIG_GT |

每个 `.pt` 文件内容为 `torch.LongTensor` shape=[N_test]，即测试集节点的 argmax 预测类别。

---

## 六、预期结果解读

堆叠柱状图中的关键观察：

1. **Text MLP**: 只有 Shared + T-Unique（T-Unique 上界）
2. **Image MLP**: 只有 Shared + V-Unique（V-Unique 上界）
3. **Late_GNN / Early_GNN**: 相比单模态 MLP 增加了 Hard 列（Synergy），但 T-Unique 和 V-Unique 可能**缩水**（图传播污染）
4. **SUPRA**: 保留完整的 T-Unique + V-Unique（Ut/Uv 通道无图传播），同时通过 C 通道获得 Synergy 增益

**核心论点**: SUPRA 的 Ut/Uv 通道不受图传播污染，因此在模态独特语义上的表现最接近单模态 MLP 上界。
