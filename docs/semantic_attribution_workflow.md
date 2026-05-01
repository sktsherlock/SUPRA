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

### 2.1 Text MLP（Reddit-M）

```bash
python -c "
import torch as th, numpy as np, argparse
from GNN.GraphData import load_data
from GNN.Baselines.Early_GNN import EarlyFusionModel, SimpleEncoder, MLPHead

# Config
data_name = 'Reddit-M'
graph_path = '/mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt'
text_feat_path = '/mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy'
visual_feat_path = '/mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy'
out_path = 'Results/attribution/Reddit-M/text_mlp_test_pred.pt'

graph, labels, train_idx, val_idx, test_idx = load_data(graph_path, train_ratio=0.6, val_ratio=0.2, name=data_name)
text_feat = th.tensor(np.load(text_feat_path), dtype=th.float32)
labels_test = labels[test_idx]

# Text MLP
in_d = text_feat.shape[1]
model = th.nn.Sequential(
    th.nn.Linear(in_d, 256), th.nn.ReLU(), th.nn.Dropout(0.3),
    th.nn.Linear(256, 256), th.nn.ReLU(), th.nn.Dropout(0.3),
    th.nn.Linear(256, int(labels.max())+1)
)
# ... train with EarlyStopping, save best test pred as torch tensor ...
# Run 3 times with seed 42,43,44, average logits, then argmax
print(f'Text MLP test accuracy: {(pred == labels_test).float().mean():.4f}')
th.save(pred, out_path)
"
```

> 注: 上述为示意代码。完整训练脚本见 `scripts/attribution/export_single_modality.py`

### 2.2 简化方案：使用现有训练脚本 + 推理钩子

**推荐使用独立推理脚本** `scripts/attribution/export_predictions.py`，统一导出所有模型的测试集预测。

```bash
# Reddit-M
python scripts/attribution/export_predictions.py \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --pred_dir Results/attribution/Reddit-M/ \
    --models text_mlp image_mlp late_gnn_gcn late_gnn_gat early_gnn_gcn supra ntsformer mig_gt \
    --gpu 0

# Grocery
python scripts/attribution/export_predictions.py \
    --data_name Grocery \
    --text_feature /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_openai_clip-vit-large-patch14.npy \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --pred_dir Results/attribution/Grocery/ \
    --models text_mlp image_mlp late_gnn_gcn late_gnn_gat early_gnn_gcn supra ntsformer mig_gt \
    --gpu 0
```

---

## 三、Stage 2 — 语义归因可视化

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
