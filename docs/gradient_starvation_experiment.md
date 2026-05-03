# Gradient Starvation Verification Experiment

验证辅助损失（Auxiliary Loss）对模态投影器（enc_t / enc_v）的梯度"复活"效果。

---

## 实验设计：4 阶段梯度追踪

| 组别 | 模型 | 配置 | 追踪目标 |
|------|------|------|---------|
| **Group 1** | MMGCN (Late_GNN) | 传统多模态 GNN，各模态独立 GNN 聚合后融合 | text_enc, vis_enc, mmgnn |
| **Group 2** | SUPRA (No Bypass) | `ablate_bypass`：强制 `logits_final = logits_C`，去除 Ut/Uv 分支 | enc_t, enc_v, gnn |
| **Group 3** | SUPRA Base | `aux_weight = 0`：三通道融合但无辅助损失 | enc_t, enc_v, gnn |
| **Group 4** | SUPRA Full | `aux_weight > 0`：三通道 + 辅助损失 | enc_t, enc_v, gnn |

**核心论点**：Group 1 → Group 2 → Group 3 → Group 4 构成严密逻辑链，逐步证明 SUPRA 每个组件对梯度的挽救作用。

---

## Grocery 数据集参数

> 来源: `Results/supra_gpu0_default_accuracy_best.csv`

| 参数 | 值 |
|------|-----|
| Backbone | GCN |
| n_layers | 3 |
| embed_dim | 256 |
| lr | 0.001 |
| wd | 0.0001 |
| dropout | 0.3 |
| mlp_variant | ablate |

### Group 1 — MMGCN (Late_GNN)

```bash
python -m GNN.Baselines.Late_GNN \
    --data_name Grocery \
    --text_feature /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --model_name GCN \
    --n_layers 3 \
    --n_hidden 256 \
    --dropout 0.3 \
    --lr 0.001 \
    --wd 0.0001 \
    --n_epochs 300 \
    --n_runs 1 \
    --eval_steps 1 \
    --early_stop_patience 50 \
    --analyze_gradients \
    --gradient_csv Results/gradient_starvation/grocery_g1_mmgen.csv \
    --disable_wandb
```

### Group 2 — SUPRA (No Bypass)

```bash
python -m GNN.SUPRA \
    --data_name Grocery \
    --text_feature /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --model_name GCN \
    --n_layers 3 \
    --embed_dim 256 \
    --lr 0.001 \
    --wd 0.0001 \
    --dropout 0.3 \
    --aux_weight 0.0 \
    --mlp_variant ablate \
    --ablate_bypass \
    --n_epochs 300 \
    --n_runs 1 \
    --eval_steps 1 \
    --early_stop_patience 50 \
    --analyze_gradients \
    --gradient_csv Results/gradient_starvation/grocery_g2_no_bypass.csv \
    --disable_wandb
```

### Group 3 — SUPRA Base（aux_weight = 0.0）

```bash
python -m GNN.SUPRA \
    --data_name Grocery \
    --text_feature /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --model_name GCN \
    --n_layers 3 \
    --embed_dim 256 \
    --lr 0.001 \
    --wd 0.0001 \
    --dropout 0.3 \
    --aux_weight 0.0 \
    --mlp_variant ablate \
    --n_epochs 300 \
    --n_runs 1 \
    --eval_steps 1 \
    --early_stop_patience 50 \
    --analyze_gradients \
    --gradient_csv Results/gradient_starvation/grocery_g3_base.csv \
    --disable_wandb
```

### Group 4 — SUPRA Full（aux_weight = 0.7）

```bash
python -m GNN.SUPRA \
    --data_name Grocery \
    --text_feature /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --model_name GCN \
    --n_layers 3 \
    --embed_dim 256 \
    --lr 0.001 \
    --wd 0.0001 \
    --dropout 0.3 \
    --aux_weight 0.7 \
    --mlp_variant ablate \
    --n_epochs 300 \
    --n_runs 1 \
    --eval_steps 1 \
    --early_stop_patience 50 \
    --analyze_gradients \
    --gradient_csv Results/gradient_starvation/grocery_g4_full.csv \
    --disable_wandb
```

**输出文件**:
```
Results/gradient_starvation/grocery_g1_mmgen_l2_norm_run1.csv
Results/gradient_starvation/grocery_g2_no_bypass_l2_norm_run1.csv
Results/gradient_starvation/grocery_g3_base_l2_norm_run1.csv
Results/gradient_starvation/grocery_g4_full_l2_norm_run1.csv
```

---

## Toys 数据集参数

> 来源: `Results/supra_gpu0_default_accuracy_best.csv`

| 参数 | 值 |
|------|-----|
| Backbone | GCN |
| n_layers | 2 |
| embed_dim | 256 |
| lr | 0.0005 |
| wd | 0.0001 |
| dropout | 0.3 |
| mlp_variant | ablate |
| aux_weight (Full) | 0.5 |

### Group 1 — MMGCN

```bash
python -m GNN.Baselines.Late_GNN \
    --data_name Toys \
    --text_feature /mnt/input/MAGB_Dataset/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Toys/ToysGraph.pt \
    --model_name GCN \
    --n_layers 2 \
    --n_hidden 256 \
    --dropout 0.3 \
    --lr 0.0005 \
    --wd 0.0001 \
    --n_epochs 300 \
    --n_runs 1 \
    --eval_steps 1 \
    --early_stop_patience 50 \
    --analyze_gradients \
    --gradient_csv Results/gradient_starvation/toys_g1_mmgen.csv \
    --disable_wandb
```

### Group 2 — SUPRA (No Bypass)

```bash
python -m GNN.SUPRA \
    --data_name Toys \
    --text_feature /mnt/input/MAGB_Dataset/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Toys/ToysGraph.pt \
    --model_name GCN \
    --n_layers 2 \
    --embed_dim 256 \
    --lr 0.0005 \
    --wd 0.0001 \
    --dropout 0.3 \
    --aux_weight 0.0 \
    --mlp_variant ablate \
    --ablate_bypass \
    --n_epochs 300 \
    --n_runs 1 \
    --eval_steps 1 \
    --early_stop_patience 50 \
    --analyze_gradients \
    --gradient_csv Results/gradient_starvation/toys_g2_no_bypass.csv \
    --disable_wandb
```

### Group 3 — SUPRA Base

```bash
python -m GNN.SUPRA \
    --data_name Toys \
    --text_feature /mnt/input/MAGB_Dataset/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Toys/ToysGraph.pt \
    --model_name GCN \
    --n_layers 2 \
    --embed_dim 256 \
    --lr 0.0005 \
    --wd 0.0001 \
    --dropout 0.3 \
    --aux_weight 0.0 \
    --mlp_variant ablate \
    --n_epochs 300 \
    --n_runs 1 \
    --eval_steps 1 \
    --early_stop_patience 50 \
    --analyze_gradients \
    --gradient_csv Results/gradient_starvation/toys_g3_base.csv \
    --disable_wandb
```

### Group 4 — SUPRA Full（aux_weight = 0.5）

```bash
python -m GNN.SUPRA \
    --data_name Toys \
    --text_feature /mnt/input/MAGB_Dataset/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Toys/ToysGraph.pt \
    --model_name GCN \
    --n_layers 2 \
    --embed_dim 256 \
    --lr 0.0005 \
    --wd 0.0001 \
    --dropout 0.3 \
    --aux_weight 0.5 \
    --mlp_variant ablate \
    --n_epochs 300 \
    --n_runs 1 \
    --eval_steps 1 \
    --early_stop_patience 50 \
    --analyze_gradients \
    --gradient_csv Results/gradient_starvation/toys_g4_full.csv \
    --disable_wandb
```

**输出文件**:
```
Results/gradient_starvation/toys_g1_mmgen_l2_norm_run1.csv
Results/gradient_starvation/toys_g2_no_bypass_l2_norm_run1.csv
Results/gradient_starvation/toys_g3_base_l2_norm_run1.csv
Results/gradient_starvation/toys_g4_full_l2_norm_run1.csv
```

---

## 生成对比图

### 4 组对比图（Grocery）

```bash
python -m GNN.Utils.plot_gradient_norm --mode 4 \
    --csv_1 Results/gradient_starvation/grocery_g1_mmgen_l2_norm_run1.csv \
    --csv_2 Results/gradient_starvation/grocery_g2_no_bypass_l2_norm_run1.csv \
    --csv_3 Results/gradient_starvation/grocery_g3_base_l2_norm_run1.csv \
    --csv_4 Results/gradient_starvation/grocery_g4_full_l2_norm_run1.csv \
    --label_1 "MMGCN\n(Late_GNN)" \
    --label_2 "SUPRA (No Bypass)\nC-only" \
    --label_3 "SUPRA Base\n(aux=0)" \
    --label_4 "SUPRA Full\n(aux=0.7)" \
    --max_epoch 50 \
    --save_plot Results/gradient_starvation/grocery_4group.pdf
```

### 4 组对比图（Toys）

```bash
python -m GNN.Utils.plot_gradient_norm --mode 4 \
    --csv_1 Results/gradient_starvation/toys_g1_mmgen_l2_norm_run1.csv \
    --csv_2 Results/gradient_starvation/toys_g2_no_bypass_l2_norm_run1.csv \
    --csv_3 Results/gradient_starvation/toys_g3_base_l2_norm_run1.csv \
    --csv_4 Results/gradient_starvation/toys_g4_full_l2_norm_run1.csv \
    --label_1 "MMGCN\n(Late_GNN)" \
    --label_2 "SUPRA (No Bypass)\nC-only" \
    --label_3 "SUPRA Base\n(aux=0)" \
    --label_4 "SUPRA Full\n(aux=0.5)" \
    --max_epoch 50 \
    --save_plot Results/gradient_starvation/toys_4group.pdf
```

**预期逻辑链现象**:
- Group 1（MMGCN）：各模态 GNN 梯度分离，无统一协调
- Group 2（SUPRA No Bypass）：共享 GNN 存在，但无 bypass 保护，整体梯度极低（拓扑瓶颈）
- Group 3（SUPRA Base）：有 bypass，但无辅助损失，三通道整体梯度衰减
- Group 4（SUPRA Full）：辅助损失激活投影器，所有模块梯度被"复活"

---

## 代码改动说明

### `--ablate_bypass` flag（SUPRA.py）

添加 `--ablate_bypass` 参数，将 `logits_final` 强制设为 `logits_C`，去除 Ut/Uv 分支：

```python
ablate_bypass = getattr(args, "ablate_bypass", False)
if ablate_bypass:
    logits_final = out.logits_C_0
    total_task_loss = cross_entropy(logits_final[idx], labels[idx], label_smoothing=ls)
    logs = {"loss/task": float(total_task_loss.detach().cpu().item())}
    return total_task_loss, logs
```

### 梯度追踪（Late_GNN.py）

Late_GNN 训练循环中新增梯度 L2 范数追踪（`text_enc`, `vis_enc`, `mmgnn`），与 SUPRA 格式兼容但列名不同（`plot_gradient_norm.py` 自动识别）。

### 可视化脚本

`GNN/Utils/plot_gradient_norm.py` 支持两种模式：
- `--mode 2`：2 组对比（左 Base，右 Full）
- `--mode 4`：4 组 2×2 子图（MMGCN → No Bypass → Base → Full）
