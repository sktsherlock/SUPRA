# Gradient Starvation Verification Experiment

验证辅助损失（Auxiliary Loss）对模态投影器（enc_t / enc_v）的梯度"复活"效果。

---

## 核心变量

| 实验组 | `aux_weight` | 预期现象 |
|--------|-------------|---------|
| SUPRA Base | `0.0` | 联合优化中任务梯度衰减，所有模块陷入低迷（梯度集体衰减） |
| SUPRA w/ aux | 最佳值 | 辅助损失强心剂作用，投影器梯度被极大放大，保持高度活跃 |

---

## Grocery 数据集

> 来源: `Results/supra_gpu0_default_accuracy_best.csv`
>
> - **Backbone**: GCN
> - **n_layers**: 3
> - **embed_dim**: 256
> - **aux_weight (Base)**: 0.0
> - **aux_weight (aux)**: 0.7
> - **lr**: 0.001
> - **wd**: 0.0001
> - **dropout**: 0.3
> - **mlp_variant**: ablate

**数据路径**（服务器路径）:
```
/mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy
/mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy
/mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt
```

### Run 1 — Grocery SUPRA Base（aux_weight = 0.0）

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
    --gradient_csv Results/gradient_starvation/grocery_base.csv \
    --disable_wandb
```

### Run 2 — Grocery SUPRA w/ aux（aux_weight = 0.7）

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
    --gradient_csv Results/gradient_starvation/grocery_aux.csv \
    --disable_wandb
```

---

## Toys 数据集

> 来源: `Results/supra_gpu0_default_accuracy_best.csv`
>
> - **Backbone**: GCN
> - **n_layers**: 2
> - **embed_dim**: 256
> - **aux_weight (Base)**: 0.0
> - **aux_weight (aux)**: 0.5
> - **lr**: 0.0005
> - **wd**: 0.0001
> - **dropout**: 0.3
> - **mlp_variant**: ablate

**数据路径**（服务器路径）:
```
/mnt/input/MAGB_Dataset/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy
/mnt/input/MAGB_Dataset/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy
/mnt/input/MAGB_Dataset/Toys/ToysGraph.pt
```

### Run 3 — Toys SUPRA Base（aux_weight = 0.0）

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
    --gradient_csv Results/gradient_starvation/toys_base.csv \
    --disable_wandb
```

### Run 4 — Toys SUPRA w/ aux（aux_weight = 0.5）

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
    --gradient_csv Results/gradient_starvation/toys_aux.csv \
    --disable_wandb
```

**输出文件**:
```
Results/gradient_starvation/grocery_base_l2_norm_run1.csv
Results/gradient_starvation/grocery_aux_l2_norm_run1.csv
Results/gradient_starvation/toys_base_l2_norm_run1.csv
Results/gradient_starvation/toys_aux_l2_norm_run1.csv
```

---

## 生成对比图

### Grocery

```bash
python -m GNN.Utils.plot_gradient_norm \
    --csv_base Results/gradient_starvation/grocery_base_l2_norm_run1.csv \
    --csv_aux Results/gradient_starvation/grocery_aux_l2_norm_run1.csv \
    --aux_label "aux_weight=0.7" \
    --max_epoch 50 \
    --save_plot Results/gradient_starvation/grocery_gradient_starvation.pdf
```

### Toys

```bash
python -m GNN.Utils.plot_gradient_norm \
    --csv_base Results/gradient_starvation/toys_base_l2_norm_run1.csv \
    --csv_aux Results/gradient_starvation/toys_aux_l2_norm_run1.csv \
    --aux_label "aux_weight=0.5" \
    --max_epoch 50 \
    --save_plot Results/gradient_starvation/toys_gradient_starvation.pdf
```

**预期现象**:
- 左图（aux_weight=0.0）：前 10 个 epoch 三条线集体冲高后迅速衰减，所有模块陷入低迷（梯度衰减，非饥饿）
- 右图（aux_weight>0）：蓝色（enc_t）和黄色（enc_v）投影器梯度被极大放大，与 GNN 保持在同一量级，实现"梯度复活"

---

## 代码实现说明

### 梯度提取逻辑（`GNN/SUPRA.py`）

```python
total_loss.backward()

if getattr(args, 'analyze_gradients', False):
    def _grad_norm_sq(m):
        return sum(
            p.grad.float().norm(2).pow(2).item()
            for p in m.parameters()
            if p.grad is not None
        )
    grad_history['enc_t'].append(_grad_norm_sq(model.enc_t) ** 0.5)
    grad_history['enc_v'].append(_grad_norm_sq(model.enc_v) ** 0.5)
    gnn_norm_sq = sum(_grad_norm_sq(layer) for layer in model.mp_C)
    grad_history['gnn'].append(gnn_norm_sq ** 0.5)

optimizer.step()
```

### 追踪的模块

| 模块 | 含义 |
|------|------|
| `model.enc_t` | 文本模态投影器（Text Projector） |
| `model.enc_v` | 视觉模态投影器（Visual Projector） |
| `model.mp_C` | 共享 GNN 通道（Shared GNN Layers） |

### 可视化脚本

`GNN/Utils/plot_gradient_norm.py` — 左右子图折线图，支持 `--max_epoch` 截断数据，输出 PDF 矢量格式。
