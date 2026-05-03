# Gradient Starvation Verification Experiment

验证辅助损失（Auxiliary Loss）对模态投影器（enc_t / enc_v）的梯度"复活"效果。

---

## 核心变量

| 实验组 | `aux_weight` | 预期现象 |
|--------|-------------|---------|
| SUPRA Base | `0.0` | GNN 通道梯度高位，enc_t/enc_v 迅速贴底（梯度饥饿） |
| SUPRA w/ aux | `0.3`（最佳） | 三条线相对均衡，投影器获得充足梯度 |

---

## Grocery 数据集最佳参数

> 来源: `Results/supra_gpu0_default_accuracy_best.csv`
>
> - **数据集**: Grocery
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

---

## 实验命令

### Run 1 — SUPRA Base（aux_weight = 0.0，无辅助损失）

```bash
python -m GNN.SUPRA \
    --data_name Grocery \
    --text_feature /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --backbone GCN \
    --n-layers 3 \
    --embed_dim 256 \
    --lr 0.001 \
    --wd 0.0001 \
    --dropout 0.3 \
    --aux_weight 0.0 \
    --mlp_variant ablate \
    --n-epochs 300 \
    --n-runs 1 \
    --eval_steps 1 \
    --analyze_gradients \
    --gradient_csv Results/gradient_starvation/grocery_base.csv
```

### Run 2 — SUPRA w/ aux（aux_weight = 0.7）

```bash
python -m GNN.SUPRA \
    --data_name Grocery \
    --text_feature /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --backbone GCN \
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
    --analyze_gradients \
    --gradient_csv Results/gradient_starvation/grocery_aux.csv
```

**输出文件**（每个 run 结束后打印路径）:
```
Results/gradient_starvation/grocery_base_l2_norm_run1.csv
Results/gradient_starvation/grocery_aux_l2_norm_run1.csv
```

---

## 生成对比图

```bash
python -m GNN.Utils.plot_gradient_norm \
    --csv_base Results/gradient_starvation/grocery_base_l2_norm_run1.csv \
    --csv_aux Results/gradient_starvation/grocery_aux_l2_norm_run1.csv \
    --aux_label "aux_weight=0.7" \
    --save_plot Results/gradient_starvation/grocery_gradient_starvation.pdf
```

**预期现象**:
- 左图（aux_weight=0.0）：GNN 线条高位平稳，enc_t / enc_v 在数个 epoch 后迅速下降趋近于 0
- 右图（aux_weight=0.7）：enc_t / enc_v 线条被明显抬升，与 GNN 保持在同一数量级

---

## 代码实现说明

### 梯度提取逻辑（`GNN/SUPRA.py`）

```python
# After loss.backward(), before optimizer.step()
total_loss.backward()

if getattr(args, 'analyze_gradients', False):
    def _grad_norm(m):
        return th.sqrt(sum(
            p.grad.float().norm(2).pow(2)
            for p in m.parameters()
            if p.grad is not None
        )).item()
    grad_history['enc_t'].append(_grad_norm(model.enc_t))
    grad_history['enc_v'].append(_grad_norm(model.enc_v))
    gnn_norm_sq = sum(_grad_norm(layer).pow(2) for layer in model.mp_C)
    grad_history['gnn'].append(th.sqrt(gnn_norm_sq).item())

optimizer.step()
```

### 追踪的模块

| 模块 | 含义 |
|------|------|
| `model.enc_t` | 文本模态投影器（Text Projector） |
| `model.enc_v` | 视觉模态投影器（Visual Projector） |
| `model.mp_C` | 共享 GNN 通道（Shared GNN Layers） |

### 可视化脚本

`GNN/Utils/plot_gradient_norm.py` — 左右子图折线图，输出 PDF 矢量格式。
