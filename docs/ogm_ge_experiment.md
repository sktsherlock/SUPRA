# OGM-GE 梯度调制实验

验证 OGM-GE（On-the-fly Gradient Modulation with Gaussian Enhancement）与 SUPRA 辅助通道（aux_weight）在平衡模态学习上的效果对比。

---

## 理论背景

### 原版 OGM-GE（CVPR 2022）

论文原文方法通过 `softmax(logits)` 在正确类上的概率之和来判断模态强弱：

```
score_a = sum(softmax(logits_audio)[i, label[i]])
score_v = sum(softmax(logits_visual)[i, label[i]])
ratio = score_a / score_v
```

如果 `ratio > 1`（audio 强于 visual），则对 audio 相关层梯度乘以 `1 - tanh(α * ratio)` 压低，反之亦然。**本质是两模态互相压制。**

### 我们的改进： C 通道作为锚点

SUPRA 的三通道架构天然包含一个稳定的锚点——C 通道（GNN 聚合后的共享语义）。我们以 C 为锚来衡量 Ut/Uv 的相对强度：

```
score_Ut = sum(softmax(logits_Ut)[i, label[i]])
score_Uv = sum(softmax(logits_Uv)[i, label[i]])
score_C  = sum(softmax(logits_C)[i, label[i]])

ratio_t = score_Ut / score_C
ratio_v = score_Uv / score_C
```

- `ratio_t > 1`：Ut 预测正确性远高于 C → Ut 可能在绕过 GNN 聚合自行收敛 → **压制 Ut**
- `ratio_t ≤ 1`：Ut 弱于或等于 C → Ut 在独立语义上学得不够 → **不压制**

C 通道本身的梯度**始终不参与调节**，作为稳定的参考基准。

### 与 aux_weight 的关系

| 机制 | 层次 | 作用方式 |
|------|------|---------|
| `aux_weight` | Loss 层面 | 给 Ut/Uv 额外的 loss 信号，让它们从各自的 logits 回传梯度强化 |
| OGM-GE | 梯度层面 | 在 backward 后、optimizer.step 前，根据 Ut/Uv 相对于 C 的强弱动态压低其 encoder/head 的梯度 |

两者可叠加（aux_weight 强化 Ut/Uv 的 loss 信号，OGM-GE 调节 encoder/head 的梯度方向）。

---

## 实验设计

在 Grocery 数据集上运行 2×2 因子实验：

| 实验 | aux_weight | use_ogm_ge | 说明 |
|------|------------|------------|------|
| **G1** | 0.0 | ✗ | 纯 C 通道（基线） |
| **G2** | 0.5 | ✗ | SUPRA Full（现有最佳配置） |
| **G3** | 0.0 | ✓ | OGM-GE only（替代 aux 强化） |
| **G4** | 0.5 | ✓ | OGM-GE + aux（叠加效果） |

预期：
- G3 vs G1：OGM-GE 能否单独带来提升（验证梯度调节的价值）
- G4 vs G2：aux + OGM-GE 叠加是否优于单独 aux
- G4 vs G3：aux_loss 提供的额外梯度强化是否比单纯梯度调节更有效

---

## Grocery 数据集超参数

> 来源: `Results/supra_gpu0_default_accuracy_best.csv`

| 参数 | 值 |
|------|-----|
| Backbone | GCN |
| n-layers | 3 |
| embed_dim | 256 |
| lr | 0.001 |
| wd | 0.0001 |
| dropout | 0.3 |
| mlp_variant | ablate |

---

## 运行命令

### G1 — 纯 C 通道基线（aux_weight=0, OGM-GE 关闭）

```bash
python -m GNN.SUPRA \
    --data_name Grocery \
    --text_feature /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --model_name GCN \
    --n_layers 3 \
    --embed_dim 256 \
    --n_hidden 256 \
    --dropout 0.3 \
    --lr 0.001 \
    --wd 0.0001 \
    --aux_weight 0.0 \
    --mlp_variant ablate \
    --selfloop False \
    --n_epochs 300 \
    --n_runs 3 \
    --seed 42 \
    --eval_steps 1 \
    --early_stop_patience 50 \
    --result_csv Results/ogm_ge/grocery_g1_baseline.csv \
    --result_csv_all Results/ogm_ge/grocery_g1_baseline_all.csv \
    --disable_wandb \
    --gpu 0
```

### G2 — SUPRA Full（aux_weight=0.5, OGM-GE 关闭）

```bash
python -m GNN.SUPRA \
    --data_name Grocery \
    --text_feature /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --model_name GCN \
    --n_layers 3 \
    --embed_dim 256 \
    --n_hidden 256 \
    --dropout 0.3 \
    --lr 0.001 \
    --wd 0.0001 \
    --aux_weight 0.5 \
    --mlp_variant ablate \
    --selfloop False \
    --n_epochs 300 \
    --n_runs 3 \
    --seed 42 \
    --eval_steps 1 \
    --early_stop_patience 50 \
    --result_csv Results/ogm_ge/grocery_g2_aux.csv \
    --result_csv_all Results/ogm_ge/grocery_g2_aux_all.csv \
    --disable_wandb \
    --gpu 0
```

### G3 — OGM-GE only（aux_weight=0, OGM-GE 开启）

```bash
python -m GNN.SUPRA \
    --data_name Grocery \
    --text_feature /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --model_name GCN \
    --n_layers 3 \
    --embed_dim 256 \
    --n_hidden 256 \
    --dropout 0.3 \
    --lr 0.001 \
    --wd 0.0001 \
    --aux_weight 0.0 \
    --mlp_variant ablate \
    --use_ogm_ge \
    --ogm_alpha 0.5 \
    --ogm_starts 0 \
    --ogm_ends 50 \
    --selfloop False \
    --n_epochs 300 \
    --n_runs 3 \
    --seed 42 \
    --eval_steps 1 \
    --early_stop_patience 50 \
    --result_csv Results/ogm_ge/grocery_g3_ogm.csv \
    --result_csv_all Results/ogm_ge/grocery_g3_ogm_all.csv \
    --disable_wandb \
    --gpu 0
```

### G4 — OGM-GE + aux（aux_weight=0.5, OGM-GE 开启）

```bash
python -m GNN.SUPRA \
    --data_name Grocery \
    --text_feature /mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt \
    --model_name GCN \
    --n_layers 3 \
    --embed_dim 256 \
    --n_hidden 256 \
    --dropout 0.3 \
    --lr 0.001 \
    --wd 0.0001 \
    --aux_weight 0.5 \
    --mlp_variant ablate \
    --use_ogm_ge \
    --ogm_alpha 0.5 \
    --ogm_starts 0 \
    --ogm_ends 50 \
    --selfloop False \
    --n_epochs 300 \
    --n_runs 3 \
    --seed 42 \
    --eval_steps 1 \
    --early_stop_patience 50 \
    --result_csv Results/ogm_ge/grocery_g4_ogm_aux.csv \
    --result_csv_all Results/ogm_ge/grocery_g4_ogm_aux_all.csv \
    --disable_wandb \
    --gpu 0
```

---

## OGM-GE 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_ogm_ge` | False | 开启 OGM-GE 梯度调制 |
| `--ogm_alpha` | 0.5 | 压制强度系数。越大 = 对强模态压制越狠。建议范围 0.1~0.8 |
| `--ogm_ge` | False | 开启 Gaussian Enhancement（GE）。在压低梯度的同时加 N(0, std) 噪声，弥补信号损失 |
| `--ogm_starts` | 0 | 开始调制的 epoch（包含） |
| `--ogm_ends` | 100 | 结束调制的 epoch（包含） |

### 调制系数公式

```
ratio_t = score_Ut / score_C
ratio_v = score_Uv / score_C

coeff = 1 - tanh(alpha * (ratio - 1))   if ratio > 1
       = 1                                 otherwise
```

---

## 输出文件

```
Results/ogm_ge/
├── grocery_g1_baseline.csv         # G1 汇总
├── grocery_g1_baseline_all.csv     # G1 每轮 raw 结果
├── grocery_g2_aux.csv              # G2 汇总
├── grocery_g2_aux_all.csv          # G2 每轮 raw 结果
├── grocery_g3_ogm.csv              # G3 汇总
├── grocery_g3_ogm_all.csv          # G3 每轮 raw 结果
└── grocery_g4_ogm_aux.csv         # G4 汇总
└── grocery_g4_ogm_aux_all.csv     # G4 每轮 raw 结果
```

---

## 预期现象

| 对比 | 预期现象 |
|------|---------|
| G2 vs G1 | aux_weight 强化 Ut/Uv 独立学习，纯 C 通道基线明显提升 |
| G3 vs G1 | OGM-GE 在梯度层面防止 Ut/Uv 过度主导，对基线有改善 |
| G4 vs G2 | 两者叠加，若 OGM-GE 调节方向与 aux_loss 互补，则叠加提升；若有冲突，则可能下降 |
| G4 vs G3 | aux_loss 提供了 Ut/Uv 独立的 loss 监督，理论上比单纯梯度压制更直接有效 |

---

## 代码实现说明

### 文件结构

```
GNN/Utils/ogm_ge.py       # OGM-GE 核心逻辑
GNN/SUPRA.py              # 训练循环集成（backward 后、optimizer.step 前）
```

### 核心函数

**`compute_ogm_coefficients`** — 计算 Ut/Uv 的调制系数：
```python
coeff_t, coeff_v = compute_ogm_coefficients(
    logits_Ut=out.logits_Ut_0[train_idx],
    logits_Uv=out.logits_Uv_0[train_idx],
    logits_C=out.logits_C_0[train_idx],
    labels=labels[train_idx],
    alpha=0.5,
)
```

**`apply_ogm_ge`** — 将系数作用于模型参数的梯度：
- `enc_t` 相关参数：乘以 `coeff_t`
- `enc_v` 相关参数：乘以 `coeff_v`
- `head_Ut` 参数：乘以 `coeff_t`
- `head_Uv` 参数：乘以 `coeff_v`
- `mp_C` 和 `head_C`：**不参与调节**

### 集成位置（SUPRA.py main() 函数）

```python
total_loss.backward()

# ← OGM-GE 插入这里
use_ogm = getattr(args, 'use_ogm_ge', False)
if use_ogm and (args.ogm_starts <= epoch <= args.ogm_ends):
    coeff_t, coeff_v = compute_ogm_coefficients(...)
    apply_ogm_ge(model, coeff_t, coeff_v, use_ge=args.ogm_ge)

optimizer.step()  # ← 调节在 step 之前生效
```

### 与 aux_weight 的兼容性

- `aux_weight > 0` 时：`total_task_loss = loss_C + aux_weight * (loss_Ut + loss_Uv)`，Ut/Uv 的 logits 直接获得 loss 强化
- OGM-GE 叠加时：loss 层面的强化照旧，encoder/head 的梯度额外被 OGM 调制
- 两者互补不冲突：**aux 是 loss 强化，OGM 是梯度调节**

---

## 后续扩展

1. **Grid Search alpha**：在 Grocery 上对 `ogm_alpha ∈ {0.1, 0.3, 0.5, 0.8}` 做 grid，找出最优压制强度
2. **GE 开关对比**：开启 `--ogm_ge` vs 关闭，对比高斯噪声对泛化的影响
3. **多数据集验证**：在 Movies / Toys / Reddit-M 上运行相同 2×2 实验，验证结论泛化性
4. **梯度追踪叠加**：开启 `--analyze_gradients` 追踪 enc_t/enc_v 的梯度 L2 范数，验证 OGM 确实在压制 Ut/Uv encoder 梯度
