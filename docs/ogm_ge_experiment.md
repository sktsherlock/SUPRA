# OGM-GE 梯度调制实验

验证 OGM-GE（On-the-fly Gradient Modulation with Gaussian Enhancement）与 SUPRA 辅助通道（aux_weight）在平衡模态学习上的效果对比。在 Movies / Grocery / Toys / Reddit-M 四个数据集上全部运行。

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

2×2 因子实验 × 4 数据集：

| 实验 | aux_weight | use_ogm_ge | 说明 |
|------|------------|------------|------|
| **G1** | 0.0 | ✗ | 纯 C 通道（基线） |
| **G2** | 0.5~0.7 | ✗ | SUPRA Full（aux 强化，现有最佳配置） |
| **G3** | 0.0 | ✓ | OGM-GE only（替代 aux 强化） |
| **G4** | 0.5~0.7 | ✓ | OGM-GE + aux（叠加效果） |

预期：
- G3 vs G1：OGM-GE 能否单独带来提升（验证梯度调节的价值）
- G4 vs G2：aux + OGM-GE 叠加是否优于单独 aux
- G4 vs G3：aux_loss 提供的额外梯度强化是否比单纯梯度调节更有效

---

## 各数据集超参数

| 数据集 | Backbone | n_layers | lr | aux_weight (G2/G4) | 数据路径 |
|--------|----------|----------|-----|---------------------|---------|
| **Movies** | GCN | 3 | 0.001 | 0.7 | `/mnt/input/MAGB_Dataset/Movies/` |
| **Grocery** | GCN | 3 | 0.001 | 0.7 | `/mnt/input/MAGB_Dataset/Grocery/` |
| **Toys** | GCN | 2 | 0.0005 | 0.5 | `/mnt/input/MAGB_Dataset/Toys/` |
| **Reddit-M** | GCN | 3 | 0.0005 | 0.7 | `/mnt/input/MAGB_Dataset/Reddit-M/` |

> 所有实验：`embed_dim=256`, `n_hidden=256`, `dropout=0.3`, `wd=0.0001`, `mlp_variant=ablate`, `selfloop=False`

---

## 运行命令

> **所有实验均添加 `--report_drop_modality --degrade_target both --degrade_alphas 1.0` 以追踪模态污染下限。汇总时使用 `tools/summarize_ogm_ge.py --dir Results/ogm_ge`。

### 批量运行脚本

建议在远程服务器上创建脚本 `run_ogm_ge.sh`：

```bash
#!/bin/bash
# =====================================================================
# OGM-GE 梯度调制实验 — 4 数据集 × 4 组别
# =====================================================================
# Usage: bash run_ogm_ge.sh [gpu_id]

GPU="${1:-0}"
SEED=42
N_RUNS=3
N_EPOCHS=300
PATIENCE=50
WD=0.0001
DROPOUT=0.3
OUTDIR="Results/ogm_ge"
mkdir -p "$OUTDIR"

declare -A LR
declare -A N_LAYERS
declare -A AUX
declare -A TEXT_PATH
declare -A VIS_PATH
declare -A GRAPH_PATH

# Movies
LR["Movies"]=0.001
N_LAYERS["Movies"]=3
AUX["Movies"]=0.7
TEXT_PATH["Movies"]="/mnt/input/MAGB_Dataset/Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy"
VIS_PATH["Movies"]="/mnt/input/MAGB_Dataset/Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy"
GRAPH_PATH["Movies"]="/mnt/input/MAGB_Dataset/Movies/MoviesGraph.pt"

# Grocery
LR["Grocery"]=0.001
N_LAYERS["Grocery"]=3
AUX["Grocery"]=0.7
TEXT_PATH["Grocery"]="/mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy"
VIS_PATH["Grocery"]="/mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy"
GRAPH_PATH["Grocery"]="/mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt"

# Toys
LR["Toys"]=0.0005
N_LAYERS["Toys"]=2
AUX["Toys"]=0.5
TEXT_PATH["Toys"]="/mnt/input/MAGB_Dataset/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy"
VIS_PATH["Toys"]="/mnt/input/MAGB_Dataset/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy"
GRAPH_PATH["Toys"]="/mnt/input/MAGB_Dataset/Toys/ToysGraph.pt"

# Reddit-M
LR["Reddit-M"]=0.0005
N_LAYERS["Reddit-M"]=3
AUX["Reddit-M"]=0.7
TEXT_PATH["Reddit-M"]="/mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy"
VIS_PATH["Reddit-M"]="/mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy"
GRAPH_PATH["Reddit-M"]="/mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt"

# =====================================================================
# 实验组别
# =====================================================================
run_group() {
    local ds="$1"
    local grp="$2"     # g1 | g2 | g3 | g4
    local aux_w="$3"   # aux_weight value
    local use_ogm="$4" # true | false
    local ogm_alpha="${5:-0.5}"

    local txt="${TEXT_PATH[$ds]}"
    local vis="${VIS_PATH[$ds]}"
    local grh="${GRAPH_PATH[$ds]}"
    local lr="${LR[$ds]}"
    local nl="${N_LAYERS[$ds]}"
    local aux="${AUX[$ds]}"

    local ogm_tag=""
    [[ "$use_ogm" == "true" ]] && ogm_tag="_ogm"

    local label="${ds,,}_${grp}${ogm_tag}"
    local csv="${OUTDIR}/${label}.csv"
    local all_csv="${OUTDIR}/${label}_all.csv"

    if [[ -f "$csv" ]]; then
        echo "[SKIP] ${label}"
        return 0
    fi

    echo "[RUN ] ${label}"

    local cmd=(python -m GNN.SUPRA
        --data_name "$ds"
        --text_feature "$txt"
        --visual_feature "$vis"
        --graph_path "$grh"
        --model_name GCN
        --n_layers "$nl"
        --embed_dim 256
        --n_hidden 256
        --dropout "$DROPOUT"
        --lr "$lr"
        --wd "$WD"
        --aux_weight "$aux_w"
        --mlp_variant ablate
        --selfloop False
        --n_epochs "$N_EPOCHS"
        --n_runs "$N_RUNS"
        --seed "$SEED"
        --eval_steps 1
        --early_stop_patience "$PATIENCE"
        --report_drop_modality
        --degrade_target both
        --degrade_alphas 1.0
        --result_csv "$csv"
        --result_csv_all "$all_csv"
        --disable_wandb
        --gpu "$GPU"
    )

    if [[ "$use_ogm" == "true" ]]; then
        cmd+=(
            --use_ogm_ge
            --ogm_alpha "$ogm_alpha"
            --ogm_starts 0
            --ogm_ends 50
        )
    fi

    "${cmd[@]}" 2>&1 | tail -n 5
    echo "[DONE] ${label}"
}

# =====================================================================
# 执行：4 datasets × 4 groups = 16 experiments
# =====================================================================
for ds in Movies Grocery Toys "Reddit-M"; do
    # G1: aux=0, no OGM-GE
    run_group "$ds" g1 0.0 false

    # G2: aux, no OGM-GE
    run_group "$ds" g2 "${AUX[$ds]}" false

    # G3: no aux, OGM-GE
    run_group "$ds" g3 0.0 true 0.5

    # G4: aux + OGM-GE
    run_group "$ds" g4 "${AUX[$ds]}" true 0.5
done

echo ""
echo "All done. Results in $OUTDIR/"
ls -la "$OUTDIR/"
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
├── movies_g1.csv            # G1 汇总
├── movies_g1_all.csv        # G1 每轮 raw 结果
├── movies_g2.csv           # G2 汇总
├── movies_g2_all.csv
├── movies_g3_ogm.csv      # G3 汇总
├── movies_g3_ogm_all.csv
├── movies_g4_ogm.csv      # G4 汇总
├── movies_g4_ogm_all.csv
├── grocery_g1.csv
├── grocery_g2.csv
├── grocery_g3_ogm.csv
├── grocery_g4_ogm.csv
├── toys_g1.csv
├── toys_g2.csv
├── toys_g3_ogm.csv
├── toys_g4_ogm.csv
├── reddit-m_g1.csv
├── reddit-m_g2.csv
├── reddit-m_g3_ogm.csv
└── reddit-m_g4_ogm.csv
```

---

## 汇总命令

```bash
# 汇总所有数据集的 Accuracy 结果
python tools/summarize_ogm_ge.py

# 汇总 F1-macro 结果（需单独跑 --metric f1_macro 实验）
python tools/summarize_ogm_ge.py --f1
```

`--dir` 默认 `Results/ogm_ge`，如目录不同请指定 `--dir`。

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

1. **Grid Search alpha**：对 `ogm_alpha ∈ {0.1, 0.3, 0.5, 0.8}` 做 grid，找出最优压制强度
2. **GE 开关对比**：开启 `--ogm_ge` vs 关闭，对比高斯噪声对泛化的影响
3. **梯度追踪叠加**：开启 `--analyze_gradients` 追踪 enc_t/enc_v 的梯度 L2 范数，验证 OGM 确实在压制 Ut/Uv encoder 梯度
