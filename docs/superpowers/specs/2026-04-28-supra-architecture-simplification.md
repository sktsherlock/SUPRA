# SUPRA Architecture Simplification - Design Spec

## Date: 2026-04-28

## Context

SUPRA (Shared-Unique Multimodal GNN) is being refined for publication. Previous ablation studies showed that `ortho_alpha` (spectral orthogonalization) and `use_aux_loss` (auxiliary loss) provided marginal benefit. The architecture simplification removes these training tricks and focuses on the core three-channel design with an enhanced MLP projection.

**Original motivation**: With high-quality encoders (CLIP, RoBERTa), modality features are strong. Traditional multimodal GNN message passing dilutes each node's native modality signal. SUPRA's unique channels (Ut/Uv) preserve modality-specific features by bypassing GNN.

**Refined design**: Three-channel architecture with independent classification heads + nonlinear MLP projection before GNN. Auxiliary loss on Ut/Uv channels (not C) as a training stabilization mechanism.

---

## 1. Architecture Changes

### 1.1 MLP Projection Enhancement (Core Change)

**Before** (SUPRA v1):
```python
# _make_mp_layers in SUPRA class
concat(e_t, e_v) → Linear(2*embed_dim, embed_dim) → GNN
```

**After** (SUPRA v2):
```python
concat(e_t, e_v) → Linear(2*embed_dim, embed_dim) → ReLU → LayerNorm → Linear(embed_dim, embed_dim) → GNN
```

**Rationale**:
- Two-layer MLP provides nonlinear transformation capacity
- LayerNorm stabilizes training
- Addresses reviewer concern: "does the dimensionality reduction from 2*d to d lose information?"
- This is the main architectural innovation beyond the three-channel design

### 1.2 Remove ortho_alpha

- Remove `ortho_alpha` parameter and `SpectralOrthogonalizer` from SUPRA class
- Remove `supra.add_argument("--ortho_alpha", ...)` from args_init
- Three-channel + independent heads already prevents low-rank collapse; orthogonalization is unnecessary complexity

### 1.3 Simplify aux_loss

**Before**:
```python
total_task_loss = total_task_loss + aux_weight * (loss_C + loss_Ut + loss_Uv)
```

**After**:
```python
total_task_loss = total_task_loss + aux_weight * (loss_Ut + loss_Uv)
```

**Rationale**:
- loss_C is redundant: C channel already learns from main logits_final loss
- loss_Ut and loss_Uv ensure independent gradient flow to enc_t and enc_v
- This avoids C channel overfitting to its own prediction while neglecting shared representation

---

## 2. Experiment Configuration

### 2.1 Hyperparameter Grid

| Parameter | Values | Notes |
|-----------|--------|-------|
| n_layers | 2, 3, 4 | GNN depth in shared channel |
| embed_dim | 256 (fixed) | Same as baselines for fair comparison |
| aux_weight | 0.0, 0.1, 0.3, 0.5, 0.7 | Ablation: full SUPRA vs lightweight |
| ortho_alpha | Removed | Not a core contribution |
| use_aux_loss | Removed | Merged into aux_weight |

### 2.2 Model Variants

- **SUPRA (full)**: aux_weight > 0 — uses auxiliary loss on Ut/Uv
- **SUPRA (lightweight)**: aux_weight = 0.0 — no auxiliary loss, pure three-channel

Both are SUPRA family; best result from all aux_weight values is reported as "SUPRA" in best.csv.

### 2.3 Backbones

- GCN (selfloop=true)
- GAT (n_heads=4, selfloop=false)

---

## 3. Script Changes

### 3.1 run_supra.sh

Changes:
- Remove `supra_shared_depths` array (already done in previous commit)
- Remove `--ortho_alpha` and `--shared_depth` from python command
- Add `aux_weight` array: `("0.0" "0.1" "0.3" "0.5" "0.7")`
- Add `--aux_weight` parameter to python command
- Update label format: `${label_prefix}-${MODEL_NAME}-lr${lr}-wd${wd}-h${h}-L${L}-do${supra_dropout}-aw${aw}`

### 3.2 run_ablation_study.sh

Changes:
- Remove `ortho_alpha` from MODE_CONFIG (4-way becomes 1-way or simplified)
- Since ortho is removed, ablation study becomes SUPRA vs Baseline comparison
- Add `aux_weight` sweep for SUPRA ablation: `0.0` vs `0.1/0.3/0.5/0.7`
- Labels: `SUPRA-aux0.0` vs `SUPRA-aux0.5`, etc.

### 3.3 GNN/SUPRA.py

Changes:
1. Remove `ortho_alpha` argument and `SpectralOrthogonalizer` / `_register_spectral_hooks`
2. Modify `_make_mp_layers`:
   - Replace single Linear with: Linear → ReLU → LayerNorm → Linear
3. Remove `use_aux_loss` argument, replace with `aux_weight` (float)
4. Modify `_compute_losses`:
   - Remove loss_C from aux loss calculation
   - aux_weight now multiplies (loss_Ut + loss_Uv) directly
5. Update `args_init()`: remove ortho_alpha, add aux_weight

---

## 4. CSV Logging

### 4.1 all.csv
Records every experiment run with full hyperparameters including aux_weight.

### 4.2 best.csv
Records best result per method:
- "SUPRA" — best among all aux_weight values (n_layers × aux_weight grid)
- "Ablate-NoAux" — not separate; aux=0.0 is part of SUPRA family

The "aux_weight" column distinguishes different SUPRA configurations.

---

## 5. Implementation Order

1. **GNN/SUPRA.py**: Modify architecture (MLP enhancement, remove ortho, simplify aux)
2. **run_supra.sh**: Update hyperparameter grid and label format
3. **run_ablation_study.sh**: Update similarly
4. **Verify**: Run a quick test on one dataset to ensure training works
5. **Commit**: Push all changes

---

## 6. Self-Review Checklist

- [ ] Placeholder scan: all TBD/TODO resolved
- [ ] Internal consistency: architecture matches experiment config
- [ ] Scope: focused on SUPRA simplification, not unrelated changes
- [ ] Ambiguity: aux_weight is clearly defined as float multiplier
- [ ] Backward compatibility: remove deprecated args (ortho_alpha, shared_depth, use_aux_loss)