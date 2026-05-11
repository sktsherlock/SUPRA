# 🌍 Language | 语言
| [English](README.md) | [中文](README_CN.md) |

---

# SUPRA: Shared-Unique Decomposition for Multimodal Graph Learning

A multimodal graph neural network framework that addresses **modality contamination** — the phenomenon where shared GNN aggregation causes one modality's signal to overwhelm another's — through a **Shared-Unique Decomposition** architecture.

## Method Overview

SUPRA decomposes multimodal node representation into three functionally distinct channels:

| Channel | Description | Gradient Source |
|---------|-------------|----------------|
| **C (Shared)** | `concat(text_enc, visual_enc)` → GNN → shared semantic representation | `logits_C` loss |
| **Uₜ (Unique Text)** | `text_enc` → head_Ut → logits_Ut (no graph propagation) | `logits_Ut` loss |
| **Uᵥ (Unique Visual)** | `visual_enc` → head_Uv → logits_Uv (no graph propagation) | `logits_Uv` loss |

**Fusion**: Equal-weight average of all three channels:
```
logits_final = (logits_C + logits_Ut + logits_Uv) / 3
```

The key insight is that Uₜ/Uᵥ channels are **bypassed from GNN message passing**, protecting modality-specific semantics from being diluted by the shared aggregator. An optional **auxiliary loss** (`aux_weight`) provides additional gradient reinforcement to the encoders.

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mlp_variant` | MLP projection before GNN: `ablate`=none (recommended), `full`=Linear→ReLU→LN→Linear | `ablate` |
| `--aux_weight` | Extra gradient strength for Uₜ/Uᵥ channels (0=disable, 0.5~0.7 recommended) | `0.0` |
| `--ablate_bypass` | Ablation flag: force `logits_final = logits_C` to verify bypass value | disabled |

### OGM-GE: Optional Gradient Modulation

OGM-GE (CVPR 2022) can be stacked on top of SUPRA as a complementary gradient-level regularizer. It uses the C channel as an anchor to detect when Uₜ/Uᵥ are becoming over-confident relative to the shared representation, and applies gradient dampening accordingly:

```bash
python -m GNN.SUPRA ... \
    --use_ogm_ge \
    --ogm_alpha 0.5 \
    --ogm_starts 0 \
    --ogm_ends 50
```

See [`docs/ogm_ge_experiment.md`](docs/ogm_ge_experiment.md) for full experimental details.

## Supported Methods

### Baseline Suite (`run_mag_baseline_suite.sh`)

| Experiment | Description | Entry |
|-----------|-------------|-------|
| **plain** | Unimodal MLP/GNN (text-only or visual-only) | `Early_GNN --backend mlp/gnn` |
| **baseline** | Early fusion: `concat(text, visual)` → GNN | `Early_GNN` |
| **late** | Late fusion: separate encoders → separate GNNs → fuse | `Late_GNN` |
| **nts** | NTSFormer multimodal graph transformer | `NTSFormer` |
| **mig** | MIG_GT multi-hop GDCF + transformer | `MIG_GT` |

### SUPRA Suite (`run_supra_suite.sh`)

| Component | Options |
|-----------|---------|
| GNN Backbone | GCN, SAGE, GAT, GCNII, JKNet |
| Embed Dim | 128, 256 |
| aux_weight | 0.0, 0.5, 0.7 |
| mlp_variant | `full`, `ablate` |

### Ablation Study (`run_ablation_study.sh`)

| Group | Model | Configuration | Purpose |
|-------|-------|----------------|---------|
| **G1** | MMGCN (Late_GNN) | Traditional multimodal GNN | Baseline |
| **G2** | SUPRA (Synergy-Only) | `ablate_bypass`: `logits_final = logits_C` | Verify bypass contribution |
| **G3** | SUPRA Base | `aux_weight = 0`: three channels, no aux loss | Architecture baseline |
| **G4** | SUPRA Full | `aux_weight > 0`: three channels + aux loss | Full model |

See [`docs/gradient_starvation_experiment.md`](docs/gradient_starvation_experiment.md) for gradient tracking experiments.

## Project Structure

```
SUPRA_2.0/
├── GNN/
│   ├── SUPRA.py              # SUPRA model (C/Ut/Uv three-channel architecture)
│   ├── Utils/
│   │   ├── ogm_ge.py         # OGM-GE gradient modulation
│   │   ├── plot_gradient_norm.py  # Gradient norm visualization
│   │   └── graph_degradation.py   # Feature noise / edge rewiring
│   ├── Baselines/
│   │   ├── Early_GNN.py      # Early fusion (concat → GNN)
│   │   ├── Late_GNN.py       # Late fusion (separate encoders → GNN → fuse)
│   │   ├── NTSFormer.py      # Multimodal graph transformer
│   │   └── MIG_GT.py         # Multi-hop GDCF + transformer
│   └── Library/              # Single-modality GNN primitives (GCN, SAGE, GAT, etc.)
├── docs/
│   ├── ogm_ge_experiment.md       # OGM-GE gradient modulation experiments
│   ├── gradient_starvation_experiment.md  # Gradient starvation verification
│   └── degradation_experiment.md  # Feature/edge degradation experiments
├── tools/
│   ├── summarize_ogm_ge.py        # OGM-GE results summarizer
│   ├── run_degradation_experiments.py  # Degradation experiments runner
│   └── plot_gradient_norm         # (invoked as python -m GNN.Utils.plot_gradient_norm)
├── run_mag_baseline_suite.sh  # Comprehensive baseline suite
├── run_supra_suite.sh         # SUPRA main experiments
├── run_ablation_study.sh      # 4-group ablation study
└── requirements.yaml           # Conda environment
```

## Setup

### 1. Environment

```bash
git clone https://github.com/sktsherlock/SUPRA.git
cd SUPRA
conda env create -f requirements.yaml
conda activate MAG
```

### 2. Data Path

**IMPORTANT**: `run_mag_baseline_suite.sh` reads `DATA_ROOT` directly (not `path_config.sh`):

```bash
export DATA_ROOT=/path/to/MAGB_Dataset
```

For other scripts, edit `path_config.sh`:

```bash
DATA_ROOT="${DATA_ROOT:-/path/to/your/MAGB_Dataset}"
export DATA_ROOT
```

Required structure:
```
${DATA_ROOT}/
├── Movies/
│   ├── MoviesGraph.pt
│   ├── TextFeature/Movies_*.npy
│   └── ImageFeature/Movies_*.npy
├── Grocery/
├── Toys/
└── Reddit-M/
```

Feature groups:
- `clip_roberta` (default): CLIP visual + RoBERTa text
- `default`: Llama 3.2 Vision features

## Running Experiments

### Baseline Suite (Recommended Entry Point)

```bash
# Run all baselines (Plain/Early/Late/NTS/MIG) on all datasets
DATA_ROOT=/path/to/MAGB_Dataset ./run_mag_baseline_suite.sh

# Dry run (preview commands only)
./run_mag_baseline_suite.sh --dry_run

# Custom selection
EXPERIMENTS="plain baseline late" DATASETS="Movies Grocery" ./run_mag_baseline_suite.sh

# With Llama features instead of CLIP
FEATURE_GROUPS="default" ./run_mag_baseline_suite.sh
```

### SUPRA Suite

```bash
# Run SUPRA on all datasets
DATA_ROOT=/path/to/MAGB_Dataset ./run_supra_suite.sh

# Single dataset
DATASETS="Movies" ./run_supra_suite.sh

# Custom result CSV
RESULT_CSV="Results/my_results.csv" ./run_supra_suite.sh
```

### Ablation Study

```bash
# 4-group ablation on all datasets
./run_ablation_study.sh

# Individual groups (Grocery example)
python -m GNN.Baselines.Late_GNN ...    # G1: MMGCN
python -m GNN.SUPRA --ablate_bypass ... # G2: Synergy-Only
python -m GNN.SUPRA --aux_weight 0.0 ... # G3: SUPRA Base
python -m GNN.SUPRA --aux_weight 0.7 ... # G4: SUPRA Full
```

See [`docs/gradient_starvation_experiment.md`](docs/gradient_starvation_experiment.md) for the full set of commands with gradient tracking.

### OGM-GE Experiments

```bash
# 2×2 factorial: aux_weight × use_ogm_ge across all 4 datasets
# See docs/ogm_ge_experiment.md for the full batch runner script

# Summarize results
python tools/summarize_ogm_ge.py
python tools/summarize_ogm_ge.py --f1
```

### Degradation Experiments

```bash
# Feature noise degradation + edge rewiring degradation
# See docs/degradation_experiment.md for full commands

python tools/run_degradation_experiments.py \
    --data_name Movies \
    --text_feature /path/to/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy \
    --visual_feature /path/to/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /path/to/MoviesGraph.pt \
    --embed_dim 256 --n_layers 3 --lr 0.001 \
    --save_dir Results/degradation --gpu 0
```

## Key Parameters

### SUPRA

| Flag | Description | Default |
|------|-------------|---------|
| `--data_name` | Dataset (Movies, Grocery, Toys, Reddit-M) | Required |
| `--model_name` | GNN backbone (GCN, SAGE, GAT, GCNII, JKNet) | GCN |
| `--embed_dim` | Embedding dimension | 256 |
| `--n_layers` | GNN layer count | 2 |
| `--mlp_variant` | `full` or `ablate` (recommended) | `ablate` |
| `--aux_weight` | Auxiliary loss weight for Uₜ/Uᵥ (0=disable) | 0.0 |
| `--ablate_bypass` | Force logits_final = logits_C | disabled |
| `--use_ogm_ge` | Enable OGM-GE gradient modulation | disabled |
| `--gpu` | GPU device ID | 0 |

### Baseline Suite

| Flag | Description | Default |
|------|-------------|---------|
| `--experiment` | Experiment type | Required |
| `--datasets` | Dataset list | All 4 |
| `--feature_groups` | Feature set (`clip_roberta`, `default`) | `clip_roberta` |
| `--metric` | Metric (`accuracy`, `f1`) | `accuracy` |
| `--n_runs` | Number of runs | 3 |

## Citation

```bibtex
@misc{supra2026,
  title={SUPRA: Shared-Unique Decomposition for Multimodal Graph Learning},
  author={},
  year={2026}
}
```

## License

See [LICENSE](LICENSE) for details.
