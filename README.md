# 🌍 Language | 语言
| [English](README.md) | [中文](README_CN.md) |

---

# SUPRA: Unified Multimodal Learning with Spectral Orthogonalization

A multimodal graph neural network framework that leverages spectral orthogonalization to mitigate gradient competition and low-rank bias in shared graph representation learning.

## Method Overview

SUPRA introduces a **Shared-Unique Decomposition** architecture with **Newton-Schulz Orthogonalization** to promote diversity between shared and modality-specific parameters:

- **Shared Channel (C)**: GNN backbone that learns common graph structure via concat(text, visual) → message passing
- **Unique Text Channel (Uₜ)**: Projects raw text features to embedding space (no graph message passing)
- **Unique Visual Channel (Uᵥ)**: Projects raw visual features to embedding space (no graph message passing)
- **Late Fusion**: Average of logits from all three channels: `(logits_C + logits_Ut + logits_Uv) / 3`
- **Spectral Orthogonalization Hook**: Gradient post-processing on shared channel to enforce orthogonal update directions
- **Auxiliary Loss (optional)**: Per-branch cross-entropy losses to strengthen gradients to unique channels

### 4-way Ablation Study

| Variant | Architecture | Orthogonalization | Auxiliary Loss |
|---------|-------------|-------------------|----------------|
| Ablate-None | C+Uₜ+Uᵥ three-channel | ❌ (α=0) | ❌ |
| Ablate-Ortho | C+Uₜ+Uᵥ three-channel | ✅ (α=1) | ❌ |
| Ablate-Aux | C+Uₜ+Uᵥ three-channel | ❌ (α=0) | ✅ |
| SUPRA-Full | C+Uₜ+Uᵥ three-channel | ✅ (α=1) | ✅ |

## Supported Methods

### Baseline Models (`run_baseline.sh`)
| Model | Description | Layers |
|-------|-------------|--------|
| MLP | Multi-layer perceptron (no graph) | 2, 3 |
| GCN | Graph Convolutional Network (via Early_GNN) | 1, 2 |
| SAGE | GraphSAGE (mean aggregator, via Early_GNN) | 2, 3, 4 |
| GAT | Graph Attention Network (via Early_GNN) | 1, 2 |
| GCNII | GCN with Initial residual + Identity mapping | 2, 3, 4 |
| JKNet | Jumping Knowledge Network | 2, 3, 4 |

### SUPRA (`run_supra.sh` / `run_ablation_study.sh`)
| Component | Description | Options |
|-----------|-------------|---------|
| GNN Backbone | GCN, SAGE, GAT, RevGAT | - |
| Embed Dim | Shared embedding dimension | 128, 256 |
| Shared Depth | Shared channel depth | 1, 2, 3, 4 |
| Ortho Alpha | Spectral orthogonalization strength | 0.0 (off), 1.0 (full) |
| Aux Loss | Enable per-branch auxiliary loss | true/false |

## Project Structure

```
SUPRA_2.0/
├── GNN/
│   ├── SUPRA.py              # SUPRA model implementation with spectral orthogonalization
│   ├── Library/              # Simple single-modality baselines
│   │   ├── MLP.py, GCN.py, GAT.py, GraphSAGE.py
│   │   ├── GCNII.py, JKNet.py, SGC.py, APPNP.py
│   ├── Baselines/            # Multimodal baselines (Early/Late fusion)
│   │   ├── Early_GNN.py      # Early fusion: concat text+visual -> GNN
│   │   ├── Late_GNN.py       # Late fusion: separate encoders -> GNN -> fuse
│   │   ├── MIG_GT.py, NTSFormer.py
│   │   └── OGM-GE/, NTSFormer/ (submodules)
│   └── GraphData.py          # Data loading utilities
├── plot/                     # Visualization scripts
│   └── plot_*.sh             # Various plotting scripts
├── scripts/                  # Additional experiment scripts
├── path_config.sh            # Data path configuration (shared by scripts)
├── run_baseline.sh           # Baseline experiment runner
├── run_supra.sh              # SUPRA experiment runner
├── run_ablation_study.sh     # 4-way ablation study runner
├── run_batch_baseline.sh     # Batch runner for all baselines
└── requirements.yaml         # Conda environment spec
```

## Setup

### 1. Environment

```bash
# Clone the repository
git clone https://github.com/sktsherlock/SUPRA.git
cd SUPRA

# Create conda environment
conda env create -f requirements.yaml

# Activate
conda activate MAG
```

### 2. Data Path Configuration

Edit `path_config.sh` to set your data root:

```bash
DATA_ROOT="${DATA_ROOT:-/path/to/your/MAGB_Dataset}"
export DATA_ROOT
```

Required data structure:
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

Feature groups supported:
- `clip_roberta` (default): CLIP visual + RoBERTa text
- `default`: Llama 3.2 Vision features

## Running Experiments

### Quick Start

```bash
# Run ablation study (4 variants including SUPRA-Full)
./run_ablation_study.sh

# Run baseline experiments
./run_batch_baseline.sh

# Run a specific model
./run_baseline.sh --model GCN --data_name Movies

# Run SUPRA with custom settings
./run_supra.sh --data_name Movies
```

### Ablation Study

```bash
# Run all 4-way ablation on all datasets (Movies, Grocery, Toys, Reddit-M)
./run_ablation_study.sh

# Run ablation on specific datasets
DATASETS="Movies Grocery" ./run_ablation_study.sh

# Run ablation with specific backbone
MODEL_NAME=GAT ./run_ablation_study.sh

# Run ablation variants individually
ORTHO_ALPHA=0.0 USE_AUX_LOSS=false ./run_supra.sh --data_name Movies   # Ablate-None
ORTHO_ALPHA=1.0 USE_AUX_LOSS=false ./run_supra.sh --data_name Movies   # Ablate-Ortho
ORTHO_ALPHA=0.0 USE_AUX_LOSS=true  ./run_supra.sh --data_name Movies   # Ablate-Aux
ORTHO_ALPHA=1.0 USE_AUX_LOSS=true  ./run_supra.sh --data_name Movies   # SUPRA-Full
```

### Baseline Experiments

```bash
# Run all baseline models on Movies dataset
./run_baseline.sh --model GCN --data_name Movies

# Run specific model with custom params
./run_baseline.sh --data_name Movies --model SAGE --n_layers 2

# Run with specific feature group
FEATURE_GROUPS="default" ./run_baseline.sh --data_name Grocery --model GAT

# Run with F1-macro metric
./run_baseline.sh --data_name Movies --model GCN --metric f1_macro --average macro
```

**Key parameters:**
| Flag | Description | Default |
|------|-------------|---------|
| `--data_name` | Dataset (Movies, Grocery, Toys, Reddit-M) | Required |
| `--model` | Model (MLP, GCN, SAGE, GAT, GCNII, JKNet) | Required |
| `--feature_group` | Feature set (clip_roberta, default) | clip_roberta |
| `--metric` | Metric (accuracy, f1_macro) | accuracy |
| `--average` | Average mode (macro, micro) | macro |
| `--gpu` | GPU device ID | 0 |
| `--n_runs` | Number of runs | 3 |

### SUPRA Experiments

```bash
# Run SUPRA with GCN backbone
./run_supra.sh --data_name Movies

# Run with GAT backbone and custom embedding
./run_supra.sh --data_name Grocery --model_name GAT --embed_dim 128

# Run specific shared_depth range
SUPRA_LAYERS="2 3 4" ./run_supra.sh --data_name Toys
```

**Key parameters:**
| Flag | Description | Default |
|------|-------------|---------|
| `--data_name` | Dataset (Movies, Grocery, Toys, Reddit-M) | Required |
| `--model_name` | GNN backbone (GCN, SAGE, GAT, RevGAT) | GCN |
| `--feature_group` | Feature set (clip_roberta, default) | clip_roberta |
| `--embed_dim` | Embedding dimension | 256 |
| `--shared_depth` | Shared channel depth | 2 |
| `--ortho_alpha` | Spectral orthogonalization strength (0=disable, 1=full) | 1.0 |
| `--use_aux_loss` | Enable auxiliary loss on each branch | off |
| `--gpu` | GPU device ID | 0 |
| `--n_runs` | Number of runs | 3 |

### Batch Experiments

```bash
# Run all baselines on all datasets with all metrics (dry run)
./run_batch_baseline.sh --dry_run

# Run only GCN and SAGE
./run_batch_baseline.sh --models "GCN SAGE"

# Run only F1-macro metric
./run_batch_baseline.sh --metrics "f1_macro"
```

## Hyperparameters

### Fixed Parameters (All Methods)
- Dropout: 0.3 (fixed)
- Weight decay: 1e-4
- Train ratio: 0.6
- Val ratio: 0.2

### Baseline Search Space
| Model | Learning Rate | Layers |
|-------|--------------|--------|
| MLP | 0.0005, 0.001 | 2, 3 |
| GCN | 0.0005, 0.001 | 1, 2 |
| SAGE | 0.0005, 0.001 | 2, 3, 4 |
| GAT | 0.0005, 0.001 | 1, 2 |
| GCNII | 0.0005, 0.001 | 2, 3, 4 |
| JKNet | 0.0005, 0.001 | 2, 3, 4 |

### SUPRA / Ablation Search Space
- Learning rate: 0.0005, 0.001
- Layers (n_layers): 2, 3, 4
- Embed dim: 128, 256
- Shared depth: 1, 2, 3, 4

## Output

Results are saved to:
- `logs_baseline/` - Training logs for baseline experiments
- `logs_supra/` - Training logs for SUPRA experiments
- `logs_ablation/` - Training logs for ablation experiments
- `results_csv/baseline_best.csv` - Best results per method/dataset/metric
- `results_csv/baseline_all.csv` - All experimental results

## Citation

If you find this work useful, please cite:

```bibtex
@misc{supra2026,
  title={SUPRA: Unified Multimodal Learning with Spectral Orthogonalization},
  author={},
  year={2026}
}
```

## License

See [LICENSE](LICENSE) for details.