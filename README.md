# SUPRA: Unified Multimodal Learning with Spectral Orthogonalization

A multimodal graph neural network framework that leverages spectral orthogonalization to mitigate gradient competition and low-rank bias in shared graph representation learning.

## Method Overview

SUPRA introduces a **Shared-Unique Decomposition** architecture with **Newton-Schulz Orthogonalization** to promote diversity between shared and modality-specific parameters:

- **Shared Channel (C)**: GNN backbone that learns common graph structure
- **Unique Encoders (Uₜ, Uᵥ)**: Modality-specific transformers for text and visual features
- **Spectral Orthogonalization Hook**: Gradient post-processing to enforce orthogonal update directions
- **Dynamic Orthogonalization Strength (α)**: Adaptive mixing of original and orthogonalized gradients

## Supported Methods

### Baseline Models (`run_baseline.sh`)
| Model | Description | Layers |
|-------|-------------|--------|
| MLP | Multi-layer perceptron (no graph) | 2, 3, 4 |
| GCN | Graph Convolutional Network | 1, 2 |
| SAGE | GraphSAGE (mean aggregator) | 2, 3, 4 |
| GAT | Graph Attention Network | 1, 2 |
| GCNII | GCN with Initial residual + Identity mapping | 2, 3, 4 |
| JKNet | Jumping Knowledge Network | 2, 3, 4 |

### SUPRA (`run_supra.sh`)
| Component | Description | Options |
|-----------|-------------|---------|
| GNN Backbone | GCN, SAGE, GAT, RevGAT | - |
| Embed Dim | Shared embedding dimension | 128, 256 |
| Shared Depth | Shared channel depth | 1, 2, 3, 4 |

## Project Structure

```
SUPRA_2.0/
├── GNN/
│   ├── SUPRA.py           # SUPRA model implementation
│   ├── Library/           # Baseline GNN implementations
│   │   ├── GCN.py, GAT.py, GraphSAGE.py, GCNII.py, JKNet.py, MLP.py
│   │   └── APPNP.py, SGC.py (reference)
│   ├── Baselines/          # Early/late fusion baselines
│   │   ├── Early_GNN.py, Late_GNN.py
│   │   ├── MIG_GT.py, NTSFormer.py
│   │   └── OGM-GE/ (submodule)
│   └── GraphData.py        # Data loading utilities
├── plot/                   # Visualization scripts
│   ├── path_config.sh      # Data path configuration
│   ├── plot_gnn.sh        # Plot training curves
│   └── plot_rank_*.sh     # Rank collapse/training plots
├── run_baseline.sh         # Baseline experiment runner
├── run_supra.sh            # SUPRA experiment runner
└── requirements.yaml       # Conda environment spec
```

## Setup

### 1. Environment

```bash
# Create conda environment
conda env create -f requirements.yaml

# Activate
conda activate MAG
```

### 2. Data Path Configuration

Edit `plot/path_config.sh` to set your data root:

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

### Baseline Experiments

```bash
# Run all baselines on Movies dataset
./run_baseline.sh --data_name Movies

# Run specific model with custom params
./run_baseline.sh --data_name Movies --model GCN --n_layers 2

# Run with specific feature group
FEATURE_GROUPS="default" ./run_baseline.sh --data_name Grocery
```

**Key parameters:**
| Flag | Description | Default |
|------|-------------|---------|
| `--data_name` | Dataset (Movies, Grocery, Toys, Reddit-M) | Required |
| `--model` | Model (MLP, GCN, SAGE, GAT, GCNII, JKNet) | GCN |
| `--feature_group` | Feature set (clip_roberta, default) | clip_roberta |
| `--n_hidden` | Hidden dimension | 256 |
| `--n_layers` | Number of layers | 2 |
| `--gpu` | GPU device ID | 0 |
| `--n_runs` | Number of runs | 3 |
| `--output_dir` | Output directory | logs_baseline |

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
| `--gpu` | GPU device ID | 0 |
| `--n_runs` | Number of runs | 3 |

## Hyperparameters

### Fixed Parameters (All Methods)
- Dropout: 0.3
- Weight decay: 1e-4
- Train ratio: 0.6
- Val ratio: 0.2
- Metric: accuracy
- Average: macro

### Baseline Search Space
| Model | Learning Rate | Layers |
|-------|--------------|--------|
| MLP | 0.001, 0.005 | 2, 3, 4 |
| GCN | 0.001, 0.005 | 1, 2 |
| SAGE | 0.0005, 0.001 | 2, 3, 4 |
| GAT | 0.001, 0.005 | 1, 2 |
| GCNII | 0.001, 0.005 | 2, 3, 4 |
| JKNet | 0.0005, 0.001 | 2, 3, 4 |

### SUPRA Search Space
- Learning rate: 0.0005, 0.001
- Layers: 2, 3, 4
- Embed dim: 128, 256
- Shared depth: 1, 2, 3, 4

## Output

Logs are saved to `logs_baseline/` and `logs_supra/` by default:
```
logs_supra/
└── Movies/
    └── SUPRA-GCN-lr0.001-wd0.0001-h256-L2-...log
```

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