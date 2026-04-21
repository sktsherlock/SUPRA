# SUPRA: 基于谱正交化的统一多模态学习

一个多模态图神经网络框架，通过谱正交化缓解共享图表示学习中的梯度竞争和低秩偏置问题。

## 方法概述

SUPRA 提出了一种**共享-独特分解**架构，结合**Newton-Schulz 正交化**来促进共享参数和模态特定参数之间的多样性：

- **共享通道 (C)**：学习通用图结构的 GNN 主干网络
- **独特编码器 (Uₜ, Uᵥ)**：用于文本和视觉特征的模态特定 Transformer
- **谱正交化 Hook**：梯度后处理，强制正交的更新方向
- **动态正交化强度 (α)**：原始梯度和正交化梯度的自适应混合

## 支持的方法

### 基线模型（`run_baseline.sh`）
| 模型 | 描述 | 层数 |
|------|------|------|
| MLP | 多层感知机（无图结构） | 2, 3 |
| GCN | 图卷积网络（通过 Early_GNN） | 1, 2 |
| SAGE | GraphSAGE（均值聚合，通过 Early_GNN） | 2, 3, 4 |
| GAT | 图注意力网络（通过 Early_GNN） | 1, 2 |
| GCNII | 带初始残差和恒等映射的 GCN | 2, 3, 4 |
| JKNet | 跳跃知识网络 | 2, 3, 4 |

### SUPRA（`run_supra.sh`）
| 组件 | 描述 | 选项 |
|------|------|------|
| GNN 主干 | GCN, SAGE, GAT, RevGAT | - |
| 嵌入维度 | 共享嵌入维度 | 128, 256 |
| 共享深度 | 共享通道深度 | 1, 2, 3, 4 |

## 项目结构

```
SUPRA_2.0/
├── GNN/
│   ├── SUPRA.py              # SUPRA 模型实现（含谱正交化）
│   ├── Library/               # 简单的单模态基线
│   │   ├── MLP.py, GCN.py, GAT.py, GraphSAGE.py
│   │   ├── GCNII.py, JKNet.py, SGC.py, APPNP.py
│   ├── Baselines/            # 多模态基线（Early/Late 融合）
│   │   ├── Early_GNN.py       # Early 融合：拼接 text+visual -> GNN
│   │   ├── Late_GNN.py       # Late 融合：独立编码器 -> GNN -> 融合
│   │   ├── MIG_GT.py, NTSFormer.py
│   │   └── OGM-GE/, NTSFormer/ (子模块)
│   └── GraphData.py          # 数据加载工具
├── plot/                      # 可视化脚本
│   ├── path_config.sh         # 数据路径配置
│   └── plot_*.sh              # 各种绘图脚本
├── scripts/                   # 额外的实验脚本
│   ├── baselines/            # 基线扫描脚本
│   └── supra/                 # SUPRA 扫描脚本
├── run_baseline.sh            # 基线实验运行器
├── run_supra.sh               # SUPRA 实验运行器
├── run_batch_baseline.sh      # 所有基线的批量运行脚本
└── requirements.yaml          # Conda 环境配置
```

## 环境配置

### 1. 环境搭建

```bash
# 克隆仓库
git clone https://github.com/sktsherlock/SUPRA.git
cd SUPRA

# 创建 conda 环境
conda env create -f requirements.yaml

# 激活环境
conda activate MAG
```

### 2. 数据路径配置

编辑 `plot/path_config.sh` 设置数据根目录：

```bash
DATA_ROOT="${DATA_ROOT:-/path/to/your/MAGB_Dataset}"
export DATA_ROOT
```

所需数据结构：
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

支持的特征组：
- `clip_roberta`（默认）：CLIP 视觉 + RoBERTa 文本
- `default`：Llama 3.2 Vision 特征

## 运行实验

### 快速开始

```bash
# 运行所有基线模型实验
./run_batch_baseline.sh

# 或运行特定模型
./run_baseline.sh --model GCN --data_name Movies

# 运行 SUPRA 实验
./run_supra.sh --data_name Movies
```

### 基线实验

```bash
# 在 Movies 数据集上运行所有基线模型
./run_baseline.sh --model GCN --data_name Movies

# 使用自定义参数运行特定模型
./run_baseline.sh --data_name Movies --model SAGE --n_layers 2

# 使用特定特征组
FEATURE_GROUPS="default" ./run_baseline.sh --data_name Grocery --model GAT

# 使用 F1-macro 指标
./run_baseline.sh --data_name Movies --model GCN --metric f1_macro --average macro
```

**关键参数：**
| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--data_name` | 数据集（Movies, Grocery, Toys, Reddit-M） | 必填 |
| `--model` | 模型（MLP, GCN, SAGE, GAT, GCNII, JKNet） | 必填 |
| `--feature_group` | 特征集（clip_roberta, default） | clip_roberta |
| `--metric` | 指标（accuracy, f1_macro） | accuracy |
| `--average` | 平均方式（macro, micro） | macro |
| `--gpu` | GPU 设备 ID | 0 |
| `--n_runs` | 运行次数 | 3 |

### SUPRA 实验

```bash
# 使用 GCN 主干运行 SUPRA
./run_supra.sh --data_name Movies

# 使用 GAT 主干和自定义嵌入维度
./run_supra.sh --data_name Grocery --model_name GAT --embed_dim 128

# 指定 shared_depth 范围
SUPRA_LAYERS="2 3 4" ./run_supra.sh --data_name Toys
```

**关键参数：**
| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--data_name` | 数据集（Movies, Grocery, Toys, Reddit-M） | 必填 |
| `--model_name` | GNN 主干（GCN, SAGE, GAT, RevGAT） | GCN |
| `--feature_group` | 特征集（clip_roberta, default） | clip_roberta |
| `--embed_dim` | 嵌入维度 | 256 |
| `--shared_depth` | 共享通道深度 | 2 |
| `--gpu` | GPU 设备 ID | 0 |
| `--n_runs` | 运行次数 | 3 |

### 批量实验

```bash
# 预览所有基线实验命令（不执行）
./run_batch_baseline.sh --dry_run

# 只运行 GCN 和 SAGE
./run_batch_baseline.sh --models "GCN SAGE"

# 只运行 F1-macro 指标
./run_batch_baseline.sh --metrics "f1_macro"
```

## 超参数

### 固定参数（所有方法）
- Dropout: 0.3（固定）
- Weight decay: 1e-4
- Train ratio: 0.6
- Val ratio: 0.2

### 基线搜索空间
| 模型 | 学习率 | 层数 |
|------|--------|------|
| MLP | 0.0005, 0.001 | 2, 3 |
| GCN | 0.0005, 0.001 | 1, 2 |
| SAGE | 0.0005, 0.001 | 2, 3, 4 |
| GAT | 0.0005, 0.001 | 1, 2 |
| GCNII | 0.0005, 0.001 | 2, 3, 4 |
| JKNet | 0.0005, 0.001 | 2, 3, 4 |

### SUPRA 搜索空间
- 学习率: 0.0005, 0.001
- 层数 (n_layers): 2, 3, 4
- 嵌入维度: 128, 256
- 共享深度: 1, 2, 3, 4

## 输出

结果保存位置：
- `logs_baseline/` - 基线实验的训练日志
- `logs_supra/` - SUPRA 实验的训练日志
- `results_csv/baseline_best.csv` - 每个方法/数据集/指标的最佳结果
- `results_csv/baseline_all.csv` - 所有实验结果

## 引用

如果觉得这个工作有帮助，请引用：

```bibtex
@misc{supra2026,
  title={SUPRA: Unified Multimodal Learning with Spectral Orthogonalization},
  author={},
  year={2026}
}
```

## 许可证

详见 [LICENSE](LICENSE)。