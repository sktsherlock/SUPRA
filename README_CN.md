# 🌍 Language | 语言
| [English](README.md) | [中文](README_CN.md) |

---

# SUPRA: 共享-独特分解的多模态图学习框架

一个多模态图神经网络框架，通过**共享-独特分解**架构解决**模态污染**（modality contamination）问题——即在 GNN 消息传递过程中，某一模态的信号压制另一模态信号的现象。

## 方法概述

SUPRA 将多模态节点表示分解为三个功能不同的通道：

| 通道 | 描述 | 梯度来源 |
|------|------|---------|
| **C（共享）** | `concat(text_enc, visual_enc)` → GNN → 共享语义表示 | `logits_C` loss |
| **Uₜ（文本独特）** | `text_enc` → head_Ut → logits_Ut（不经过图传播） | `logits_Ut` loss |
| **Uᵥ（视觉独特）** | `visual_enc` → head_Uv → logits_Uv（不经过图传播） | `logits_Uv` loss |

**融合方式**：三个通道 logits 等权重平均：
```
logits_final = (logits_C + logits_Ut + logits_Uv) / 3
```

核心设计：Uₜ/Uᵥ 通道**绕过了 GNN 消息传递**，保护模态独特语义不被共享聚合器稀释。可选的**辅助损失**（`aux_weight`）为编码器提供额外的梯度强化。

### 关键参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--mlp_variant` | GNN 前 MLP 投影：`ablate`=无投影（推荐），`full`=Linear→ReLU→LN→Linear | `ablate` |
| `--aux_weight` | Uₜ/Uᵥ 通道的额外梯度强化（0=关闭，推荐 0.5~0.7） | `0.0` |
| `--ablate_bypass` | 消融标志：强制 `logits_final = logits_C` | 关闭 |

### OGM-GE：可选的梯度调制

OGM-GE（CVPR 2022）可作为补充的梯度层面正则化器叠加在 SUPRA 上。它以 C 通道为锚点，检测 Uₜ/Uᵥ 相对于共享表示是否过度自信，并相应地抑制其梯度：

```bash
python -m GNN.SUPRA ... \
    --use_ogm_ge \
    --ogm_alpha 0.5 \
    --ogm_starts 0 \
    --ogm_ends 50
```

详见 [`docs/ogm_ge_experiment.md`](docs/ogm_ge_experiment.md)。

## 支持的方法

### 基线套件（`run_mag_baseline_suite.sh`）

| 实验类型 | 描述 | 调用方式 |
|---------|------|---------|
| **plain** | 单模态 MLP/GNN（纯文本或纯视觉） | `Early_GNN --backend mlp/gnn` |
| **baseline** | 早期融合：`concat(text, visual)` → GNN | `Early_GNN` |
| **late** | 晚期融合：独立编码器 → 独立 GNN → 融合 | `Late_GNN` |
| **nts** | NTSFormer 多模态图 Transformer | `NTSFormer` |
| **mig** | MIG_GT 多跳 GDCF + Transformer | `MIG_GT` |

### SUPRA 套件（`run_supra_suite.sh`）

| 组件 | 选项 |
|------|------|
| GNN 主干 | GCN, SAGE, GAT, GCNII, JKNet |
| 嵌入维度 | 128, 256 |
| aux_weight | 0.0, 0.5, 0.7 |
| mlp_variant | `full`, `ablate` |

### 消融实验（`run_ablation_study.sh`）

| 组别 | 模型 | 配置 | 目的 |
|------|------|------|------|
| **G1** | MMGCN（Late_GNN） | 传统多模态 GNN | 基线 |
| **G2** | SUPRA（Synergy-Only） | `ablate_bypass`：`logits_final = logits_C` | 验证 bypass 价值 |
| **G3** | SUPRA Base | `aux_weight = 0`：三通道无辅助损失 | 架构基线 |
| **G4** | SUPRA Full | `aux_weight > 0`：三通道 + 辅助损失 | 完整模型 |

详见 [`docs/gradient_starvation_experiment.md`](docs/gradient_starvation_experiment.md) 梯度追踪实验。

## 项目结构

```
SUPRA_2.0/
├── GNN/
│   ├── SUPRA.py              # SUPRA 模型（C/Ut/Uv 三通道架构）
│   ├── Utils/
│   │   ├── ogm_ge.py         # OGM-GE 梯度调制
│   │   ├── plot_gradient_norm.py  # 梯度范数可视化
│   │   └── graph_degradation.py   # 特征噪声 / 边重连
│   ├── Baselines/
│   │   ├── Early_GNN.py      # 早期融合（concat → GNN）
│   │   ├── Late_GNN.py       # 晚期融合（独立编码器 → GNN → 融合）
│   │   ├── NTSFormer.py      # 多模态图 Transformer
│   │   └── MIG_GT.py         # 多跳 GDCF + Transformer
│   └── Library/              # 单模态 GNN 基础模块（GCN, SAGE, GAT 等）
├── docs/
│   ├── ogm_ge_experiment.md       # OGM-GE 梯度调制实验
│   ├── gradient_starvation_experiment.md  # 梯度饥饿验证实验
│   └── degradation_experiment.md  # 特征/边退化实验
├── tools/
│   ├── summarize_ogm_ge.py        # OGM-GE 结果汇总
│   ├── run_degradation_experiments.py  # 退化实验运行器
│   └── plot_gradient_norm         # (调用方式: python -m GNN.Utils.plot_gradient_norm)
├── run_mag_baseline_suite.sh  # 综合基线套件
├── run_supra_suite.sh         # SUPRA 主实验
├── run_ablation_study.sh      # 4 组消融实验
└── requirements.yaml           # Conda 环境配置
```

## 环境配置

### 1. 环境搭建

```bash
git clone https://github.com/sktsherlock/SUPRA.git
cd SUPRA
conda env create -f requirements.yaml
conda activate MAG
```

### 2. 数据路径

**重要**：`run_mag_baseline_suite.sh` 直接读取 `DATA_ROOT` 环境变量（不是 `path_config.sh`）：

```bash
export DATA_ROOT=/path/to/MAGB_Dataset
```

其他脚本请编辑 `path_config.sh`：

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

### 基线套件（推荐入口）

```bash
# 在所有数据集上运行所有基线（Plain/Early/Late/NTS/MIG）
DATA_ROOT=/path/to/MAGB_Dataset ./run_mag_baseline_suite.sh

# 预览命令（不执行）
./run_mag_baseline_suite.sh --dry_run

# 自定义选择
EXPERIMENTS="plain baseline late" DATASETS="Movies Grocery" ./run_mag_baseline_suite.sh

# 使用 Llama 特征
FEATURE_GROUPS="default" ./run_mag_baseline_suite.sh
```

### SUPRA 套件

```bash
# 在所有数据集上运行 SUPRA
DATA_ROOT=/path/to/MAGB_Dataset ./run_supra_suite.sh

# 单数据集
DATASETS="Movies" ./run_supra_suite.sh

# 指定结果 CSV
RESULT_CSV="Results/my_results.csv" ./run_supra_suite.sh
```

### 消融实验

```bash
# 4 组消融覆盖所有数据集
./run_ablation_study.sh

# 单独运行各组（Grocery 数据集示例）
python -m GNN.Baselines.Late_GNN ...    # G1: MMGCN
python -m GNN.SUPRA --ablate_bypass ... # G2: Synergy-Only
python -m GNN.SUPRA --aux_weight 0.0 ... # G3: SUPRA Base
python -m GNN.SUPRA --aux_weight 0.7 ... # G4: SUPRA Full
```

详见 [`docs/gradient_starvation_experiment.md`](docs/gradient_starvation_experiment.md) 获取带梯度追踪的完整命令。

### OGM-GE 实验

```bash
# 2×2 因子实验：aux_weight × use_ogm_ge，覆盖 4 个数据集
# 详见 docs/ogm_ge_experiment.md 中的批量运行脚本

# 汇总结果
python tools/summarize_ogm_ge.py
python tools/summarize_ogm_ge.py --f1
```

### 退化实验

```bash
# 特征噪声退化 + 边重连退化
# 详见 docs/degradation_experiment.md 获取完整命令

python tools/run_degradation_experiments.py \
    --data_name Movies \
    --text_feature /path/to/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy \
    --visual_feature /path/to/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /path/to/MoviesGraph.pt \
    --embed_dim 256 --n_layers 3 --lr 0.001 \
    --save_dir Results/degradation --gpu 0
```

## 关键参数

### SUPRA

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--data_name` | 数据集（Movies, Grocery, Toys, Reddit-M） | 必填 |
| `--model_name` | GNN 主干（GCN, SAGE, GAT, GCNII, JKNet） | GCN |
| `--embed_dim` | 嵌入维度 | 256 |
| `--n_layers` | GNN 层数 | 2 |
| `--mlp_variant` | `full` 或 `ablate`（推荐） | `ablate` |
| `--aux_weight` | Uₜ/Uᵥ 辅助损失权重（0=关闭） | 0.0 |
| `--ablate_bypass` | 强制 logits_final = logits_C | 关闭 |
| `--use_ogm_ge` | 启用 OGM-GE 梯度调制 | 关闭 |
| `--gpu` | GPU 设备 ID | 0 |

### 基线套件

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--experiment` | 实验类型 | 必填 |
| `--datasets` | 数据集列表 | 全部 4 个 |
| `--feature_groups` | 特征集（`clip_roberta`, `default`） | `clip_roberta` |
| `--metric` | 指标（`accuracy`, `f1`） | `accuracy` |
| `--n_runs` | 运行次数 | 3 |

## 引用

```bibtex
@misc{supra2026,
  title={SUPRA: Shared-Unique Decomposition for Multimodal Graph Learning},
  author={},
  year={2026}
}
```

## 许可证

详见 [LICENSE](LICENSE)。
