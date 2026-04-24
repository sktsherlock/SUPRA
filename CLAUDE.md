# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SUPRA is a multimodal graph neural network framework for node classification. The core innovation is a **Shared-Unique Decomposition** architecture that addresses the "modality contamination" problem in multimodal graphs—where one modality's signal overwhelms another's during GNN message passing.

## Architecture

```
text ──► enc_t ──┬──► concat ──► GNN ──► C 通道 (共享语义)
                │
visual ──► enc_v ┘   │
                      ├──► Ut 通道 (文本独特语义, 无GNN)
                      └──► Uv 通道 (视觉独特语义, 无GNN)

融合: logits_final = (logits_C + logits_Ut + logits_Uv) / 3
```

**关键设计**:
- **C 通道**: concat(t,v) → GNN → 预测, 学习跨模态共享信息
- **Ut/Uv 通道**: 直接编码 → 预测, 无图传播, 学习各模态独特信息
- **Spectral Orthogonalization**: 梯度后处理, 防止共享表示低秩崩溃

## Running Experiments

```bash
# 激活环境
conda activate MAG

# 配置数据路径 (编辑 path_config.sh)
DATA_ROOT="/path/to/MAGB_Dataset"

# SUPRA 完整实验
./run_supra.sh --data_name Movies --model_name GCN --embed_dim 256 --shared_depth 2 --ortho_alpha 1.0

# 4路消融实验
./run_ablation_study.sh

# 基线对比
./run_baseline.sh --model GCN --data_name Movies

# 综合基线 (包含 NTSFormer, MIG-GT 等)
./run_comprehensive_baseline.sh
```

## Key Parameters

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_name` | 数据集 (Movies, Grocery, Toys, Reddit-M) | 必需 |
| `--model_name` | GNN backbone (GCN, SAGE, GAT, RevGAT) | GCN |
| `--embed_dim` | embedding 维度 | 256 |
| `--shared_depth` | C通道 GNN层数 | 2 |
| `--ortho_alpha` | 正交化强度 (0=关闭, 1=开启) | 1.0 |
| `--use_aux_loss` | 启用辅助损失 | off |
| `--use_gate` | 启用通道门控 (已移除, 不推荐) | off |

## Code Structure

- `GNN/SUPRA.py` - 主模型, 包含三通道架构和谱正交化
- `GNN/Baselines/Early_GNN.py` - 早期融合基线 (concat → GNN)
- `GNN/Baselines/Late_GNN.py` - 晚期融合基线 (各模态独立GNN → 融合)
- `GNN/Library/` - 基础 GNN 实现 (GCN, GAT, SAGE, GCNII 等)
- `GNN/Utils/model_config.py` - 通用参数定义 (add_common_args)
- `GNN/Utils/result_logger.py` - CSV 结果保存接口
- `path_config.sh` - 数据路径配置

## Common Issues

1. **CSV结果不保存**: NTSFormer/MIG-GT 需要 `--result_csv` 和 `--result_csv_all` 参数
2. **CUDA版本**: requirements.yaml 指定 CUDA 12.1, 需匹配 PyTorch 版本
3. **DGL版本**: 依赖特定 DGL 版本, 勿随意升级

## 研究背景

- **模态独特语义 (Modality-Unique Semantics)**: 某些节点只有特定模态能正确预测
- **模态公平性 (Modality Fairness)**: 保证弱模态也有平等的学习机会, 不被强模态压制
- **MLP_s的作用**: 在concat后对异构模态进行投影, 使其适合图传播 (仅在SUPRA中有效, EarlyGNN中反而有害)
