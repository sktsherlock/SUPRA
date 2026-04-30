# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SUPRA is a multimodal graph neural network framework for node classification. The core innovation is a **Shared-Unique Decomposition** architecture that addresses the "modality contamination" problem in multimodal graphs—where one modality's signal overwhelms another's during GNN message passing.

## Architecture

```
text ──► enc_t ──┬──► concat ──► MLP ──► GNN ──► C 通道 (共享语义)
                 │                              (可选mlp_variant=ablate则无MLP)
visual ──► enc_v ┘                                    │
                                                      ├──► Ut 通道 (文本独特语义, 无GNN)
                                                      └──► Uv 通道 (视觉独特语义, 无GNN)

融合: logits_final = (logits_C + logits_Ut + logits_Uv) / 3
```

**三通道设计**:
- **C 通道**: concat(enc_t, enc_v) → (可选MLP投影) → GNN层 → head_C → logits_C，学习跨模态共享信息
- **Ut 通道**: enc_t → head_Ut → logits_Ut，无图传播，学习文本模态独特语义
- **Uv 通道**: enc_v → head_Uv → logits_Uv，无图传播，学习视觉模态独特语义
- **融合**: 三个通道的 logits 等权重相加

**关键设计要点**:
- Ut/Uv 通道**不参与图消息传递**，保证独特语义不被邻居信息污染
- 每个通道的梯度来源独立：C 通道从 `logits_C` 的 loss 回传；Ut/Uv 从各自的 `logits_Ut/Uv` 回传
- `aux_weight > 0` 时，Ut/Uv 会额外获得 `aux_weight * (loss_Ut + loss_Uv)` 的梯度强化
- `mlp_variant`:
  - `full`（默认）：concat 后接 Linear→ReLU→LN→Linear 再进 GNN
  - `ablate`：concat 直接进 GNN，无额外投影层

## Running Experiments

### Data Path Configuration

**IMPORTANT**: `run_mag_baseline_suite.sh` does NOT use `path_config.sh`. It reads the `DATA_ROOT` environment variable directly:

```bash
DATA_ROOT=/your/data/path ./run_mag_baseline_suite.sh
```

To set permanently: `echo 'export DATA_ROOT=/your/path' >> ~/.bashrc && source ~/.bashrc`

For other scripts that use `path_config.sh` (e.g., `run_supra.sh`, `run_baseline.sh`), edit `path_config.sh` directly.

### Main Experiment Scripts

```bash
# Comprehensive baseline suite (main script for all baselines)
# Sets DATA_ROOT env var - does NOT use path_config.sh
DATA_ROOT=/mnt/input/MAGB_Dataset ./run_mag_baseline_suite.sh

# SUPRA + ablation experiments (use path_config.sh)
./run_supra.sh --data_name Movies --model_name GCN
./run_ablation_study.sh

# Baseline experiments (use path_config.sh)
./run_baseline.sh --model GCN --data_name Movies
./run_comprehensive_baseline.sh
```

### Single Model Run Examples

```bash
# NTSFormer
python -m GNN.Baselines.NTSFormer --data_name Movies --metric accuracy \
  --text_feature /path/Movies_roberta_base_512_mean.npy \
  --visual_feature /path/Movies_openai_clip-vit-large-patch14.npy \
  --graph_path /path/MoviesGraph.pt --result_csv Results/best.csv --result_csv_all Results/all.csv

# MIG_GT
python -m GNN.Baselines.MIG_GT --data_name Movies --metric accuracy \
  --text_feature /path/Movies_roberta_base_512_mean.npy \
  --visual_feature /path/Movies_openai_clip-vit-large-patch14.npy \
  --graph_path /path/MoviesGraph.pt --result_csv Results/best.csv --result_csv_all Results/all.csv

# Early_GNN (single modality: --single_modality text|visual)
python -m GNN.Baselines.Early_GNN --data_name Movies --backend mlp --model_name MLP \
  --single_modality text --result_csv Results/best.csv
```

## Code Structure

- `GNN/SUPRA.py` - 主模型, 三通道架构 (C/Ut/Uv) + 等权重融合
- `GNN/Baselines/Early_GNN.py` - 早期融合基线 (concat → GNN); 支持 `--single_modality text|visual` 单模态实验
- `GNN/Baselines/Late_GNN.py` - 晚期融合基线 (各模态独立GNN → 融合)
- `GNN/Baselines/NTSFormer.py` - 多模态图Transformer (SIGN pre-compute + Transformer)
- `GNN/Baselines/MIG_GT.py` - Multi-hop GDCF + Transformer
- `GNN/Library/` - 基础 GNN 实现 (GCN, GAT, SAGE, GCNII, JKNet, SGC, APPNP 等)
- `GNN/Utils/model_config.py` - 通用参数 (add_common_args); 所有 baseline 共享
- `GNN/Utils/NodeClassification.py` - 训练循环、degrade 指标计算 (`_compute_degrade_metrics_mag`)
- `GNN/Utils/result_logger.py` - CSV 结果保存 (`build_result_row`, `update_best_result_csv`)
- `path_config.sh` - 数据路径配置 (被 run_supra.sh 等使用, 但 NOT run_mag_baseline_suite.sh)
- `scripts/baselines/run_mag_baseline_suite.sh` - 综合基线实验主脚本

## Key Parameters

| 参数 | 说明 | 适用模型 |
|------|------|---------|
| `--data_name` | 数据集 (Movies, Grocery, Toys, Reddit-M) | 全部 |
| `--single_modality` | 单模态模式: `text` 或 `visual` | Early_GNN (MLP/GNN) |
| `--backend` | `mlp` 或 `gnn` | Early_GNN |
| `--model_name` | GNN backbone (GCN, SAGE, GAT, GCNII, JKNet) | Early_GNN |
| `--nts_num_tf_layers` | NTSFormer Transformer层数 | NTSFormer |
| `--nts_sign_k` | SIGN pre-compute 的跳数 | NTSFormer |
| `--k_t`, `--k_v` | MGDCF 文本/视觉模态的接收域 hops | MIG_GT |
| `--num_samples` | SGT 全局采样数 C | MIG_GT |
| `--report_drop_modality` | 计算模态降级指标 | 全部 |
| `--degrade_target` | 降级哪个模态: `text`, `visual`, `both` | 全部 |
| `--degrade_alphas` | 噪声强度, 如 `0.2,0.4,0.6,0.8,1.0` | 全部 |
| `--result_csv` | Best 结果写入路径 | NTSFormer/MIG_GT |
| `--result_csv_all` | All runs 结果写入路径 | NTSFormer/MIG_GT |
| `--mlp_variant` | MLP投影模式: `full`（有投影）或 `ablate`（无投影）| SUPRA |
| `--aux_weight` | 额外梯度强化 Ut/Uv 通道的权重, 0=关闭 | SUPRA |

## Common Issues

1. **模态降级指标三者相同 (full=degrade_text=degrade_visual)**
   - Early_GNN: 检查 `--single_modality` 是否正确传递给了模型
   - NTSFormer: `_compute_degrade_metrics_mag` 对 NTSFormer 无效, 因为 `text_h_list` 来自闭包捕获的原始特征; 需要在 degrade 计算时重新 `sign_pre_compute`
   - 确认 `--degrade_alphas` 非空 (默认 `""` 在 `add_common_args` 中, 需要显式设置如 `--degrade_alphas 1.0`)

2. **CSV 结果不保存**: NTSFormer/MIG_GT 必须同时指定 `--result_csv` 和 `--result_csv_all`

3. **bash `for do` 语法错误**: `do` 是 bash 保留关键字, 循环变量不能用 `do`, 应改为其他名如 `drop`

4. **argparse 参数重复定义**: `add_common_args` 已定义 `--report_drop_modality`, `--degrade_target`, `--degrade_alphas`, 不要在各个 baseline 里重复 `parser.add_argument`

5. **JKNet `--jknet_aggr` 无效值**: 有效值为 `concat`, `max`, `last`, 不是 `mean`

6. **DGL 自连接: `graph.remove_self_loop().add_self_loop()`** 必须在 `graph.create_formats_()` 之前调用

7. **`get_metric` 返回数组**: `f1` 等 metric 在 `average!=None` 时返回 per-class 数组, 必须用 `float(np.asarray(score).mean())` 转为标量再比较

8. **MIG_GT `val_score > best_val_score` 类型错误**: `get_metric` 返回 numpy 数组时直接比较会报错, 需要先转为 float

## 研究背景

- **模态独特语义 (Modality-Unique Semantics)**: 某些节点只有特定模态能正确预测
- **模态公平性 (Modality Fairness)**: 保证弱模态也有平等的学习机会, 不被强模态压制
- **MLP投影的作用** (`mlp_variant=full`): 在concat后对异构模态进行投影, 使其适合图传播 (仅在SUPRA中有效, Early_GNN中反而有害)
- **模态编码器 (enc_t/enc_v)**: Linear→ReLU→Dropout, 将原始高维特征投影到 embed_dim 公共空间; 同时作为 Ut/Uv 通道的特征提取器
