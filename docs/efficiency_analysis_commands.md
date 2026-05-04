# Efficiency Analysis Commands

效率分析实验命令，记录于 2026-05-02。效率统计（Train(s)、Eval(s)、Total Time、Train(s/ep)、Eval(s/call)、Epochs Needed）已集成到各模型训练代码中，直接运行即可在末尾看到 Efficiency Profile。

**数据集**: Reddit-M
**特征组**: Llama (default)
**数据路径**: `/mnt/input/MAGB_Dataset`
**运行设置**: 2 runs 平均，种子=42

---

## 1. SUPRA-GCN（3层，aux=0.1）

```bash
python -m GNN.SUPRA \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --embed_dim 256 --n-layers 3 --n-hidden 256 \
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
    --aux_weight 0.1 \
    --mlp_variant ablate \
    --n-runs 2 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/efficiency/supra_gcn_reddit_m.csv \
    --disable_wandb \
    --gpu 0
```

## 2. Early_GNN-GCN（3层，与SUPRA参数一致）

```bash
python -m GNN.Baselines.Early_GNN \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --backend gnn --model_name GCN \
    --n-hidden 256 --n-layers 3 \
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
    --n-runs 2 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/efficiency/early_gnn_gcn_reddit_m.csv \
    --disable_wandb \
    --gpu 0
```

## 3. Early_GNN-GraphSAGE（3层，与SUPRA参数一致）

```bash
python -m GNN.Baselines.Early_GNN \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --backend gnn --model_name SAGE \
    --n-hidden 256 --n-layers 3 \
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
    --n-runs 2 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/efficiency/early_gnn_sage_reddit_m.csv \
    --disable_wandb \
    --gpu 0
```

> 注: Early_GNN 为早期融合基线（无编码器，raw concat），与 SUPRA/Late_GNN 架构完全不同。

## 4. Late_GNN-GCN（3层，与Early_GNN参数一致）

```bash
python -m GNN.Baselines.Late_GNN \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --model_name GCN \
    --n-hidden 256 --n-layers 3 \
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
    --n-runs 2 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/efficiency/late_gnn_gcn_reddit_m.csv \
    --disable_wandb \
    --gpu 0
```

## 5. Late_GNN-GAT（3层，lr=0.001）

```bash
python -m GNN.Baselines.Late_GNN \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --model_name GAT \
    --n-hidden 256 --n-layers 3 \
    --dropout 0.3 --lr 0.001 --wd 0.0001 \
    --n-heads 4 --attn-drop 0.0 --edge-drop 0.0 \
    --n-runs 2 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --result_csv Results/efficiency/late_gnn_gat_reddit_m.csv \
    --disable_wandb \
    --gpu 0
```

## 6. NTSFormer（lr=0.0005，sign_k=1，num_heads=2，eval_steps=5）

```bash
python -m GNN.Baselines.NTSFormer \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --n-hidden 256 --n-layers 2 \
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
    --nts_num_heads 2 \
    --nts_sign_k 1 \
    --n-runs 2 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --eval_steps 5 \
    --result_csv Results/efficiency/ntsformer_reddit_m.csv \
    --disable_wandb \
    --gpu 0
```

> 注: NTSFormer 默认不使用 inductive（`inductive=False`），所以 `observe_graph == graph`（不移除边）。最佳参数: `--nts_num_heads=2`, `--nts_sign_k=1`（来自 gpu1_default_accuracy_best.csv）。

## 7. MIG_GT（lr=0.001，k_t=3, k_v=2，eval_steps=5）

```bash
python -m GNN.Baselines.MIG_GT \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --n-hidden 256 --n-layers 2 \
    --dropout 0.3 --lr 0.001 --wd 0.0001 \
    --k_t 3 --k_v 2 \
    --mgdcf_alpha 0.1 --mgdcf_beta 0.9 \
    --num_samples 10 --tur_weight 1.0 \
    --n-runs 2 --seed 42 \
    --n-epochs 1000 --early_stop_patience 30 \
    --eval_steps 5 \
    --result_csv Results/efficiency/mig_gt_reddit_m.csv \
    --disable_wandb \
    --gpu 0
```

---
## 输出说明

每个模型运行结束后会输出：

```
============================================================
Efficiency Profile: <ModelName> on <data_name>
============================================================
  Parameters:       X.XXX M
  Peak Reserved:   XXXX.XX ± XX.XX MB
  Train(s):       XXXX.XX ± XX.XX s
  Eval(s):        XXXX.XX ± XX.XX s
  Total Time:     XXXX.XX ± XXXX.XX s  (XX.X min)
  Train(s/ep):    X.XXXX ± X.XXXX s/epoch
  Eval(s/call):   X.XXXX ± X.XXXX s/call
  Epochs Needed:  XXX.X ± XX.X
============================================================
```

同时每 run 结束会打印详细时间分解：
```
[TIME] train_total=XXX.Xs(avg=X.XXXs×XXXep)  eval_total=XXX.Xs(avg=X.XXXs×XXep)
```

---

## 实验结果

| 模型 | eval_steps | Params (M) | Peak Reserved (MB) | Train(s) | Eval(s) | Total(s) | Train(s/ep) | Eval(s/call) | Epochs |
|------|-----------|------------|-------------------|----------|---------|----------|-------------|--------------|--------|
| SUPRA-GCN | 1 | 2.399 | 7956.00 ± 0.00 | - | - | - | - | - | - |
| GCN | 1 | 2.176 | 13196.00 ± 10.00 | - | - | - | - | - | - |
| GraphSAGE | 1 | 4.352 | 10406.00 ± 10.00 | - | - | - | - | - | - |
| MMGCN | 1 | 2.386 | 20024.00 ± 34.00 | - | - | - | - | - | - |
| MMGAT | 1 | 11.041 | 24811.00 ± 781.00 | - | - | - | - | - | - |
| NTSFormer | 5 | 5.788 | 32115 (nvidia-smi) | - | - | - | - | - | - |
| MIG_GT | 5 | 2.441 | 26854.00 ± 0.00 | - | - | - | - | - | - |

> **注意**: 表格中 "-" 表示该实验尚未用新参数重新运行以获得准确数值。NTSFormer 和 MIG_GT 的 eval_steps=5，其他模型 eval_steps=1。
>
> **Peak Reserved**：PyTorch `memory_reserved` 统计的 allocator 预留总量，对应 nvidia-smi 显示的显存占用（**NTSFormer 除外**，该值为 nvidia-smi 实测值，因 PyTorch allocator 碎片化导致 reserved 偏高）。
>
> **Train/Eval 分离说明**：`Train(s)` 为累计训练时长，`Eval(s)` 为累计验证时长，`Total(s) = Train(s) + Eval(s)`，`Train(s/ep)` 和 `Eval(s/call)` 为平均值。
>
> **数据集**：Reddit-M，2 runs 平均，seed=42。

---

### Peak Reserved 统计说明

PyTorch CUDA allocator 采用 caching 策略：`memory_reserved` 表示 allocator 已从 GPU 获取但尚未分配给 tensor 的缓存内存，以及已分配的 tensor 内存总和。这与 `nvidia-smi` 报告的显存占用（显存池 + 已分配）完全一致，是实际硬件显存占用的真实反映。

`memory_allocated` 仅统计当前 live tensor 显存，不包含 allocator 内部碎片和已释放但未归还的缓存块。因此论文汇报统一使用 **Peak Reserved** 作为显存指标。

## 参数汇总表

| 模型 | n_layers | eval_steps | lr | wd | 特殊参数 |
|------|----------|-----------|-----|-----|---------|
| SUPRA-GCN | 3 | 1 | 0.0005 | 0.0001 | aux=0.1, mlp=ablate |
| GCN | 3 | 1 | 0.0005 | 0.0001 | 无编码器，raw concat |
| GraphSAGE | 3 | 1 | 0.0005 | 0.0001 | 无编码器，raw concat |
| MMGCN | 3 | 1 | 0.0005 | 0.0001 | 无编码器，raw 各自 GNN |
| MMGAT | 3 | 1 | 0.001 | 0.0001 | heads=4, 无编码器 |
| NTSFormer | 2 | 5 | 0.0005 | 0.0001 | sign_k=1, num_heads=2 |
| MIG_GT | 2 | 5 | 0.001 | 0.0001 | k_t=3, k_v=2 |

## 共同超参数

- `n_runs`: 2（取平均）
- `seed`: 42
- `dropout`: 0.3
- `n_hidden`: 256
- `n_epochs`: 1000
- `early_stop_patience`: 30
- `train_ratio`: 0.6
- `val_ratio`: 0.2
- `label_smoothing`: 0.1
- `undirected`: True
- `selfloop`: True
