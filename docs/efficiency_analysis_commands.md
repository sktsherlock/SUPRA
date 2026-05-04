# Efficiency Analysis Commands

效率分析实验命令，记录于 2026-05-02。效率统计（Parameters、Peak Memory、Total Time、Avg Epoch、Epochs Needed）已集成到各模型训练代码中，直接运行即可在末尾看到 Efficiency Profile。

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
    --n-epochs 1000 --early_stop_patience 20 \
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
    --n-epochs 1000 --early_stop_patience 20 \
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
    --n-epochs 1000 --early_stop_patience 20 \
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
    --n-epochs 1000 --early_stop_patience 20 \
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
    --n-epochs 1000 --early_stop_patience 20 \
    --result_csv Results/efficiency/late_gnn_gat_reddit_m.csv \
    --disable_wandb \
    --gpu 0
```

## 6. NTSFormer（lr=0.0005，sign_k=1，num_heads=2）

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
    --n-epochs 1000 --early_stop_patience 20 \
    --result_csv Results/efficiency/ntsformer_reddit_m.csv \
    --disable_wandb \
    --gpu 0
```

> 注: NTSFormer 默认不使用 inductive（`inductive=False`），所以 `observe_graph == graph`（不移除边）。最佳参数: `--nts_num_heads=2`, `--nts_sign_k=1`（来自 gpu1_default_accuracy_best.csv）。

## 7. MIG_GT（lr=0.001，k_t=3, k_v=2）

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
    --n-epochs 1000 --early_stop_patience 20 \
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
  Peak Memory:     XXXX.XX ± XX.XX MB
  Total Time(est): XXXX.XX ± XXX.XX s  (XX.X min)
  Avg Epoch:        X.XXXX ± X.XXXX s/epoch
  Epochs Needed:    XXX.X ± XX.X
============================================================
```

同时每 run 结束会打印详细时间分解：
```
[TIME] train_total=XXX.Xs(avg=X.XXXs×XXXep)  eval_total=XXX.Xs(avg=X.XXXs×XXep)
```

---

## 实验结果

| 模型 | Params (M) | Peak Memory (MB) | Peak Reserved (MB) | Total Time (s) | Avg Epoch (s/ep) | Epochs Needed |
|------|------------|-----------------|-------------------|----------------|-----------------|--------------|
| SUPRA-GCN | 2.399 | 5619.27 ± 28.78 | 7956.00 ± 0.00 | 14.88 ± 0.87 | 0.0580 ± 0.0040 | 256.5 ± 2.5 |
| Early_GNN-GCN | 2.176 | 12691.49 ± 0.00 | 13196.00 ± 10.00 | 8.18 ± 0.95 | 0.0520 ± 0.0025 | 157.5 ± 10.5 |
| Early_GNN-GraphSAGE | 4.352 | 10002.26 ± 0.33 | 10406.00 ± 10.00 | 23.43 ± 3.41 | 0.0588 ± 0.0014 | 398.5 ± 67.5 |
| Late_GNN-GCN | 2.386 | 19473.32 ± 0.00 | 20024.00 ± 34.00 | 10.94 ± 0.99 | 0.0990 ± 0.0059 | 110.5 ± 16.5 |
| Late_GNN-GAT | 11.041 | 20898.52 ± 4.29 | 24811.00 ± 781.00 | 24.55 ± 1.01 | 0.2838 ± 0.0035 | 86.5 ± 2.5 |
| NTSFormer | 5.788 | 62447.03 ± 0.00 | 32115 (nvidia-smi) | 838.23 ± 0.00 | 5.5512 ± 0.0000 | 151.0 ± 0.0 |
| MIG_GT | 2.441 | 21751.33 ± 0.00 | 26854.00 ± 0.00 | 113.32 ± 0.00 | 0.2529 ± 0.0000 | 448.0 ± 0.0 |

> **Peak Memory**：PyTorch `memory_allocated` 统计的真实 tensor 显存（不含 allocator 缓存）。
> **Peak Reserved**：PyTorch `memory_reserved` 统计的 allocator 预留总量（**NTSFormer 除外**，该值为 nvidia-smi 实测值，因 PyTorch allocator 碎片化导致 reserved 偏高）。
> **数据集**：Reddit-M，2 runs 平均，seed=42。

## 参数汇总表

| 模型 | n_layers | lr | wd | 特殊参数 |
|------|----------|-----|-----|---------|
| SUPRA-GCN | 3 | 0.0005 | 0.0001 | aux=0.1, mlp=ablate |
| Early_GNN-GCN | 3 | 0.0005 | 0.0001 | 无编码器，raw concat |
| Early_GNN-GraphSAGE | 3 | 0.0005 | 0.0001 | 无编码器，raw concat |
| Late_GNN-GCN | 3 | 0.0005 | 0.0001 | 无编码器，raw 各自 GNN |
| Late_GNN-GAT | 3 | 0.001 | 0.0001 | heads=4, 无编码器 |
| NTSFormer | 2 | 0.0005 | 0.0001 | sign_k=1, num_heads=2 |
| MIG_GT | 2 | 0.001 | 0.0001 | k_t=3, k_v=2 |

## 共同超参数

- `n_runs`: 2（取平均）
- `seed`: 42
- `dropout`: 0.3
- `n_hidden`: 256
- `n_epochs`: 1000
- `early_stop_patience`: 20
- `train_ratio`: 0.6
- `val_ratio`: 0.2
- `label_smoothing`: 0.1
- `undirected`: True
- `selfloop`: True
