# Efficiency Analysis Commands

效率分析实验命令，记录于 2026-05-01。

**数据集**: Reddit-M
**特征组**: Llama (default)
**数据路径**: `/mnt/input/MAGB_Dataset`
**度量**: Parameters (M), Peak Memory (MB ± std), Total Time (s ± std), Avg Epoch (s/ep ± std), Epochs Needed (± std)
**运行设置**: 2 runs 平均，种子=42

---

## 1. SUPRA-GCN（3层，aux=0.1）

```bash
python tools/profile_efficiency.py \
    --model SUPRA \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --embed_dim 256 --n_layers 3 --n_hidden 256 \
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
    --aux_weight 0.1 \
    --mlp_variant ablate \
    --n_runs 2 --seed 42 \
    --n_profile_epochs 10 --n_epochs 1000 --early_stop_patience 20 \
    --gpu 0
```

## 2. Early_GNN-GCN（3层，与SUPRA参数一致）

```bash
python tools/profile_efficiency.py \
    --model Early_GNN_GCN \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --n_hidden 256 --n_layers 3 \
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
    --n_runs 2 --seed 42 \
    --n_profile_epochs 10 --n_epochs 1000 --early_stop_patience 20 \
    --gpu 0
```

> 注: Early_GNN 为早期融合基线（无编码器，raw concat），与 SUPRA/Late_GNN 架构完全不同。

## 3. Late_GNN-GCN（3层，与Early_GNN参数一致）

```bash
python tools/profile_efficiency.py \
    --model Late_GNN_GCN \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --n_hidden 256 --n_layers 3 \
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
    --n_runs 2 --seed 42 \
    --n_profile_epochs 10 --n_epochs 1000 --early_stop_patience 20 \
    --gpu 0
```

## 4. Late_GNN-GAT（3层，lr=0.001）

```bash
python tools/profile_efficiency.py \
    --model Late_GNN_GAT \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --n_hidden 256 --n_layers 3 \
    --dropout 0.3 --lr 0.001 --wd 0.0001 \
    --n_heads 4 --attn_drop 0.0 --edge_drop 0.0 \
    --n_runs 2 --seed 42 \
    --n_profile_epochs 10 --n_epochs 1000 --early_stop_patience 20 \
    --gpu 0
```

## 5. NTSFormer（lr=0.0005，sign_k=2）

```bash
python tools/profile_efficiency.py \
    --model NTSFormer \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --n_hidden 256 --n_layers 2 \
    --dropout 0.3 --lr 0.0005 --wd 0.0001 \
    --n_heads 4 \
    --nts_sign_k 2 \
    --n_runs 2 --seed 42 \
    --n_profile_epochs 10 --n_epochs 1000 --early_stop_patience 20 \
    --gpu 0
```

## 6. MIG_GT（lr=0.001，k_t=3, k_v=2）

```bash
python tools/profile_efficiency.py \
    --model MIG_GT \
    --data_name Reddit-M \
    --text_feature /mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
    --visual_feature /mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
    --graph_path /mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt \
    --n_hidden 256 --n_layers 2 \
    --dropout 0.3 --lr 0.001 --wd 0.0001 \
    --k_t 3 --k_v 2 \
    --mgdcf_alpha 0.1 --mgdcf_beta 0.9 \
    --num_samples 10 --tur_weight 1.0 \
    --n_runs 2 --seed 42 \
    --n_profile_epochs 10 --n_epochs 1000 --early_stop_patience 20 \
    --gpu 0
```

---

## 实验结果

| 模型 | Params (M) | Peak Memory (MB) | Total Time (s) | Avg Epoch (s/ep) | Epochs Needed |
|------|------------|-----------------|----------------|-----------------|--------------|
| SUPRA-GCN | 2.399 | 4772.53 ± 0.00 | 3.63 ± 0.07 | 0.0579 ± 0.0047 | 63.0 ± 4.0 |
| Early_GNN-GCN | 2.176 | 10048.46 ± 0.00 | 2.94 ± 1.29 | 0.0548 ± 0.0017 | 53.0 ± 22.0 |
| Late_GNN-GCN | — | — | — | — | — |
| Late_GNN-GAT | — | — | — | — | — |
| NTSFormer | — | — | — | — | — |
| MIG_GT | — | — | — | — | — |

---

## 参数汇总表

| 模型 | n_layers | lr | wd | 特殊参数 |
|------|----------|-----|-----|---------|
| SUPRA-GCN | 3 | 0.0005 | 0.0001 | aux=0.1, mlp=ablate |
| Early_GNN | 3 | 0.0005 | 0.0001 | 无编码器，raw concat |
| Late_GNN-GCN | 3 | 0.0005 | 0.0001 | 无编码器，raw 各自 GNN |
| Late_GNN-GAT | 3 | 0.001 | 0.0001 | heads=4, 无编码器 |
| NTSFormer | 2 | 0.0005 | 0.0001 | sign_k=2 |
| MIG_GT | 2 | 0.001 | 0.0001 | k_t=3, k_v=2 |

## 共同超参数

- `n_runs`: 2（取平均）
- `seed`: 42
- `dropout`: 0.3
- `n_hidden`: 256
- `n_profile_epochs`: 10（用于测量 per-epoch 时间和峰值显存）
- `n_epochs`: 1000
- `early_stop_patience`: 20
- `train_ratio`: 0.6
- `val_ratio`: 0.2
- `label_smoothing`: 0.1
- `undirected`: True
- `selfloop`: True
