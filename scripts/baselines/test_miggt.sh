#!/bin/bash

DATA_ROOT="/mnt/input/MAGB_Dataset"

python GNN/Library/MAG/MIG_GT.py \
  --data_name Reddit-M \
  --graph_path ${DATA_ROOT}/Reddit-M/RedditMGraph.pt \
  --text_feature ${DATA_ROOT}/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
  --visual_feature ${DATA_ROOT}/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
  --gpu 0 \
  --inductive false \
  --undirected true \
  --selfloop true \
  --metric accuracy \
  --average macro \
  --n-epochs 500 \
  --n-runs 1 \
  --eval_steps 1 \
  --lr 0.001 \
  --wd 0.0 \
  --n-hidden 256 \
  --dropout 0.2 \
  --label-smoothing 0.1 \
  --disable_wandb \
  --log-every 1 \
  --k_t 3 \
  --k_v 2 \
  --mgdcf_alpha 0.1 \
  --mgdcf_beta 0.9 \
  --num_samples 10 \
  --tur_weight 1.0 \
  --tur_sample_edges 8000


# python GNN/Library/MAG/Late_GNN.py \
#   --data_name Reddit-M \
#   --graph_path ${DATA_ROOT}/Reddit-M/RedditMGraph.pt \
#   --text_feature ${DATA_ROOT}/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
#   --visual_feature ${DATA_ROOT}/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
#   --gpu 0 \
#   --inductive false \
#   --undirected true \
#   --selfloop true \
#   --metric accuracy \
#   --average macro \
#   --n-epochs 2 \
#   --n-runs 1 \
#   --eval_steps 1 \
#   --lr 0.001 \
#   --wd 0.0 \
#   --n-hidden 256 \
#   --dropout 0.2 \
#   --label-smoothing 0.1 \
#   --disable_wandb \
#   --log-every 1 \
