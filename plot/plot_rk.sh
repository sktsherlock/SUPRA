#!/usr/bin/env bash
set -euo pipefail

# Load centralized path config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path_config.sh"

export DGLBACKEND=${DGLBACKEND:-pytorch}

# Minimal plain-GNN rank-over-training test (single run, single dataset)
python plot_rank_during_training.py \
  --model_type plain \
  --data_name Reddit-M \
  --graph_path "${DATA_ROOT}/Reddit-M/RedditMGraph.pt" \
  --text_feature "${DATA_ROOT}/Reddit-M/TextFeature/RedditM_roberta_base_100_mean.npy" \
  --visual_feature "${DATA_ROOT}/Reddit-M/ImageFeature/RedditM_openai_clip-vit-large-patch14.npy" \
  --model_name GCN \
  --n_hidden 256 \
  --n_layers 2 \
  --dropout 0.5 \
  --lr 0.001 \
  --wd 0.0 \
  --n_epochs 10 \
  --eval_steps 2 \
  --degrade_alpha 1.0 \
  --metric accuracy \
  --average macro \
  --train_ratio 0.6 \
  --val_ratio 0.2 \
  --output_dir ./figures/rank_training/quick \
  --output_name rank_over_training_plain_GCN_Reddit-M_clip_roberta.pdf \
  --gpu 0 \
  --undirected \
  --selfloop

echo "Done. Output: ./figures/rank_training/quick/rank_over_training_plain_GCN_Reddit-M_clip_roberta.pdf"