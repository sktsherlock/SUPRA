#!/usr/bin/env bash
set -euo pipefail

# Load centralized path config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path_config.sh"

# python test_plain_gnn.py \
#   --graph_path "${DATA_ROOT}/Movies/MoviesGraph.pt" \
#   --text_feature "${DATA_ROOT}/Movies/TextFeature/Movies_roberta_base_512_mean.npy" \
#   --visual_feature "${DATA_ROOT}/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy" \
#   --selfloop --undirected \
#   --n_epochs 50 \
#   --output_dir ./figures

python test_ogm_gnn.py \
  --graph_path "${DATA_ROOT}/Movies/MoviesGraph.pt" \
  --text_feature "${DATA_ROOT}/Movies/TextFeature/Movies_roberta_base_512_mean.npy" \
  --visual_feature "${DATA_ROOT}/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy" \
  --selfloop --undirected \
  --n_epochs 50 \
  --output_dir ./figures