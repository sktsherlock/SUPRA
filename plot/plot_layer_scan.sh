#!/usr/bin/env bash
set -euo pipefail

export DGLBACKEND=${DGLBACKEND:-pytorch}

# Load centralized path config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path_config.sh"

OUTPUT_DIR="./figures/layer_scan"

# Default Dataset: Movies
DATASET=${DATASET:-"Movies"}
FEATURE_GROUP="clip_roberta"

# Paths
# Adjusted paths based on ls exploration and error correction
GRAPH_PATH="${DATA_ROOT}/${DATASET}/${DATASET}Graph.pt" 
# Note: GraphData.py uses dgl.load_graphs. Usually .pt implies pickling for DGL but dgl.save_graphs output.
# If .mat failed, likely it wasn't there. Found 'MoviesGraph.pt' in ls output.

TEXT_FEAT="${DATA_ROOT}/${DATASET}/TextFeature/${DATASET}_roberta_base_512_mean.npy"
VIS_FEAT="${DATA_ROOT}/${DATASET}/ImageFeature/${DATASET}_openai_clip-vit-large-patch14.npy"

# Scan Parameters
LAYERS_SCAN="1,2,3,4,5,6"
N_EPOCHS=${N_EPOCHS:-200}
LR=${LR:-0.001}
DROPOUT=${DROPOUT:-0.5}
DEGRADE_ALPHA=${DEGRADE_ALPHA:-1.0}

MODEL_TYPE=${MODEL_TYPE:-"plain"} # plain implies GCN on [Text||Vis]
MODEL_NAME=${MODEL_NAME:-"GCN"}

# Check if files exist
if [ ! -f "$GRAPH_PATH" ]; then
    echo "Error: Graph file not found at $GRAPH_PATH"
    exit 1
fi
if [ ! -f "$TEXT_FEAT" ]; then
    echo "Error: Text feature not found at $TEXT_FEAT"
    exit 1
fi

echo ">>> Running Layer Scan on ${DATASET} with ${MODEL_NAME} (${MODEL_TYPE})..."
echo ">>> Layers: ${LAYERS_SCAN}"

python plot_layer_scan.py \
    --data_name "${DATASET}" \
    --graph_path "${GRAPH_PATH}" \
    --text_feature "${TEXT_FEAT}" \
    --visual_feature "${VIS_FEAT}" \
    --model_type "${MODEL_TYPE}" \
    --model_name "${MODEL_NAME}" \
    --layers_scan "${LAYERS_SCAN}" \
    --n_hidden 256 \
    --dropout "${DROPOUT}" \
    --lr "${LR}" \
    --n_epochs "${N_EPOCHS}" \
    --metric "accuracy" \
    --average "macro" \
    --degrade_alpha "${DEGRADE_ALPHA}" \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --eval_split "test" \
    --output_dir "${OUTPUT_DIR}/${DATASET}" \
    --output_name "layer_scan_${MODEL_TYPE}_${MODEL_NAME}.pdf" \
    --save_csv \
    --selfloop \
    --undirected \
    --gpu 0

echo ">>> Scan Complete. Results saved to ${OUTPUT_DIR}/${DATASET}"
