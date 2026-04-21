#!/usr/bin/env bash
set -euo pipefail

# Load centralized path config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/path_config.sh"

# Plain demo: plot multimodal rank curves for a single dataset
DATASET=${DATASET:-Movies}
FEATURE_GROUP=${FEATURE_GROUP:-clip_roberta}

# ---------------- Feature group mappings ----------------
declare -A TEXT_FEATURE_BY_DS_GROUP
declare -A VIS_FEATURE_BY_DS_GROUP

TEXT_FEATURE_BY_DS_GROUP["Movies|default"]='TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Movies|default"]='ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Grocery|default"]='TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Grocery|default"]='ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Toys|default"]='TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Toys|default"]='ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Reddit-M|default"]='TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-M|default"]='ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Movies|clip_roberta"]='TextFeature/Movies_roberta_base_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Movies|clip_roberta"]='ImageFeature/Movies_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Grocery|clip_roberta"]='TextFeature/Grocery_roberta_base_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Grocery|clip_roberta"]='ImageFeature/Grocery_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Toys|clip_roberta"]='TextFeature/Toys_roberta_base_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Toys|clip_roberta"]='ImageFeature/Toys_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Reddit-M|clip_roberta"]='TextFeature/RedditM_roberta_base_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-M|clip_roberta"]='ImageFeature/RedditM_openai_clip-vit-large-patch14.npy'

ds_dir="${DATA_ROOT}/${DATASET}"
ds_prefix="${DATASET//-/}"
graph_path="${ds_dir}/${ds_prefix}Graph.pt"

key="${DATASET}|${FEATURE_GROUP}"
text_rel="${TEXT_FEATURE_BY_DS_GROUP[${key}]:-}"
vis_rel="${VIS_FEATURE_BY_DS_GROUP[${key}]:-}"

if [[ -z "${text_rel}" || -z "${vis_rel}" ]]; then
  echo "[Error] Missing feature mapping for dataset='${DATASET}', FEATURE_GROUP='${FEATURE_GROUP}'." >&2
  exit 1
fi

text_feature="${ds_dir}/${text_rel}"
visual_feature="${ds_dir}/${vis_rel}"

if [[ ! -f "${graph_path}" ]]; then
  echo "[Error] Missing graph: ${graph_path}" >&2
  exit 1
fi
if [[ ! -f "${text_feature}" ]]; then
  echo "[Error] Missing text feature: ${text_feature}" >&2
  exit 1
fi
if [[ ! -f "${visual_feature}" ]]; then
  echo "[Error] Missing visual feature: ${visual_feature}" >&2
  exit 1
fi

OUTPUT_DIR=${OUTPUT_DIR:-./figures}
GPU_ID=${GPU_ID:-0}
BETA_MIN=${BETA_MIN:-0.0}
BETA_MAX=${BETA_MAX:-8.0}
BETA_STEPS=${BETA_STEPS:-17}
SPLIT=${SPLIT:-test}
PLAIN_HIDDEN=${PLAIN_HIDDEN:-256}
PLAIN_LAYERS=${PLAIN_LAYERS:-2}
PLAIN_DROPOUT=${PLAIN_DROPOUT:-0.5}

mkdir -p "${OUTPUT_DIR}"

python plot_rank_collapse.py \
  --data_name "${DATASET}" \
  --graph_path "${graph_path}" \
  --text_feature "${text_feature}" \
  --visual_feature "${visual_feature}" \
  --plain_demo \
  --beta_min "${BETA_MIN}" \
  --beta_max "${BETA_MAX}" \
  --beta_steps "${BETA_STEPS}" \
  --split "${SPLIT}" \
  --plain_hidden "${PLAIN_HIDDEN}" \
  --plain_layers "${PLAIN_LAYERS}" \
  --plain_dropout "${PLAIN_DROPOUT}" \
  --output_dir "${OUTPUT_DIR}" \
  --gpu "${GPU_ID}"

echo "✓ Plain demo complete. Output: ${OUTPUT_DIR}/plain_gnn_rank_collapse.pdf"