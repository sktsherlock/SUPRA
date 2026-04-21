#!/usr/bin/env bash
set -euo pipefail

# 遍历 4 个包含 Llama3/CLIP特征的数据集 * 3 种架构 (plain/early/late) * 2 种 GNN (GCN/SAGE)

# Run rank-over-training plots for all datasets and model types.
# Uses plot_rank_during_training.py (no changes to model source files).

# Load centralized path config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/path_config.sh"

export DGLBACKEND=${DGLBACKEND:-pytorch}

# ---------------- Feature group support ----------------
FEATURE_GROUPS=${FEATURE_GROUPS:-"clip_roberta"}

declare -A TEXT_FEATURE_BY_DS_GROUP
declare -A VIS_FEATURE_BY_DS_GROUP

# Default group
TEXT_FEATURE_BY_DS_GROUP["Movies|default"]='TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Movies|default"]='ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Grocery|default"]='TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Grocery|default"]='ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Toys|default"]='TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Toys|default"]='ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Reddit-M|default"]='TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-M|default"]='ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy'

# CLIP (image) + RoBERTa (text)
TEXT_FEATURE_BY_DS_GROUP["Movies|clip_roberta"]='TextFeature/Movies_roberta_base_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Movies|clip_roberta"]='ImageFeature/Movies_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Grocery|clip_roberta"]='TextFeature/Grocery_roberta_base_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Grocery|clip_roberta"]='ImageFeature/Grocery_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Toys|clip_roberta"]='TextFeature/Toys_roberta_base_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Toys|clip_roberta"]='ImageFeature/Toys_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Reddit-M|clip_roberta"]='TextFeature/RedditM_roberta_base_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-M|clip_roberta"]='ImageFeature/RedditM_openai_clip-vit-large-patch14.npy'

# ---------------- Datasets ----------------
DEFAULT_DATASETS="Movies Grocery Reddit-M Toys"
DATASETS=${DATASETS:-"${DEFAULT_DATASETS}"}
read -r -a DATASETS_ARR <<< "${DATASETS}"
read -r -a FEATURE_GROUPS_ARR <<< "${FEATURE_GROUPS}"

# ---------------- Models ----------------
MODEL_TYPES=${MODEL_TYPES:-"plain early late"}
GNN_MODELS=${GNN_MODELS:-"GCN SAGE"}
read -r -a MODEL_TYPES_ARR <<< "${MODEL_TYPES}"
read -r -a GNN_MODELS_ARR <<< "${GNN_MODELS}"

# ---------------- Training knobs ----------------
GPU_ID=${GPU_ID:-0}
N_EPOCHS=${N_EPOCHS:-100}
EVAL_STEPS=${EVAL_STEPS:-5}
METRIC=${METRIC:-accuracy}
AVERAGE=${AVERAGE:-macro}
SELFLOOP=${SELFLOOP:-true}
UNDIRECTED=${UNDIRECTED:-true}
TRAIN_RATIO=${TRAIN_RATIO:-0.6}
VAL_RATIO=${VAL_RATIO:-0.2}
INDUCTIVE=${INDUCTIVE:-false}

# ---------------- GNN hyperparams (shared) ----------------
N_HIDDEN=${N_HIDDEN:-256}
N_LAYERS=${N_LAYERS:-2}
DROPOUT=${DROPOUT:-0.5}
LR=${LR:-0.001}
WD=${WD:-0.0}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.0}
MM_PROJ_DIM=${MM_PROJ_DIM:-}
LATE_EMBED_DIM=${LATE_EMBED_DIM:-}
MODALITY_DROPOUT=${MODALITY_DROPOUT:-0.0}

OUTPUT_DIR=${OUTPUT_DIR:-./figures/rank_training}

require_file() {
  local path="$1"
  local desc="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[Error] Missing ${desc}: ${path}" >&2
    return 1
  fi
  echo "${path}"
}

dataset_config() {
  local ds="$1"
  local ds_dir="${DATA_ROOT}/${ds}"
  local ds_prefix="${ds//-/}"

  GRAPH_PATH["${ds}"]=$(require_file "${ds_dir}/${ds_prefix}Graph.pt" "graph")

  local key="${ds}|${FEATURE_GROUP}"
  local text_rel="${TEXT_FEATURE_BY_DS_GROUP[${key}]:-}"
  local vis_rel="${VIS_FEATURE_BY_DS_GROUP[${key}]:-}"
  if [[ -z "${text_rel}" || -z "${vis_rel}" ]]; then
    echo "[Error] Missing feature mapping for dataset='${ds}', FEATURE_GROUP='${FEATURE_GROUP}'." >&2
    exit 1
  fi

  TEXT_FEAT["${ds}"]=$(require_file "${ds_dir}/${text_rel}" "text feature (group=${FEATURE_GROUP})")
  VIS_FEAT["${ds}"]=$(require_file "${ds_dir}/${vis_rel}" "visual feature (group=${FEATURE_GROUP})")
}

declare -A GRAPH_PATH
declare -A TEXT_FEAT
declare -A VIS_FEAT

for fg in "${FEATURE_GROUPS_ARR[@]}"; do
  FEATURE_GROUP="${fg}"
  for ds in "${DATASETS_ARR[@]}"; do
    dataset_config "${ds}"
  done

  for ds in "${DATASETS_ARR[@]}"; do
    for model_type in "${MODEL_TYPES_ARR[@]}"; do
      for gnn in "${GNN_MODELS_ARR[@]}"; do
        out_name="rank_over_training_${model_type}_${gnn}_${ds}_${FEATURE_GROUP}.pdf"
        python plot_rank_during_training.py \
          --model_type "${model_type}" \
          --data_name "${ds}" \
          --graph_path "${GRAPH_PATH[$ds]}" \
          --text_feature "${TEXT_FEAT[$ds]}" \
          --visual_feature "${VIS_FEAT[$ds]}" \
          --model_name "${gnn}" \
          --n_hidden "${N_HIDDEN}" \
          --n_layers "${N_LAYERS}" \
          --dropout "${DROPOUT}" \
          --lr "${LR}" \
          --wd "${WD}" \
          --n_epochs "${N_EPOCHS}" \
          --eval_steps "${EVAL_STEPS}" \
          --label_smoothing "${LABEL_SMOOTHING}" \
          --metric "${METRIC}" \
          --average "${AVERAGE}" \
          --train_ratio "${TRAIN_RATIO}" \
          --val_ratio "${VAL_RATIO}" \
          --output_dir "${OUTPUT_DIR}/fg_${FEATURE_GROUP}/${ds}" \
          --output_name "${out_name}" \
          --gpu "${GPU_ID}" \
          --modality_dropout "${MODALITY_DROPOUT}" \
          --inductive "${INDUCTIVE}" \
          --undirected "${UNDIRECTED}" \
          --selfloop "${SELFLOOP}" \
          ${MM_PROJ_DIM:+--mm_proj_dim "${MM_PROJ_DIM}"} \
          ${LATE_EMBED_DIM:+--late_embed_dim "${LATE_EMBED_DIM}"}
      done
    done
  done

done

echo "All runs finished. Output root: ${OUTPUT_DIR}"
