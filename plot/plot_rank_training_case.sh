#!/usr/bin/env bash
set -euo pipefail

# 针对某几个特定的情况（Case Study），绘制 rank-over-training 曲线

# Case run: rank-over-training for all datasets & models (outputs per-dataset folders)
export DGLBACKEND=${DGLBACKEND:-pytorch}

DATA_ROOT=${DATA_ROOT:-/hyperai/input/input0/MAGB_Dataset}
FEATURE_GROUPS=${FEATURE_GROUPS:-"clip_roberta"}

declare -A TEXT_FEATURE_BY_DS_GROUP
declare -A VIS_FEATURE_BY_DS_GROUP

# CLIP (image) + RoBERTa (text)
TEXT_FEATURE_BY_DS_GROUP["Movies|clip_roberta"]='TextFeature/Movies_roberta_base_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Movies|clip_roberta"]='ImageFeature/Movies_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Grocery|clip_roberta"]='TextFeature/Grocery_roberta_base_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Grocery|clip_roberta"]='ImageFeature/Grocery_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Toys|clip_roberta"]='TextFeature/Toys_roberta_base_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Toys|clip_roberta"]='ImageFeature/Toys_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Reddit-M|clip_roberta"]='TextFeature/RedditM_roberta_base_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-M|clip_roberta"]='ImageFeature/RedditM_openai_clip-vit-large-patch14.npy'

DEFAULT_DATASETS="Movies Grocery Reddit-M Toys"
DATASETS=${DATASETS:-"${DEFAULT_DATASETS}"}
read -r -a DATASETS_ARR <<< "${DATASETS}"
read -r -a FEATURE_GROUPS_ARR <<< "${FEATURE_GROUPS}"

MODEL_TYPES=${MODEL_TYPES:-"plain early late"}
GNN_MODELS=${GNN_MODELS:-"GCN SAGE"}
read -r -a MODEL_TYPES_ARR <<< "${MODEL_TYPES}"
read -r -a GNN_MODELS_ARR <<< "${GNN_MODELS}"

GPU_ID=${GPU_ID:-0}
N_EPOCHS=${N_EPOCHS:-100}
EVAL_STEPS=${EVAL_STEPS:-5}
METRIC=${METRIC:-accuracy}
AVERAGE=${AVERAGE:-macro}
SELFLOOP=${SELFLOOP:-true}
UNDIRECTED=${UNDIRECTED:-true}
TRAIN_RATIO=${TRAIN_RATIO:-0.6}
VAL_RATIO=${VAL_RATIO:-0.2}
DEGRADE_ALPHA=${DEGRADE_ALPHA:-1.0}

N_HIDDEN=${N_HIDDEN:-256}
N_LAYERS=${N_LAYERS:-2}
DROPOUT=${DROPOUT:-0.5}
LR=${LR:-0.001}
WD=${WD:-0.0}
MM_PROJ_DIM=${MM_PROJ_DIM:-}
LATE_EMBED_DIM=${LATE_EMBED_DIM:-}
MODALITY_DROPOUT=${MODALITY_DROPOUT:-0.0}

OUTPUT_ROOT=${OUTPUT_ROOT:-./figures/case}

add_flag() {
  local flag="$1"
  local val="$2"
  if [[ "${val}" == "true" ]]; then
    echo "${flag}"
  fi
}

require_file() {
  local path="$1"
  local desc="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[Error] Missing ${desc}: ${path}" >&2
    exit 1
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
        out_name="rank_case_${model_type}_${gnn}_${ds}_${FEATURE_GROUP}.pdf"
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
          --degrade_alpha "${DEGRADE_ALPHA}" \
          --metric "${METRIC}" \
          --average "${AVERAGE}" \
          --train_ratio "${TRAIN_RATIO}" \
          --val_ratio "${VAL_RATIO}" \
          --output_dir "${OUTPUT_ROOT}/${ds}" \
          --output_name "${out_name}" \
          --gpu "${GPU_ID}" \
          --modality_dropout "${MODALITY_DROPOUT}" \
          $(add_flag --undirected "${UNDIRECTED}") \
          $(add_flag --selfloop "${SELFLOOP}") \
          ${MM_PROJ_DIM:+--mm_proj_dim "${MM_PROJ_DIM}"} \
          ${LATE_EMBED_DIM:+--late_embed_dim "${LATE_EMBED_DIM}"}
      done
    done
  done
done

echo "All case runs finished. Output root: ${OUTPUT_ROOT}"
