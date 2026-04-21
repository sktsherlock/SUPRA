#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# SUPRA Experiment Script
# Our method: Unified Multimodal Learning with Spectral Orthogonalization
# =============================================================================

# Load centralized path config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/plot/path_config.sh"

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

# Default group (Llama3 vision features)
TEXT_FEATURE_BY_DS_GROUP["Movies|default"]='TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Movies|default"]='ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Grocery|default"]='TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Grocery|default"]='ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Toys|default"]='TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Toys|default"]='ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Reddit-M|default"]='TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-M|default"]='ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy'

GPU_ID=${GPU_ID:-0}
N_RUNS=${N_RUNS:-3}
N_EPOCHS=${N_EPOCHS:-1000}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-50}
EVAL_STEPS=${EVAL_STEPS:-1}
METRIC=${METRIC:-accuracy}
AVERAGE=${AVERAGE:-macro}
UNDIRECTED=${UNDIRECTED:-true}
TRAIN_RATIO=${TRAIN_RATIO:-0.6}
VAL_RATIO=${VAL_RATIO:-0.2}
INDUCTIVE=${INDUCTIVE:-false}

supra_dropout="0.3"
supra_lrs=("0.0005" "0.001")
supra_wds=("1e-4")
supra_n_hidden=("256")
SUPRA_LAYERS=${SUPRA_LAYERS:-"2 3 4"}
read -r -a supra_n_layers <<< "${SUPRA_LAYERS}"
supra_label_smoothing="0.1"
supra_early_stop_patience="50"

supra_embed_dims=("128" "256")
supra_shared_depths=("1" "2" "3" "4")

gat_n_heads=4
gat_attn_drop=0.0
gat_edge_drop=0.0

show_help() {
  cat << 'HELPEOF'
Usage: run_supra.sh --data_name DATASET [options]

Required:
  --data_name DATASET  Movies, Grocery, Reddit-M, Toys

Optional:
  --gpu ID             GPU device ID (default: 0)
  --n_runs N           Number of runs (default: 3)
  --feature_group NAME  clip_roberta, default (default: clip_roberta)
  --model_name NAME    GNN backbone: GCN, SAGE, GAT, RevGAT (default: GCN)
  --embed_dim N        SUPRA embedding dimension (default: 256)
  --shared_depth N     Shared channel depth (default: 2)
  --output_dir DIR     Output directory (default: logs_supra)

Examples:
  ./run_supra.sh --data_name Movies
  ./run_supra.sh --data_name Grocery --model_name GAT --embed_dim 128
  ./run_supra.sh --data_name Reddit-M --n_runs 5
HELPEOF
}

DATA_NAME=""
FEATURE_GROUP="clip_roberta"
MODEL_NAME="GCN"
EMBED_DIM=""
SHARED_DEPTH=""
OUTPUT_DIR="logs_supra"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_name) DATA_NAME="$2"; shift 2 ;;
    --feature_group) FEATURE_GROUP="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --embed_dim) EMBED_DIM="$2"; shift 2 ;;
    --shared_depth) SHARED_DEPTH="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    --n_runs) N_RUNS="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --help) show_help; exit 0 ;;
    *) echo "Unknown option: $1"; show_help; exit 1 ;;
  esac
done

if [[ -z "${DATA_NAME}" ]]; then
  echo "Error: --data_name is required"
  show_help
  exit 1
fi

EMBED_DIM="${EMBED_DIM:-256}"
SHARED_DEPTH="${SHARED_DEPTH:-2}"

case "${MODEL_NAME}" in
  "GCN") SELFLOOP="true" ;;
  "SAGE") SELFLOOP="false" ;;
  "GAT") SELFLOOP="false" ;;
  "RevGAT") SELFLOOP="false" ;;
  *) SELFLOOP="true" ;;
esac

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
  local fg="$2"
  local ds_dir="${DATA_ROOT}/${ds}"
  local ds_prefix="${ds//-/}"

  local key="${ds}|${fg}"
  local text_rel="${TEXT_FEATURE_BY_DS_GROUP[${key}]:-}"
  local vis_rel="${VIS_FEATURE_BY_DS_GROUP[${key}]:-}"

  if [[ -z "${text_rel}" || -z "${vis_rel}" ]]; then
    echo "[Error] Missing feature mapping for dataset='${ds}', feature_group='${fg}'." >&2
    exit 1
  fi

  GRAPH_PATH=$(require_file "${ds_dir}/${ds_prefix}Graph.pt" "graph")
  TEXT_FEAT=$(require_file "${ds_dir}/${text_rel}" "text feature")
  VIS_FEAT=$(require_file "${ds_dir}/${vis_rel}" "visual feature")
}

run_model() {
  local label="$1"
  shift

  local log_file="${OUTPUT_DIR}/${DATA_NAME}/${label}.log"
  mkdir -p "$(dirname "${log_file}")"

  echo "[${label}] log=${log_file}"

  local start_s=${SECONDS}
  ( "$@" ) >> "${log_file}" 2>&1
  local rc=$?
  local dur_s=$((SECONDS - start_s))

  if [[ "${rc}" -ne 0 ]]; then
    echo "[FAIL] ${label} (exit=${rc}, ${dur_s}s)" >&2
    return "${rc}"
  fi

  echo "[OK] ${label} (${dur_s}s)"
  return 0
}

dataset_config "${DATA_NAME}" "${FEATURE_GROUP}"

echo ">>> Dataset: ${DATA_NAME}, Feature Group: ${FEATURE_GROUP}"
echo ">>> Model: SUPRA (${MODEL_NAME} backbone)"
echo ">>> Self-loop: ${SELFLOOP}, Undirected: ${UNDIRECTED}"

EXTRA_ARGS=""
case "${MODEL_NAME}" in
  "GAT")
    EXTRA_ARGS="--n-heads ${gat_n_heads} --attn-drop ${gat_attn_drop} --edge-drop ${gat_edge_drop}"
    ;;
  "SAGE")
    EXTRA_ARGS="--aggregator mean"
    ;;
esac

for lr in "${supra_lrs[@]}"; do
  for wd in "${supra_wds[@]}"; do
    for h in "${supra_n_hidden[@]}"; do
      for L in "${supra_n_layers[@]}"; do
        for ed in "${supra_embed_dims[@]}"; do
          for sd in "${supra_shared_depths[@]}"; do
            label="SUPRA-${MODEL_NAME}-lr${lr}-wd${wd}-h${h}-L${L}-do${supra_dropout}-ed${ed}-sd${sd}"
            run_model "${label}" \
              python GNN/SUPRA.py \
                --data_name "${DATA_NAME}" \
                --graph_path "${GRAPH_PATH}" \
                --text_feature "${TEXT_FEAT}" \
                --visual_feature "${VIS_FEAT}" \
                --gpu "${GPU_ID}" \
                --n-runs "${N_RUNS}" \
                --n-epochs "${N_EPOCHS}" \
                --warmup_epochs "${WARMUP_EPOCHS}" \
                --eval_steps "${EVAL_STEPS}" \
                --early_stop_patience "${supra_early_stop_patience}" \
                --lr "${lr}" --wd "${wd}" \
                --n-layers "${L}" --n-hidden "${h}" --dropout "${supra_dropout}" \
                --label-smoothing "${supra_label_smoothing}" \
                --metric "${METRIC}" --average "${AVERAGE}" \
                --train_ratio "${TRAIN_RATIO}" --val_ratio "${VAL_RATIO}" \
                --undirected \
                --selfloop "${SELFLOOP}" \
                --inductive "${INDUCTIVE}" \
                --model_name "${MODEL_NAME}" \
                --embed_dim "${ed}" \
                --shared_depth "${sd}" \
                --disable_wandb \
                ${EXTRA_ARGS}
          done
        done
      done
    done
  done
done

echo ">>> All SUPRA runs completed. Logs: ${OUTPUT_DIR}/${DATA_NAME}/"