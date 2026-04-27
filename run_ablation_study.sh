#!/usr/bin/env bash
# =============================================================================
# SUPRA Ablation Study Runner (standalone - no external script calls)
# Runs 4-way ablation: none, ortho, aux, full
# on all datasets with same hyperparameter grid as SUPRA-Full
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve symlinks to actual path (handles /hyperai/home/SUPRA -> /output/SUPRA)
SCRIPT_DIR="$(readlink -f "${SCRIPT_DIR}" 2>/dev/null || echo "${SCRIPT_DIR}")"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/path_config.sh"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# =============================================================================
# Configurable defaults
# =============================================================================
DATASETS=${DATASETS:-"Movies Grocery Toys Reddit-M"}
MODEL_NAME=${MODEL_NAME:-"GCN"}
N_RUNS=${N_RUNS:-3}
OUTPUT_DIR=${OUTPUT_DIR:-"logs_ablation"}
GPU_ID=${GPU_ID:-0}
DATA_NAME=""

METRICS=${METRICS:-"accuracy f1_macro"}
FEATURE_GROUPS=${FEATURE_GROUPS:-"clip_roberta default"}
AVERAGE=${AVERAGE:-macro}

mkdir -p results_csv

# =============================================================================
# Parse arguments
# =============================================================================
show_help() {
  cat << 'HELPEOF'
Usage: run_ablation_study.sh [options]

Options:
  --gpu ID             GPU device ID (default: 0)
  --datasets LIST       Space-separated list of datasets (default: Movies Grocery Toys Reddit-M)
  --metrics LIST       Space-separated list of metrics (default: accuracy f1_macro)
  --feature_groups LIST Space-separated list of feature groups (default: clip_roberta default)
  --n_runs N           Number of runs per experiment (default: 3)
  --output_dir DIR      Output log directory (default: logs_ablation)
  --help               Show this help message

Examples:
  ./run_ablation_study.sh --gpu 1
  ./run_ablation_study.sh --datasets "Movies Grocery" --metrics "accuracy f1_macro" --feature_groups "clip_roberta default"
HELPEOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_ID="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --metrics) METRICS="$2"; shift 2 ;;
    --feature_groups) FEATURE_GROUPS="$2"; shift 2 ;;
    --n_runs) N_RUNS="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --help) show_help; exit 0 ;;
    *) echo "Unknown option: $1"; show_help; exit 1 ;;
  esac
done

read -r -a METRIC_ARR <<< "${METRICS}"
read -r -a FG_ARR <<< "${FEATURE_GROUPS}"

N_EPOCHS=${N_EPOCHS:-1000}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-50}
EVAL_STEPS=${EVAL_STEPS:-1}
AVERAGE=${AVERAGE:-macro}
UNDIRECTED=${UNDIRECTED:-true}
TRAIN_RATIO=${TRAIN_RATIO:-0.6}
VAL_RATIO=${VAL_RATIO:-0.2}
INDUCTIVE=${INDUCTIVE:-false}

# =============================================================================
# Feature path mappings
# =============================================================================
declare -A TEXT_FEATURE_BY_DS_GROUP
declare -A VIS_FEATURE_BY_DS_GROUP

TEXT_FEATURE_BY_DS_GROUP["Movies|clip_roberta"]='TextFeature/Movies_roberta_base_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Movies|clip_roberta"]='ImageFeature/Movies_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Grocery|clip_roberta"]='TextFeature/Grocery_roberta_base_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Grocery|clip_roberta"]='ImageFeature/Grocery_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Toys|clip_roberta"]='TextFeature/Toys_roberta_base_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Toys|clip_roberta"]='ImageFeature/Toys_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Reddit-M|clip_roberta"]='TextFeature/RedditM_roberta_base_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-M|clip_roberta"]='ImageFeature/RedditM_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Movies|default"]='TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Movies|default"]='ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Grocery|default"]='TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Grocery|default"]='ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Toys|default"]='TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Toys|default"]='ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Reddit-M|default"]='TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-M|default"]='ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy'

# =============================================================================
# SUPRA hyperparameter grid
# =============================================================================
supra_dropout="0.3"
supra_lrs=("0.0005" "0.001")
supra_wds=("1e-4")
supra_n_hidden=("256")
SUPRA_LAYERS=${SUPRA_LAYERS:-"2 3 4"}
read -r -a supra_n_layers <<< "${SUPRA_LAYERS}"
supra_label_smoothing="0.1"
supra_early_stop_patience="50"
supra_embed_dims=("256")
supra_aux_weights=("0.0" "0.1" "0.3" "0.5" "0.7")

gat_n_heads=4
gat_attn_drop=0.0
gat_edge_drop=0.0

# =============================================================================
# Helpers
# =============================================================================
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

  echo "    [${label}] log=${log_file}"

  local start_s=${SECONDS}
  ( "$@" ) >> "${log_file}" 2>&1
  local rc=$?
  local dur_s=$((SECONDS - start_s))

  if [[ "${rc}" -ne 0 ]]; then
    echo "    [FAIL] ${label} (exit=${rc}, ${dur_s}s)" >&2
    return "${rc}"
  fi

  echo "    [OK] ${label} (${dur_s}s)"
  return 0
}

# =============================================================================
# Main loop
# =============================================================================
echo "================================================"
echo "SUPRA Ablation Study (standalone)"
echo "Datasets: ${DATASETS}"
echo "Model: ${MODEL_NAME}"
echo "Metrics: ${METRICS}"
echo "Feature Groups: ${FEATURE_GROUPS}"
echo "Output: ${OUTPUT_DIR}"
echo "GPU: ${GPU_ID}"
echo "================================================"

for metric in "${METRIC_ARR[@]}"; do
  for fg in "${FG_ARR[@]}"; do
    # Set CSV paths based on metric and feature group
    RESULT_CSV="results_csv/ablation_best_${metric}_${fg}.csv"
    RESULT_CSV_ALL="results_csv/ablation_all_${metric}_${fg}.csv"

    for ds in ${DATASETS}; do
      echo ""
      echo "=============================================="
      echo "[${fg}] ${ds} (${metric})"
      echo "=============================================="

      if ! dataset_config "${ds}" "${fg}"; then
        echo "  [SKIP] Cannot configure dataset ${ds}"
        continue
      fi

      echo "  Graph: ${GRAPH_PATH}"
      echo "  Text:  ${TEXT_FEAT}"
      echo "  Visual: ${VIS_FEAT}"

      DATA_NAME="${ds}"

      case "${MODEL_NAME}" in
        "GCN") SELFLOOP="true" ;;
        "SAGE") SELFLOOP="false" ;;
        "GAT") SELFLOOP="false" ;;
        "RevGAT") SELFLOOP="false" ;;
        *) SELFLOOP="true" ;;
      esac

      EXTRA_ARGS=""
      case "${MODEL_NAME}" in
        "GAT")
          EXTRA_ARGS="--n-heads ${gat_n_heads} --attn-drop ${gat_attn_drop} --edge-drop ${gat_edge_drop}"
          ;;
        "SAGE")
          EXTRA_ARGS="--aggregator mean"
          ;;
      esac

      for aw in "${supra_aux_weights[@]}"; do
        for lr in "${supra_lrs[@]}"; do
          for wd in "${supra_wds[@]}"; do
            for h in "${supra_n_hidden[@]}"; do
              for L in "${supra_n_layers[@]}"; do
                for ed in "${supra_embed_dims[@]}"; do
                  label="SUPRA-${MODEL_NAME}-aw${aw}-lr${lr}-wd${wd}-h${h}-L${L}"
                  run_model "${label}" \
                    python GNN/SUPRA.py \
                      --data_name "${ds}" \
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
                      --metric "${metric}" --average "${AVERAGE}" \
                      --train_ratio "${TRAIN_RATIO}" --val_ratio "${VAL_RATIO}" \
                      --undirected "${UNDIRECTED}" \
                      --selfloop "${SELFLOOP}" \
                      --inductive "${INDUCTIVE}" \
                      --model_name "${MODEL_NAME}" \
                      --embed_dim "${ed}" \
                      --aux_weight "${aw}" \
                      --result_tag "SUPRA" \
                      --result_csv "${RESULT_CSV}" \
                      --result_csv_all "${RESULT_CSV_ALL}" \
                      --disable_wandb \
                      ${EXTRA_ARGS}
                done
              done
            done
          done
        done
      done

    done
  done
done

echo ""
echo "================================================"
echo "Ablation study complete. Results in: ${OUTPUT_DIR}/"
echo "================================================"