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

N_EPOCHS=${N_EPOCHS:-1000}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-50}
EVAL_STEPS=${EVAL_STEPS:-1}
METRIC=${METRIC:-accuracy}
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
# 4-way ablation config: ortho_alpha + use_aux_loss
# =============================================================================
declare -A MODE_CONFIG
MODE_CONFIG["none"]="0.0 false"
MODE_CONFIG["ortho"]="1.0 false"
MODE_CONFIG["aux"]="0.0 true"
MODE_CONFIG["full"]="1.0 true"

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
supra_embed_dims=("128" "256")
supra_shared_depths=("1" "2" "3" "4")

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
echo "Output: ${OUTPUT_DIR}"
echo "GPU: ${GPU_ID}"
echo "================================================"

FEATURE_GROUP="clip_roberta"

for ds in ${DATASETS}; do
  echo ""
  echo "=============================================="
  echo "[${ds}]"
  echo "=============================================="

  if ! dataset_config "${ds}" "${FEATURE_GROUP}"; then
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

  for mode in none ortho aux full; do
    read -r ortho_alpha aux_loss <<< "${MODE_CONFIG[${mode}]}"

    case "${mode}" in
      none)  label_prefix="Ablate-None" ;;
      ortho) label_prefix="Ablate-Ortho" ;;
      aux)   label_prefix="Ablate-Aux" ;;
      full)  label_prefix="SUPRA-Full" ;;
    esac

    echo ""
    echo "  >>> [${label_prefix}] OrthoAlpha=${ortho_alpha}, AuxLoss=${aux_loss}"

    AUX_ARGS=""
    if [[ "${aux_loss}" == "true" ]]; then
      AUX_ARGS="--use_aux_loss"
    fi

    for lr in "${supra_lrs[@]}"; do
      for wd in "${supra_wds[@]}"; do
        for h in "${supra_n_hidden[@]}"; do
          for L in "${supra_n_layers[@]}"; do
            for ed in "${supra_embed_dims[@]}"; do
              for sd in "${supra_shared_depths[@]}"; do
                label="${label_prefix}-${MODEL_NAME}-lr${lr}-wd${wd}-h${h}-L${L}-do${supra_dropout}-ed${ed}-sd${sd}"
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
                    --metric "${METRIC}" --average "${AVERAGE}" \
                    --train_ratio "${TRAIN_RATIO}" --val_ratio "${VAL_RATIO}" \
                    --undirected \
                    --selfloop "${SELFLOOP}" \
                    --inductive "${INDUCTIVE}" \
                    --model_name "${MODEL_NAME}" \
                    --embed_dim "${ed}" \
                    --shared_depth "${sd}" \
                    --ortho_alpha "${ortho_alpha}" \
                    ${AUX_ARGS} \
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

echo ""
echo "================================================"
echo "Ablation study complete. Results in: ${OUTPUT_DIR}/"
echo "================================================"