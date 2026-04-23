#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Comprehensive Baseline Experiment Runner
# Runs ALL baselines across ALL datasets, ALL metrics, ALL feature groups
#
# Covers:
#   - Early fusion GNNs (GCN, SAGE, GAT, GCNII, JKNet) via Early_GNN.py
#   - Late fusion baselines: Late_GNN (MMGCN/MGAT-style via Late_GNN.py)
#   - NTSFormer (via NTSFormer.py)
#   - MIG-GT (via MIG_GT.py)
#
# Results aggregated to results_csv/baseline_best.csv and baseline_all.csv
# =============================================================================
# Usage:
#   ./run_comprehensive_baseline.sh              # Run everything (default)
#   ./run_comprehensive_baseline.sh --gpu 1       # Use GPU 1
#   ./run_comprehensive_baseline.sh --dry_run     # Show commands without executing
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/path_config.sh"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# ---------------- Configurable defaults ----------------
DATASETS=${DATASETS:-"Movies Grocery Reddit-M Toys"}
METRICS=${METRICS:-"accuracy f1_macro"}
FEATURE_GROUPS=${FEATURE_GROUPS:-"clip_roberta default"}
N_RUNS=${N_RUNS:-3}
GPU_ID=${GPU_ID:-0}

DRY_RUN=${DRY_RUN:-false}
OUTPUT_DIR=${OUTPUT_DIR:-logs_baseline}

# Result CSV paths (per metric + feature_group)
# Will be set dynamically inside the loop based on metric and feature_group
RESULT_CSV=""
RESULT_CSV_ALL=""

# Create results directory once
mkdir -p results_csv

# =============================================================================
# Feature path mappings (same as run_baseline.sh)
# =============================================================================
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

# =============================================================================
# Common training knobs
# =============================================================================
N_EPOCHS=${N_EPOCHS:-1000}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-50}
EVAL_STEPS=${EVAL_STEPS:-1}
AVERAGE=${AVERAGE:-macro}
SELFLOOP=${SELFLOOP:-true}
UNDIRECTED=${UNDIRECTED:-true}
TRAIN_RATIO=${TRAIN_RATIO:-0.6}
VAL_RATIO=${VAL_RATIO:-0.2}
INDUCTIVE=${INDUCTIVE:-false}

# =============================================================================
# Per-model hyperparameter grids
# =============================================================================

# --- Early fusion GNNs (GCN, SAGE, GAT) ---
early_gnn_dropout="0.3"
early_gnn_lrs=("0.0005" "0.001")
early_gnn_wds=("1e-4")
early_gnn_n_hidden=("256")
early_gnn_label_smoothing="0.1"
early_gnn_early_stop="50"

GCN_LAYERS=${GCN_LAYERS:-"1 2"}
SAGE_LAYERS=${SAGE_LAYERS:-"2 3 4"}
GAT_LAYERS=${GAT_LAYERS:-"1 2"}
GCNII_LAYERS=${GCNII_LAYERS:-"2 3 4"}
JKNET_LAYERS=${JKNET_LAYERS:-"2 3 4"}
read -r -a gcn_n_layers <<< "${GCN_LAYERS}"
read -r -a sage_n_layers <<< "${SAGE_LAYERS}"
read -r -a gat_n_layers <<< "${GAT_LAYERS}"
read -r -a gcnii_n_layers <<< "${GCNII_LAYERS}"
read -r -a jknet_n_layers <<< "${JKNET_LAYERS}"
sage_aggregator="mean"
gat_n_heads=4
gat_attn_drop=0.0
gat_edge_drop=0.0

# --- Late_GNN (MMGCN/MGAT) ---
late_dropout="0.3"
late_lrs=("0.0005" "0.001")
late_wds=("1e-4")
late_n_hidden=("256")
late_label_smoothing="0.1"
late_early_stop="50"
late_n_layers="2"  # Late_GNN uses fixed 2-layer GNN

# --- NTSFormer ---
nts_lr="0.001"
nts_n_hidden="256"
nts_epochs="500"

# --- MIG-GT ---
mig_lr="0.001"
mig_wd="0.0"
mig_n_hidden="256"
mig_epochs="500"

# =============================================================================
# Parse arguments
# =============================================================================
show_help() {
  cat << 'HELPEOF'
Usage: run_comprehensive_baseline.sh [options]

Options:
  --datasets LIST    Space-separated list of datasets (default: Movies Grocery Reddit-M Toys)
  --metrics LIST     Space-separated list of metrics (default: accuracy f1_macro)
  --feature_groups LIST  Space-separated list of feature groups (default: clip_roberta default)
  --n_runs N         Number of runs per experiment (default: 3)
  --gpu ID           GPU device ID (default: 0)
  --output_dir DIR   Output log directory (default: logs_baseline)
  --dry_run          Show commands without executing
  --help             Show this help message

Examples:
  # Run all baselines on all datasets with all metrics and feature groups
  ./run_comprehensive_baseline.sh

  # Use GPU 1 and run only accuracy metric
  ./run_comprehensive_baseline.sh --gpu 1 --metrics "accuracy"

  # Dry run to see what would be executed
  ./run_comprehensive_baseline.sh --dry_run
HELPEOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --metrics) METRICS="$2"; shift 2 ;;
    --feature_groups) FEATURE_GROUPS="$2"; shift 2 ;;
    --n_runs) N_RUNS="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --dry_run) DRY_RUN="true"; shift ;;
    --help) show_help; exit 0 ;;
    *) echo "Unknown option: $1"; show_help; exit 1 ;;
  esac
done

read -r -a DATASET_ARR <<< "${DATASETS}"
read -r -a METRIC_ARR <<< "${METRICS}"
read -r -a FG_ARR <<< "${FEATURE_GROUPS}"

# =============================================================================
# Utility functions
# =============================================================================
_ts() { date '+%F %T'; }

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
# Header
# =============================================================================
echo "=============================================="
echo "  Comprehensive Baseline Runner"
echo "=============================================="
echo "Datasets:     ${DATASETS}"
echo "Metrics:      ${METRICS}"
echo "Features:     ${FEATURE_GROUPS}"
echo "Runs per exp: ${N_RUNS}"
echo "GPU:          ${GPU_ID}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "=============================================="

# =============================================================================
# Main experiment loops
# =============================================================================
total_exp=0
exp_count=0
failed=()

for metric in "${METRIC_ARR[@]}"; do
  for fg in "${FG_ARR[@]}"; do
    # Set CSV paths based on metric and feature group
    RESULT_CSV="results_csv/baseline_best_${metric}_${fg}.csv"
    RESULT_CSV_ALL="results_csv/baseline_all_${metric}_${fg}.csv"
    for ds in "${DATASET_ARR[@]}"; do
      DATA_NAME="${ds}"

      echo ""
      echo "=============================================="
      echo "[${fg}] ${ds} (${metric})"
      echo "=============================================="

      # Validate paths
      if ! dataset_config "${ds}" "${fg}"; then
        echo "  [SKIP] Cannot configure dataset ${ds} with feature group ${fg}"
        continue
      fi

      echo "  Graph: ${GRAPH_PATH}"
      echo "  Text:  ${TEXT_FEAT}"
      echo "  Visual: ${VIS_FEAT}"

      # ============================================
      # 1. Early_GNN: GCN, SAGE, GAT, GCNII, JKNet
      # ============================================
      for model_name in GCN SAGE GAT GCNII JKNet; do
        exp_count=$((exp_count + 1))
        echo ""
        echo "  [${exp_count}] Early_GNN/${model_name}"

        case "${model_name}" in
          GCN)   layers_array=("${gcn_n_layers[@]}");   SELFLOOP="true";  EXTRA_ARGS="--backend gnn --model_name GCN --early_fuse concat" ;;
          SAGE)  layers_array=("${sage_n_layers[@]}");  SELFLOOP="false"; EXTRA_ARGS="--backend gnn --model_name SAGE --early_fuse concat --aggregator ${sage_aggregator}" ;;
          GAT)   layers_array=("${gat_n_layers[@]}");   SELFLOOP="false"; EXTRA_ARGS="--backend gnn --model_name GAT --early_fuse concat --n-heads ${gat_n_heads} --attn-drop ${gat_attn_drop} --edge-drop ${gat_edge_drop}" ;;
          GCNII) layers_array=("${gcnii_n_layers[@]}"); SELFLOOP="true";  EXTRA_ARGS="--backend gnn --model_name GCNII --early_fuse concat" ;;
          JKNet) layers_array=("${jknet_n_layers[@]}"); SELFLOOP="true";  EXTRA_ARGS="--backend gnn --model_name JKNet --early_fuse concat" ;;
        esac

        for L in "${layers_array[@]}"; do
          for lr in "${early_gnn_lrs[@]}"; do
            label="Early-${model_name}-L${L}-lr${lr}-do${early_gnn_dropout}-${fg}-${metric}"
            cmd="python GNN/Baselines/Early_GNN.py \
              --data_name \"${ds}\" \
              --graph_path \"${GRAPH_PATH}\" \
              --text_feature \"${TEXT_FEAT}\" \
              --visual_feature \"${VIS_FEAT}\" \
              --gpu \"${GPU_ID}\" \
              --n-runs \"${N_RUNS}\" \
              --n-epochs \"${N_EPOCHS}\" \
              --warmup_epochs \"${WARMUP_EPOCHS}\" \
              --eval_steps \"${EVAL_STEPS}\" \
              --early_stop_patience \"${early_gnn_early_stop}\" \
              --lr \"${lr}\" --wd \"${early_gnn_wds}\" \
              --n-layers \"${L}\" --n-hidden \"${early_gnn_n_hidden}\" --dropout \"${early_gnn_dropout}\" \
              --label-smoothing \"${early_gnn_label_smoothing}\" \
              --metric \"${metric}\" --average \"${AVERAGE}\" \
              --train_ratio \"${TRAIN_RATIO}\" --val_ratio \"${VAL_RATIO}\" \
              --undirected \"${UNDIRECTED}\" --selfloop \"${SELFLOOP}\" \
              --inductive \"${INDUCTIVE}\" \
              --disable_wandb \
              --result_csv \"${RESULT_CSV}\" \
              --result_csv_all \"${RESULT_CSV_ALL}\" \
              ${EXTRA_ARGS}"

            if [[ "${DRY_RUN}" == "true" ]]; then
              echo "    CMD: ${cmd}"
            else
              if run_model "${label}" bash -c "${cmd}"; then
                : # ok
              else
                failed+=("Early-${model_name}@${ds}@${fg}@${metric}")
              fi
            fi
          done
        done
      done

      # ============================================
      # 2. Late_GNN (MMGCN/MGAT-style late fusion)
      # ============================================
      for lr in "${late_lrs[@]}"; do
        exp_count=$((exp_count + 1))
        label="Late_GNN-lr${lr}-do${late_dropout}-${fg}-${metric}"
        echo ""
        echo "  [${exp_count}] ${label}"

        if [[ "${DRY_RUN}" == "true" ]]; then
          echo "    CMD: python GNN/Baselines/Late_GNN.py ..."
        else
          if run_model "${label}" python GNN/Baselines/Late_GNN.py \
            --data_name "${ds}" \
            --graph_path "${GRAPH_PATH}" \
            --text_feature "${TEXT_FEAT}" \
            --visual_feature "${VIS_FEAT}" \
            --gpu "${GPU_ID}" \
            --n-runs "${N_RUNS}" \
            --n-epochs "${N_EPOCHS}" \
            --warmup_epochs "${WARMUP_EPOCHS}" \
            --eval_steps "${EVAL_STEPS}" \
            --early_stop_patience "${late_early_stop}" \
            --lr "${lr}" --wd "${late_wds}" \
            --n-hidden "${late_n_hidden}" --dropout "${late_dropout}" \
            --label-smoothing "${late_label_smoothing}" \
            --metric "${metric}" --average "${AVERAGE}" \
            --train_ratio "${TRAIN_RATIO}" --val_ratio "${VAL_RATIO}" \
            --undirected "${UNDIRECTED}" --selfloop "${SELFLOOP}" \
            --inductive "${INDUCTIVE}" \
            --model_name GCN \
            --disable_wandb \
            --result_csv "${RESULT_CSV}" \
            --result_csv_all "${RESULT_CSV_ALL}"; then
            :
          else
            failed+=("Late_GNN@${ds}@${fg}@${metric}")
          fi
        fi
      done

      # ============================================
      # 3. NTSFormer
      # ============================================
      exp_count=$((exp_count + 1))
      label="NTSFormer-lr${nts_lr}-${fg}-${metric}"
      echo ""
      echo "  [${exp_count}] ${label}"

      if [[ "${DRY_RUN}" == "true" ]]; then
        echo "    CMD: python GNN/Baselines/NTSFormer.py ..."
      else
        if run_model "${label}" python GNN/Baselines/NTSFormer.py \
          --data_name "${ds}" \
          --graph_path "${GRAPH_PATH}" \
          --text_feature "${TEXT_FEAT}" \
          --visual_feature "${VIS_FEAT}" \
          --gpu "${GPU_ID}" \
          --n-runs "${N_RUNS}" \
          --n-epochs "${nts_epochs}" \
          --warmup_epochs "${WARMUP_EPOCHS}" \
          --eval_steps "${EVAL_STEPS}" \
          --early_stop_patience "50" \
          --lr "${nts_lr}" --wd "1e-4" \
          --n-hidden "${nts_n_hidden}" --dropout "0.3" \
          --label-smoothing "0.1" \
          --metric "${metric}" --average "${AVERAGE}" \
          --train_ratio "${TRAIN_RATIO}" --val_ratio "${VAL_RATIO}" \
          --undirected "${UNDIRECTED}" --selfloop "${SELFLOOP}" \
          --inductive "${INDUCTIVE}" \
          --disable_wandb \
          --result_csv_all "${RESULT_CSV_ALL}"; then
          :
        else
          failed+=("NTSFormer@${ds}@${fg}@${metric}")
        fi
      fi

      # ============================================
      # 4. MIG-GT
      # ============================================
      exp_count=$((exp_count + 1))
      label="MIGGT-lr${mig_lr}-${fg}-${metric}"
      echo ""
      echo "  [${exp_count}] ${label}"

      if [[ "${DRY_RUN}" == "true" ]]; then
        echo "    CMD: python GNN/Baselines/MIG_GT.py ..."
      else
        if run_model "${label}" python GNN/Baselines/MIG_GT.py \
          --data_name "${ds}" \
          --graph_path "${GRAPH_PATH}" \
          --text_feature "${TEXT_FEAT}" \
          --visual_feature "${VIS_FEAT}" \
          --gpu "${GPU_ID}" \
          --n-runs "${N_RUNS}" \
          --n-epochs "${mig_epochs}" \
          --eval_steps 1 \
          --lr "${mig_lr}" --wd "${mig_wd}" \
          --n-hidden "${mig_n_hidden}" --dropout "0.2" \
          --label-smoothing "0.1" \
          --metric "${metric}" --average "${AVERAGE}" \
          --train_ratio "${TRAIN_RATIO}" --val_ratio "${VAL_RATIO}" \
          --undirected "${UNDIRECTED}" --selfloop "${SELFLOOP}" \
          --inductive "${INDUCTIVE}" \
          --k_t 3 --k_v 2 \
          --mgdcf_alpha 0.1 --mgdcf_beta 0.9 \
          --num_samples 10 \
          --tur_weight 1.0 \
          --tur_sample_edges 8000 \
          --result_csv "${RESULT_CSV}" \
          --result_csv_all "${RESULT_CSV_ALL}" \
          --disable_wandb; then
          :
        else
          failed+=("MIGGT@${ds}@${fg}@${metric}")
        fi
      fi

    done
  done
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "  Experiment Summary"
echo "=============================================="
echo "Total experiments: ${exp_count}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Results CSV: ${RESULT_CSV}"
echo ""
if [[ ${#failed[@]} -eq 0 ]]; then
  echo "All ${exp_count} experiments completed successfully!"
else
  echo "${#failed[@]} experiments failed:"
  for f in "${failed[@]}"; do
    echo "  - ${f}"
  done
fi
echo "=============================================="