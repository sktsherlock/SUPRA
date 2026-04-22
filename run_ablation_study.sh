#!/usr/bin/env bash
# =============================================================================
# SUPRA Ablation Study Runner
# Runs 4-way ablation: none, ortho, aux, full
# on all datasets with same hyperparameter grid as SUPRA-Full
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/path_config.sh"

# Default: run GCN backbone on all 4 datasets
DATASETS=${DATASETS:-"Movies Grocery Toys Reddit-M"}
MODEL_NAME=${MODEL_NAME:-"GCN"}
N_RUNS=${N_RUNS:-3}
OUTPUT_DIR=${OUTPUT_DIR:-"logs_ablation"}

declare -A MODE_CONFIG
MODE_CONFIG["none"]="0.0 false"
MODE_CONFIG["ortho"]="1.0 false"
MODE_CONFIG["aux"]="0.0 true"
MODE_CONFIG["full"]="1.0 true"

echo "================================================"
echo "SUPRA Ablation Study"
echo "Datasets: ${DATASETS}"
echo "Model: ${MODEL_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "================================================"

for ds in ${DATASETS}; do
  for mode in none ortho aux full; do
    read -r ortho_alpha aux_loss <<< "${MODE_CONFIG[${mode}]}"

    case "${mode}" in
      none)   label_tag="Ablate-None" ;;
      ortho)  label_tag="Ablate-Ortho" ;;
      aux)   label_tag="Ablate-Aux" ;;
      full)   label_tag="SUPRA-Full" ;;
    esac

    echo ""
    echo ">>> [${label_tag}] Dataset=${ds}, OrthoAlpha=${ortho_alpha}, AuxLoss=${aux_loss}"

    ORTHO_ALPHA="${ortho_alpha}" \
    USE_AUX_LOSS="${aux_loss}" \
    ./run_supra.sh \
      --data_name "${ds}" \
      --model_name "${MODEL_NAME}" \
      --n_runs "${N_RUNS}" \
      --output_dir "${OUTPUT_DIR}"
  done
done

echo ""
echo "================================================"
echo "Ablation study complete. Results in: ${OUTPUT_DIR}/"
echo "================================================"