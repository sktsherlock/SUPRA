#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Batch Baseline Experiment Runner
# Runs all baseline models on all datasets for all metrics
# =============================================================================
# Usage:
#   ./run_batch_baseline.sh              # Run everything (default)
#   ./run_batch_baseline.sh --models GCN SAGE    # Run specific models
#   ./run_batch_baseline.sh --datasets Movies     # Run specific datasets
#   ./run_batch_baseline.sh --metrics accuracy    # Run specific metric only
#   ./run_batch_baseline.sh --dry_run             # Show commands without executing
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/plot/path_config.sh"

# ---------------- Configurable defaults ----------------
MODELS=${MODELS:-"MLP GCN SAGE GAT GCNII JKNet"}
DATASETS=${DATASETS:-"Movies Grocery Reddit-M Toys"}
METRICS=${METRICS:-"accuracy f1_macro"}
N_RUNS=${N_RUNS:-3}
FEATURE_GROUP=${FEATURE_GROUP:-"clip_roberta"}
GPU_ID=${GPU_ID:-0}

DRY_RUN=${DRY_RUN:-false}

# =============================================================================
# Parse arguments
# =============================================================================
show_help() {
  cat << 'HELPEOF'
Usage: run_batch_baseline.sh [options]

Options:
  --models LIST      Space-separated list of models (default: MLP GCN SAGE GAT GCNII JKNet)
  --datasets LIST    Space-separated list of datasets (default: Movies Grocery Reddit-M Toys)
  --metrics LIST     Space-separated list of metrics (default: accuracy f1_macro)
  --n_runs N         Number of runs per experiment (default: 3)
  --feature_group    Feature group: clip_roberta, default (default: clip_roberta)
  --gpu ID           GPU device ID (default: 0)
  --dry_run          Show commands without executing
  --help             Show this help message

Examples:
  # Run all baselines on all datasets with all metrics
  ./run_batch_baseline.sh

  # Run only GCN and SAGE on Movies and Grocery
  ./run_batch_baseline.sh --models "GCN SAGE" --datasets "Movies Grocery"

  # Run only F1-macro metric
  ./run_batch_baseline.sh --metrics "f1_macro"

  # Dry run to see what would be executed
  ./run_batch_baseline.sh --dry_run
HELPEOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --metrics) METRICS="$2"; shift 2 ;;
    --n_runs) N_RUNS="$2"; shift 2 ;;
    --feature_group) FEATURE_GROUP="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    --dry_run) DRY_RUN="true"; shift ;;
    --help) show_help; exit 0 ;;
    *) echo "Unknown option: $1"; show_help; exit 1 ;;
  esac
done

read -r -a MODEL_ARR <<< "${MODELS}"
read -r -a DATASET_ARR <<< "${DATASETS}"
read -r -a METRIC_ARR <<< "${METRICS}"

# =============================================================================
# Calculate total experiments
# =============================================================================
MODEL_COUNT=${#MODEL_ARR[@]}
DATASET_COUNT=${#DATASET_ARR[@]}
METRIC_COUNT=${#METRIC_ARR[@]}

echo "=============================================="
echo "  Batch Baseline Experiment Runner"
echo "=============================================="
echo "Models:      ${MODELS}"
echo "Datasets:    ${DATASETS}"
echo "Metrics:     ${METRICS}"
echo "Feature:     ${FEATURE_GROUP}"
echo "Runs per exp: ${N_RUNS}"
echo "GPU:         ${GPU_ID}"
echo "=============================================="

# Count total experiments
total_models=${#MODEL_ARR[@]}
total_datasets=${#DATASET_ARR[@]}
total_metrics=${#METRIC_ARR[@]}

# Show estimated time (rough estimate: ~5min per experiment)
est_time_min=$((total_models * total_datasets * total_metrics * 5))
echo "Total experiments: $((total_models * total_datasets * total_metrics))"
echo "Estimated time: ~${est_time_min} minutes"
echo "=============================================="

# =============================================================================
# Run experiments
# =============================================================================
exp_count=0
failed=()

for metric in "${METRIC_ARR[@]}"; do
  for ds in "${DATASET_ARR[@]}"; do
    for model in "${MODEL_ARR[@]}"; do
      exp_count=$((exp_count + 1))
      echo ""
      echo "[${exp_count}/$((total_models * total_datasets * total_metrics))] ${model} on ${ds} (${metric})"

      cmd="bash run_baseline.sh \
        --model ${model} \
        --data_name ${ds} \
        --metric ${metric} \
        --average macro \
        --feature_group ${FEATURE_GROUP} \
        --gpu ${GPU_ID} \
        --n_runs ${N_RUNS}"

      if [[ "${DRY_RUN}" == "true" ]]; then
        echo "  CMD: ${cmd}"
      else
        if ${cmd} 2>&1 | tee "logs_baseline/${ds}/${model}-${metric}.log"; then
          echo "  [OK] ${model} on ${ds} (${metric})"
        else
          echo "  [FAIL] ${model} on ${ds} (${metric})"
          failed+=("${model}@${ds}@${metric}")
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
if [[ ${#failed[@]} -eq 0 ]]; then
  echo "All ${exp_count} experiments completed successfully!"
else
  echo "${#failed[@]} experiments failed:"
  for f in "${failed[@]}"; do
    echo "  - ${f}"
  done
fi
echo "=============================================="
echo "Logs: logs_baseline/"
echo "Results: results_csv/baseline_best.csv"
echo "=============================================="