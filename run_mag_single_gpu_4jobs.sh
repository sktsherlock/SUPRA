#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU runner for:
# - 4 datasets: Movies Grocery Toys Reddit-M
# - 2 feature groups: clip_roberta, default
# - 2 metrics: accuracy, f1
# Total: 4 jobs (feature_group x metric), each job writes one best CSV.
# Default behavior: run 4 jobs in parallel on 4 GPUs.

GPU_IDS=${GPU_IDS:-"0 1 2 3"}
WANDB_RUN_MODE=${WANDB_RUN_MODE:-disabled}
export WANDB_DISABLED=${WANDB_DISABLED:-true}

DATASETS=${DATASETS:-"Movies Grocery Toys Reddit-M"}
EXPERIMENTS=${EXPERIMENTS:-"plain baseline late nts mig"}
GNN_MODELS=${GNN_MODELS:-"GCN SAGE GAT GCNII JKNet"}
AVERAGE=${AVERAGE:-macro}

RESULT_DIR=${RESULT_DIR:-"Results"}
mkdir -p "${RESULT_DIR}"

echo "[INFO] Start multi-GPU MAG baseline suite"
echo "[INFO] GPU_IDS=${GPU_IDS}, DATASETS=${DATASETS}, EXPERIMENTS=${EXPERIMENTS}"

read -r -a GPU_ARR <<< "${GPU_IDS}"
if [[ ${#GPU_ARR[@]} -lt 4 ]]; then
  echo "[Error] Need at least 4 GPU ids in GPU_IDS, got: '${GPU_IDS}'" >&2
  echo "        Example: GPU_IDS=\"0 1 2 3\" bash run_mag_single_gpu_4jobs.sh" >&2
  exit 1
fi

run_one() {
  local gpu_id="$1"
  local feature_group="$2"
  local metric="$3"

  local tag="${feature_group}_${metric}"
  local result_csv="${RESULT_DIR}/gpu${gpu_id}_${tag}_best.csv"
  local result_csv_all="${RESULT_DIR}/gpu${gpu_id}_${tag}_all.csv"
  local log_dir="logs_gpu${gpu_id}_${tag}"

  echo "[INFO] Running gpu=${gpu_id}, feature_group=${feature_group}, metric=${metric}"
  echo "[INFO] -> best: ${result_csv}"

  GPU_ID="${gpu_id}" \
  FEATURE_GROUPS="${feature_group}" \
  DATASETS="${DATASETS}" \
  EXPERIMENTS="${EXPERIMENTS}" \
  GNN_MODELS="${GNN_MODELS}" \
  METRIC="${metric}" \
  AVERAGE="${AVERAGE}" \
  RESULT_CSV="${result_csv}" \
  RESULT_CSV_ALL="${result_csv_all}" \
  WANDB_RUN_MODE="${WANDB_RUN_MODE}" \
  LOG_DIR="${log_dir}" \
  bash run_mag_baseline_suite.sh
}

# 2 feature groups x 2 metrics = 4 jobs (parallel on 4 GPUs)
declare -a PIDS=()
declare -a LABELS=()

run_one "${GPU_ARR[0]}" "clip_roberta" "accuracy" &
PIDS+=("$!")
LABELS+=("gpu=${GPU_ARR[0]} clip_roberta accuracy")

run_one "${GPU_ARR[1]}" "default" "accuracy" &
PIDS+=("$!")
LABELS+=("gpu=${GPU_ARR[1]} default accuracy")

run_one "${GPU_ARR[2]}" "clip_roberta" "f1" &
PIDS+=("$!")
LABELS+=("gpu=${GPU_ARR[2]} clip_roberta f1")

run_one "${GPU_ARR[3]}" "default" "f1" &
PIDS+=("$!")
LABELS+=("gpu=${GPU_ARR[3]} default f1")

FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[OK] ${LABELS[$i]}"
  else
    echo "[FAIL] ${LABELS[$i]}" >&2
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "[Error] At least one parallel job failed." >&2
  exit 1
fi

echo "[INFO] All jobs finished. Best CSV files:"
echo "  ${RESULT_DIR}/gpu${GPU_ARR[0]}_clip_roberta_accuracy_best.csv"
echo "  ${RESULT_DIR}/gpu${GPU_ARR[1]}_default_accuracy_best.csv"
echo "  ${RESULT_DIR}/gpu${GPU_ARR[2]}_clip_roberta_f1_best.csv"
echo "  ${RESULT_DIR}/gpu${GPU_ARR[3]}_default_f1_best.csv"
