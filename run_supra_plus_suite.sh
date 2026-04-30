#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# SUPRA+ Deep Residual Modality Encoder Experiment Suite
# Compares linear vs deep modality encoder on Movies and Toys datasets
# =============================================================================
# Usage:
#   GPU_ID=0 METRIC="accuracy" DATASETS="Movies Toys" bash run_supra_plus_suite.sh
#
# Environment variables:
#   GPU_ID          - GPU device ID (default: 0)
#   METRIC          - "accuracy" or "f1" (default: accuracy)
#   FEATURE_GROUPS - "default" or "clip_roberta" (default: default)
#   DATASETS        - Space-separated dataset list (default: Movies Toys)
# =============================================================================

GPU_ID=${GPU_ID:-0}
METRIC=${METRIC:-accuracy}
FEATURE_GROUPS=${FEATURE_GROUPS:-default}
DATASETS=${DATASETS:-"Movies Toys"}

for variant in "linear" "deep"; do
  if [[ "$variant" == "deep" ]]; then
    ENC_N_LAYERS=2
    ENC_HIDDEN_DIM=1024
    LABEL="supra_plus"
  else
    ENC_N_LAYERS=2
    ENC_HIDDEN_DIM=1024
    LABEL="supra_linear"
  fi

  echo "=== Running $LABEL (modality_encoder=$variant, n_layers=$ENC_N_LAYERS, hidden=$ENC_HIDDEN_DIM) ==="
  MODALITY_ENCODER="$variant" ENC_N_LAYERS=$ENC_N_LAYERS ENC_HIDDEN_DIM=$ENC_HIDDEN_DIM \
    METRIC="$METRIC" FEATURE_GROUPS="$FEATURE_GROUPS" DATASETS="$DATASETS" GPU_ID=$GPU_ID \
    RESULT_CSV="Results/0430/${LABEL}_gpu${GPU_ID}_${METRIC}_best.csv" \
    RESULT_CSV_ALL="Results/0430/${LABEL}_gpu${GPU_ID}_${METRIC}_all.csv" \
    LOG_DIR="logs_${LABEL}_gpu${GPU_ID}_${METRIC}" \
    bash run_supra_suite.sh
done

echo "=== All experiments complete. Results saved to Results/0430/ ==="
