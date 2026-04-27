#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# SUPRA Comprehensive Suite
# Runs SUPRA method on multimodal node classification with GCN/GAT backbones
# =============================================================================
# Usage:
#   GPU_ID=0 FEATURE_GROUPS="clip_roberta" DATASETS="Movies Grocery Toys Reddit-M" \
#   METRIC="accuracy" AVERAGE="macro" \
#   RESULT_CSV="Results/supra_gpu0_clip_roberta_accuracy_best.csv" \
#   RESULT_CSV_ALL="Results/supra_gpu0_clip_roberta_accuracy_all.csv" \
#   LOG_DIR="logs_supra_gpu0_clip_roberta_accuracy" bash run_supra_suite.sh
#
# Environment variables:
#   GPU_ID          - GPU device ID (default: 0)
#   DATA_ROOT       - Data root path (default: /mnt/input/MAGB_Dataset)
#   FEATURE_GROUPS  - "clip_roberta" or "default" or both (default: clip_roberta)
#   DATASETS        - Space-separated dataset list (default: Movies Grocery Reddit-M Toys)
#   METRIC          - "accuracy" or "f1" (default: accuracy)
#   AVERAGE         - "macro" or "micro" (default: macro)
#   N_RUNS          - Number of runs (default: 3)
# =============================================================================

export DGLBACKEND=${DGLBACKEND:-pytorch}

# ---------------- W&B logging mode ----------------
WANDB_RUN_MODE=${WANDB_RUN_MODE:-offline}

# ---------------- User-editable: paths ----------------
DATA_ROOT=${DATA_ROOT:-/mnt/input/MAGB_Dataset}

# ---------------- Feature group support ----------------
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

# ---------------- Datasets ----------------
DEFAULT_DATASETS="Movies Grocery Reddit-M Toys"
DATASETS=${DATASETS:-"${DEFAULT_DATASETS}"}
read -r -a DATASETS_ARR <<< "${DATASETS}"
read -r -a FEATURE_GROUPS_ARR <<< "${FEATURE_GROUPS}"

# ---------------- Model backbones ----------------
SUPRA_MODELS=${SUPRA_MODELS:-"GCN GAT"}  # Only GCN and GAT for now

# ---------------- Common training knobs ----------------
GPU_ID=${GPU_ID:-0}
N_RUNS=${N_RUNS:-3}
N_EPOCHS=${N_EPOCHS:-1000}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-50}
EVAL_STEPS=${EVAL_STEPS:-1}
METRIC=${METRIC:-accuracy}
AVERAGE=${AVERAGE:-macro}
RESULT_CSV=${RESULT_CSV:-}
RESULT_CSV_ALL=${RESULT_CSV_ALL:-}

SELFLOOP=${SELFLOOP:-true}
UNDIRECTED=${UNDIRECTED:-true}
TRAIN_RATIO=${TRAIN_RATIO:-0.6}
VAL_RATIO=${VAL_RATIO:-0.2}
INDUCTIVE=${INDUCTIVE:-false}

# ---------------- SUPRA hyperparameter grid ----------------
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
supra_mlp_variants=("full" "ablate")  # full=MLP投影, ablate=无投影(对比基线)

# GAT parameters
gat_n_heads=4
gat_attn_drop=0.0
gat_edge_drop=0.0

# ---------------- Logging setup ----------------
LOG_DIR=${LOG_DIR:-logs_supra}
LOG_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/${LOG_DIR}
mkdir -p "${LOG_ROOT}"

# =============================================================================
# Helper: count job
# =============================================================================
count_job() {
  local log_file="$1"
  local label="$(basename "${log_file}" .log)"
  local dir="$(dirname "${log_file}")"

  # Check if done
  if [[ -f "${log_file}.done" ]]; then
    echo "  [SKIP] ${label} (done)"
    return 0
  fi

  echo "  + ${label}"
  return 0
}

# =============================================================================
# Main sweep loop
# =============================================================================

# Pre-count total jobs for progress display
total_jobs=0
for fg in "${FEATURE_GROUPS_ARR[@]}"; do
  for ds in "${DATASETS_ARR[@]}"; do
    for model_name in ${SUPRA_MODELS}; do
      for dropout in "${supra_dropout}"; do
        for lr in "${supra_lrs[@]}"; do
          for wd in "${supra_wds[@]}"; do
            for h in "${supra_n_hidden[@]}"; do
              for L in "${supra_n_layers[@]}"; do
                for ed in "${supra_embed_dims[@]}"; do
                  for aw in "${supra_aux_weights[@]}"; do
                    for mlp_var in "${supra_mlp_variants[@]}"; do
                      total_jobs=$((total_jobs + 1))
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
echo "=============================================="
echo "Total jobs: ${total_jobs}"
echo "  - Models: ${SUPRA_MODELS}"
echo "  - Layers: ${SUPRA_LAYERS}"
echo "  - Aux weights: ${supra_aux_weights[*]}"
echo "  - MLP variants: ${supra_mlp_variants[*]}"
echo "=============================================="

job_counter=0
for fg in "${FEATURE_GROUPS_ARR[@]}"; do
  for ds in "${DATASETS_ARR[@]}"; do
    echo ""
    echo "=============================================="
    echo "[${fg}] ${ds} (${METRIC})"
    echo "=============================================="

    # Resolve feature paths
    key="${ds}|${fg}"
    text_rel="${TEXT_FEATURE_BY_DS_GROUP[${key}]:-}"
    vis_rel="${VIS_FEATURE_BY_DS_GROUP[${key}]:-}"

    if [[ -z "${text_rel}" || -z "${vis_rel}" ]]; then
      echo "  [SKIP] Missing feature mapping for ${ds}|${fg}"
      continue
    fi

    ds_dir="${DATA_ROOT}/${ds}"
    ds_prefix="${ds//-/}"
    graph_path="${ds_dir}/${ds_prefix}Graph.pt"
    text_feat="${ds_dir}/${text_rel}"
    vis_feat="${ds_dir}/${vis_rel}"

    if [[ ! -f "${graph_path}" ]]; then
      echo "  [SKIP] Graph not found: ${graph_path}"
      continue
    fi
    if [[ ! -f "${text_feat}" ]]; then
      echo "  [SKIP] Text feature not found: ${text_feat}"
      continue
    fi
    if [[ ! -f "${vis_feat}" ]]; then
      echo "  [SKIP] Visual feature not found: ${vis_feat}"
      continue
    fi

    echo "  Graph: ${graph_path}"
    echo "  Text:  ${text_feat}"
    echo "  Visual: ${vis_feat}"

    # Determine selfloop based on model
    SELFLOOP_GCN="true"
    SELFLOOP_GAT="false"

    for model_name in ${SUPRA_MODELS}; do
      echo ""
      echo "  --- ${model_name} ---"

      if [[ "${model_name}" == "GCN" ]]; then
        SELFLOOP="${SELFLOOP_GCN}"
      elif [[ "${model_name}" == "GAT" ]]; then
        SELFLOOP="${SELFLOOP_GAT}"
      fi

      for dropout in "${supra_dropout}"; do
        for lr in "${supra_lrs[@]}"; do
          for wd in "${supra_wds[@]}"; do
            for h in "${supra_n_hidden[@]}"; do
              for L in "${supra_n_layers[@]}"; do
                for ed in "${supra_embed_dims[@]}"; do
                  for aw in "${supra_aux_weights[@]}"; do
                    for mlp_var in "${supra_mlp_variants[@]}"; do
                      ((++job_counter))
                      label="SUPRA-${model_name}-L${L}-aw${aw}-mlp${mlp_var}"
                      log_file="${LOG_ROOT}/fg_${fg}/${ds}/${label}.log"

                    mkdir -p "$(dirname "${log_file}")"

                    # Check if done
                    if [[ -f "${log_file}.done" ]]; then
                      echo "    [SKIP] ${job_counter}/${total_jobs} ${label} (done)"
                      continue
                    fi

                    echo "    + ${job_counter}/${total_jobs} ${label}"

                    start_time=${SECONDS}

                    cmd=(
                      python -m GNN.SUPRA
                      --data_name "${ds}"
                      --graph_path "${graph_path}"
                      --text_feature "${text_feat}"
                      --visual_feature "${vis_feat}"
                      --gpu "${GPU_ID}"
                      --n-runs "${N_RUNS}"
                      --n-epochs "${N_EPOCHS}"
                      --warmup_epochs "${WARMUP_EPOCHS}"
                      --eval_steps "${EVAL_STEPS}"
                      --early_stop_patience "${supra_early_stop_patience}"
                      --lr "${lr}"
                      --wd "${wd}"
                      --n-layers "${L}"
                      --n-hidden "${h}"
                      --dropout "${dropout}"
                      --label-smoothing "${supra_label_smoothing}"
                      --metric "${METRIC}"
                      --average "${AVERAGE}"
                      --train_ratio "${TRAIN_RATIO}"
                      --val_ratio "${VAL_RATIO}"
                      --undirected "${UNDIRECTED}"
                      --selfloop "${SELFLOOP}"
                      --inductive "${INDUCTIVE}"
                      --model_name "${model_name}"
                      --embed_dim "${ed}"
                      --aux_weight "${aw}"
                      --mlp_variant "${mlp_var}"
                      --result_csv "${RESULT_CSV}"
                      --result_csv_all "${RESULT_CSV_ALL}"
                      --disable_wandb
                    )

                    if [[ "${model_name}" == "GAT" ]]; then
                      cmd+=(
                        --n-heads "${gat_n_heads}"
                        --attn-drop "${gat_attn_drop}"
                        --edge-drop "${gat_edge_drop}"
                      )
                    fi

                    if "${cmd[@]}" >> "${log_file}" 2>&1; then
                      dur_s=$((SECONDS - start_time))
                      touch "${log_file}.done"
                      echo "    [OK] ${label} (${dur_s}s)"
                    else
                      rc=$?
                      dur_s=$((SECONDS - start_time))
                      echo "    [FAIL] ${label} (exit=${rc}, ${dur_s}s) log=${log_file}" >&2
                      echo "    --- tail ${log_file} ---" >&2
                      tail -n 80 "${log_file}" >&2 || true
                      echo "    --- end tail ---" >&2
                    fi
                    done
                  done
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
echo "SUPRA suite complete. Logs: ${LOG_ROOT}/"