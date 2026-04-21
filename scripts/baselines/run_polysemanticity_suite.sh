#!/usr/bin/env bash
set -euo pipefail

export DGLBACKEND=pytorch

# ==========================================================
# Env config (can be overridden externally)
# ==========================================================
GPU_ID=${GPU_ID:-0}
DRY_RUN=${DRY_RUN:-false}           # true: only print commands
DO_TRAIN=${DO_TRAIN:-true}          # true: train models + save ckpts
DO_ANALYZE=${DO_ANALYZE:-true}      # true: run analyze_polysemanticity_mag.py
DO_TRAIN_EARLY=${DO_TRAIN_EARLY:-true}  # true: train Early_GNN (only if missing)
DO_TRAIN_TRI=${DO_TRAIN_TRI:-false}     # true: train Tri_GNN (only if missing)

# Datasets / feature groups
DATASETS=${DATASETS:-"Movies Grocery Toys Reddit-M"}
FEATURE_GROUPS=${FEATURE_GROUPS:-"default clip_roberta"}

# Backbones and search grid (keep small by default)
BACKBONES=${BACKBONES:-"GCN SAGE GAT"}
# Optional override: explicit layer list to run (e.g., "2" or "1 2").
# If empty, uses backbone-specific defaults in layers_for_backbone().
LAYERS=${LAYERS:-""}
LRS=${LRS:-"0.0005 0.001"}

# Common training hyperparams (match your previous SUPRA script defaults)
N_EPOCHS=${N_EPOCHS:-1000}
N_RUNS=${N_RUNS:-3}
EVAL_STEPS=${EVAL_STEPS:-10}
EARLY_STOP=${EARLY_STOP:-50}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-0}
WD=${WD:-0.0}
N_HIDDEN=${N_HIDDEN:-256}
DROPOUT=${DROPOUT:-0.2}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.1}
METRIC=${METRIC:-accuracy}
AVERAGE=${AVERAGE:-macro}

# SUPRA-specific
AUX_LOSS=${AUX_LOSS:-0.1}
PID_DROPOUT=${PID_DROPOUT:-0.2}
PID_LU=${PID_LU:-0}
PID_L=${PID_L:-3}

# ==========================================================
# Fair / pure comparison switches (optional)
# ==========================================================
# PURE_COMPARE=true:
# - turn off SUPRA auxiliary loss (AUX_LOSS=0)
# - let SUPRA pid_L follow n_layers by default (no explicit --pid_L)
PURE_COMPARE=${PURE_COMPARE:-false}
# How to set SUPRA shared propagation depth relative to --n-layers (L)
# - fixed:  pass --pid_L ${PID_L} (current default behavior)
# - match:  pass --pid_L ${L}
# - auto:   do NOT pass --pid_L (SUPRA defaults pid_L = n_layers)
SUPRA_PID_L_MODE=${SUPRA_PID_L_MODE:-fixed}

if [[ "${PURE_COMPARE}" == "true" ]]; then
  AUX_LOSS=0.0
  SUPRA_PID_L_MODE=auto
fi

# Late-GNN specific
MM_PROJ_DIM=${MM_PROJ_DIM:-${N_HIDDEN}}
LATE_EMBED_DIM=${LATE_EMBED_DIM:-${N_HIDDEN}}

# Early-GNN specific
EARLY_FUSE=${EARLY_FUSE:-"concat"}      # concat | sum
EARLY_EMBED_DIM=${EARLY_EMBED_DIM:-${N_HIDDEN}}

# Tri-GNN specific
TRI_EMBED_DIM=${TRI_EMBED_DIM:-${N_HIDDEN}}

# Analysis controls
TOP_RATIO=${TOP_RATIO:-0.1}
STRICT_ZERO=${STRICT_ZERO:-true}
DEGREE_BINS=${DEGREE_BINS:-"0,1,2,4,8,16,32,64,128,1000000000"}
SPACES=${SPACES:-"late/text_h,late/vis_h,late/fused,early/h,tri/early_h,tri/text_h,tri/vis_h,tri/fused,supra/Ut,supra/Uv,supra/C"}
WRITE_PRED_EVAL=${WRITE_PRED_EVAL:-true}   # true: write pred_eval.csv (per-channel prediction metrics)

# Data root
DATA_ROOT=${DATA_ROOT:-"/openbayes/input/input0/MAGB_Dataset"}

# Output root
OUT_ROOT=${OUT_ROOT:-"polysemanticity_runs"}

# ---------------- Feature group definition ----------------
# (keep identical to your previous script)

declare -A TEXT_FEATURE_BY_DS_GROUP
declare -A VIS_FEATURE_BY_DS_GROUP

# Default group (Llama 3.2)
TEXT_FEATURE_BY_DS_GROUP["Movies|default"]='TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Movies|default"]='ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Grocery|default"]='TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Grocery|default"]='ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Toys|default"]='TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Toys|default"]='ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Reddit-M|default"]='TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-M|default"]='ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy'

# CLIP + RoBERTa group
TEXT_FEATURE_BY_DS_GROUP["Movies|clip_roberta"]='TextFeature/Movies_roberta_base_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Movies|clip_roberta"]='ImageFeature/Movies_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Grocery|clip_roberta"]='TextFeature/Grocery_roberta_base_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Grocery|clip_roberta"]='ImageFeature/Grocery_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Toys|clip_roberta"]='TextFeature/Toys_roberta_base_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Toys|clip_roberta"]='ImageFeature/Toys_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Reddit-M|clip_roberta"]='TextFeature/RedditM_roberta_base_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-M|clip_roberta"]='ImageFeature/RedditM_openai_clip-vit-large-patch14.npy'

# ==========================================================
# Helpers
# ==========================================================

_ts() { date '+%H:%M:%S'; }

run_cmd() {
  local cmd="$1"
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY RUN: ${cmd}"
  else
    echo "[$(_ts)] CMD: ${cmd}" >> "${GPU_MASTER_LOG}"
    eval "${cmd}"
  fi
}

layers_for_backbone() {
  local gnn="$1"
  if [[ "${gnn}" == "SAGE" ]]; then
    echo "3"
  else
    echo "1 2"
  fi
}

extra_args_for_backbone() {
  local gnn="$1"
  if [[ "${gnn}" == "SAGE" ]]; then
    echo "--aggregator mean"
  elif [[ "${gnn}" == "GAT" ]]; then
    echo "--n-heads 3 --attn-drop 0.0 --edge-drop 0.0 --no-attn-dst true"
  else
    echo ""
  fi
}

# ==========================================================
# Main
# ==========================================================

echo "=========================================================="
echo "RUNNING POLYSEMANTICITY SUITE"
echo "GPU ID        : ${GPU_ID}"
echo "DO_TRAIN      : ${DO_TRAIN}"
echo "DO_TRAIN_EARLY: ${DO_TRAIN_EARLY}"
echo "DO_TRAIN_TRI  : ${DO_TRAIN_TRI}"
echo "DO_ANALYZE    : ${DO_ANALYZE}"
echo "DRY_RUN       : ${DRY_RUN}"
echo "DATASETS      : ${DATASETS}"
echo "FEATURE_GROUPS: ${FEATURE_GROUPS}"
echo "BACKBONES     : ${BACKBONES}"
echo "OUT_ROOT      : ${OUT_ROOT}"
echo "=========================================================="

mkdir -p "${OUT_ROOT}"
GPU_MASTER_LOG="${OUT_ROOT}/suite_gpu${GPU_ID}.log"
echo "[$(_ts)] START | GPU=${GPU_ID} | OUT_ROOT=${OUT_ROOT}" >> "${GPU_MASTER_LOG}"


run_one_job() {
  local job_gpu="$1"
  local FEATURE_GROUP="$2"
  local ds="$3"
  local gnn="$4"
  local lr="$5"
  local L="$6"

  local GRAPH TEXT_FEAT VIS_FEAT RUN_DIR LATE_CKPT SUPRA_CKPT ANALYZE_OUT
  local EARLY_CKPT TRI_CKPT
  GRAPH="${DATA_ROOT}/${ds}/${ds//-/}Graph.pt"
  TEXT_FEAT="${DATA_ROOT}/${ds}/${TEXT_FEATURE_BY_DS_GROUP["${ds}|${FEATURE_GROUP}"]}"
  VIS_FEAT="${DATA_ROOT}/${ds}/${VIS_FEATURE_BY_DS_GROUP["${ds}|${FEATURE_GROUP}"]}"

  if [[ -z "${TEXT_FEAT}" || -z "${VIS_FEAT}" ]]; then
    echo "[WARN] Missing feature mapping for ${ds}|${FEATURE_GROUP}, skip"
    return 0
  fi

  RUN_DIR="${OUT_ROOT}/${FEATURE_GROUP}/${ds}/${gnn}/L${L}_lr${lr}"
  mkdir -p "${RUN_DIR}"

  LATE_CKPT="${RUN_DIR}/late_best.pt"
  EARLY_CKPT="${RUN_DIR}/early_best.pt"
  TRI_CKPT="${RUN_DIR}/tri_best.pt"
  SUPRA_CKPT_BASE="${RUN_DIR}/supra_best.pt"
  SUPRA_CKPT_RUN1="${RUN_DIR}/supra_best_run1.pt"
  SUPRA_CKPT="${SUPRA_CKPT_BASE}"
  if [[ -f "${SUPRA_CKPT_RUN1}" && ! -f "${SUPRA_CKPT_BASE}" ]]; then
    SUPRA_CKPT="${SUPRA_CKPT_RUN1}"
  fi
  ANALYZE_OUT="${RUN_DIR}/analysis"

  COMMON_ARGS=(
    --data_name "${ds}"
    --graph_path "${GRAPH}"
    --text_feature "${TEXT_FEAT}"
    --visual_feature "${VIS_FEAT}"
    --gpu "${job_gpu}"
    --metric "${METRIC}"
    --average "${AVERAGE}"
    --inductive false
    --undirected true
    --selfloop true
    --n-epochs "${N_EPOCHS}"
    --n-runs "${N_RUNS}"
    --warmup_epochs "${WARMUP_EPOCHS}"
    --eval_steps "${EVAL_STEPS}"
    --early_stop_patience "${EARLY_STOP}"
    --wd "${WD}"
    --n-hidden "${N_HIDDEN}"
    --dropout "${DROPOUT}"
    --label-smoothing "${LABEL_SMOOTHING}"
    --log-every 1
  )

  local extra_args
  extra_args=$(extra_args_for_backbone "${gnn}")

  local SUPRA_PID_L_ARGS
  SUPRA_PID_L_ARGS=""
  case "${SUPRA_PID_L_MODE}" in
    fixed)
      SUPRA_PID_L_ARGS="--pid_L ${PID_L}"
      ;;
    match)
      SUPRA_PID_L_ARGS="--pid_L ${L}"
      ;;
    auto)
      SUPRA_PID_L_ARGS=""
      ;;
    *)
      echo "[ERROR] Unknown SUPRA_PID_L_MODE=${SUPRA_PID_L_MODE} (expected: fixed|match|auto)"
      exit 2
      ;;
  esac

  echo "[$(_ts)] GPU=${job_gpu} | DS=${ds} | FG=${FEATURE_GROUP} | ${gnn} | L=${L} | lr=${lr}"
  echo "[$(_ts)] JOB: GPU=${job_gpu} DS=${ds} FG=${FEATURE_GROUP} model=${gnn} L=${L} lr=${lr}" >> "${GPU_MASTER_LOG}"

  if [[ "${DO_TRAIN}" == "true" ]]; then
    if [[ ! -f "${LATE_CKPT}" ]]; then
      cmd="python GNN/Library/MAG/Late_GNN.py ${COMMON_ARGS[*]} --disable_wandb --model_name ${gnn} --n-layers ${L} --lr ${lr} --mm_proj_dim ${MM_PROJ_DIM} --late_embed_dim ${LATE_EMBED_DIM} ${extra_args} --save_best_ckpt ${LATE_CKPT} > ${RUN_DIR}/late_train_gpu${job_gpu}.log 2>&1"
      run_cmd "${cmd}"
    else
      echo "[Info] Skip Late_GNN training (ckpt exists): ${LATE_CKPT}"
    fi

    if [[ ! -f "${SUPRA_CKPT}" ]]; then
      # SUPRA saves per-run ckpts and may create *_run1.pt rather than the base name.
      if [[ -f "${SUPRA_CKPT_BASE}" || -f "${SUPRA_CKPT_RUN1}" ]]; then
        echo "[Info] Skip SUPRA training (ckpt exists): ${SUPRA_CKPT_BASE} or ${SUPRA_CKPT_RUN1}"
      else
        cmd="python GNN/Library/MAG/SUPRA.py ${COMMON_ARGS[*]} --disable_wandb --model_name ${gnn} --n-layers ${L} --lr ${lr} ${extra_args} --pid_dropout ${PID_DROPOUT} --pid_lambda_aux ${AUX_LOSS} --pid_lu ${PID_LU} ${SUPRA_PID_L_ARGS} --save_best_ckpt ${SUPRA_CKPT_BASE} > ${RUN_DIR}/supra_train_gpu${job_gpu}.log 2>&1"
        run_cmd "${cmd}"
      fi
    else
      echo "[Info] Skip SUPRA training (ckpt exists): ${SUPRA_CKPT}"
    fi

    if [[ "${DO_TRAIN_EARLY}" == "true" ]]; then
      if [[ ! -f "${EARLY_CKPT}" ]]; then
        cmd="python GNN/Library/MAG/Early_GNN.py ${COMMON_ARGS[*]} --disable_wandb --backend gnn --model_name ${gnn} --n-layers ${L} --lr ${lr} --mm_proj_dim ${MM_PROJ_DIM} --early_fuse ${EARLY_FUSE} --early_embed_dim ${EARLY_EMBED_DIM} --separate_classifier ${extra_args} --save_best_ckpt ${EARLY_CKPT} > ${RUN_DIR}/early_train_gpu${job_gpu}.log 2>&1"
        run_cmd "${cmd}"
      else
        echo "[Info] Skip Early_GNN training (ckpt exists): ${EARLY_CKPT}"
      fi
    else
      echo "[Info] Skip Early_GNN training (DO_TRAIN_EARLY=false)"
    fi

    if [[ "${DO_TRAIN_TRI}" == "true" ]]; then
      if [[ ! -f "${TRI_CKPT}" ]]; then
        cmd="python GNN/Library/MAG/Tri_GNN.py ${COMMON_ARGS[*]} --disable_wandb --model_name ${gnn} --n-layers ${L} --lr ${lr} --mm_proj_dim ${MM_PROJ_DIM} --tri_embed_dim ${TRI_EMBED_DIM} ${extra_args} --save_best_ckpt ${TRI_CKPT} > ${RUN_DIR}/tri_train_gpu${job_gpu}.log 2>&1"
        run_cmd "${cmd}"
      else
        echo "[Info] Skip Tri_GNN training (ckpt exists): ${TRI_CKPT}"
      fi
    else
      echo "[Info] Skip Tri_GNN training (DO_TRAIN_TRI=false)"
    fi
  fi

  if [[ "${DO_ANALYZE}" == "true" ]]; then
    mkdir -p "${ANALYZE_OUT}"
    strict_flag=""
    if [[ "${STRICT_ZERO}" == "true" ]]; then
      strict_flag="--strict_zero"
    fi

    pred_flag=""
    if [[ "${WRITE_PRED_EVAL}" == "true" ]]; then
      pred_flag="--write_pred_eval"
    fi
    spaces_to_use="${SPACES}"
    # If early ckpt is missing, drop early spaces to avoid hard failure.
    if [[ ! -f "${EARLY_CKPT}" ]]; then
      spaces_to_use=$(echo "${spaces_to_use}" | sed 's/\(^\|,\)early\/[^,]*//g' | sed 's/^,//' | sed 's/,,*/,/g' | sed 's/,$//')
    fi
    # If tri ckpt is missing, drop tri spaces to avoid hard failure.
    if [[ ! -f "${TRI_CKPT}" ]]; then
      spaces_to_use=$(echo "${spaces_to_use}" | sed 's/\(^\|,\)tri\/[^,]*//g' | sed 's/^,//' | sed 's/,,*/,/g' | sed 's/,$//')
    fi

    cmd="python analyze_polysemanticity_mag.py ${COMMON_ARGS[*]} --model_name ${gnn} --n-layers ${L} --lr ${lr} ${extra_args} --mm_proj_dim ${MM_PROJ_DIM} --late_embed_dim ${LATE_EMBED_DIM} --early_fuse ${EARLY_FUSE} --early_embed_dim ${EARLY_EMBED_DIM} --tri_embed_dim ${TRI_EMBED_DIM} --pid_embed_dim ${N_HIDDEN} --pid_dropout ${PID_DROPOUT} --pid_lu ${PID_LU} ${SUPRA_PID_L_ARGS} --late_ckpt ${LATE_CKPT} --supra_ckpt ${SUPRA_CKPT} --early_ckpt ${EARLY_CKPT} --tri_ckpt ${TRI_CKPT} --out_dir ${ANALYZE_OUT} --top_ratio ${TOP_RATIO} --degree_bins '${DEGREE_BINS}' --spaces '${spaces_to_use}' ${strict_flag} ${pred_flag} > ${RUN_DIR}/analysis_gpu${job_gpu}.log 2>&1"
    run_cmd "${cmd}"
    echo "[Done] Analysis outputs: ${ANALYZE_OUT}"
  fi
}

for FEATURE_GROUP in ${FEATURE_GROUPS}; do
  echo "=========================================================="
  echo "Feature Group : ${FEATURE_GROUP}"
  echo "=========================================================="

  for ds in ${DATASETS}; do
    for gnn in ${BACKBONES}; do
      if [[ -n "${LAYERS}" ]]; then
        layers="${LAYERS}"
      else
        layers=$(layers_for_backbone "${gnn}")
      fi
      for lr in ${LRS}; do
        for L in ${layers}; do
          run_one_job "${GPU_ID}" "${FEATURE_GROUP}" "${ds}" "${gnn}" "${lr}" "${L}"
        done
      done
    done
  done

done

echo "All done!"
echo "[$(_ts)] DONE | GPU=${GPU_ID}" >> "${GPU_MASTER_LOG}"
