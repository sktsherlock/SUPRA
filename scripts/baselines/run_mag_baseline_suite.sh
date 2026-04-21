#!/usr/bin/env bash
set -euo pipefail

# MAG baseline suite: per-modality encoders + concat -> GNN/MLP.
# Runs: Text/Visual/Multimodal x (MLP, GCN, SAGE) by default.

export DGLBACKEND=${DGLBACKEND:-pytorch}

# ---------------- W&B logging mode ----------------
WANDB_RUN_MODE=${WANDB_RUN_MODE:-offline}  # disabled|offline|online

# ---------------- User-editable: paths ----------------
DATA_ROOT=${DATA_ROOT:-/hyperai/input/input0/MAGB_Dataset}

# ---------------- Feature group support ----------------
FEATURE_GROUPS=${FEATURE_GROUPS:-"clip_roberta"}

declare -A TEXT_FEATURE_BY_DS_GROUP
declare -A VIS_FEATURE_BY_DS_GROUP

# Default group (repo naming convention)
TEXT_FEATURE_BY_DS_GROUP["Movies|default"]='TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Movies|default"]='ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Grocery|default"]='TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Grocery|default"]='ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Toys|default"]='TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Toys|default"]='ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Reddit-M|default"]='TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-M|default"]='ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Reddit-S|default"]='TextFeature/RedditS_Llama_3.2_11B_Vision_Instruct_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-S|default"]='ImageFeature/RedditS_Llama-3.2-11B-Vision-Instruct_visual.npy'

# CLIP (image) + RoBERTa (text)
TEXT_FEATURE_BY_DS_GROUP["Movies|clip_roberta"]='TextFeature/Movies_roberta_base_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Movies|clip_roberta"]='ImageFeature/Movies_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Grocery|clip_roberta"]='TextFeature/Grocery_roberta_base_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Grocery|clip_roberta"]='ImageFeature/Grocery_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Toys|clip_roberta"]='TextFeature/Toys_roberta_base_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Toys|clip_roberta"]='ImageFeature/Toys_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Reddit-M|clip_roberta"]='TextFeature/RedditM_roberta_base_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-M|clip_roberta"]='ImageFeature/RedditM_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Reddit-S|clip_roberta"]='TextFeature/RedditS_roberta_base_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-S|clip_roberta"]='ImageFeature/RedditS_openai_clip-vit-large-patch14.npy'

# ---------------- Datasets ----------------
DEFAULT_DATASETS="Movies Grocery Reddit-M Reddit-S Toys"
DATASETS=${DATASETS:-"${DEFAULT_DATASETS}"}
read -r -a DATASETS_ARR <<< "${DATASETS}"
read -r -a FEATURE_GROUPS_ARR <<< "${FEATURE_GROUPS}"

# ---------------- Experiment grid ----------------
EXPERIMENTS=${EXPERIMENTS:-"plain baseline late ogm"}        # plain|baseline|late|tri|ogm
# Plain_GNN: by default only run unimodal (text/visual). To include concat, set MODALITIES="text visual none".
MODALITIES=${MODALITIES:-"text visual"}             # text|visual|none (plain)
EARLY_LATE_MODALITIES=${EARLY_LATE_MODALITIES:-"none"}   # usually "none"
BACKENDS=${BACKENDS:-"mlp gnn"}                          # mlp|gnn (baseline)
PLAIN_BACKENDS=${PLAIN_BACKENDS:-"mlp gnn"}             # mlp|gnn (plain)
OGM_BACKENDS=${OGM_BACKENDS:-"gnn"}                     # gnn|mlp (ogm)
# ModDrop is optional for baselines; keep it opt-in by default.
BASELINE_VARIANTS=${BASELINE_VARIANTS:-"base"} # base|drop (baseline/EF)
LATE_VARIANTS=${LATE_VARIANTS:-"base"}         # base|drop (late/tri)
EARLY_FUSE_MODES=${EARLY_FUSE_MODES:-"concat"}      # concat|sum (baseline/EF)
GNN_MODELS=${GNN_MODELS:-"GCN SAGE GAT"}                 # GNN backbones

read -r -a EXPERIMENTS_ARR <<< "${EXPERIMENTS}"
read -r -a MODALITIES_ARR <<< "${MODALITIES}"
read -r -a EARLY_LATE_MODALITIES_ARR <<< "${EARLY_LATE_MODALITIES}"
read -r -a BACKENDS_ARR <<< "${BACKENDS}"
read -r -a PLAIN_BACKENDS_ARR <<< "${PLAIN_BACKENDS}"
read -r -a OGM_BACKENDS_ARR <<< "${OGM_BACKENDS}"
read -r -a BASELINE_VARIANTS_ARR <<< "${BASELINE_VARIANTS}"
read -r -a LATE_VARIANTS_ARR <<< "${LATE_VARIANTS}"
read -r -a EARLY_FUSE_MODES_ARR <<< "${EARLY_FUSE_MODES}"
read -r -a GNN_MODELS_ARR <<< "${GNN_MODELS}"

# ---------------- Common training knobs ----------------
GPU_ID=${GPU_ID:-0}
N_RUNS=${N_RUNS:-3}
N_EPOCHS=${N_EPOCHS:-1000}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-50}
EVAL_STEPS=${EVAL_STEPS:-1}
METRIC=${METRIC:-accuracy}
AVERAGE=${AVERAGE:-macro}
REPORT_DROP_MODALITY=${REPORT_DROP_MODALITY:-false}
REPORT_DROP_MODE=${REPORT_DROP_MODE:-best}
RESULT_CSV=${RESULT_CSV:-}
RESULT_CSV_ALL=${RESULT_CSV_ALL:-}
EARLY_MODALITY_DROPOUT=${EARLY_MODALITY_DROPOUT:-0.0}
# By default, keep Early/Late ModDrop ratios aligned unless explicitly overridden.
LATE_MODALITY_DROPOUT=${LATE_MODALITY_DROPOUT:-${EARLY_MODALITY_DROPOUT}}
KD_WEIGHT=${KD_WEIGHT:-0.0}
KD_TEMPERATURE=${KD_TEMPERATURE:-1.0}
DEGRADE_ALPHA=${DEGRADE_ALPHA:-1.0}
DEGRADE_TARGET=${DEGRADE_TARGET:-both}
DEGRADE_ALPHAS=${DEGRADE_ALPHAS:-}

SELFLOOP=${SELFLOOP:-true}
UNDIRECTED=${UNDIRECTED:-true}
TRAIN_RATIO=${TRAIN_RATIO:-0.6}
VAL_RATIO=${VAL_RATIO:-0.2}
INDUCTIVE=${INDUCTIVE:-false}

# Optional per-dataset split ratios (override defaults above)
declare -A TRAIN_RATIO_BY_DS
declare -A VAL_RATIO_BY_DS
# Reddit-S: 2:2:6 split
TRAIN_RATIO_BY_DS["Reddit-S"]=${TRAIN_RATIO_BY_DS["Reddit-S"]:-0.2}
VAL_RATIO_BY_DS["Reddit-S"]=${VAL_RATIO_BY_DS["Reddit-S"]:-0.2}

MM_PROJ_DIM=${MM_PROJ_DIM:-}

# ---------------- Sweep (grid search) ----------------
# Per-backbone grids (edit manually like run_imok_new.sh)
# ---------------- MLP sweep ----------------
mlp_dropouts=("0.2")
mlp_lrs=("0.0005" "0.001")
mlp_wds=("0.0")
mlp_n_hidden=("256")
mlp_n_layers=("2" "3")
mlp_label_smoothing="0.1"
mlp_early_stop_patience="50"

# ---------------- GCN sweep ----------------
gcn_dropouts=("0.2")
gcn_lrs=("0.0005" "0.001")
gcn_wds=("0.0")
gcn_n_hidden=("256")
GCN_LAYERS=${GCN_LAYERS:-"1 2"}
read -r -a gcn_n_layers <<< "${GCN_LAYERS}"
gcn_label_smoothing="0.1"
gcn_early_stop_patience="50"

# ---------------- GraphSAGE sweep ----------------
sage_dropouts=("0.2")
sage_lrs=("0.0005" "0.001")
sage_wds=("0.0")
sage_n_hidden=("256")
SAGE_LAYERS=${SAGE_LAYERS:-"3"}
read -r -a sage_n_layers <<< "${SAGE_LAYERS}"
sage_label_smoothing="0.1"
sage_early_stop_patience="50"
sage_aggregator="mean"

# ---------------- GAT sweep ----------------
gat_dropouts=("0.2")
gat_lrs=("0.0005" "0.001")
gat_wds=("0.0")
gat_n_hidden=("256")
GAT_LAYERS=${GAT_LAYERS:-"1 2"}
read -r -a gat_n_layers <<< "${GAT_LAYERS}"
gat_label_smoothing="0.1"
gat_early_stop_patience="25"
gat_n_heads=3
gat_attn_drop=0.0
gat_edge_drop=0.0
gat_no_attn_dst=true

# ---------------- OGM sweep ----------------
ogm_modulations=("OGM_GE")
ogm_projection_dims=("256" "512")
ogm_modulation_starts=("0")
ogm_modulation_ends=("100")
ogm_alphas=("1.0")

# ---------------- Logging ----------------
RESUME_FROM=${RESUME_FROM:-}
if [[ -n "${RESUME_FROM}" ]]; then
  LOG_ROOT="${RESUME_FROM}"
else
  LOG_ROOT=${LOG_DIR:-logs_mag_baseline}
fi
mkdir -p "${LOG_ROOT}"

WANDB_BASE_DIR=${WANDB_BASE_DIR:-/openbayes/home/MAGB}
WANDB_PROJECT=${WANDB_PROJECT:-PIDMAG}
WANDB_ENTITY=${WANDB_ENTITY:-}

case "${WANDB_RUN_MODE}" in
  disabled)
    export WANDB_DISABLED=true
    ;;
  offline)
    unset WANDB_DISABLED || true
    export WANDB_MODE=offline
    ;;
  online)
    unset WANDB_DISABLED || true
    unset WANDB_MODE || true
    ;;
  *)
    echo "[Error] Unknown WANDB_RUN_MODE=${WANDB_RUN_MODE} (use disabled|offline|online)" >&2
    exit 1
    ;;
esac

# ---------------- Utilities ----------------
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

list_groups_for_dataset() {
  local ds="$1"
  local g=()
  local k
  for k in "${!TEXT_FEATURE_BY_DS_GROUP[@]}"; do
    if [[ "${k}" == "${ds}|"* ]]; then
      g+=("${k#${ds}|}")
    fi
  done
  if [[ ${#g[@]} -eq 0 ]]; then
    echo "(none configured)"
  else
    printf '%s' "${g[*]}"
  fi
}

dataset_config() {
  local ds="$1"
  local ds_dir="${DATA_ROOT}/${ds}"
  local ds_key="${ds//-/_}"
  local ds_prefix="${ds//-/}"

  local graph_var="GRAPH_PATH_${ds_key}"
  local text_var="TEXT_FEAT_${ds_key}"
  local vis_var="VIS_FEAT_${ds_key}"

  if [[ -n "${!graph_var:-}" ]]; then
    GRAPH_PATH["${ds}"]="${!graph_var}"
  else
    GRAPH_PATH["${ds}"]=$(require_file "${ds_dir}/${ds_prefix}Graph.pt" "graph")
  fi

  if [[ -n "${!text_var:-}" && -n "${!vis_var:-}" ]]; then
    TEXT_FEAT["${ds}"]="${!text_var}"
    VIS_FEAT["${ds}"]="${!vis_var}"
    return
  fi

  local key="${ds}|${FEATURE_GROUP}"
  local text_rel="${TEXT_FEATURE_BY_DS_GROUP[${key}]:-}"
  local vis_rel="${VIS_FEATURE_BY_DS_GROUP[${key}]:-}"
  if [[ -z "${text_rel}" || -z "${vis_rel}" ]]; then
    echo "[Error] Missing feature mapping for dataset='${ds}', FEATURE_GROUP='${FEATURE_GROUP}'." >&2
    echo "        Configured groups for '${ds}': $(list_groups_for_dataset "${ds}")" >&2
    exit 1
  fi

  TEXT_FEAT["${ds}"]=$(require_file "${ds_dir}/${text_rel}" "text feature (group=${FEATURE_GROUP})")
  VIS_FEAT["${ds}"]=$(require_file "${ds_dir}/${vis_rel}" "visual feature (group=${FEATURE_GROUP})")
}

RUN_IDX=0
TOTAL_JOBS_ALL=0
JOB_TOTAL=0
CURRENT_LOG_FILE=""
PROGRESS_LOG="${LOG_ROOT}/progress.log"
SKIP_DONE=${SKIP_DONE:-true}
RESULT_CSV_ARG=()
if [[ -n "${RESULT_CSV}" ]]; then
  RESULT_CSV_ARG=(--result_csv "${RESULT_CSV}")
fi
RESULT_CSV_ALL_ARG=()
if [[ -n "${RESULT_CSV_ALL}" ]]; then
  RESULT_CSV_ALL_ARG=(--result_csv_all "${RESULT_CSV_ALL}")
fi
DEGRADE_TARGET_ARG=(--degrade_target "${DEGRADE_TARGET}")
DEGRADE_ALPHAS_ARG=()
if [[ -n "${DEGRADE_ALPHAS}" ]]; then
  DEGRADE_ALPHAS_ARG=(--degrade_alphas "${DEGRADE_ALPHAS}")
fi

run_cmd() {
  local log_file="$1"
  local label="$2"
  shift 2

  RUN_IDX=$((RUN_IDX + 1))
  CURRENT_LOG_FILE="${log_file}"

  mkdir -p "$(dirname "${log_file}")"
  {
    echo ""
    echo "===== $(_ts) | START | ${RUN_IDX}/${JOB_TOTAL} | ${label} ====="
    echo "[CMD] $*"
  } >> "${log_file}"

  echo "[${RUN_IDX}/${JOB_TOTAL}] ${label} | log=${log_file}"

  local start_s=${SECONDS}
  ( "$@" ) >> "${log_file}" 2>&1
  local rc=$?
  local dur_s=$((SECONDS - start_s))

  { echo "===== $(_ts) | END   | ${RUN_IDX}/${JOB_TOTAL} | ${label} | ${dur_s}s | rc=${rc} ====="; } >> "${log_file}"

  if [[ "${rc}" -ne 0 ]]; then
    echo "$(_ts) | ${RUN_IDX}/${JOB_TOTAL} | FAIL(exit=${rc}) | ${label} | ${log_file}" >> "${PROGRESS_LOG}" || true
    return "${rc}"
  fi

  touch "${log_file}.done"
  echo "$(_ts) | ${RUN_IDX}/${JOB_TOTAL} | OK | ${label} | ${log_file}" >> "${PROGRESS_LOG}" || true
}

count_job() {
  local log_file="$1"
  if [[ "${SKIP_DONE}" == "true" && -f "${log_file}.done" ]]; then
    return
  fi
  TOTAL_JOBS_ALL=$((TOTAL_JOBS_ALL + 1))
  JOB_TOTAL=$((JOB_TOTAL + 1))
}

declare -A GRAPH_PATH
declare -A TEXT_FEAT
declare -A VIS_FEAT

N_DS=${#DATASETS_ARR[@]}
N_FG=${#FEATURE_GROUPS_ARR[@]}
N_MOD=${#MODALITIES_ARR[@]}
N_GNN=${#GNN_MODELS_ARR[@]}

TOTAL_JOBS_ALL=0
JOB_TOTAL=0
for fg in "${FEATURE_GROUPS_ARR[@]}"; do
  for ds in "${DATASETS_ARR[@]}"; do
    for exp in "${EXPERIMENTS_ARR[@]}"; do
      case "${exp}" in
        "plain")
          for modality in "${MODALITIES_ARR[@]}"; do
            for backend in "${PLAIN_BACKENDS_ARR[@]}"; do
              if [[ "${backend}" == "mlp" ]]; then
                for do in "${mlp_dropouts[@]}"; do
                  for lr in "${mlp_lrs[@]}"; do
                    for wd in "${mlp_wds[@]}"; do
                      for h in "${mlp_n_hidden[@]}"; do
                        for L in "${mlp_n_layers[@]}"; do
                          model_label="MLP"
                          run_label="Plain-${modality^}-${model_label}-lr${lr}-wd${wd}-h${h}-L${L}-do${do}"
                          log_file="${LOG_ROOT}/fg_${fg}/${ds}/${run_label}.log"
                          count_job "${log_file}"
                        done
                      done
                    done
                  done
                done
              elif [[ "${backend}" == "gnn" ]]; then
                for gnn in "${GNN_MODELS_ARR[@]}"; do
                  case "${gnn}" in
                    "GCN")
                      dropouts=("${gcn_dropouts[@]}")
                      lrs=("${gcn_lrs[@]}")
                      wds=("${gcn_wds[@]}")
                      hiddens=("${gcn_n_hidden[@]}")
                      layers=("${gcn_n_layers[@]}")
                      ;;
                    "SAGE")
                      dropouts=("${sage_dropouts[@]}")
                      lrs=("${sage_lrs[@]}")
                      wds=("${sage_wds[@]}")
                      hiddens=("${sage_n_hidden[@]}")
                      layers=("${sage_n_layers[@]}")
                      ;;
                    "GAT")
                      dropouts=("${gat_dropouts[@]}")
                      lrs=("${gat_lrs[@]}")
                      wds=("${gat_wds[@]}")
                      hiddens=("${gat_n_hidden[@]}")
                      layers=("${gat_n_layers[@]}")
                      ;;
                    *)
                      echo "[Error] Unsupported gnn_model: ${gnn}" >&2
                      exit 1
                      ;;
                  esac
                  for do in "${dropouts[@]}"; do
                    for lr in "${lrs[@]}"; do
                      for wd in "${wds[@]}"; do
                        for h in "${hiddens[@]}"; do
                          for L in "${layers[@]}"; do
                            model_label="${gnn}"
                            run_label="Plain-${modality^}-${model_label}-lr${lr}-wd${wd}-h${h}-L${L}-do${do}"
                            log_file="${LOG_ROOT}/fg_${fg}/${ds}/${run_label}.log"
                            count_job "${log_file}"
                          done
                        done
                      done
                    done
                  done
                done
              else
                echo "[Error] Unknown backend: ${backend}" >&2
                exit 1
              fi
            done
          done
          ;;
        "baseline")
          modality="none"
          for variant in "${BASELINE_VARIANTS_ARR[@]}"; do
            variant_suffix=""
            case "${variant}" in
              base)
                ;;
              drop)
                variant_suffix="-ModDrop"
                ;;
              *)
                echo "[Error] Unknown baseline variant: ${variant}" >&2
                exit 1
                ;;
            esac
            for early_fuse in "${EARLY_FUSE_MODES_ARR[@]}"; do
              fuse_suffix=""
              case "${early_fuse}" in
                concat)
                  ;;
                sum)
                  fuse_suffix="-sum"
                  ;;
                *)
                  echo "[Error] Unknown EARLY_FUSE_MODES entry: ${early_fuse} (use concat|sum)" >&2
                  exit 1
                  ;;
              esac
              for backend in "${BACKENDS_ARR[@]}"; do
                  if [[ "${backend}" == "mlp" ]]; then
                    for do in "${mlp_dropouts[@]}"; do
                      for lr in "${mlp_lrs[@]}"; do
                        for wd in "${mlp_wds[@]}"; do
                          for h in "${mlp_n_hidden[@]}"; do
                            for L in "${mlp_n_layers[@]}"; do
                              model_label="MLP"
                              run_label="${modality^}-${model_label}-lr${lr}-wd${wd}-h${h}-L${L}-do${do}${variant_suffix}${fuse_suffix}"
                              log_file="${LOG_ROOT}/fg_${fg}/${ds}/${run_label}.log"
                              count_job "${log_file}"
                            done
                          done
                        done
                      done
                    done
                  elif [[ "${backend}" == "gnn" ]]; then
                    for gnn in "${GNN_MODELS_ARR[@]}"; do
                      case "${gnn}" in
                        "GCN")
                          dropouts=("${gcn_dropouts[@]}")
                          lrs=("${gcn_lrs[@]}")
                          wds=("${gcn_wds[@]}")
                          hiddens=("${gcn_n_hidden[@]}")
                          layers=("${gcn_n_layers[@]}")
                          ;;
                        "SAGE")
                          dropouts=("${sage_dropouts[@]}")
                          lrs=("${sage_lrs[@]}")
                          wds=("${sage_wds[@]}")
                          hiddens=("${sage_n_hidden[@]}")
                          layers=("${sage_n_layers[@]}")
                          ;;
                        "GAT")
                          dropouts=("${gat_dropouts[@]}")
                          lrs=("${gat_lrs[@]}")
                          wds=("${gat_wds[@]}")
                          hiddens=("${gat_n_hidden[@]}")
                          layers=("${gat_n_layers[@]}")
                          ;;
                        *)
                          echo "[Error] Unsupported gnn_model: ${gnn}" >&2
                          exit 1
                          ;;
                      esac
                      for do in "${dropouts[@]}"; do
                        for lr in "${lrs[@]}"; do
                          for wd in "${wds[@]}"; do
                            for h in "${hiddens[@]}"; do
                              for L in "${layers[@]}"; do
                                model_label="${gnn}"
                                run_label="${modality^}-${model_label}-lr${lr}-wd${wd}-h${h}-L${L}-do${do}${variant_suffix}${fuse_suffix}"
                                log_file="${LOG_ROOT}/fg_${fg}/${ds}/${run_label}.log"
                                count_job "${log_file}"
                              done
                            done
                          done
                        done
                      done
                    done
                  else
                    echo "[Error] Unknown backend: ${backend}" >&2
                    exit 1
                  fi
              done
            done
          done
          ;;
        "late")
          for modality in "${EARLY_LATE_MODALITIES_ARR[@]}"; do
            if [[ "${modality}" != "none" ]]; then
              echo "[Error] Late only supports modality=none (got ${modality})." >&2
              exit 1
            fi
            for variant in "${LATE_VARIANTS_ARR[@]}"; do
              variant_suffix=""
              case "${variant}" in
                base)
                  ;;
                drop)
                  variant_suffix="-ModDrop"
                  ;;
                *)
                  echo "[Error] Unknown late variant: ${variant}" >&2
                  exit 1
                  ;;
              esac
              for gnn in "${GNN_MODELS_ARR[@]}"; do
                case "${gnn}" in
                  "GCN")
                    dropouts=("${gcn_dropouts[@]}")
                    lrs=("${gcn_lrs[@]}")
                    wds=("${gcn_wds[@]}")
                    hiddens=("${gcn_n_hidden[@]}")
                    layers=("${gcn_n_layers[@]}")
                    ;;
                  "SAGE")
                    dropouts=("${sage_dropouts[@]}")
                    lrs=("${sage_lrs[@]}")
                    wds=("${sage_wds[@]}")
                    hiddens=("${sage_n_hidden[@]}")
                    layers=("${sage_n_layers[@]}")
                    ;;
                  "GAT")
                    dropouts=("${gat_dropouts[@]}")
                    lrs=("${gat_lrs[@]}")
                    wds=("${gat_wds[@]}")
                    hiddens=("${gat_n_hidden[@]}")
                    layers=("${gat_n_layers[@]}")
                    ;;
                  *)
                    echo "[Error] Unsupported gnn_model: ${gnn}" >&2
                    exit 1
                    ;;
                esac
                for do in "${dropouts[@]}"; do
                  for lr in "${lrs[@]}"; do
                    for wd in "${wds[@]}"; do
                      for h in "${hiddens[@]}"; do
                        for L in "${layers[@]}"; do
                          model_label="Late-${gnn}"
                          run_label="${model_label}-lr${lr}-wd${wd}-h${h}-L${L}-do${do}${variant_suffix}"
                          log_file="${LOG_ROOT}/fg_${fg}/${ds}/${run_label}.log"
                          count_job "${log_file}"
                        done
                      done
                    done
                  done
                done
              done
            done
          done
          ;;
        "tri")
          for modality in "${EARLY_LATE_MODALITIES_ARR[@]}"; do
            if [[ "${modality}" != "none" ]]; then
              echo "[Error] Tri only supports modality=none (got ${modality})." >&2
              exit 1
            fi
            for variant in "${LATE_VARIANTS_ARR[@]}"; do
              variant_suffix=""
              case "${variant}" in
                base)
                  ;;
                drop)
                  variant_suffix="-ModDrop"
                  ;;
                *)
                  echo "[Error] Unknown tri variant: ${variant}" >&2
                  exit 1
                  ;;
              esac
              for gnn in "${GNN_MODELS_ARR[@]}"; do
                case "${gnn}" in
                  "GCN")
                    dropouts=("${gcn_dropouts[@]}")
                    lrs=("${gcn_lrs[@]}")
                    wds=("${gcn_wds[@]}")
                    hiddens=("${gcn_n_hidden[@]}")
                    layers=("${gcn_n_layers[@]}")
                    ;;
                  "SAGE")
                    dropouts=("${sage_dropouts[@]}")
                    lrs=("${sage_lrs[@]}")
                    wds=("${sage_wds[@]}")
                    hiddens=("${sage_n_hidden[@]}")
                    layers=("${sage_n_layers[@]}")
                    ;;
                  "GAT")
                    dropouts=("${gat_dropouts[@]}")
                    lrs=("${gat_lrs[@]}")
                    wds=("${gat_wds[@]}")
                    hiddens=("${gat_n_hidden[@]}")
                    layers=("${gat_n_layers[@]}")
                    ;;
                  *)
                    echo "[Error] Unsupported gnn_model: ${gnn}" >&2
                    exit 1
                    ;;
                esac
                for do in "${dropouts[@]}"; do
                  for lr in "${lrs[@]}"; do
                    for wd in "${wds[@]}"; do
                      for h in "${hiddens[@]}"; do
                        for L in "${layers[@]}"; do
                          model_label="Tri-${gnn}"
                          run_label="${model_label}-lr${lr}-wd${wd}-h${h}-L${L}-do${do}${variant_suffix}"
                          log_file="${LOG_ROOT}/fg_${fg}/${ds}/${run_label}.log"
                          count_job "${log_file}"
                        done
                      done
                    done
                  done
                done
              done
            done
          done
          ;;
        "ogm")
          for backend in "${OGM_BACKENDS_ARR[@]}"; do
            if [[ "${backend}" == "mlp" ]]; then
              for do in "${mlp_dropouts[@]}"; do
                for lr in "${mlp_lrs[@]}"; do
                  for wd in "${mlp_wds[@]}"; do
                    for h in "${mlp_n_hidden[@]}"; do
                      for L in "${mlp_n_layers[@]}"; do
                        for om in "${ogm_modulations[@]}"; do
                          for opd in "${ogm_projection_dims[@]}"; do
                            for os in "${ogm_modulation_starts[@]}"; do
                              for oe in "${ogm_modulation_ends[@]}"; do
                                for oa in "${ogm_alphas[@]}"; do
                                  model_label="MLP"
                                  run_label="OGMMLP-lr${lr}-wd${wd}-h${h}-L${L}-do${do}-mod${om}-p${opd}-ms${os}-me${oe}-a${oa}"
                                  log_file="${LOG_ROOT}/fg_${fg}/${ds}/${run_label}.log"
                                  count_job "${log_file}"
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
            elif [[ "${backend}" == "gnn" ]]; then
              for gnn in "${GNN_MODELS_ARR[@]}"; do
                case "${gnn}" in
                  "GCN")
                    dropouts=("${gcn_dropouts[@]}")
                    lrs=("${gcn_lrs[@]}")
                    wds=("${gcn_wds[@]}")
                    hiddens=("${gcn_n_hidden[@]}")
                    layers=("${gcn_n_layers[@]}")
                    ;;
                  "SAGE")
                    dropouts=("${sage_dropouts[@]}")
                    lrs=("${sage_lrs[@]}")
                    wds=("${sage_wds[@]}")
                    hiddens=("${sage_n_hidden[@]}")
                    layers=("${sage_n_layers[@]}")
                    ;;
                  "GAT")
                    dropouts=("${gat_dropouts[@]}")
                    lrs=("${gat_lrs[@]}")
                    wds=("${gat_wds[@]}")
                    hiddens=("${gat_n_hidden[@]}")
                    layers=("${gat_n_layers[@]}")
                    ;;
                  *)
                    echo "[Error] Unsupported gnn_model: ${gnn}" >&2
                    exit 1
                    ;;
                esac
                for do in "${dropouts[@]}"; do
                  for lr in "${lrs[@]}"; do
                    for wd in "${wds[@]}"; do
                      for h in "${hiddens[@]}"; do
                        for L in "${layers[@]}"; do
                          for om in "${ogm_modulations[@]}"; do
                            for opd in "${ogm_projection_dims[@]}"; do
                              for os in "${ogm_modulation_starts[@]}"; do
                                for oe in "${ogm_modulation_ends[@]}"; do
                                  for oa in "${ogm_alphas[@]}"; do
                                    run_label="OGMGNN-${gnn}-lr${lr}-wd${wd}-h${h}-L${L}-do${do}-mod${om}-p${opd}-ms${os}-me${oe}-a${oa}"
                                    log_file="${LOG_ROOT}/fg_${fg}/${ds}/${run_label}.log"
                                    count_job "${log_file}"
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
            else
              echo "[Error] Unknown backend: ${backend}" >&2
              exit 1
            fi
          done
          ;;
        *)
          echo "[Error] Unknown experiment: ${exp}" >&2
          exit 1
          ;;
      esac
    done
  done
done

echo "===== $(_ts) | SUITE START | planned_jobs_total=${TOTAL_JOBS_ALL} =====" > "${PROGRESS_LOG}"

DRY_RUN=${DRY_RUN:-false}
RUN_IDX=0

for fg in "${FEATURE_GROUPS_ARR[@]}"; do
  FEATURE_GROUP="${fg}"

  # Per-group wandb directory (offline mode) - always reset to avoid mixing.
  if [[ "${WANDB_RUN_MODE}" == "offline" ]]; then
    WANDB_DIR="${WANDB_BASE_DIR}/${LOG_ROOT}/fg_${FEATURE_GROUP}/wandb_offline"
    mkdir -p "${WANDB_DIR}"
    export WANDB_DIR
  fi
  if [[ -n "${WANDB_PROJECT}" ]]; then
    export WANDB_PROJECT
  fi
  if [[ -n "${WANDB_ENTITY}" ]]; then
    export WANDB_ENTITY
  fi

  # Validate dataset files for this feature group
  for ds in "${DATASETS_ARR[@]}"; do
    dataset_config "${ds}"
  done

  for ds in "${DATASETS_ARR[@]}"; do
    for exp in "${EXPERIMENTS_ARR[@]}"; do
      case "${exp}" in
        "plain")
          for modality in "${MODALITIES_ARR[@]}"; do
            for backend in "${PLAIN_BACKENDS_ARR[@]}"; do
              if [[ "${backend}" == "mlp" ]]; then
                for do in "${mlp_dropouts[@]}"; do
                  for lr in "${mlp_lrs[@]}"; do
                    for wd in "${mlp_wds[@]}"; do
                      for h in "${mlp_n_hidden[@]}"; do
                        for L in "${mlp_n_layers[@]}"; do
                          model_label="MLP"
                          run_label="Plain-${modality^}-${model_label}-lr${lr}-wd${wd}-h${h}-L${L}-do${do}"
                          log_file="${LOG_ROOT}/fg_${FEATURE_GROUP}/${ds}/${run_label}.log"
                          if [[ "${DRY_RUN}" == "true" ]]; then
                            echo "[DRY_RUN] fg=${FEATURE_GROUP} ds=${ds} exp=plain modality=${modality} backend=mlp model=MLP lr=${lr} wd=${wd} h=${h} L=${L} do=${do}"
                            continue
                          fi
                          if [[ "${SKIP_DONE}" == "true" && -f "${log_file}.done" ]]; then
                            echo "[SKIP] ${run_label} (done) | ${log_file}"
                            continue
                          fi
                          run_cmd "${log_file}" "fg=${FEATURE_GROUP} ds=${ds} ${run_label}" \
                            env WANDB_NAME="${run_label}-${ds}-${FEATURE_GROUP}" \
                              WANDB_GROUP="MAG/${ds}/${FEATURE_GROUP}/plain" \
                              WANDB_TAGS="exp=plain,ds=${ds},fg=${FEATURE_GROUP},backend=mlp,model=MLP,modality=${modality}" \
                              python GNN/Library/MAG/Plain_GNN.py \
                                --data_name "${ds}" \
                                --graph_path "${GRAPH_PATH[$ds]}" \
                                --text_feature "${TEXT_FEAT[$ds]}" \
                                --visual_feature "${VIS_FEAT[$ds]}" \
                                --gpu "${GPU_ID}" \
                                --inductive "${INDUCTIVE}" \
                                --undirected "${UNDIRECTED}" --selfloop "${SELFLOOP}" \
                                --metric "${METRIC}" --average "${AVERAGE}" \
                                "${RESULT_CSV_ARG[@]}" \
                                "${RESULT_CSV_ALL_ARG[@]}" \
                                --report_drop_modality "${REPORT_DROP_MODALITY}" \
                                --report_drop_mode "${REPORT_DROP_MODE}" \
                                --n-epochs "${N_EPOCHS}" --n-runs "${N_RUNS}" \
                                --warmup_epochs "${WARMUP_EPOCHS}" --eval_steps "${EVAL_STEPS}" \
                                --early_stop_patience "${mlp_early_stop_patience}" \
                                --lr "${lr}" --wd "${wd}" \
                                --n-layers "${L}" --n-hidden "${h}" --dropout "${do}" \
                                --label-smoothing "${mlp_label_smoothing}" \
                                --train_ratio "${TRAIN_RATIO_BY_DS[${ds}]:-${TRAIN_RATIO}}" --val_ratio "${VAL_RATIO_BY_DS[${ds}]:-${VAL_RATIO}}" \
                                --degrade_alpha "${DEGRADE_ALPHA}" \
                                "${DEGRADE_TARGET_ARG[@]}" \
                                "${DEGRADE_ALPHAS_ARG[@]}" \
                                --backend "mlp" --model_name "MLP" \
                                --single_modality "${modality}"
                        done
                      done
                    done
                  done
                done
              elif [[ "${backend}" == "gnn" ]]; then
                for gnn in "${GNN_MODELS_ARR[@]}"; do
                  extra_args=()
                  case "${gnn}" in
                    "GCN")
                      dropouts=("${gcn_dropouts[@]}")
                      lrs=("${gcn_lrs[@]}")
                      wds=("${gcn_wds[@]}")
                      hiddens=("${gcn_n_hidden[@]}")
                      layers=("${gcn_n_layers[@]}")
                      label_smoothing="${gcn_label_smoothing}"
                      early_stop_patience="${gcn_early_stop_patience}"
                      ;;
                    "SAGE")
                      dropouts=("${sage_dropouts[@]}")
                      lrs=("${sage_lrs[@]}")
                      wds=("${sage_wds[@]}")
                      hiddens=("${sage_n_hidden[@]}")
                      layers=("${sage_n_layers[@]}")
                      label_smoothing="${sage_label_smoothing}"
                      early_stop_patience="${sage_early_stop_patience}"
                      extra_args=(--aggregator "${sage_aggregator}")
                      ;;
                    "GAT")
                      dropouts=("${gat_dropouts[@]}")
                      lrs=("${gat_lrs[@]}")
                      wds=("${gat_wds[@]}")
                      hiddens=("${gat_n_hidden[@]}")
                      layers=("${gat_n_layers[@]}")
                      label_smoothing="${gat_label_smoothing}"
                      early_stop_patience="${gat_early_stop_patience}"
                      extra_args=(--n-heads "${gat_n_heads}" --attn-drop "${gat_attn_drop}" --edge-drop "${gat_edge_drop}" --no-attn-dst "${gat_no_attn_dst}")
                      ;;
                    *)
                      echo "[Error] Unsupported gnn_model: ${gnn}" >&2
                      exit 1
                      ;;
                  esac
                  for do in "${dropouts[@]}"; do
                    for lr in "${lrs[@]}"; do
                      for wd in "${wds[@]}"; do
                        for h in "${hiddens[@]}"; do
                          for L in "${layers[@]}"; do
                            model_label="${gnn}"
                            run_label="Plain-${modality^}-${model_label}-lr${lr}-wd${wd}-h${h}-L${L}-do${do}"
                            log_file="${LOG_ROOT}/fg_${FEATURE_GROUP}/${ds}/${run_label}.log"
                            if [[ "${DRY_RUN}" == "true" ]]; then
                              echo "[DRY_RUN] fg=${FEATURE_GROUP} ds=${ds} exp=plain modality=${modality} backend=gnn model=${gnn} lr=${lr} wd=${wd} h=${h} L=${L} do=${do}"
                              continue
                            fi
                            if [[ "${SKIP_DONE}" == "true" && -f "${log_file}.done" ]]; then
                              echo "[SKIP] ${run_label} (done) | ${log_file}"
                              continue
                            fi
                            run_cmd "${log_file}" "fg=${FEATURE_GROUP} ds=${ds} ${run_label}" \
                              env WANDB_NAME="${run_label}-${ds}-${FEATURE_GROUP}" \
                                WANDB_GROUP="MAG/${ds}/${FEATURE_GROUP}/plain" \
                                WANDB_TAGS="exp=plain,ds=${ds},fg=${FEATURE_GROUP},backend=gnn,model=${gnn},modality=${modality}" \
                                python GNN/Library/MAG/Plain_GNN.py \
                                  --data_name "${ds}" \
                                  --graph_path "${GRAPH_PATH[$ds]}" \
                                  --text_feature "${TEXT_FEAT[$ds]}" \
                                  --visual_feature "${VIS_FEAT[$ds]}" \
                                  --gpu "${GPU_ID}" \
                                  --inductive "${INDUCTIVE}" \
                                  --undirected "${UNDIRECTED}" --selfloop "${SELFLOOP}" \
                                  --metric "${METRIC}" --average "${AVERAGE}" \
                                  "${RESULT_CSV_ARG[@]}" \
                                  "${RESULT_CSV_ALL_ARG[@]}" \
                                  --report_drop_modality "${REPORT_DROP_MODALITY}" \
                                  --report_drop_mode "${REPORT_DROP_MODE}" \
                                  --n-epochs "${N_EPOCHS}" --n-runs "${N_RUNS}" \
                                  --warmup_epochs "${WARMUP_EPOCHS}" --eval_steps "${EVAL_STEPS}" \
                                  --early_stop_patience "${early_stop_patience}" \
                                  --lr "${lr}" --wd "${wd}" \
                                  --n-layers "${L}" --n-hidden "${h}" --dropout "${do}" \
                                  --label-smoothing "${label_smoothing}" \
                                  --train_ratio "${TRAIN_RATIO_BY_DS[${ds}]:-${TRAIN_RATIO}}" --val_ratio "${VAL_RATIO_BY_DS[${ds}]:-${VAL_RATIO}}" \
                                  --degrade_alpha "${DEGRADE_ALPHA}" \
                                  "${DEGRADE_TARGET_ARG[@]}" \
                                  "${DEGRADE_ALPHAS_ARG[@]}" \
                                  --backend "gnn" --model_name "${gnn}" \
                                  --single_modality "${modality}" \
                                  "${extra_args[@]}"
                          done
                        done
                      done
                    done
                  done
                done
              else
                echo "[Error] Unknown backend: ${backend}" >&2
                exit 1
              fi
            done
          done
          ;;
        "baseline")
          modality="none"
          for variant in "${BASELINE_VARIANTS_ARR[@]}"; do
              variant_drop=0.0
              variant_suffix=""
              case "${variant}" in
                base)
                  ;;
                drop)
                  variant_drop="${EARLY_MODALITY_DROPOUT}"
                  variant_suffix="-ModDrop"
                  ;;
                *)
                  echo "[Error] Unknown baseline variant: ${variant}" >&2
                  exit 1
                  ;;
              esac

              for early_fuse in "${EARLY_FUSE_MODES_ARR[@]}"; do
                fuse_suffix=""
                fuse_tag_suffix=""
                case "${early_fuse}" in
                  concat)
                    ;;
                  sum)
                    fuse_suffix="-sum"
                    fuse_tag_suffix="-sum"
                    ;;
                  *)
                    echo "[Error] Unknown EARLY_FUSE_MODES entry: ${early_fuse} (use concat|sum)" >&2
                    exit 1
                    ;;
                esac

              for backend in "${BACKENDS_ARR[@]}"; do
                if [[ "${backend}" == "mlp" ]]; then
                  variant_tag="EF-MLP${fuse_tag_suffix}"
                  case "${variant}" in
                    base) ;;
                    drop) variant_tag="${variant_tag}+ModDrop" ;;
                  esac
                  for do in "${mlp_dropouts[@]}"; do
                    for lr in "${mlp_lrs[@]}"; do
                      for wd in "${mlp_wds[@]}"; do
                        for h in "${mlp_n_hidden[@]}"; do
                          for L in "${mlp_n_layers[@]}"; do
                            model_label="MLP"
                            run_label="${modality^}-${model_label}-lr${lr}-wd${wd}-h${h}-L${L}-do${do}${variant_suffix}${fuse_suffix}"
                            log_file="${LOG_ROOT}/fg_${FEATURE_GROUP}/${ds}/${run_label}.log"
                            if [[ "${DRY_RUN}" == "true" ]]; then
                              echo "[DRY_RUN] fg=${FEATURE_GROUP} ds=${ds} exp=baseline variant=${variant} early_fuse=${early_fuse} backend=mlp model=MLP lr=${lr} wd=${wd} h=${h} L=${L} do=${do}"
                              continue
                            fi
                            if [[ "${SKIP_DONE}" == "true" && -f "${log_file}.done" ]]; then
                              echo "[SKIP] ${run_label} (done) | ${log_file}"
                              continue
                            fi
                            run_cmd "${log_file}" "fg=${FEATURE_GROUP} ds=${ds} ${run_label}" \
                              env WANDB_NAME="${run_label}-${ds}-${FEATURE_GROUP}" \
                                WANDB_GROUP="MAG/${ds}/${FEATURE_GROUP}/baseline" \
                                WANDB_TAGS="exp=baseline,variant=${variant},early_fuse=${early_fuse},ds=${ds},fg=${FEATURE_GROUP},backend=mlp,model=MLP" \
                                python GNN/Library/MAG/Early_GNN.py \
                                  --data_name "${ds}" \
                                  --graph_path "${GRAPH_PATH[$ds]}" \
                                  --text_feature "${TEXT_FEAT[$ds]}" \
                                  --visual_feature "${VIS_FEAT[$ds]}" \
                                  --gpu "${GPU_ID}" \
                                  --inductive "${INDUCTIVE}" \
                                  --undirected "${UNDIRECTED}" --selfloop "${SELFLOOP}" \
                                  --metric "${METRIC}" --average "${AVERAGE}" \
                                  "${RESULT_CSV_ARG[@]}" \
                                  "${RESULT_CSV_ALL_ARG[@]}" \
                                  --report_drop_modality "${REPORT_DROP_MODALITY}" \
                                  --report_drop_mode "${REPORT_DROP_MODE}" \
                                  --degrade_alpha "${DEGRADE_ALPHA}" \
                                  --result_tag "${variant_tag}" \
                                  --n-epochs "${N_EPOCHS}" --n-runs "${N_RUNS}" \
                                  --warmup_epochs "${WARMUP_EPOCHS}" --eval_steps "${EVAL_STEPS}" \
                                  --early_stop_patience "${mlp_early_stop_patience}" \
                                  --lr "${lr}" --wd "${wd}" \
                                  --n-layers "${L}" --n-hidden "${h}" --dropout "${do}" \
                                  --label-smoothing "${mlp_label_smoothing}" \
                                  --train_ratio "${TRAIN_RATIO_BY_DS[${ds}]:-${TRAIN_RATIO}}" --val_ratio "${VAL_RATIO_BY_DS[${ds}]:-${VAL_RATIO}}" \
                                  --backend "mlp" --model_name "MLP" \
                                  --early_fuse "${early_fuse}" \
                                  --modality_dropout "${variant_drop}" \
                                  --kd_weight "${KD_WEIGHT}" \
                                  --kd_temperature "${KD_TEMPERATURE}" \
                                  "${DEGRADE_TARGET_ARG[@]}" \
                                  "${DEGRADE_ALPHAS_ARG[@]}" \
                                  ${MM_PROJ_DIM:+--mm_proj_dim "${MM_PROJ_DIM}"}
                          done
                        done
                      done
                    done
                  done
                elif [[ "${backend}" == "gnn" ]]; then
                  variant_tag="EF-GNN${fuse_tag_suffix}"
                  case "${variant}" in
                    base) ;;
                    drop) variant_tag="${variant_tag}+ModDrop" ;;
                  esac
                  for gnn in "${GNN_MODELS_ARR[@]}"; do
                    extra_args=()
                    case "${gnn}" in
                      "GCN")
                        dropouts=("${gcn_dropouts[@]}")
                        lrs=("${gcn_lrs[@]}")
                        wds=("${gcn_wds[@]}")
                        hiddens=("${gcn_n_hidden[@]}")
                        layers=("${gcn_n_layers[@]}")
                        label_smoothing="${gcn_label_smoothing}"
                        early_stop_patience="${gcn_early_stop_patience}"
                        ;;
                      "SAGE")
                        dropouts=("${sage_dropouts[@]}")
                        lrs=("${sage_lrs[@]}")
                        wds=("${sage_wds[@]}")
                        hiddens=("${sage_n_hidden[@]}")
                        layers=("${sage_n_layers[@]}")
                        label_smoothing="${sage_label_smoothing}"
                        early_stop_patience="${sage_early_stop_patience}"
                        extra_args=(--aggregator "${sage_aggregator}")
                        ;;
                      "GAT")
                        dropouts=("${gat_dropouts[@]}")
                        lrs=("${gat_lrs[@]}")
                        wds=("${gat_wds[@]}")
                        hiddens=("${gat_n_hidden[@]}")
                        layers=("${gat_n_layers[@]}")
                        label_smoothing="${gat_label_smoothing}"
                        early_stop_patience="${gat_early_stop_patience}"
                        extra_args=(--n-heads "${gat_n_heads}" --attn-drop "${gat_attn_drop}" --edge-drop "${gat_edge_drop}" --no-attn-dst "${gat_no_attn_dst}")
                        ;;
                      *)
                        echo "[Error] Unsupported gnn_model: ${gnn}" >&2
                        exit 1
                        ;;
                    esac
                    for do in "${dropouts[@]}"; do
                      for lr in "${lrs[@]}"; do
                        for wd in "${wds[@]}"; do
                          for h in "${hiddens[@]}"; do
                            for L in "${layers[@]}"; do
                              model_label="${gnn}"
                              run_label="${modality^}-${model_label}-lr${lr}-wd${wd}-h${h}-L${L}-do${do}${variant_suffix}${fuse_suffix}"
                              log_file="${LOG_ROOT}/fg_${FEATURE_GROUP}/${ds}/${run_label}.log"
                              if [[ "${DRY_RUN}" == "true" ]]; then
                                echo "[DRY_RUN] fg=${FEATURE_GROUP} ds=${ds} exp=baseline variant=${variant} early_fuse=${early_fuse} backend=gnn model=${gnn} lr=${lr} wd=${wd} h=${h} L=${L} do=${do}"
                                continue
                              fi
                              if [[ "${SKIP_DONE}" == "true" && -f "${log_file}.done" ]]; then
                                echo "[SKIP] ${run_label} (done) | ${log_file}"
                                continue
                              fi
                              run_cmd "${log_file}" "fg=${FEATURE_GROUP} ds=${ds} ${run_label}" \
                                env WANDB_NAME="${run_label}-${ds}-${FEATURE_GROUP}" \
                                  WANDB_GROUP="MAG/${ds}/${FEATURE_GROUP}/baseline" \
                                  WANDB_TAGS="exp=baseline,variant=${variant},early_fuse=${early_fuse},ds=${ds},fg=${FEATURE_GROUP},backend=gnn,model=${gnn}" \
                                  python GNN/Library/MAG/Early_GNN.py \
                                    --data_name "${ds}" \
                                    --graph_path "${GRAPH_PATH[$ds]}" \
                                    --text_feature "${TEXT_FEAT[$ds]}" \
                                    --visual_feature "${VIS_FEAT[$ds]}" \
                                    --gpu "${GPU_ID}" \
                                    --inductive "${INDUCTIVE}" \
                                    --undirected "${UNDIRECTED}" --selfloop "${SELFLOOP}" \
                                    --metric "${METRIC}" --average "${AVERAGE}" \
                                    "${RESULT_CSV_ARG[@]}" \
                                    "${RESULT_CSV_ALL_ARG[@]}" \
                                    --report_drop_modality "${REPORT_DROP_MODALITY}" \
                                    --report_drop_mode "${REPORT_DROP_MODE}" \
                                    --degrade_alpha "${DEGRADE_ALPHA}" \
                                    --result_tag "${variant_tag}" \
                                    --n-epochs "${N_EPOCHS}" --n-runs "${N_RUNS}" \
                                    --warmup_epochs "${WARMUP_EPOCHS}" --eval_steps "${EVAL_STEPS}" \
                                    --early_stop_patience "${early_stop_patience}" \
                                    --lr "${lr}" --wd "${wd}" \
                                    --n-layers "${L}" --n-hidden "${h}" --dropout "${do}" \
                                    --label-smoothing "${label_smoothing}" \
                                    --train_ratio "${TRAIN_RATIO_BY_DS[${ds}]:-${TRAIN_RATIO}}" --val_ratio "${VAL_RATIO_BY_DS[${ds}]:-${VAL_RATIO}}" \
                                    --backend "gnn" --model_name "${gnn}" \
                                    --early_fuse "${early_fuse}" \
                                    --modality_dropout "${variant_drop}" \
                                    --kd_weight "${KD_WEIGHT}" \
                                    --kd_temperature "${KD_TEMPERATURE}" \
                                    "${DEGRADE_TARGET_ARG[@]}" \
                                    "${DEGRADE_ALPHAS_ARG[@]}" \
                                    "${extra_args[@]}" \
                                    ${MM_PROJ_DIM:+--mm_proj_dim "${MM_PROJ_DIM}"}
                            done
                          done
                        done
                      done
                    done
                  done
                else
                  echo "[Error] Unknown backend: ${backend}" >&2
                  exit 1
                fi
            done
            done
          done
          ;;
        "late")
          for modality in "${EARLY_LATE_MODALITIES_ARR[@]}"; do
            if [[ "${modality}" != "none" ]]; then
              echo "[Error] Late only supports modality=none (got ${modality})." >&2
              exit 1
            fi
            for variant in "${LATE_VARIANTS_ARR[@]}"; do
              variant_drop=0.0
              variant_suffix=""
              case "${variant}" in
                base)
                  ;;
                drop)
                  variant_drop="${LATE_MODALITY_DROPOUT}"
                  variant_suffix="-ModDrop"
                  ;;
                *)
                  echo "[Error] Unknown late variant: ${variant}" >&2
                  exit 1
                  ;;
              esac
              case "${variant}" in
                base) variant_tag="LF-GNN" ;;
                drop) variant_tag="LF-GNN+ModDrop" ;;
              esac
              for gnn in "${GNN_MODELS_ARR[@]}"; do
                extra_args=()
                case "${gnn}" in
                  "GCN")
                    dropouts=("${gcn_dropouts[@]}")
                    lrs=("${gcn_lrs[@]}")
                    wds=("${gcn_wds[@]}")
                    hiddens=("${gcn_n_hidden[@]}")
                    layers=("${gcn_n_layers[@]}")
                    label_smoothing="${gcn_label_smoothing}"
                    early_stop_patience="${gcn_early_stop_patience}"
                    ;;
                  "SAGE")
                    dropouts=("${sage_dropouts[@]}")
                    lrs=("${sage_lrs[@]}")
                    wds=("${sage_wds[@]}")
                    hiddens=("${sage_n_hidden[@]}")
                    layers=("${sage_n_layers[@]}")
                    label_smoothing="${sage_label_smoothing}"
                    early_stop_patience="${sage_early_stop_patience}"
                    extra_args=(--aggregator "${sage_aggregator}")
                    ;;
                  "GAT")
                    dropouts=("${gat_dropouts[@]}")
                    lrs=("${gat_lrs[@]}")
                    wds=("${gat_wds[@]}")
                    hiddens=("${gat_n_hidden[@]}")
                    layers=("${gat_n_layers[@]}")
                    label_smoothing="${gat_label_smoothing}"
                    early_stop_patience="${gat_early_stop_patience}"
                    extra_args=(--n-heads "${gat_n_heads}" --attn-drop "${gat_attn_drop}" --edge-drop "${gat_edge_drop}" --no-attn-dst "${gat_no_attn_dst}")
                    ;;
                  *)
                    echo "[Error] Unsupported gnn_model: ${gnn}" >&2
                    exit 1
                    ;;
                esac
                for do in "${dropouts[@]}"; do
                  for lr in "${lrs[@]}"; do
                    for wd in "${wds[@]}"; do
                      for h in "${hiddens[@]}"; do
                        for L in "${layers[@]}"; do
                          model_label="Late-${gnn}"
                          run_label="${model_label}-lr${lr}-wd${wd}-h${h}-L${L}-do${do}${variant_suffix}"
                          log_file="${LOG_ROOT}/fg_${FEATURE_GROUP}/${ds}/${run_label}.log"
                          if [[ "${DRY_RUN}" == "true" ]]; then
                            echo "[DRY_RUN] fg=${FEATURE_GROUP} ds=${ds} exp=late variant=${variant} model=${gnn} lr=${lr} wd=${wd} h=${h} L=${L} do=${do}"
                            continue
                          fi
                          if [[ "${SKIP_DONE}" == "true" && -f "${log_file}.done" ]]; then
                            echo "[SKIP] ${run_label} (done) | ${log_file}"
                            continue
                          fi
                          run_cmd "${log_file}" "fg=${FEATURE_GROUP} ds=${ds} ${run_label}" \
                            env WANDB_NAME="${run_label}-${ds}-${FEATURE_GROUP}" \
                              WANDB_GROUP="MAG/${ds}/${FEATURE_GROUP}/late" \
                              WANDB_TAGS="exp=late,variant=${variant},ds=${ds},fg=${FEATURE_GROUP},model=${gnn},modality=${modality}" \
                              python GNN/Library/MAG/Late_GNN.py \
                                --data_name "${ds}" \
                                --graph_path "${GRAPH_PATH[$ds]}" \
                                --text_feature "${TEXT_FEAT[$ds]}" \
                                --visual_feature "${VIS_FEAT[$ds]}" \
                                --gpu "${GPU_ID}" \
                                --inductive "${INDUCTIVE}" \
                                --undirected "${UNDIRECTED}" --selfloop "${SELFLOOP}" \
                                --metric "${METRIC}" --average "${AVERAGE}" \
                                "${RESULT_CSV_ARG[@]}" \
                                "${RESULT_CSV_ALL_ARG[@]}" \
                                --report_drop_modality "${REPORT_DROP_MODALITY}" \
                                --report_drop_mode "${REPORT_DROP_MODE}" \
                                --degrade_alpha "${DEGRADE_ALPHA}" \
                                --result_tag "${variant_tag}" \
                                --n-epochs "${N_EPOCHS}" --n-runs "${N_RUNS}" \
                                --warmup_epochs "${WARMUP_EPOCHS}" --eval_steps "${EVAL_STEPS}" \
                                --early_stop_patience "${early_stop_patience}" \
                                --lr "${lr}" --wd "${wd}" \
                                --n-layers "${L}" --n-hidden "${h}" --dropout "${do}" \
                                --label-smoothing "${label_smoothing}" \
                                --train_ratio "${TRAIN_RATIO_BY_DS[${ds}]:-${TRAIN_RATIO}}" --val_ratio "${VAL_RATIO_BY_DS[${ds}]:-${VAL_RATIO}}" \
                                --model_name "${gnn}" \
                                --modality_dropout "${variant_drop}" \
                                --kd_weight "${KD_WEIGHT}" \
                                --kd_temperature "${KD_TEMPERATURE}" \
                                "${DEGRADE_TARGET_ARG[@]}" \
                                "${DEGRADE_ALPHAS_ARG[@]}" \
                                "${extra_args[@]}"
                        done
                      done
                    done
                  done
                done
              done
            done
          done
          ;;
        "tri")
          for modality in "${EARLY_LATE_MODALITIES_ARR[@]}"; do
            if [[ "${modality}" != "none" ]]; then
              echo "[Error] Tri only supports modality=none (got ${modality})." >&2
              exit 1
            fi
            for variant in "${LATE_VARIANTS_ARR[@]}"; do
              variant_drop=0.0
              variant_suffix=""
              case "${variant}" in
                base)
                  ;;
                drop)
                  variant_drop="${LATE_MODALITY_DROPOUT}"
                  variant_suffix="-ModDrop"
                  ;;
                *)
                  echo "[Error] Unknown tri variant: ${variant}" >&2
                  exit 1
                  ;;
              esac
              case "${variant}" in
                base) variant_tag="Tri-GNN" ;;
                drop) variant_tag="Tri-GNN+ModDrop" ;;
              esac
              for gnn in "${GNN_MODELS_ARR[@]}"; do
                extra_args=()
                case "${gnn}" in
                  "GCN")
                    dropouts=("${gcn_dropouts[@]}")
                    lrs=("${gcn_lrs[@]}")
                    wds=("${gcn_wds[@]}")
                    hiddens=("${gcn_n_hidden[@]}")
                    layers=("${gcn_n_layers[@]}")
                    label_smoothing="${gcn_label_smoothing}"
                    early_stop_patience="${gcn_early_stop_patience}"
                    ;;
                  "SAGE")
                    dropouts=("${sage_dropouts[@]}")
                    lrs=("${sage_lrs[@]}")
                    wds=("${sage_wds[@]}")
                    hiddens=("${sage_n_hidden[@]}")
                    layers=("${sage_n_layers[@]}")
                    label_smoothing="${sage_label_smoothing}"
                    early_stop_patience="${sage_early_stop_patience}"
                    extra_args=(--aggregator "${sage_aggregator}")
                    ;;
                  "GAT")
                    dropouts=("${gat_dropouts[@]}")
                    lrs=("${gat_lrs[@]}")
                    wds=("${gat_wds[@]}")
                    hiddens=("${gat_n_hidden[@]}")
                    layers=("${gat_n_layers[@]}")
                    label_smoothing="${gat_label_smoothing}"
                    early_stop_patience="${gat_early_stop_patience}"
                    extra_args=(--n-heads "${gat_n_heads}" --attn-drop "${gat_attn_drop}" --edge-drop "${gat_edge_drop}" --no-attn-dst "${gat_no_attn_dst}")
                    ;;
                  *)
                    echo "[Error] Unsupported gnn_model: ${gnn}" >&2
                    exit 1
                    ;;
                esac
                for do in "${dropouts[@]}"; do
                  for lr in "${lrs[@]}"; do
                    for wd in "${wds[@]}"; do
                      for h in "${hiddens[@]}"; do
                        for L in "${layers[@]}"; do
                          model_label="Tri-${gnn}"
                          run_label="${model_label}-lr${lr}-wd${wd}-h${h}-L${L}-do${do}${variant_suffix}"
                          log_file="${LOG_ROOT}/fg_${FEATURE_GROUP}/${ds}/${run_label}.log"
                          if [[ "${DRY_RUN}" == "true" ]]; then
                            echo "[DRY_RUN] fg=${FEATURE_GROUP} ds=${ds} exp=tri variant=${variant} model=${gnn} lr=${lr} wd=${wd} h=${h} L=${L} do=${do}"
                            continue
                          fi
                          if [[ "${SKIP_DONE}" == "true" && -f "${log_file}.done" ]]; then
                            echo "[SKIP] ${run_label} (done) | ${log_file}"
                            continue
                          fi
                          run_cmd "${log_file}" "fg=${FEATURE_GROUP} ds=${ds} ${run_label}" \
                            env WANDB_NAME="${run_label}-${ds}-${FEATURE_GROUP}" \
                              WANDB_GROUP="MAG/${ds}/${FEATURE_GROUP}/tri" \
                              WANDB_TAGS="exp=tri,variant=${variant},ds=${ds},fg=${FEATURE_GROUP},model=${gnn},modality=${modality}" \
                              python GNN/Library/MAG/Tri_GNN.py \
                                --data_name "${ds}" \
                                --graph_path "${GRAPH_PATH[$ds]}" \
                                --text_feature "${TEXT_FEAT[$ds]}" \
                                --visual_feature "${VIS_FEAT[$ds]}" \
                                --gpu "${GPU_ID}" \
                                --inductive "${INDUCTIVE}" \
                                --undirected "${UNDIRECTED}" --selfloop "${SELFLOOP}" \
                                --metric "${METRIC}" --average "${AVERAGE}" \
                                "${RESULT_CSV_ARG[@]}" \
                                "${RESULT_CSV_ALL_ARG[@]}" \
                                --report_drop_modality "${REPORT_DROP_MODALITY}" \
                                --report_drop_mode "${REPORT_DROP_MODE}" \
                                --degrade_alpha "${DEGRADE_ALPHA}" \
                                --result_tag "${variant_tag}" \
                                --n-epochs "${N_EPOCHS}" --n-runs "${N_RUNS}" \
                                --warmup_epochs "${WARMUP_EPOCHS}" --eval_steps "${EVAL_STEPS}" \
                                --early_stop_patience "${early_stop_patience}" \
                                --lr "${lr}" --wd "${wd}" \
                                --n-layers "${L}" --n-hidden "${h}" --dropout "${do}" \
                                --label-smoothing "${label_smoothing}" \
                                --train_ratio "${TRAIN_RATIO_BY_DS[${ds}]:-${TRAIN_RATIO}}" --val_ratio "${VAL_RATIO_BY_DS[${ds}]:-${VAL_RATIO}}" \
                                --model_name "${gnn}" \
                                --modality_dropout "${variant_drop}" \
                                --kd_weight "${KD_WEIGHT}" \
                                --kd_temperature "${KD_TEMPERATURE}" \
                                "${DEGRADE_TARGET_ARG[@]}" \
                                "${DEGRADE_ALPHAS_ARG[@]}" \
                                "${extra_args[@]}"
                        done
                      done
                    done
                  done
                done
              done
            done
          done
          ;;
        "ogm")
          for backend in "${OGM_BACKENDS_ARR[@]}"; do
            if [[ "${backend}" == "mlp" ]]; then
              for do in "${mlp_dropouts[@]}"; do
                for lr in "${mlp_lrs[@]}"; do
                  for wd in "${mlp_wds[@]}"; do
                    for h in "${mlp_n_hidden[@]}"; do
                      for L in "${mlp_n_layers[@]}"; do
                        for om in "${ogm_modulations[@]}"; do
                          for opd in "${ogm_projection_dims[@]}"; do
                            for os in "${ogm_modulation_starts[@]}"; do
                              for oe in "${ogm_modulation_ends[@]}"; do
                                for oa in "${ogm_alphas[@]}"; do
                                  model_label="MLP"
                                  run_label="OGMMLP-lr${lr}-wd${wd}-h${h}-L${L}-do${do}-mod${om}-p${opd}-ms${os}-me${oe}-a${oa}"
                                  log_file="${LOG_ROOT}/fg_${FEATURE_GROUP}/${ds}/${run_label}.log"
                                  if [[ "${DRY_RUN}" == "true" ]]; then
                                    echo "[DRY_RUN] fg=${FEATURE_GROUP} ds=${ds} exp=ogm backend=mlp model=MLP lr=${lr} wd=${wd} h=${h} L=${L} do=${do} mod=${om} p=${opd} ms=${os} me=${oe} a=${oa}"
                                    continue
                                  fi
                                  if [[ "${SKIP_DONE}" == "true" && -f "${log_file}.done" ]]; then
                                    echo "[SKIP] ${run_label} (done) | ${log_file}"
                                    continue
                                  fi
                                  run_cmd "${log_file}" "fg=${FEATURE_GROUP} ds=${ds} ${run_label}" \
                                    env WANDB_NAME="${run_label}-${ds}-${FEATURE_GROUP}" \
                                      WANDB_GROUP="MAG/${ds}/${FEATURE_GROUP}/ogm" \
                                      WANDB_TAGS="exp=ogm,ds=${ds},fg=${FEATURE_GROUP},backend=mlp,model=MLP" \
                                      python GNN/Library/MAG/OGM_GNN.py \
                                        --data_name "${ds}" \
                                        --graph_path "${GRAPH_PATH[$ds]}" \
                                        --text_feature "${TEXT_FEAT[$ds]}" \
                                        --visual_feature "${VIS_FEAT[$ds]}" \
                                        --gpu "${GPU_ID}" \
                                        --inductive "${INDUCTIVE}" \
                                        --undirected "${UNDIRECTED}" --selfloop "${SELFLOOP}" \
                                        --metric "${METRIC}" --average "${AVERAGE}" \
                                        "${RESULT_CSV_ARG[@]}" \
                                        "${RESULT_CSV_ALL_ARG[@]}" \
                                      --report_drop_modality "${REPORT_DROP_MODALITY}" \
                                      --report_drop_mode "${REPORT_DROP_MODE}" \
                                      --degrade_alpha "${DEGRADE_ALPHA}" \
                                      --n-epochs "${N_EPOCHS}" --n-runs "${N_RUNS}" \
                                        --warmup_epochs "${WARMUP_EPOCHS}" --eval_steps "${EVAL_STEPS}" \
                                        --early_stop_patience "${mlp_early_stop_patience}" \
                                        --lr "${lr}" --wd "${wd}" \
                                        --n-layers "${L}" --n-hidden "${h}" --dropout "${do}" \
                                        --label-smoothing "${mlp_label_smoothing}" \
                                        --train_ratio "${TRAIN_RATIO_BY_DS[${ds}]:-${TRAIN_RATIO}}" --val_ratio "${VAL_RATIO_BY_DS[${ds}]:-${VAL_RATIO}}" \
                                        --backend "mlp" --model_name "MLP" \
                                        --result_tag "OGM-MLP" \
                                        --ogm_modulation "${om}" \
                                        --ogm_projection_dim "${opd}" \
                                        --ogm_modulation_starts "${os}" \
                                        --ogm_modulation_ends "${oe}" \
                                        --ogm_alpha "${oa}"
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
            elif [[ "${backend}" == "gnn" ]]; then
              for gnn in "${GNN_MODELS_ARR[@]}"; do
                extra_args=()
                case "${gnn}" in
                  "GCN")
                    dropouts=("${gcn_dropouts[@]}")
                    lrs=("${gcn_lrs[@]}")
                    wds=("${gcn_wds[@]}")
                    hiddens=("${gcn_n_hidden[@]}")
                    layers=("${gcn_n_layers[@]}")
                    label_smoothing="${gcn_label_smoothing}"
                    early_stop_patience="${gcn_early_stop_patience}"
                    ;;
                  "SAGE")
                    dropouts=("${sage_dropouts[@]}")
                    lrs=("${sage_lrs[@]}")
                    wds=("${sage_wds[@]}")
                    hiddens=("${sage_n_hidden[@]}")
                    layers=("${sage_n_layers[@]}")
                    label_smoothing="${sage_label_smoothing}"
                    early_stop_patience="${sage_early_stop_patience}"
                    extra_args=(--aggregator "${sage_aggregator}")
                    ;;
                  "GAT")
                    dropouts=("${gat_dropouts[@]}")
                    lrs=("${gat_lrs[@]}")
                    wds=("${gat_wds[@]}")
                    hiddens=("${gat_n_hidden[@]}")
                    layers=("${gat_n_layers[@]}")
                    label_smoothing="${gat_label_smoothing}"
                    early_stop_patience="${gat_early_stop_patience}"
                    extra_args=(--n-heads "${gat_n_heads}" --attn-drop "${gat_attn_drop}" --edge-drop "${gat_edge_drop}" --no-attn-dst "${gat_no_attn_dst}")
                    ;;
                  *)
                    echo "[Error] Unsupported gnn_model: ${gnn}" >&2
                    exit 1
                    ;;
                esac
                for do in "${dropouts[@]}"; do
                  for lr in "${lrs[@]}"; do
                    for wd in "${wds[@]}"; do
                      for h in "${hiddens[@]}"; do
                        for L in "${layers[@]}"; do
                          for om in "${ogm_modulations[@]}"; do
                            for opd in "${ogm_projection_dims[@]}"; do
                              for os in "${ogm_modulation_starts[@]}"; do
                                for oe in "${ogm_modulation_ends[@]}"; do
                                  for oa in "${ogm_alphas[@]}"; do
                                    run_label="OGMGNN-${gnn}-lr${lr}-wd${wd}-h${h}-L${L}-do${do}-mod${om}-p${opd}-ms${os}-me${oe}-a${oa}"
                                    log_file="${LOG_ROOT}/fg_${FEATURE_GROUP}/${ds}/${run_label}.log"
                                    if [[ "${DRY_RUN}" == "true" ]]; then
                                      echo "[DRY_RUN] fg=${FEATURE_GROUP} ds=${ds} exp=ogm backend=gnn model=${gnn} lr=${lr} wd=${wd} h=${h} L=${L} do=${do} mod=${om} p=${opd} ms=${os} me=${oe} a=${oa}"
                                      continue
                                    fi
                                    if [[ "${SKIP_DONE}" == "true" && -f "${log_file}.done" ]]; then
                                      echo "[SKIP] ${run_label} (done) | ${log_file}"
                                      continue
                                    fi
                                    run_cmd "${log_file}" "fg=${FEATURE_GROUP} ds=${ds} ${run_label}" \
                                      env WANDB_NAME="${run_label}-${ds}-${FEATURE_GROUP}" \
                                        WANDB_GROUP="MAG/${ds}/${FEATURE_GROUP}/ogm" \
                                        WANDB_TAGS="exp=ogm,ds=${ds},fg=${FEATURE_GROUP},backend=gnn,model=${gnn}" \
                                        python GNN/Library/MAG/OGM_GNN.py \
                                          --data_name "${ds}" \
                                          --graph_path "${GRAPH_PATH[$ds]}" \
                                          --text_feature "${TEXT_FEAT[$ds]}" \
                                          --visual_feature "${VIS_FEAT[$ds]}" \
                                          --gpu "${GPU_ID}" \
                                          --inductive "${INDUCTIVE}" \
                                          --undirected "${UNDIRECTED}" --selfloop "${SELFLOOP}" \
                                          --metric "${METRIC}" --average "${AVERAGE}" \
                                          "${RESULT_CSV_ARG[@]}" \
                                          "${RESULT_CSV_ALL_ARG[@]}" \
                                          --report_drop_modality "${REPORT_DROP_MODALITY}" \
                                          --report_drop_mode "${REPORT_DROP_MODE}" \
                                          --degrade_alpha "${DEGRADE_ALPHA}" \
                                          --n-epochs "${N_EPOCHS}" --n-runs "${N_RUNS}" \
                                          --warmup_epochs "${WARMUP_EPOCHS}" --eval_steps "${EVAL_STEPS}" \
                                          --early_stop_patience "${early_stop_patience}" \
                                          --lr "${lr}" --wd "${wd}" \
                                          --n-layers "${L}" --n-hidden "${h}" --dropout "${do}" \
                                          --label-smoothing "${label_smoothing}" \
                                          --train_ratio "${TRAIN_RATIO_BY_DS[${ds}]:-${TRAIN_RATIO}}" --val_ratio "${VAL_RATIO_BY_DS[${ds}]:-${VAL_RATIO}}" \
                                          --backend "gnn" --model_name "${gnn}" \
                                          --result_tag "OGM-GNN" \
                                          --ogm_modulation "${om}" \
                                          --ogm_projection_dim "${opd}" \
                                          --ogm_modulation_starts "${os}" \
                                          --ogm_modulation_ends "${oe}" \
                                          --ogm_alpha "${oa}" \
                                          "${DEGRADE_TARGET_ARG[@]}" \
                                          "${DEGRADE_ALPHAS_ARG[@]}" \
                                          "${extra_args[@]}"
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
            else
              echo "[Error] Unknown backend: ${backend}" >&2
              exit 1
            fi
          done
          ;;
        *)
          echo "[Error] Unknown experiment: ${exp}" >&2
          exit 1
          ;;
      esac
    done
  done
done

echo "All runs finished. Logs root: ${LOG_ROOT}"
