#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Baseline Experiment Script
# Runs: GCN, GraphSAGE, GAT, RevGAT, MLP, SGC, APPNP, GCNII, JKNet
# =============================================================================
# Usage:
#   ./run_baseline.sh --model GCN --data_name Movies
#   DATA_ROOT=/custom/path ./run_baseline.sh --model GAT --data_name Grocery
#
# Environment variables to override:
#   DATA_ROOT       - Data root path (default: /hyperai/input/input0/MAGB_Dataset)
#   GPU_ID          - GPU device ID (default: 0)
#   N_RUNS          - Number of runs (default: 3)
# =============================================================================

# Load centralized path config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/plot/path_config.sh"

# ---------------- User-editable: paths ----------------
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

# =============================================================================
# Default parameters (shared across methods for fair comparison)
# =============================================================================
GPU_ID=${GPU_ID:-0}
N_RUNS=${N_RUNS:-3}
N_EPOCHS=${N_EPOCHS:-1000}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-50}
EVAL_STEPS=${EVAL_STEPS:-1}
METRIC=${METRIC:-accuracy}
AVERAGE=${AVERAGE:-macro}
SELFLOOP=${SELFLOOP:-true}
UNDIRECTED=${UNDIRECTED:-true}
TRAIN_RATIO=${TRAIN_RATIO:-0.6}
VAL_RATIO=${VAL_RATIO:-0.2}
INDUCTIVE=${INDUCTIVE:-false}

# =============================================================================
# Per-model parameter grids (tight ranges from run_mag_baseline_suite.sh)
# =============================================================================

# ---------------- MLP sweep ----------------
mlp_dropout="0.3"
mlp_lrs=("0.0005" "0.001")
mlp_wds=("1e-4")
mlp_n_hidden=("256")
mlp_n_layers=("2" "3")
mlp_label_smoothing="0.1"
mlp_early_stop_patience="50"

# ---------------- GCN sweep ----------------
gcn_dropout="0.3"
gcn_lrs=("0.0005" "0.001")
gcn_wds=("1e-4")
gcn_n_hidden=("256")
GCN_LAYERS=${GCN_LAYERS:-"1 2"}
read -r -a gcn_n_layers <<< "${GCN_LAYERS}"
gcn_label_smoothing="0.1"
gcn_early_stop_patience="50"

# ---------------- GraphSAGE sweep ----------------
sage_dropout="0.3"
sage_lrs=("0.0005" "0.001")
sage_wds=("1e-4")
sage_n_hidden=("256")
SAGE_LAYERS=${SAGE_LAYERS:-"2 3 4"}
read -r -a sage_n_layers <<< "${SAGE_LAYERS}"
sage_label_smoothing="0.1"
sage_early_stop_patience="50"
sage_aggregator="mean"

# ---------------- GAT sweep ----------------
gat_dropout="0.3"
gat_lrs=("0.0005" "0.001")
gat_wds=("1e-4")
gat_n_hidden=("256")
GAT_LAYERS=${GAT_LAYERS:-"1 2"}
read -r -a gat_n_layers <<< "${GAT_LAYERS}"
gat_label_smoothing="0.1"
gat_early_stop_patience="25"
gat_n_heads=4
gat_attn_drop=0.0
gat_edge_drop=0.0
gat_no_attn_dst=true

# ---------------- GCNII sweep ----------------
gcnii_dropout="0.3"
gcnii_lrs=("0.0005" "0.001")
gcnii_wds=("1e-4")
gcnii_n_hidden=("256")
GCNII_LAYERS=${GCNII_LAYERS:-"2 3 4"}
read -r -a gcnii_n_layers <<< "${GCNII_LAYERS}"
gcnii_label_smoothing="0.1"
gcnii_early_stop_patience="50"
gcnii_lamda="0.5"
gcnii_alpha="0.5"
gcnii_variant=false

# ---------------- JKNet sweep ----------------
jknet_dropout="0.3"
jknet_lrs=("0.0005" "0.001")
jknet_wds=("1e-4")
jknet_n_hidden=("256")
JKNET_LAYERS=${JKNET_LAYERS:-"2 3 4"}
read -r -a jknet_n_layers <<< "${JKNET_LAYERS}"
jknet_label_smoothing="0.1"
jknet_early_stop_patience="50"
jknet_aggr="concat"


# =============================================================================
# Parse command line arguments
# =============================================================================
show_help() {
  cat << EOF
Usage: $0 --model MODEL --data_name DATASET [options]

Required:
  --model MODEL       GCN, SAGE, GAT, RevGAT, MLP, SGC, APPNP, GCNII, JKNet
  --data_name DATASET  Movies, Grocery, Reddit-M, Toys

Optional:
  --gpu ID            GPU device ID (default: 0)
  --n_runs N          Number of runs (default: 3)
  --feature_group NAME  clip_roberta, default (default: clip_roberta)
  --selfloop          Enable self-loop (default: true for GCN/GCNII/JKNet)
  --no-selfloop       Disable self-loop (default: false)
  --undirected        Make graph undirected (default: true)
  --output_dir DIR    Output directory (default: logs_baseline)

Examples:
  $0 --model GCN --data_name Movies
  $0 --model GAT --data_name Grocery --gpu 1
  $0 --model MLP --data_name Reddit-M --n_runs 5
EOF
}

MODEL=""
DATA_NAME=""
FEATURE_GROUP="clip_roberta"
OUTPUT_DIR=${OUTPUT_DIR:-logs_baseline}
RESULT_CSV=${RESULT_CSV:-results_csv/baseline_best.csv}
RESULT_CSV_ALL=${RESULT_CSV_ALL:-results_csv/baseline_all.csv}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --data_name) DATA_NAME="$2"; shift 2 ;;
    --feature_group) FEATURE_GROUP="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    --n_runs) N_RUNS="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --result_csv) RESULT_CSV="$2"; shift 2 ;;
    --result_csv_all) RESULT_CSV_ALL="$2"; shift 2 ;;
    --selfloop) SELFLOOP="true"; shift ;;
    --no-selfloop) SELFLOOP="false"; shift ;;
    --undirected) UNDIRECTED="true"; shift ;;
    --no-undirected) UNDIRECTED="false"; shift ;;
    --help) show_help; exit 0 ;;
    *) echo "Unknown option: $1"; show_help; exit 1 ;;
  esac
done

if [[ -z "${MODEL}" || -z "${DATA_NAME}" ]]; then
  echo "Error: --model and --data_name are required"
  show_help
  exit 1
fi

# =============================================================================
# Helper functions
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

  echo "[${label}] log=${log_file}"

  local start_s=${SECONDS}
  ( "$@" ) >> "${log_file}" 2>&1
  local rc=$?
  local dur_s=$((SECONDS - start_s))

  if [[ "${rc}" -ne 0 ]]; then
    echo "[FAIL] ${label} (exit=${rc}, ${dur_s}s)" >&2
    return "${rc}"
  fi

  echo "[OK] ${label} (${dur_s}s)"
  return 0
}

# =============================================================================
# Configure dataset paths
# =============================================================================
dataset_config "${DATA_NAME}" "${FEATURE_GROUP}"

echo ">>> Dataset: ${DATA_NAME}, Feature Group: ${FEATURE_GROUP}"
echo ">>> Model: ${MODEL}, Self-loop: ${SELFLOOP}, Undirected: ${UNDIRECTED}"
echo ">>> Graph: ${GRAPH_PATH}"
echo ">>> Text: ${TEXT_FEAT}"
echo ">>> Visual: ${VIS_FEAT}"

# =============================================================================
# Run based on model type
# =============================================================================
case "${MODEL}" in
  "MLP")
    for lr in "${mlp_lrs[@]}"; do
      for wd in "${mlp_wds[@]}"; do
        for h in "${mlp_n_hidden[@]}"; do
          for L in "${mlp_n_layers[@]}"; do
            label="MLP-lr${lr}-wd${wd}-h${h}-L${L}-do${mlp_dropout}"
            run_model "${label}" \
              python GNN/Library/MLP.py \
                --data_name "${DATA_NAME}" \
                --graph_path "${GRAPH_PATH}" \
                --feature "${TEXT_FEAT}" \
                --gpu "${GPU_ID}" \
                --n-runs "${N_RUNS}" \
                --n-epochs "${N_EPOCHS}" \
                --warmup_epochs "${WARMUP_EPOCHS}" \
                --eval_steps "${EVAL_STEPS}" \
                --early_stop_patience "${mlp_early_stop_patience}" \
                --lr "${lr}" --wd "${wd}" \
                --n-layers "${L}" --n-hidden "${h}" --dropout "${mlp_dropout}" \
                --label-smoothing "${mlp_label_smoothing}" \
                --metric "${METRIC}" --average "${AVERAGE}" \
                --train_ratio "${TRAIN_RATIO}" --val_ratio "${VAL_RATIO}" \
                --inductive "${INDUCTIVE}" \
                --disable_wandb
          done
        done
      done
    done
    ;;

  "GCN")
    SELFLOOP="true"  # GCN always uses self-loop
    for lr in "${gcn_lrs[@]}"; do
      for wd in "${gcn_wds[@]}"; do
        for h in "${gcn_n_hidden[@]}"; do
          for L in "${gcn_n_layers[@]}"; do
            label="GCN-lr${lr}-wd${wd}-h${h}-L${L}-do${gcn_dropout}"
            run_model "${label}" \
              python GNN/Baselines/Early_GNN.py \
                --data_name "${DATA_NAME}" \
                --graph_path "${GRAPH_PATH}" \
                --text_feature "${TEXT_FEAT}" \
                --visual_feature "${VIS_FEAT}" \
                --gpu "${GPU_ID}" \
                --n-runs "${N_RUNS}" \
                --n-epochs "${N_EPOCHS}" \
                --warmup_epochs "${WARMUP_EPOCHS}" \
                --eval_steps "${EVAL_STEPS}" \
                --early_stop_patience "${gcn_early_stop_patience}" \
                --lr "${lr}" --wd "${wd}" \
                --n-layers "${L}" --n-hidden "${h}" --dropout "${gcn_dropout}" \
                --label-smoothing "${gcn_label_smoothing}" \
                --metric "${METRIC}" --average "${AVERAGE}" \
                --train_ratio "${TRAIN_RATIO}" --val_ratio "${VAL_RATIO}" \
                --undirected "${UNDIRECTED}" --selfloop "${SELFLOOP}" \
                --inductive "${INDUCTIVE}" \
                --backend gnn --model_name GCN --early_fuse concat \
                --disable_wandb \
                --result_csv "${RESULT_CSV}" \
                --result_csv_all "${RESULT_CSV_ALL}"
          done
        done
      done
    done
    ;;

  "SAGE")
    SELFLOOP="false"  # SAGE does not use self-loop
    for lr in "${sage_lrs[@]}"; do
      for wd in "${sage_wds[@]}"; do
        for h in "${sage_n_hidden[@]}"; do
          for L in "${sage_n_layers[@]}"; do
            label="SAGE-lr${lr}-wd${wd}-h${h}-L${L}-do${sage_dropout}"
            run_model "${label}" \
              python GNN/Baselines/Early_GNN.py \
                --data_name "${DATA_NAME}" \
                --graph_path "${GRAPH_PATH}" \
                --text_feature "${TEXT_FEAT}" \
                --visual_feature "${VIS_FEAT}" \
                --gpu "${GPU_ID}" \
                --n-runs "${N_RUNS}" \
                --n-epochs "${N_EPOCHS}" \
                --warmup_epochs "${WARMUP_EPOCHS}" \
                --eval_steps "${EVAL_STEPS}" \
                --early_stop_patience "${sage_early_stop_patience}" \
                --lr "${lr}" --wd "${wd}" \
                --n-layers "${L}" --n-hidden "${h}" --dropout "${sage_dropout}" \
                --label-smoothing "${sage_label_smoothing}" \
                --metric "${METRIC}" --average "${AVERAGE}" \
                --train_ratio "${TRAIN_RATIO}" --val_ratio "${VAL_RATIO}" \
                --undirected "${UNDIRECTED}" --selfloop "${SELFLOOP}" \
                --inductive "${INDUCTIVE}" \
                --backend gnn --model_name SAGE --early_fuse concat \
                --aggregator "${sage_aggregator}" \
                --disable_wandb \
                --result_csv "${RESULT_CSV}" \
                --result_csv_all "${RESULT_CSV_ALL}"
          done
        done
      done
    done
    ;;

  "GAT")
    SELFLOOP="false"  # GAT does not use self-loop
    for lr in "${gat_lrs[@]}"; do
      for wd in "${gat_wds[@]}"; do
        for h in "${gat_n_hidden[@]}"; do
          for L in "${gat_n_layers[@]}"; do
            label="GAT-lr${lr}-wd${wd}-h${h}-L${L}-do${gat_dropout}"
            run_model "${label}" \
              python GNN/Baselines/Early_GNN.py \
                --data_name "${DATA_NAME}" \
                --graph_path "${GRAPH_PATH}" \
                --text_feature "${TEXT_FEAT}" \
                --visual_feature "${VIS_FEAT}" \
                --gpu "${GPU_ID}" \
                --n-runs "${N_RUNS}" \
                --n-epochs "${N_EPOCHS}" \
                --warmup_epochs "${WARMUP_EPOCHS}" \
                --eval_steps "${EVAL_STEPS}" \
                --early_stop_patience "${gat_early_stop_patience}" \
                --lr "${lr}" --wd "${wd}" \
                --n-layers "${L}" --n-hidden "${h}" --dropout "${gat_dropout}" \
                --label-smoothing "${gat_label_smoothing}" \
                --metric "${METRIC}" --average "${AVERAGE}" \
                --train_ratio "${TRAIN_RATIO}" --val_ratio "${VAL_RATIO}" \
                --undirected "${UNDIRECTED}" --selfloop "${SELFLOOP}" \
                --inductive "${INDUCTIVE}" \
                --backend gnn --model_name GAT --early_fuse concat \
                --n-heads "${gat_n_heads}" --attn-drop "${gat_attn_drop}" --edge-drop "${gat_edge_drop}" \
                --disable_wandb \
                --result_csv "${RESULT_CSV}" \
                --result_csv_all "${RESULT_CSV_ALL}"
          done
        done
      done
    done
    ;;

  "GCNII")
    SELFLOOP="true"  # GCNII uses self-loop
    for lr in "${gcnii_lrs[@]}"; do
      for wd in "${gcnii_wds[@]}"; do
        for h in "${gcnii_n_hidden[@]}"; do
          for L in "${gcnii_n_layers[@]}"; do
            label="GCNII-lr${lr}-wd${wd}-h${h}-L${L}-do${gcnii_dropout}"
            run_model "${label}" \
              python -c "
import sys; sys.path.insert(0, 'GNN/Library')
from GNN.Baselines.Early_GNN import Early_GNN as mag_base
import torch as th
from GNN.GraphData import load_data
from GNN.Utils.NodeClassification import classification
import argparse, numpy as np

parser = argparse.ArgumentParser('GCNII Config')
from GNN.Utils.model_config import add_common_args; add_common_args(parser)
args = parser.parse_args()

device = th.device('cuda:%d' % args.gpu if th.cuda.is_available() else 'cpu')
graph, labels, train_idx, val_idx, test_idx = load_data(args.graph_path, train_ratio=args.train_ratio, val_ratio=args.val_ratio, name=args.data_name)

text_feat = th.from_numpy(np.load(args.text_feature).astype(np.float32)).to(device)
vis_feat = th.from_numpy(np.load(args.visual_feature).astype(np.float32)).to(device)

if args.undirected:
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)
if args.selfloop:
    graph = graph.remove_self_loop().add_self_loop()

graph.create_formats_()
train_idx, val_idx, test_idx = train_idx.to(device), val_idx.to(device), test_idx.to(device)
labels = labels.to(device)
graph = graph.to(device)

n_classes = int((labels.max() + 1).item())
feat = th.cat([text_feat, vis_feat], dim=1)

model = mag_base(text_in_dim=feat.shape[1], vis_in_dim=0, n_classes=n_classes,
                 n_layers=${L}, n_hidden=${h}, dropout=${gcnii_dropout},
                 lr=${lr}, wd=${wd}, jknet_aggr=None,
                 gcnii_lamda=${gcnii_lamda}, gcnii_alpha=${gcnii_alpha}, gcnii_variant=${gcnii_variant},
                 backend='gnn', model_name='GCNII', early_fuse='concat',
                 aggregator=None, n_heads=1, attn_drop=0.0, edge_drop=0.0,
                 use_symmetric_norm=False).to(device)

classification(args, graph, graph, model, feat, labels, train_idx, val_idx, test_idx, 1)
" \
                --data_name "${DATA_NAME}" \
                --graph_path "${GRAPH_PATH}" \
                --text_feature "${TEXT_FEAT}" \
                --visual_feature "${VIS_FEAT}" \
                --gpu "${GPU_ID}" \
                --n-runs "${N_RUNS}" \
                --n-epochs "${N_EPOCHS}" \
                --warmup_epochs "${WARMUP_EPOCHS}" \
                --eval_steps "${EVAL_STEPS}" \
                --early_stop_patience "${gcnii_early_stop_patience}" \
                --lr "${lr}" --wd "${wd}" \
                --n-layers "${L}" --n-hidden "${h}" --dropout "${gcnii_dropout}" \
                --label-smoothing "${gcnii_label_smoothing}" \
                --metric "${METRIC}" --average "${AVERAGE}" \
                --train_ratio "${TRAIN_RATIO}" --val_ratio "${VAL_RATIO}" \
                --undirected "${UNDIRECTED}" --selfloop "${SELFLOOP}" \
                --inductive "${INDUCTIVE}" \
                --disable_wandb \
                --result_csv "${RESULT_CSV}" \
                --result_csv_all "${RESULT_CSV_ALL}"
          done
        done
      done
    done
    ;;

  "JKNet")
    SELFLOOP="true"  # JKNet uses self-loop
    for lr in "${jknet_lrs[@]}"; do
      for wd in "${jknet_wds[@]}"; do
        for h in "${jknet_n_hidden[@]}"; do
          for L in "${jknet_n_layers[@]}"; do
            label="JKNet-lr${lr}-wd${wd}-h${h}-L${L}-do${jknet_dropout}"
            run_model "${label}" \
              python -c "
import sys; sys.path.insert(0, 'GNN/Library')
from GNN.Baselines.Early_GNN import Early_GNN as mag_base
import torch as th
from GNN.GraphData import load_data
from GNN.Utils.NodeClassification import classification
import argparse, numpy as np

parser = argparse.ArgumentParser('JKNet Config')
from GNN.Utils.model_config import add_common_args; add_common_args(parser)
args = parser.parse_args()

device = th.device('cuda:%d' % args.gpu if th.cuda.is_available() else 'cpu')
graph, labels, train_idx, val_idx, test_idx = load_data(args.graph_path, train_ratio=args.train_ratio, val_ratio=args.val_ratio, name=args.data_name)

text_feat = th.from_numpy(np.load(args.text_feature).astype(np.float32)).to(device)
vis_feat = th.from_numpy(np.load(args.visual_feature).astype(np.float32)).to(device)

if args.undirected:
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)
if args.selfloop:
    graph = graph.remove_self_loop().add_self_loop()

graph.create_formats_()
train_idx, val_idx, test_idx = train_idx.to(device), val_idx.to(device), test_idx.to(device)
labels = labels.to(device)
graph = graph.to(device)

n_classes = int((labels.max() + 1).item())
feat = th.cat([text_feat, vis_feat], dim=1)

model = mag_base(text_in_dim=feat.shape[1], vis_in_dim=0, n_classes=n_classes,
                 n_layers=${L}, n_hidden=${h}, dropout=${jknet_dropout},
                 lr=${lr}, wd=${wd}, jknet_aggr='${jknet_aggr}',
                 gcnii_lamda=None, gcnii_alpha=None, gcnii_variant=False,
                 backend='gnn', model_name='JKNet', early_fuse='concat',
                 aggregator=None, n_heads=1, attn_drop=0.0, edge_drop=0.0,
                 use_symmetric_norm=False).to(device)

classification(args, graph, graph, model, feat, labels, train_idx, val_idx, test_idx, 1)
" \
                --data_name "${DATA_NAME}" \
                --graph_path "${GRAPH_PATH}" \
                --text_feature "${TEXT_FEAT}" \
                --visual_feature "${VIS_FEAT}" \
                --gpu "${GPU_ID}" \
                --n-runs "${N_RUNS}" \
                --n-epochs "${N_EPOCHS}" \
                --warmup_epochs "${WARMUP_EPOCHS}" \
                --eval_steps "${EVAL_STEPS}" \
                --early_stop_patience "${jknet_early_stop_patience}" \
                --lr "${lr}" --wd "${wd}" \
                --n-layers "${L}" --n-hidden "${h}" --dropout "${jknet_dropout}" \
                --label-smoothing "${jknet_label_smoothing}" \
                --metric "${METRIC}" --average "${AVERAGE}" \
                --train_ratio "${TRAIN_RATIO}" --val_ratio "${VAL_RATIO}" \
                --undirected "${UNDIRECTED}" --selfloop "${SELFLOOP}" \
                --inductive "${INDUCTIVE}" \
                --disable_wandb \
                --result_csv "${RESULT_CSV}" \
                --result_csv_all "${RESULT_CSV_ALL}"
          done
        done
      done
    done
    ;;

  *)
    echo "Error: Unknown model '${MODEL}'"
    echo "Supported: MLP, GCN, SAGE, GAT, GCNII, JKNet"
    exit 1
    ;;
esac

echo ">>> All ${MODEL} runs completed. Logs: ${OUTPUT_DIR}/${DATA_NAME}/"