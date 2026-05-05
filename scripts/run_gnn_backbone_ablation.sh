#!/bin/bash
# =====================================================================
# GNN Backbone Ablation Experiment — Batch Runner
# =====================================================================
# Runs all 36 accuracy + 36 F1 experiments for:
#   Group 1: Early_GNN (GAT/SAGE/JKNet × 4 datasets)
#   Group 2: SUPRA    (GAT/SAGE/JKNet × 4 datasets × aux={0.0,0.5})
#
# Usage:
#   # Run accuracy experiments only
#   bash scripts/run_gnn_backbone_ablation.sh accuracy
#
#   # Run F1 experiments only
#   bash scripts/run_gnn_backbone_ablation.sh f1
#
#   # Run both
#   bash scripts/run_gnn_backbone_ablation.sh all
#
#   # Dry run (print commands without executing)
#   DRY_RUN=1 bash scripts/run_gnn_backbone_ablation.sh all
#
# Output: Results/ablation/*.csv
# =====================================================================

set -euo pipefail

MODE="${1:-all}"   # accuracy | f1 | all
DRY_RUN="${DRY_RUN:-0}"

GPU="${GPU:-0}"
N_RUNS=3
SEED=42
N_EPOCHS=1000
PATIENCE=30

# Shared hyperparameters
EMBED_DIM=256
N_HIDDEN=256
DROPOUT=0.3
WD=0.0001
LABEL_SMOOTHING=0.1
TRAIN_RATIO=0.6
VAL_RATIO=0.2

OUTDIR="Results/ablation"
mkdir -p "$OUTDIR"

# -----------------------------------------------------------------------
# Dataset definitions: name | text_path | vis_path | graph_path | lr | n_layers
# -----------------------------------------------------------------------
declare -A DATASETS
DATASETS["Reddit-M"]="/mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy"
DATASETS["Reddit-M"]+="|/mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy"
DATASETS["Reddit-M"]+="|/mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt"
DATASETS["Reddit-M"]+="|0.0005|3"

DATASETS["Movies"]="/mnt/input/MAGB_Dataset/Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy"
DATASETS["Movies"]+="|/mnt/input/MAGB_Dataset/Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy"
DATASETS["Movies"]+="|/mnt/input/MAGB_Dataset/Movies/MoviesGraph.pt"
DATASETS["Movies"]+="|0.001|3"

DATASETS["Grocery"]="/mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy"
DATASETS["Grocery"]+="|/mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy"
DATASETS["Grocery"]+="|/mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt"
DATASETS["Grocery"]+="|0.001|3"

DATASETS["Toys"]="/mnt/input/MAGB_Dataset/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy"
DATASETS["Toys"]+="|/mnt/input/MAGB_Dataset/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy"
DATASETS["Toys"]+="|/mnt/input/MAGB_Dataset/Toys/ToysGraph.pt"
DATASETS["Toys"]+="|0.0005|2"

# -----------------------------------------------------------------------
# Run a single experiment
# -----------------------------------------------------------------------
run_exp() {
    local label="$1"
    shift
    local cmd=("$@")
    local outfile="${OUTDIR}/${label}.csv"

    if [[ -f "$outfile" ]]; then
        echo "[SKIP] ${label} (already exists)"
        return 0
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY] ${cmd[*]}"
        return 0
    fi

    echo "[RUN ] ${label}"
    local start
    start=$(date +%s)
    if "${cmd[@]}" 2>&1 | tail -n 20; then
        local end dur
        end=$(date +%s)
        dur=$(( end - start ))
        echo "[DONE] ${label} — ${dur}s"
    else
        echo "[FAIL] ${label}"
        return 1
    fi
}

# -----------------------------------------------------------------------
# Group 1: Early_GNN
# -----------------------------------------------------------------------
run_early_gnn() {
    local gnn="$1"       # GAT | SAGE | JKNet
    local ds="$2"        # dataset name
    local metric_val="${3:-}" # f1_macro or empty

    IFS='|' read -r txt vis grh lr n_layers <<< "${DATASETS[$ds]}"

    local extra_gnn_args=""
    case "$gnn" in
        GAT)  extra_gnn_args="--n_heads 4 --attn_drop 0.0 --edge_drop 0.0" ;;
        SAGE) extra_gnn_args="--aggregator mean" ;;
        JKNet) extra_gnn_args="--jknet_aggr concat" ;;
    esac

    local metric_file_tag=""
    [[ -n "$metric_val" ]] && metric_file_tag="_f1macro"

    local label="early_gnn_$(echo $gnn | tr '[:upper:]' '[:lower:]')_${ds}${metric_file_tag}"

    local cmd=(
        python -m GNN.Baselines.Early_GNN
        --data_name "$ds"
        --text_feature "$txt"
        --visual_feature "$vis"
        --graph_path "$grh"
        --backend gnn
        --model_name "$gnn"
        --early_no_encoder True
        --n-hidden "$N_HIDDEN"
        --n-layers "$n_layers"
        --dropout "$DROPOUT"
        --lr "$lr"
        --wd "$WD"
        --n-runs "$N_RUNS"
        --seed "$SEED"
        --n-epochs "$N_EPOCHS"
        --early_stop_patience "$PATIENCE"
        --selfloop False
        --result_csv "${OUTDIR}/${label}.csv"
        --result_csv_all "${OUTDIR}/${label}_all.csv"
        --disable_wandb
        --gpu "$GPU"
    )

    [[ -n "$metric_val" ]] && cmd+=(--metric "$metric_val")

    run_exp "$label" "${cmd[@]}"
}

# -----------------------------------------------------------------------
# Group 2: SUPRA
# -----------------------------------------------------------------------
run_supra() {
    local gnn="$1"       # GAT | SAGE | JKNet
    local ds="$2"        # dataset name
    local aux="$3"        # 0.0 or 0.5
    local metric_val="${4:-}" # f1_macro or empty

    IFS='|' read -r txt vis grh lr n_layers <<< "${DATASETS[$ds]}"

    local extra_gnn_args=""
    case "$gnn" in
        GAT)  extra_gnn_args="--n_heads 4 --attn_drop 0.0 --edge_drop 0.0" ;;
        SAGE) extra_gnn_args="--aggregator mean" ;;
        JKNet) extra_gnn_args="--jknet_aggr concat" ;;
    esac

    local metric_file_tag=""
    [[ -n "$metric_val" ]] && metric_file_tag="_f1macro"

    local aux_tag="${aux}"  # e.g. "0.0" or "0.5"
    local label="supra_$(echo $gnn | tr '[:upper:]' '[:lower:]')_${ds}_aux${aux_tag}${metric_file_tag}"

    local cmd=(
        python -m GNN.SUPRA
        --data_name "$ds"
        --text_feature "$txt"
        --visual_feature "$vis"
        --graph_path "$grh"
        --model_name "$gnn"
        --embed_dim "$EMBED_DIM"
        --n-layers "$n_layers"
        --n-hidden "$N_HIDDEN"
        --dropout "$DROPOUT"
        --lr "$lr"
        --wd "$WD"
        --aux_weight "$aux"
        --mlp_variant ablate
        --n-runs "$N_RUNS"
        --seed "$SEED"
        --n-epochs "$N_EPOCHS"
        --early_stop_patience "$PATIENCE"
        --selfloop False
        --result_csv "${OUTDIR}/${label}.csv"
        --result_csv_all "${OUTDIR}/${label}_all.csv"
        --disable_wandb
        --gpu "$GPU"
    )

    [[ -n "$metric_val" ]] && cmd+=(--metric "$metric_val")

    run_exp "$label" "${cmd[@]}"
}

# -----------------------------------------------------------------------
# Execute
# -----------------------------------------------------------------------
echo "============================================================"
echo "GNN Backbone Ablation — Mode: $MODE | GPU: $GPU"
echo "============================================================"

run_all() {
    # ----------------------------------------------------------
    # Group 1: Early_GNN
    # ----------------------------------------------------------
    for gnn in GAT SAGE JKNet; do
        for ds in Reddit-M Movies Grocery Toys; do
            [[ "$MODE" == "accuracy" || "$MODE" == "all" ]] && \
                run_early_gnn "$gnn" "$ds"
            [[ "$MODE" == "f1" || "$MODE" == "all" ]] && \
                run_early_gnn "$gnn" "$ds" "f1_macro"
        done
    done

    # ----------------------------------------------------------
    # Group 2: SUPRA
    # ----------------------------------------------------------
    for gnn in GAT SAGE JKNet; do
        for ds in Reddit-M Movies Grocery Toys; do
            for aux in 0.0 0.5; do
                [[ "$MODE" == "accuracy" || "$MODE" == "all" ]] && \
                    run_supra "$gnn" "$ds" "$aux"
                [[ "$MODE" == "f1" || "$MODE" == "all" ]] && \
                    run_supra "$gnn" "$ds" "$aux" "f1_macro"
            done
        done
    done
}

run_all

echo ""
echo "============================================================"
echo "All done. Results in $OUTDIR/"
echo "============================================================"
ls -la "$OUTDIR"/
