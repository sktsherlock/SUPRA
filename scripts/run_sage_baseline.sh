#!/bin/bash
# =====================================================================
# GraphSAGE Baseline Experiments (Early_GNN + Late_GNN)
# =====================================================================
# Runs SAGE across:
#   - 4 datasets: Movies, Grocery, Toys, Reddit-M
#   - 2 feature groups: clip_roberta (RoBERTa+CLIP), default (Llama)
#   - 2 metrics: accuracy, f1_macro
#   - Hyperparameter grid: layers∈{2,3,4}, lr∈{0.0005,0.001}
#   - n_runs=3 per configuration
#
# Output: Results/0506/*.csv
#   *_best.csv   — aggregated best result across all runs/hyperparams
#   *_all.csv    — per-run results (for mean±std computation)
#
# Usage:
#   bash scripts/run_sage_baseline.sh
#   GPU=1 bash scripts/run_sage_baseline.sh   # use GPU 1
# =====================================================================

set -uo pipefail

GPU="${GPU:-0}"
N_RUNS=3
SEED=42
N_EPOCHS=1000
PATIENCE=50
DROPOUT=0.3
WD=0.0001
LABEL_SMOOTHING=0.1
TRAIN_RATIO=0.6
VAL_RATIO=0.2
METRICS="accuracy f1_macro"
FEATURE_GROUPS="clip_roberta default"
DATASETS="Movies Grocery Toys Reddit-M"

OUTDIR="Results/0506"
mkdir -p "$OUTDIR"

# -----------------------------------------------------------------------
# Dataset paths
# -----------------------------------------------------------------------
declare -A TEXT_PATH
declare -A VIS_PATH
declare -A GRAPH_PATH

# clip_roberta
TEXT_PATH["Movies|clip_roberta"]="/mnt/input/MAGB_Dataset/Movies/TextFeature/Movies_roberta_base_512_mean.npy"
VIS_PATH["Movies|clip_roberta"]="/mnt/input/MAGB_Dataset/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy"
TEXT_PATH["Grocery|clip_roberta"]="/mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_roberta_base_256_mean.npy"
VIS_PATH["Grocery|clip_roberta"]="/mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_openai_clip-vit-large-patch14.npy"
TEXT_PATH["Toys|clip_roberta"]="/mnt/input/MAGB_Dataset/Toys/TextFeature/Toys_roberta_base_512_mean.npy"
VIS_PATH["Toys|clip_roberta"]="/mnt/input/MAGB_Dataset/Toys/ImageFeature/Toys_openai_clip-vit-large-patch14.npy"
TEXT_PATH["Reddit-M|clip_roberta"]="/mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_roberta_base_100_mean.npy"
VIS_PATH["Reddit-M|clip_roberta"]="/mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_openai_clip-vit-large-patch14.npy"

# default (Llama)
TEXT_PATH["Movies|default"]="/mnt/input/MAGB_Dataset/Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy"
VIS_PATH["Movies|default"]="/mnt/input/MAGB_Dataset/Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy"
TEXT_PATH["Grocery|default"]="/mnt/input/MAGB_Dataset/Grocery/TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy"
VIS_PATH["Grocery|default"]="/mnt/input/MAGB_Dataset/Grocery/ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy"
TEXT_PATH["Toys|default"]="/mnt/input/MAGB_Dataset/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy"
VIS_PATH["Toys|default"]="/mnt/input/MAGB_Dataset/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy"
TEXT_PATH["Reddit-M|default"]="/mnt/input/MAGB_Dataset/Reddit-M/TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy"
VIS_PATH["Reddit-M|default"]="/mnt/input/MAGB_Dataset/Reddit-M/ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy"

GRAPH_PATH["Movies"]="/mnt/input/MAGB_Dataset/Movies/MoviesGraph.pt"
GRAPH_PATH["Grocery"]="/mnt/input/MAGB_Dataset/Grocery/GroceryGraph.pt"
GRAPH_PATH["Toys"]="/mnt/input/MAGB_Dataset/Toys/ToysGraph.pt"
GRAPH_PATH["Reddit-M"]="/mnt/input/MAGB_Dataset/Reddit-M/RedditMGraph.pt"

# -----------------------------------------------------------------------
# Run a single experiment config; skip if result exists
# -----------------------------------------------------------------------
run_config() {
    local label="$1"
    shift
    local cmd=("$@")
    local csv="${OUTDIR}/${label}.csv"

    if [[ -f "$csv" ]]; then
        echo "[SKIP] ${label}"
        return 0
    fi

    echo "[RUN ] ${label}"
    local start_sec=$SECONDS
    if "$@" 2>&1 | tail -n 5; then
        echo "[DONE] ${label} — $(( SECONDS - start_sec ))s"
    else
        echo "[FAIL] ${label}"
        return 1
    fi
}

# -----------------------------------------------------------------------
# Early_GNN SAGE
# -----------------------------------------------------------------------
run_early_sage() {
    local ds="$1"
    local fg="$2"
    local metric="$3"
    local n_layers="$4"
    local lr="$5"

    local txt="${TEXT_PATH[${ds}|${fg}]}"
    local vis="${VIS_PATH[${ds}|${fg}]}"
    local grh="${GRAPH_PATH[$ds]}"

    local mtag=""
    [[ "$metric" == "f1_macro" ]] && mtag="_f1macro"

    local label="early_sage_${ds}_${fg}_L${n_layers}_lr${lr}${mtag}"
    local csv="${OUTDIR}/${label}.csv"
    local all_csv="${OUTDIR}/${label}_all.csv"

    run_config "$label" \
        python -m GNN.Baselines.Early_GNN \
        --data_name "$ds" \
        --text_feature "$txt" \
        --visual_feature "$vis" \
        --graph_path "$grh" \
        --backend gnn \
        --model_name SAGE \
        --early_no_encoder True \
        --n-hidden 256 \
        --n-layers "$n_layers" \
        --dropout "$DROPOUT" \
        --lr "$lr" \
        --wd "$WD" \
        --n-runs "$N_RUNS" \
        --seed "$SEED" \
        --n-epochs "$N_EPOCHS" \
        --early_stop_patience "$PATIENCE" \
        --selfloop False \
        --result_csv "$csv" \
        --result_csv_all "$all_csv" \
        --disable_wandb \
        --gpu "$GPU" \
        --metric "$metric" \
        --label-smoothing "$LABEL_SMOOTHING" \
        --train_ratio "$TRAIN_RATIO" --val_ratio "$VAL_RATIO" \
        --aggregator mean
}

# -----------------------------------------------------------------------
# Late_GNN SAGE
# -----------------------------------------------------------------------
run_late_sage() {
    local ds="$1"
    local fg="$2"
    local metric="$3"
    local n_layers="$4"
    local lr="$5"

    local txt="${TEXT_PATH[${ds}|${fg}]}"
    local vis="${VIS_PATH[${ds}|${fg}]}"
    local grh="${GRAPH_PATH[$ds]}"

    local mtag=""
    [[ "$metric" == "f1_macro" ]] && mtag="_f1macro"

    local label="late_sage_${ds}_${fg}_L${n_layers}_lr${lr}${mtag}"
    local csv="${OUTDIR}/${label}.csv"
    local all_csv="${OUTDIR}/${label}_all.csv"

    run_config "$label" \
        python -m GNN.Baselines.Late_GNN \
        --data_name "$ds" \
        --text_feature "$txt" \
        --visual_feature "$vis" \
        --graph_path "$grh" \
        --model_name SAGE \
        --n-hidden 256 \
        --n-layers "$n_layers" \
        --dropout "$DROPOUT" \
        --lr "$lr" \
        --wd "$WD" \
        --n-runs "$N_RUNS" \
        --seed "$SEED" \
        --n-epochs "$N_EPOCHS" \
        --early_stop_patience "$PATIENCE" \
        --selfloop False \
        --result_csv "$csv" \
        --result_csv_all "$all_csv" \
        --disable_wandb \
        --gpu "$GPU" \
        --metric "$metric" \
        --label-smoothing "$LABEL_SMOOTHING" \
        --train_ratio "$TRAIN_RATIO" --val_ratio "$VAL_RATIO" \
        --aggregator mean
}

# -----------------------------------------------------------------------
# Collect best result across all hyperparameter configs for a given
# (model, dataset, feature_group, metric) combination
# -----------------------------------------------------------------------
collect_best() {
    local model="$1"   # early or late
    local ds="$2"
    local fg="$3"
    local metric="$4"

    local prefix="${model}_sage_${ds}_${fg}"
    local mtag=""
    [[ "$metric" == "f1_macro" ]] && mtag="_f1macro"

    # Find all CSV files matching this combination
    local best_score=-999
    local best_all_csv=""
    local best_tag=""

    for csv in "${OUTDIR}"/${prefix}_L*_lr*${mtag}.csv; do
        [[ -f "$csv" ]] || continue
        # Extract score from full column (format: "75.123")
        score_line=$(cut -d',' -f3 "$csv" | grep -v "^full$" | grep -v "^$" | head -1)
        score=$(echo "$score_line" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        if [[ -n "$score" ]]; then
            # Use python for reliable float comparison
            cmp=$(python3 -c "print(1 if float('$score') > float('$best_score') else 0)" 2>/dev/null || echo 0)
            if [[ "$cmp" -eq 1 ]]; then
                best_score=$score
                # Extract hyperparam tag from filename, e.g. "L2_lr0.0005" from "early_sage_Movies_default_L2_lr0.0005_all.csv"
                cfg=$(basename "$csv" .csv | sed "s/${mtag}//" | sed "s/${prefix}_//")
                best_all_csv="${OUTDIR}/${prefix}_${cfg}${mtag}_all.csv"
                best_tag="$cfg"
            fi
        fi
    done

    if [[ -n "$best_all_csv" && -f "$best_all_csv" ]]; then
        cp "$best_all_csv" "${OUTDIR}/${prefix}${mtag}_best_all.csv"
        echo "[BEST] ${prefix}${mtag}: cfg=${best_tag} score=${best_score}"
    else
        echo "[WARN] No CSV found for ${prefix}${mtag}"
    fi
}

# -----------------------------------------------------------------------
# Execute
# -----------------------------------------------------------------------
echo "============================================================"
echo "SAGE Baseline Experiments"
echo "Datasets:   $DATASETS"
echo "FG:         $FEATURE_GROUPS"
echo "Metrics:    $METRICS"
echo "GPU:        $GPU"
echo "Output:     $OUTDIR"
echo "============================================================"

for metric in $METRICS; do
    for ds in $DATASETS; do
        for fg in $FEATURE_GROUPS; do
            echo ""
            echo ">>> $ds / $fg / $metric"

            # Run all hyperparameter configs
            for n_layers in 2 3 4; do
                for lr in 0.0005 0.001; do
                    run_early_sage "$ds" "$fg" "$metric" "$n_layers" "$lr"
                    run_late_sage  "$ds" "$fg" "$metric" "$n_layers" "$lr"
                done
            done

            # Collect best across all configs
            collect_best "early" "$ds" "$fg" "$metric"
            collect_best "late"  "$ds" "$fg" "$metric"
        done
    done
done

echo ""
echo "============================================================"
echo "All done. Results in $OUTDIR/"
echo "============================================================"
ls -la "$OUTDIR"/

echo ""
echo "Total files: $(ls "$OUTDIR" | wc -l)"
echo "Config count per (ds,fg,metric): $(ls "$OUTDIR"/early_sage_*_L2_lr0.0005*.csv 2>/dev/null | wc -l) configs found"
