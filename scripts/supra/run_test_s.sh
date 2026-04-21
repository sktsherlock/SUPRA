#!/usr/bin/env bash
set -euo pipefail

export DGLBACKEND=pytorch

declare -A TEXT_FEATURE_BY_DS_GROUP
declare -A VIS_FEATURE_BY_DS_GROUP

TEXT_FEATURE_BY_DS_GROUP["Movies|default"]='TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Movies|default"]='ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Grocery|default"]='TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Grocery|default"]='ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Toys|default"]='TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Toys|default"]='ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Reddit-M|default"]='TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-M|default"]='ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS_GROUP["Movies|clip_roberta"]='TextFeature/Movies_roberta_base_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Movies|clip_roberta"]='ImageFeature/Movies_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Grocery|clip_roberta"]='TextFeature/Grocery_roberta_base_256_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Grocery|clip_roberta"]='ImageFeature/Grocery_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Toys|clip_roberta"]='TextFeature/Toys_roberta_base_512_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Toys|clip_roberta"]='ImageFeature/Toys_openai_clip-vit-large-patch14.npy'

TEXT_FEATURE_BY_DS_GROUP["Reddit-M|clip_roberta"]='TextFeature/RedditM_roberta_base_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-M|clip_roberta"]='ImageFeature/RedditM_openai_clip-vit-large-patch14.npy'

# ==========================================
# 用法: ./run.sh [数据集名] [特征组名]
# ==========================================
DS=${1:-"Reddit-M"}               # 第一个参数，默认为 Reddit-M
FEAT_GROUP=${2:-"default"}        # 第二个参数，默认为 default
GPU_ID=${GPU_ID:-0}               # 支持通过环境变量传入 GPU_ID，比如 GPU_ID=1 ./run.sh

DATA_ROOT="/mnt/input/MAGB_Dataset"
LOOKUP_KEY="${DS}|${FEAT_GROUP}"

GRAPH="${DATA_ROOT}/${DS}/${DS//-/}Graph.pt"
TEXT_FEAT="${DATA_ROOT}/${DS}/${TEXT_FEATURE_BY_DS_GROUP[$LOOKUP_KEY]}"
VIS_FEAT="${DATA_ROOT}/${DS}/${VIS_FEATURE_BY_DS_GROUP[$LOOKUP_KEY]}"

echo "==========================================="
echo "数据集:    ${DS}"
echo "特征组:    ${FEAT_GROUP}"
echo "GPU ID:    ${GPU_ID}"
echo "==========================================="

# 通用基准参数
COMMON_ARGS=(
    --data_name "${DS}"
    --graph_path "${GRAPH}"
    --text_feature "${TEXT_FEAT}"
    --visual_feature "${VIS_FEAT}"
    --gpu "${GPU_ID}"
    --metric "accuracy"
    --undirected true
    --n-epochs 1000
    --n-runs 1
    --eval_steps 10
    --early_stop_patience 50
    --wd 0.0
    --lr 0.001
    --n-hidden 256
    --model_name "GCN"
    --disable_wandb
    --n-layers 1
    --shared_depth 1
)

echo ">> 运行 SUPRA.py..."
python GNN/Library/MAG/SUPRA.py "${COMMON_ARGS[@]}" --selfloop true

echo -e "\n>> 运行 SUPRA.py..."
python GNN/Library/MAG/SUPRA.py "${COMMON_ARGS[@]}" --selfloop false

