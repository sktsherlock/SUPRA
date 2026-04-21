#!/usr/bin/env bash
set -euo pipefail

export DGLBACKEND=${DGLBACKEND:-pytorch}
cd "$(dirname "$0")"

# 数据集根目录
DATA_ROOT=${DATA_ROOT:-/openbayes/input/input0/MAGB_Dataset}

# 定义各个数据集的特征路径 (Llama)
declare -A TEXT_FEATURE_BY_DS
declare -A VIS_FEATURE_BY_DS

TEXT_FEATURE_BY_DS["Movies"]='TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy'
VIS_FEATURE_BY_DS["Movies"]='ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS["Toys"]='TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy'
VIS_FEATURE_BY_DS["Toys"]='ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS["Grocery"]='TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy'
VIS_FEATURE_BY_DS["Grocery"]='ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy'

TEXT_FEATURE_BY_DS["Reddit-M"]='TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy'
VIS_FEATURE_BY_DS["Reddit-M"]='ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy'

DATASETS=("Movies" "Toys" "Grocery" "Reddit-M")

GPU_ID=${GPU_ID:-2}
N_RUNS=3
LRS=("0.0005" "0.001" "0.002")

LOG_DIR="logs_ntsformer_mini_a_full"
mkdir -p "${LOG_DIR}"

echo "Logs will be saved in ${LOG_DIR}"

for ds in "${DATASETS[@]}"; do
    # Graph 文件名的前缀会去掉连字符，如 RedditMGraph.pt
    graph_prefix="${ds//-/}"
    graph_path="${DATA_ROOT}/${ds}/${graph_prefix}Graph.pt"
    text_feat="${DATA_ROOT}/${ds}/${TEXT_FEATURE_BY_DS["${ds}"]}"
    vis_feat="${DATA_ROOT}/${ds}/${VIS_FEATURE_BY_DS["${ds}"]}"
    
    for metric in accuracy f1; do
        result_csv="a_nts_full_${metric}.csv"
        
        for lr in "${LRS[@]}"; do
            echo "=========================================================="
            echo "Running NTSFormer_mini on ${ds} [metric=${metric}] with lr=${lr}"
            echo "=========================================================="
            
            log_file="${LOG_DIR}/${ds}_${metric}_lr${lr}.log"
            echo "Log at: ${log_file}"
            
            # 由于规定了参数按NTSFormer内部默认，将不显式传递 n-layers, n-hidden等模型结构参数
            python GNN/Library/MAG/NTSFormer_mini.py \
                --data_name "${ds}" \
                --graph_path "${graph_path}" \
                --text_feature "${text_feat}" \
                --visual_feature "${vis_feat}" \
                --gpu "${GPU_ID}" \
                --inductive false \
                --undirected true \
                --selfloop true \
                --metric "${metric}" \
                --average macro \
                --n-runs "${N_RUNS}" \
                --result_csv "${result_csv}" \
                --disable_wandb \
                --lr "${lr}" \
                --log-every 1 > "${log_file}" 2>&1
                
        done
    done
done

echo "All NTSFormer full experiments finished!"