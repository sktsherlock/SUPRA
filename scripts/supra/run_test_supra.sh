#!/usr/bin/env bash
set -euo pipefail

export DGLBACKEND=pytorch

# ==========================================
# 环境变量配置
# ==========================================
GPU_ID=${GPU_ID:-0}
DRY_RUN=${DRY_RUN:-false}                   # 如果设为 true, 只打印命令不执行

# 目标数据集 (仅保留 Reddit-M)
DATASETS="Reddit-M"
DATA_ROOT="/mnt/input/MAGB_Dataset"

# ---------------- Feature group definition ----------------
declare -A TEXT_FEATURE_BY_DS_GROUP
declare -A VIS_FEATURE_BY_DS_GROUP

# Default group (Llama 3.2)
TEXT_FEATURE_BY_DS_GROUP["Reddit-M|default"]='TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-M|default"]='ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy'

# CLIP + RoBERTa group
TEXT_FEATURE_BY_DS_GROUP["Reddit-M|clip_roberta"]='TextFeature/RedditM_roberta_base_100_mean.npy'
VIS_FEATURE_BY_DS_GROUP["Reddit-M|clip_roberta"]='ImageFeature/RedditM_openai_clip-vit-large-patch14.npy'

# ==========================================
# 实验执行流程
# ==========================================

echo "=========================================================="
echo "RUNNING SUPRA EXPERIMENTS ON REDDIT-M"
echo "GPU ID        : ${GPU_ID}"
echo "=========================================================="

# 遍历两种特征组
for FEATURE_GROUP in default clip_roberta; do
    # 遍历两种指标
    for METRIC in accuracy f1; do
        if [ "$METRIC" == "accuracy" ]; then
            out_metric="acc"
        else
            out_metric="f1"
        fi

        # 四种组合下的独立 CSV 保存路径
        RESULT_CSV="results_pid_test_${FEATURE_GROUP}_${out_metric}.csv"
        
        # 独立的日志目录，防止文件覆写
        LOG_DIR="logs_supra_test/${FEATURE_GROUP}/${out_metric}"
        mkdir -p "${LOG_DIR}/${DATASETS}"

        echo "----------------------------------------------------------"
        echo "Feature Group : ${FEATURE_GROUP}"
        echo "Metric        : ${METRIC}"
        echo "Output CSV    : ${RESULT_CSV}"
        echo "----------------------------------------------------------"

        for ds in ${DATASETS}; do
            # 提取特征路径
            GRAPH="${DATA_ROOT}/${ds}/${ds//-/}Graph.pt"
            TEXT_FEAT="${DATA_ROOT}/${ds}/${TEXT_FEATURE_BY_DS_GROUP["${ds}|${FEATURE_GROUP}"]}"
            VIS_FEAT="${DATA_ROOT}/${ds}/${VIS_FEATURE_BY_DS_GROUP["${ds}|${FEATURE_GROUP}"]}"
            
            COMMON_ARGS=(
                --data_name "${ds}"
                --graph_path "${GRAPH}"
                --text_feature "${TEXT_FEAT}"
                --visual_feature "${VIS_FEAT}"
                --gpu "${GPU_ID}"
                --metric "${METRIC}"
                --inductive false
                --undirected true
                --selfloop true
                --average macro
                --n-epochs 1000
                --n-runs 3
                --warmup_epochs 0
                --eval_steps 1
                --early_stop_patience 50
                --wd 0.0
                --n-hidden 256
                --dropout 0.2
                --label-smoothing 0.1
                --unique_depth 0
                --shared_depth 3
                --disable_wandb
                --log-every 1
                --result_csv "${RESULT_CSV}"
            )

            # 遍历三种 Backbone
            for gnn in GCN SAGE GAT; do
                # 决定层数搜索空间 (SAGE 固定 3 层，GCN/GAT 搜索 1-2 层)
                if [ "$gnn" == "SAGE" ]; then
                    layers=(3)
                else
                    layers=(1 2)
                fi
                
                # 遍历两种学习率
                for lr in 0.0005 0.001; do
                    for L in "${layers[@]}"; do
                        run_label="${gnn}_L${L}_lr${lr}"
                        log_file="${LOG_DIR}/${ds}/${run_label}.log"
                        
                        echo "[$(date '+%H:%M:%S')] Model=${gnn} | L=${L} | lr=${lr}"
                        
                        # 配置模型特定的参数
                        extra_args=()
                        if [ "$gnn" == "SAGE" ]; then
                            extra_args+=(--aggregator mean)
                        elif [ "$gnn" == "GAT" ]; then
                            extra_args+=(--n-heads 3 --attn-drop 0.0 --edge-drop 0.0 --no-attn-dst true)
                        fi

                        if [ "${DRY_RUN}" == "true" ]; then
                            echo "DRY RUN: python GNN/Library/MAG/SUPRA.py ${COMMON_ARGS[*]} --model_name ${gnn} --n-layers ${L} --lr ${lr} ${extra_args[*]}"
                        else
                            python GNN/Library/MAG/SUPRA.py \
                                "${COMMON_ARGS[@]}" \
                                --model_name "${gnn}" \
                                --n-layers "${L}" \
                                --lr "${lr}" \
                                "${extra_args[@]}" > "${log_file}" 2>&1
                        fi
                    done
                done
            done
        done
    done
done

echo "All done! Experiments finished. Results saved to the 4 CSV files."