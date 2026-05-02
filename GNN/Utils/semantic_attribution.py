"""
Semantic Attribution Analysis — Inference & Visualization
======================================================

此脚本仅负责：
1. 从 .pt 文件加载各模型在测试集上的预测结果
2. 按节点集合（Shared / T-Unique / V-Unique / Hard）分解准确率
3. 绘制堆叠柱状图

训练和各模型预测导出需使用各自的主训练脚本（见 docs/semantic_attribution_workflow.md）

用法:
  python -m GNN.Utils.semantic_attribution \
      --data_name Reddit-M \
      --pred_dir Results/attribution/Reddit-M/ \
      --result_csv Results/attribution/Reddit-M/summary.csv \
      --save_plot Results/attribution/Reddit-M/attribution.png
"""

import argparse
import csv
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from GNN.GraphData import load_data


# ---------------------------------------------------------------------------
# Core attribution logic
# ---------------------------------------------------------------------------

def classify_nodes(
    preds_text: th.Tensor,
    preds_image: th.Tensor,
    labels: th.Tensor,
) -> Dict[str, th.Tensor]:
    """
    根据单模态 MLP 预测将测试节点分为 4 个互斥集合。
    """
    correct_t = preds_text == labels
    correct_v = preds_image == labels
    shared   = correct_t & correct_v
    t_unique = correct_t & (~correct_v)
    v_unique = (~correct_t) & correct_v
    hard     = ~correct_t & ~correct_v
    return dict(shared=shared, t_unique=t_unique, v_unique=v_unique, hard=hard)


def compute_contributions(
    preds: th.Tensor,
    node_sets: Dict[str, th.Tensor],
    labels: th.Tensor,
    total_nodes: int,
) -> Dict[str, float]:
    """
    计算每个节点集合上的绝对准确率贡献（= 该集合中正确预测的节点数 / 总节点数）。
    """
    correct = preds == labels
    return {
        "shared":   (correct & node_sets["shared"]).sum().item() / total_nodes,
        "t_unique": (correct & node_sets["t_unique"]).sum().item() / total_nodes,
        "v_unique": (correct & node_sets["v_unique"]).sum().item() / total_nodes,
        "hard":     (correct & node_sets["hard"]).sum().item() / total_nodes,
        "total":    correct.sum().item() / total_nodes,
    }


# ---------------------------------------------------------------------------
# Model prediction loading
# ---------------------------------------------------------------------------

def load_predictions(pred_dir: str) -> Dict[str, th.Tensor]:
    """
    从 pred_dir 加载所有模型的测试集预测。

    期望文件命名:
      text_mlp_test_pred.pt
      image_mlp_test_pred.pt
      late_gnn_gcn_test_pred.pt
      late_gnn_gat_test_pred.pt
      early_gnn_gcn_test_pred.pt
      supra_test_pred.pt
      ntsformer_test_pred.pt
      mig_gt_test_pred.pt

    每个文件保存的是测试节点上的 argmax 预测向量 (LongTensor, shape=[N_test])
    """
    mapping = {
        "text_mlp":      "text_mlp_test_pred.pt",
        "image_mlp":     "image_mlp_test_pred.pt",
        "late_gnn_gcn":  "late_gnn_gcn_test_pred.pt",
        "late_gnn_gat":  "late_gnn_gat_test_pred.pt",
        "early_gnn_gcn": "early_gnn_gcn_test_pred.pt",
        "supra":         "supra_test_pred.pt",
        "ntsformer":     "ntsformer_test_pred.pt",
        "mig_gt":        "mig_gt_test_pred.pt",
    }

    preds = {}
    for key, fname in mapping.items():
        fpath = os.path.join(pred_dir, fname)
        if os.path.exists(fpath):
            preds[key] = th.load(fpath).cpu()
            print(f"  [Loaded] {key}: {preds[key].shape}")
        else:
            print(f"  [Skip]   {key} ({fname} not found)")
    return preds


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_stacked_bar(
    results: Dict[str, Dict[str, float]],
    set_sizes: Dict[str, float],
    data_name: str,
    save_path: Optional[str] = None,
):
    """
    绘制堆叠柱状图。

    X轴: 模型名称
    Y轴: 绝对准确率贡献（各堆叠块之和 = 模型总准确率）
    颜色:
      Shared(灰)      — 文本和视觉 MLP 都预测正确的节点
      Text-Unique(蓝) — 仅文本 MLP 预测正确
      Visual-Unique(橙)— 仅视觉 MLP 预测正确
      Hard/绿        — 两者都错的节点
    """
    # 固定顺序：MGCN/MGAT 替代 Late_GNN-GCN/GAT，移除 Early_GNN-GCN，SUPRA 放最后
    ordered = [
        "Text MLP",
        "Image MLP",
        "MGCN",
        "MGAT",
        "NTSFormer",
        "MIG-GT",
        "SUPRA",
    ]
    models = [m for m in ordered if m in results]
    n_models = len(models)

    # 参考图配色：蓝/橙/灰/黄
    colors = {
        "shared":   "#A6A6A6",   # gray
        "t_unique": "#4472B4",   # blue
        "v_unique": "#ED7D31",   # orange
        "hard":     "#FFC000",   # yellow
    }
    labels_text = {
        "shared":   "Shared Semantics",
        "t_unique": "Text-Unique",
        "v_unique": "Visual-Unique",
        "hard":     "Synergy",
    }

    bar_width = 0.6
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.6), 6))

    bottom = np.zeros(n_models)
    for set_key in ["shared", "t_unique", "v_unique", "hard"]:
        heights = [results[m][set_key] for m in models]
        ax.bar(x, heights, bar_width, bottom=bottom,
               label=labels_text[set_key], color=colors[set_key],
               edgecolor="white", linewidth=0.5)
        bottom += np.array(heights)

    # Y 轴从 Shared 最低分附近开始，留足顶部空间给总准确率标注
    shared_min = min(results[m]["shared"] for m in models)
    y_lo = max(0, shared_min - 0.06)
    y_top = bottom.max()
    ax.set_ylim(y_lo, y_top * 1.15)

    # 顶部标注总准确率
    for i, m in enumerate(models):
        ax.text(i, bottom[i] + y_top * 0.015, f"{results[m]['total']:.1%}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy Contribution", fontsize=11)
    ax.set_title(f"Semantic Attribution Analysis — {data_name}", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 底部标注节点集合分布
    for i, (k, v) in enumerate(set_sizes.items()):
        ax.text(i, y_lo - 0.01, f"{labels_text[k]}: {v:.1%}",
                ha="center", va="top", fontsize=7, color=colors[k],
                transform=ax.get_xaxis_transform())

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 保存为 PDF 矢量格式
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
        print(f"\n[Saved] Plot → {pdf_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_attribution(args):
    # Load data to get labels and test_idx
    graph, labels, train_idx, val_idx, test_idx = load_data(
        args.graph_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        name=args.data_name,
    )
    labels_test = labels[test_idx]
    n_test = test_idx.shape[0]

    print(f"\n[Data] {args.data_name}: {n_test} test nodes")
    print(f"[Predictions] Loading from {args.pred_dir}")

    preds = load_predictions(args.pred_dir)
    if "text_mlp" not in preds or "image_mlp" not in preds:
        raise ValueError("text_mlp and image_mlp predictions are required")

    # Classify nodes
    node_sets = classify_nodes(preds["text_mlp"], preds["image_mlp"], labels_test)
    set_sizes = {k: v.sum().item() / n_test for k, v in node_sets.items()}
    print(f"\n[Node Sets]")
    for k, v in set_sizes.items():
        print(f"  {k:10s}: {v:.1%}  ({node_sets[k].sum().item():.0f} nodes)")

    # Compute contributions
    results = {}
    print(f"\n[Results]")
    print(f"{'Model':<22} {'Shared':>8} {'T-Unique':>9} {'V-Unique':>9} {'Hard':>8} {'Total':>8}")
    print("-" * 70)
    # display_name → prediction file key（late_gnn_gcn/gat 映射到 MGCN/MGAT）
    display_to_key = {
        "Text MLP":   "text_mlp",
        "Image MLP":  "image_mlp",
        "MGCN":       "late_gnn_gcn",
        "MGAT":       "late_gnn_gat",
        "NTSFormer":  "ntsformer",
        "MIG-GT":     "mig_gt",
        "SUPRA":      "supra",
    }
    ordered = [
        "Text MLP", "Image MLP", "MGCN", "MGAT", "NTSFormer", "MIG-GT", "SUPRA",
    ]
    for name in ordered:
        key = display_to_key.get(name, name)
        if key not in preds:
            print(f"{name:<22}  [SKIP — prediction file not found]")
            continue
        c = compute_contributions(preds[key], node_sets, labels_test, n_test)
        results[name] = c
        print(f"{name:<22} {c['shared']:>8.4f} {c['t_unique']:>9.4f} "
              f"{c['v_unique']:>9.4f} {c['hard']:>8.4f} {c['total']:>8.4f}")
    print("-" * 70)

    # Save CSV
    if args.result_csv:
        os.makedirs(os.path.dirname(args.result_csv), exist_ok=True)
        with open(args.result_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Shared", "T-Unique", "V-Unique", "Hard", "Total"])
            for name, c in results.items():
                writer.writerow([name, f"{c['shared']:.4f}", f"{c['t_unique']:.4f}",
                               f"{c['v_unique']:.4f}", f"{c['hard']:.4f}", f"{c['total']:.4f}"])
        print(f"\n[Saved] CSV → {args.result_csv}")

    # Plot
    if args.plot:
        plot_stacked_bar(results, set_sizes, args.data_name, args.save_plot)

    return results


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Attribution — Inference & Visualization")
    parser.add_argument("--data_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--graph_path", type=str, required=True, help="Path to graph (.pt)")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing *_test_pred.pt prediction files")
    parser.add_argument("--result_csv", type=str, default=None)
    parser.add_argument("--save_plot", type=str, default=None)
    parser.add_argument("--plot", type=lambda x: x.lower() in ("true", "1", "yes"), default=True)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    args = parser.parse_args()
    run_attribution(args)
