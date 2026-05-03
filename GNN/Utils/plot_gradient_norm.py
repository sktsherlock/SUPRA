"""
Gradient Starvation Verification — Per-Epoch L2 Norm Visualization
================================================================

读取两次 SUPRA 训练（aux_weight=0 vs aux_weight>0）的梯度 L2 范数 CSV，
绘制左右两个子图的折线图，验证辅助损失对模态投影器的梯度"复活"效果。

用法:
  python -m GNN.Utils.plot_gradient_norm \
      --csv_base Results/gradient_l2_norm_base.csv \
      --csv_aux Results/gradient_l2_norm_aux.csv \
      --save_plot Results/gradient_starvation.pdf
"""

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gradient_csv(csv_path: str) -> dict:
    """Load gradient L2 norm CSV into a dict of lists."""
    import csv
    epochs, enc_t, enc_v, gnn = [], [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            enc_t.append(float(row["enc_t"]))
            enc_v.append(float(row["enc_v"]))
            gnn.append(float(row["gnn"]))
    return dict(epochs=epochs, enc_t=enc_t, enc_v=enc_v, gnn=gnn)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_gradient_comparison(
    data_base: dict,
    data_aux: dict,
    aux_label: str,
    save_path: Optional[str] = None,
):
    """
    绘制梯度饥饿验证折线图。

    左图：SUPRA Base (aux_weight=0) — 预期 gnn 高位，enc_t/enc_v 迅速贴底
    右图：SUPRA w/ aux              — 预期三条线相对均衡
    """
    plt.rcParams["font.family"] = "serif"

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    plt.subplots_adjust(wspace=0.30)

    datasets = [
        ("SUPRA Base\n(aux_weight = 0)", data_base),
        (f"SUPRA w/ {aux_label}", data_aux),
    ]

    # 线条样式
    styles = {
        "enc_t": {"color": "#7294D4", "linestyle": "-",  "linewidth": 1.8, "marker": "o", "markersize": 3},
        "enc_v": {"color": "#FFD966", "linestyle": "-",  "linewidth": 1.8, "marker": "s", "markersize": 3},
        "gnn":   {"color": "#A9D18E", "linestyle": "--", "linewidth": 2.0, "marker": "^", "markersize": 3},
    }
    labels = {
        "enc_t": "Text Projector (enc_t)",
        "enc_v": "Visual Projector (enc_v)",
        "gnn":   "Shared GNN (mp_C)",
    }

    for ax, (title, data) in zip(axes, datasets):
        epochs = data["epochs"]

        for key in ["enc_t", "enc_v", "gnn"]:
            s = styles[key]
            ax.plot(
                epochs,
                data[key],
                label=labels[key],
                color=s["color"],
                linestyle=s["linestyle"],
                linewidth=s["linewidth"],
                marker=s["marker"],
                markersize=s["markersize"],
                markevery=max(1, len(epochs) // 12),
            )

        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

    # Y-axis label on left subplot only
    axes[0].set_ylabel("Gradient L2 Norm", fontsize=10)

    # Legend below figure
    fig.legend(
        [labels["enc_t"], labels["enc_v"], labels["gnn"]],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        framealpha=0.9,
        fontsize=9,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
        print(f"[Saved] Plot → {pdf_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gradient Starvation Verification — Per-Epoch L2 Norm Plot"
    )
    parser.add_argument(
        "--csv_base",
        type=str,
        required=True,
        help="Path to gradient L2 norm CSV from SUPRA Base (aux_weight=0)",
    )
    parser.add_argument(
        "--csv_aux",
        type=str,
        required=True,
        help="Path to gradient L2 norm CSV from SUPRA w/ aux",
    )
    parser.add_argument(
        "--aux_label",
        type=str,
        default="aux",
        help="Label for the aux experiment (e.g., 'aux_weight=0.3')",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="Path to save the plot (PDF format)",
    )
    args = parser.parse_args()

    data_base = load_gradient_csv(args.csv_base)
    data_aux = load_gradient_csv(args.csv_aux)

    plot_gradient_comparison(
        data_base=data_base,
        data_aux=data_aux,
        aux_label=args.aux_label,
        save_path=args.save_plot,
    )
