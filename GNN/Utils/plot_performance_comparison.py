"""
Performance Comparison Plot — Faceted Grouped Bar Chart
=====================================================

展示不同特征质量（Low-Conf vs High-Conf）下，各模型在 Movies / RedditM 上的性能对比。
用于论文 Intro，说明高质量特征下多模态图学习收益递减的现象。

Usage:
  python -m GNN.Utils.plot_performance_comparison --save_plot Results/perf_comparison.pdf
"""

import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Structure: data[dataset][encoder] = {model: accuracy}
DATA: Dict[str, Dict[str, Dict[str, float]]] = {
    "Movies": {
        "Low-Conf": {
            "MLP": 51.29,
            "MGCN": 53.95,
            "MGAT": 53.88,
            "MIGGT": 49.89,
            "NTS": 54.07,
        },
        "High-Conf": {
            "MLP": 57.51,
            "MGCN": 55.19,
            "MGAT": 54.44,
            "MIGGT": 55.93,
            "NTS": 56.14,
        },
    },
    "RedditM": {
        "Low-Conf": {
            "MLP": 83.20,
            "MGCN": 78.46,
            "MGAT": 78.11,
            "MIGGT": 82.91,
            "NTS": 86.13,
        },
        "High-Conf": {
            "MLP": 84.80,
            "MGCN": 79.25,
            "MGAT": 77.90,
            "MIGGT": 84.69,
            "NTS": 87.28,
        },
    },
}

MODELS = ["MLP", "MGCN", "MGAT", "MIGGT", "NTS"]
DATASETS = ["Movies", "RedditM"]
ENCODERS = ["Low-Conf", "High-Conf"]

# Y-axis ranges per dataset (independent scales)
Y_RANGES = {
    "Movies":  (45, 62),
    "RedditM": (75, 90),
}

COLORS = {
    "Low-Conf":  "#4C72B0",  # blue
    "High-Conf": "#DD8452",  # orange
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(save_path: str = None) -> None:
    """
    Render the faceted grouped bar chart.
    """
    n_datasets = len(DATASETS)
    n_models = len(MODELS)
    bar_width = 0.35

    fig, axes = plt.subplots(
        1, n_datasets,
        figsize=(9, 4.5),
        gridspec_kw={"wspace": 0.30},
    )
    if n_datasets == 1:
        axes = [axes]

    for ax_idx, dataset in enumerate(DATASETS):
        ax = axes[ax_idx]
        y_lo, y_hi = Y_RANGES[dataset]

        # x positions for each model group
        group_centers = np.arange(n_models)

        offset = bar_width + 0.05  # fixed gap between the two bars

        for enc_idx, encoder in enumerate(ENCODERS):
            values = [DATA[dataset][encoder][m] for m in MODELS]
            offsets = (enc_idx - 0.5) * offset  # -0.5*offset = left, +0.5*offset = right
            bars = ax.bar(
                group_centers + offsets,
                values,
                width=bar_width,
                label=encoder,
                color=COLORS[encoder],
                edgecolor="white",
                linewidth=0.5,
            )
            # NTS: bold outline
            for bar in bars:
                if bar.get_label() == "NTS":
                    bar.set_edgecolor("#333333")
                    bar.set_linewidth(1.5)

        # Y axis
        ax.set_ylim(y_lo, y_hi)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.set_yticks(np.linspace(y_lo, y_hi, 6))

        # Grid
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # X axis
        ax.set_xticks(group_centers)
        ax.set_xticklabels(MODELS, fontsize=10)
        ax.tick_params(axis="x", length=0)

        # Title
        ax.set_title(dataset, fontsize=12, fontweight="bold", pad=8)

        # Y-axis label on leftmost subplot only
        if ax_idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=11)

    # Global legend below figure
    fig.legend(
        ENCODERS,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=2,
        framealpha=0.9,
        fontsize=10,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

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
    parser = argparse.ArgumentParser(description="Plot performance comparison bar chart")
    parser.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="Path to save the plot (PDF format)",
    )
    args = parser.parse_args()
    plot_comparison(save_path=args.save_plot)
