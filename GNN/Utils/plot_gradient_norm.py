"""
Gradient Starvation Verification — Per-Epoch L2 Norm Visualization
================================================================

支持 2 组或 4 组对比实验的梯度 L2 范数折线图。

用法（2组）:
  python -m GNN.Utils.plot_gradient_norm --mode 2 \
      --csv_1 Results/gradient_starvation/grocery_base.csv \
      --csv_2 Results/gradient_starvation/grocery_aux.csv \
      --label_1 "Base (aux=0)" --label_2 "aux_weight=0.7" \
      --save_plot Results/gradient_starvation/grocery.pdf

用法（4组）:
  python -m GNN.Utils.plot_gradient_norm --mode 4 \
      --csv_1 Results/gradient_starvation/mmgen_base.csv \
      --csv_2 Results/gradient_starvation/supra_no_bypass.csv \
      --csv_3 Results/gradient_starvation/supra_base.csv \
      --csv_4 Results/gradient_starvation/supra_full.csv \
      --save_plot Results/gradient_starvation/4group.pdf
"""

import argparse
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_supa_csv(csv_path: str) -> Dict[str, List[float]]:
    """Load SUPRA gradient CSV (enc_t, enc_v, gnn)."""
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


def load_late_csv(csv_path: str) -> Dict[str, List[float]]:
    """Load Late_GNN (MMGCN) gradient CSV (text_gnn, vis_gnn, mmgnn)."""
    import csv
    epochs, text_gnn, vis_gnn, mmgnn = [], [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            text_gnn.append(float(row["text_gnn"]))
            vis_gnn.append(float(row["vis_gnn"]))
            mmgnn.append(float(row["mmgnn"]))
    return dict(epochs=epochs, text_gnn=text_gnn, vis_gnn=vis_gnn, mmgnn=mmgnn)


def truncate(data: Dict, max_epoch: int) -> None:
    """Truncate data dict in-place to first max_epoch entries."""
    for key in list(data.keys()):
        if key == "epochs":
            data[key] = list(range(1, min(max_epoch, len(data[key])) + 1))
        else:
            data[key] = data[key][:max_epoch]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_2group(
    csv_1: str,
    csv_2: str,
    label_1: str,
    label_2: str,
    save_path: Optional[str] = None,
    max_epoch: Optional[int] = None,
):
    """2组对比：左=Base(aux=0)，右=Full(aux>0)。"""
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    plt.subplots_adjust(wspace=0.30)

    data_1 = load_supa_csv(csv_1)
    data_2 = load_supa_csv(csv_2)
    if max_epoch:
        truncate(data_1, max_epoch)
        truncate(data_2, max_epoch)

    styles = {
        "enc_t": {"color": "#7294D4", "linestyle": "-",  "linewidth": 1.8, "marker": "o", "markersize": 3},
        "enc_v": {"color": "#FFD966", "linestyle": "-",  "linewidth": 1.8, "marker": "s", "markersize": 3},
        "gnn":   {"color": "#A9D18E", "linestyle": "--", "linewidth": 2.0, "marker": "^", "markersize": 3},
    }
    legend_labels = {
        "enc_t": "Text Projector (enc_t)",
        "enc_v": "Visual Projector (enc_v)",
        "gnn":   "Shared GNN (mp_C)",
    }

    for ax, data, title in zip(
        axes,
        [data_1, data_2],
        [label_1, label_2],
    ):
        epochs = data["epochs"]
        for key in ["enc_t", "enc_v", "gnn"]:
            s = styles[key]
            ax.plot(
                epochs, data[key],
                label=legend_labels[key],
                color=s["color"], linestyle=s["linestyle"],
                linewidth=s["linewidth"],
                marker=s["marker"], markersize=s["markersize"],
                markevery=max(1, len(epochs) // 12),
            )
        ax.set_title(title.replace("\\n", "\n"), fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

    axes[0].set_ylabel("Gradient L2 Norm", fontsize=10)
    fig.legend(
        list(legend_labels.values()),
        loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=3, framealpha=0.9, fontsize=9,
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


def plot_4group(
    csv_1: str,
    csv_2: str,
    csv_3: str,
    csv_4: str,
    label_1: str,
    label_2: str,
    label_3: str,
    label_4: str,
    save_path: Optional[str] = None,
    max_epoch: Optional[int] = None,
):
    """4组对比：MMGCN → SUPRA(No Bypass) → SUPRA(Base) → SUPRA(Full)。"""
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    plt.subplots_adjust(wspace=0.25)

    # csv_1 = Late_GNN (MMGCN): text_enc, vis_enc, mmgnn
    # csv_2,3,4 = SUPRA: enc_t, enc_v, gnn
    configs = [
        (csv_1, "late", label_1),
        (csv_2, "supra", label_2),
        (csv_3, "supra", label_3),
        (csv_4, "supra", label_4),
    ]

    supra_styles = {
        "enc_t": {"color": "#7294D4", "linestyle": "-",  "linewidth": 1.8, "marker": "o", "markersize": 3},
        "enc_v": {"color": "#FFD966", "linestyle": "-",  "linewidth": 1.8, "marker": "s", "markersize": 3},
        "gnn":   {"color": "#A9D18E", "linestyle": "--", "linewidth": 2.0, "marker": "^", "markersize": 3},
    }
    supra_legend = {
        "enc_t": "Text Projector",
        "enc_v": "Visual Projector",
        "gnn":   "Shared GNN",
    }

    late_styles = {
        "text_gnn": {"color": "#7294D4", "linestyle": "-",  "linewidth": 1.8, "marker": "o", "markersize": 3},
        "vis_gnn":  {"color": "#FFD966", "linestyle": "-",  "linewidth": 1.8, "marker": "s", "markersize": 3},
    }
    late_legend = {
        "text_gnn": "Text GNN",
        "vis_gnn":  "Visual GNN",
    }

    for idx, (csv_path, mode, label) in enumerate(configs):
        ax = axes[idx]
        if mode == "late":
            data = load_late_csv(csv_path)
            keys = ["text_gnn", "vis_gnn"]
            styles_map = late_styles
            legend_map = late_legend
        else:
            data = load_supa_csv(csv_path)
            keys = ["enc_t", "enc_v", "gnn"]
            styles_map = supra_styles
            legend_map = supra_legend

        if max_epoch:
            truncate(data, max_epoch)

        epochs = data["epochs"]
        handles = []
        for key in keys:
            s = styles_map[key]
            line, = ax.plot(
                epochs, data[key],
                color=s["color"], linestyle=s["linestyle"],
                linewidth=s["linewidth"],
                marker=s["marker"], markersize=s["markersize"],
                markevery=max(1, len(epochs) // 12),
            )
            handles.append(line)

        ax.set_title(label, fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        if idx == 0:
            ax.set_ylabel("Gradient L2 Norm", fontsize=10)

        ax.legend(
            handles, list(legend_map.values()),
            loc="best", framealpha=0.9, fontsize=7,
        )

    plt.tight_layout()

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
        "--mode", type=int, default=2, choices=[2, 4],
        help="2 = two-group comparison, 4 = four-group evolutionary comparison",
    )
    parser.add_argument("--csv_1", type=str, required=True, help="CSV for group 1")
    parser.add_argument("--csv_2", type=str, required=True, help="CSV for group 2")
    parser.add_argument("--csv_3", type=str, default=None, help="CSV for group 3 (mode 4 only)")
    parser.add_argument("--csv_4", type=str, default=None, help="CSV for group 4 (mode 4 only)")
    parser.add_argument("--label_1", type=str, default="Group 1", help="Label for group 1")
    parser.add_argument("--label_2", type=str, default="Group 2", help="Label for group 2")
    parser.add_argument("--label_3", type=str, default="Group 3", help="Label for group 3")
    parser.add_argument("--label_4", type=str, default="Group 4", help="Label for group 4")
    parser.add_argument("--save_plot", type=str, default=None, help="Path to save PDF")
    parser.add_argument(
        "--max_epoch", type=int, default=None,
        help="Only plot the first N epochs",
    )
    args = parser.parse_args()

    if args.mode == 2:
        plot_2group(
            csv_1=args.csv_1,
            csv_2=args.csv_2,
            label_1=args.label_1,
            label_2=args.label_2,
            save_path=args.save_plot,
            max_epoch=args.max_epoch,
        )
    else:
        plot_4group(
            csv_1=args.csv_1,
            csv_2=args.csv_2,
            csv_3=args.csv_3,
            csv_4=args.csv_4,
            label_1=args.label_1,
            label_2=args.label_2,
            label_3=args.label_3,
            label_4=args.label_4,
            save_path=args.save_plot,
            max_epoch=args.max_epoch,
        )
