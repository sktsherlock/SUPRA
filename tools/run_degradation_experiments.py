#!/usr/bin/env python
"""
Controlled Degradation Experiments for SUPRA Theory Verification
================================================================

Single experiment validating the Feature Dimension (σ²_ε) of the
Generalized Performance Inversion Threshold (GPIT) theorem:

  Noise ratios: [0.0, 0.1, 0.3, 0.5, 0.8, 1.2, 2.0]
  Formula: X_noisy = X + α · std(X) · N(0, 1)

Three models compared:
  1. Pure MLP  — Early_GNN (backend=mlp), no graph edges used
  2. MMGCN     — Late_GNN (model_name=GCN), forced topology
  3. SUPRA     — SUPRA (aux_weight=0.0), decoupled channels (no auxiliary boost)

Usage:
    # Toy dataset (quick test)
    python tools/run_degradation_experiments.py \
        --data_name Toys \
        --text_feature /path/Toys_text.npy \
        --visual_feature /path/Toys_visual.npy \
        --graph_path /path/ToysGraph.pt \
        --save_dir Results/degradation \
        --n_runs 3

    # Full experiment
    python tools/run_degradation_experiments.py \
        --data_name Reddit-M \
        --text_feature /path/RedditM_text.npy \
        --visual_feature /path/RedditM_visual.npy \
        --graph_path /path/RedditMGraph.pt \
        --embed_dim 256 --n_layers 3 --lr 0.0005 --dropout 0.3 \
        --save_dir Results/degradation \
        --n_runs 3 --n_epochs 300
"""

import argparse
import gc
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch as th
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GNN.Utils.model_config import add_common_args
from GNN.Utils.LossFunction import cross_entropy, get_metric
from GNN.Utils.graph_degradation import inject_feature_noise
from GNN.GraphData import load_data


# -----------------------------------------------------------------------
# Model builders (replicating training code without full CLI complexity)
# -----------------------------------------------------------------------

def build_early_mlp(args, text_dim, vis_dim, n_classes, device):
    from GNN.Baselines.Early_GNN import SimpleMAGMLP
    mlp = nn.Sequential(
        nn.Linear(text_dim + vis_dim, args.n_hidden),
        nn.ReLU(),
        nn.LayerNorm(args.n_hidden),
        nn.Dropout(args.dropout),
        nn.Linear(args.n_hidden, args.n_hidden),
        nn.ReLU(),
        nn.LayerNorm(args.n_hidden),
        nn.Dropout(args.dropout),
        nn.Linear(args.n_hidden, n_classes),
    )
    model = SimpleMAGMLP(
        text_encoder=None,
        visual_encoder=None,
        mlp=mlp,
        early_fuse="concat",
        single_modality=None,
        use_no_encoder=True,
    )
    return model.to(device)


def build_late_gnn(args, text_dim, vis_dim, n_classes, device):
    from GNN.Baselines.Late_GNN import LateFusionMAG
    from GNN.Library.GCN import GCN
    embed_dim = int(getattr(args, "late_embed_dim", args.n_hidden))
    text_encoder = nn.Linear(text_dim, embed_dim).to(device)
    visual_encoder = nn.Linear(vis_dim, embed_dim).to(device)
    text_gnn = GCN(
        in_feats=embed_dim, n_hidden=args.n_hidden, n_classes=args.n_hidden,
        n_layers=args.n_layers, activation=nn.ReLU(), dropout=args.dropout,
    ).to(device)
    vis_gnn = GCN(
        in_feats=embed_dim, n_hidden=args.n_hidden, n_classes=args.n_hidden,
        n_layers=args.n_layers, activation=nn.ReLU(), dropout=args.dropout,
    ).to(device)
    classifier = nn.Linear(args.n_hidden * 2, n_classes).to(device)
    model = LateFusionMAG(
        text_encoder, visual_encoder, text_gnn, vis_gnn, classifier,
        use_mlp_before_fusion=False, use_no_encoder=False,
    )
    return model.to(device)


def build_supra(args, text_dim, vis_dim, n_classes, device):
    from GNN.SUPRA import SUPRA
    model = SUPRA(
        text_in_dim=text_dim, vis_in_dim=vis_dim,
        embed_dim=args.embed_dim, n_classes=n_classes,
        dropout=args.dropout, args=args, device=device,
    ).to(device)
    return model


# -----------------------------------------------------------------------
# Simplified training loop for a single run
# -----------------------------------------------------------------------

def train_and_evaluate(
    model, graph, text_feat, visual_feat, labels,
    train_idx, val_idx, test_idx,
    n_epochs, lr, wd, dropout,
    model_type, label_smoothing=0.1,
):
    """Train one model for one run and return best test accuracy."""
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val = -1.0
    best_test = 0.0
    patience = 0
    max_patience = 50

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        if model_type == "early_mlp":
            feat = th.cat([text_feat, visual_feat], dim=1)
            logits = model.mlp(feat)
        elif model_type == "late_gnn":
            text_h, vis_h = model.forward_branches(graph, text_feat, visual_feat)
            fused = model.fuse_embeddings(text_h, vis_h)
            logits = model.classifier(fused)
        elif model_type == "supra":
            out = model.forward_multiple(graph, text_feat, visual_feat)
            logits = out.logits_final_0

        loss = cross_entropy(logits[train_idx], labels[train_idx], label_smoothing=label_smoothing)
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with th.no_grad():
            if model_type == "early_mlp":
                feat = th.cat([text_feat, visual_feat], dim=1)
                logits = model.mlp(feat)
            elif model_type == "late_gnn":
                text_h, vis_h = model.forward_branches(graph, text_feat, visual_feat)
                fused = model.fuse_embeddings(text_h, vis_h)
                logits = model.classifier(fused)
            elif model_type == "supra":
                out = model.forward_multiple(graph, text_feat, visual_feat)
                logits = out.logits_final_0

            val_pred = th.argmax(logits[val_idx], dim=1)
            test_pred = th.argmax(logits[test_idx], dim=1)
            val_score = float(get_metric(val_pred, labels[val_idx], "accuracy"))
            test_score = float(get_metric(test_pred, labels[test_idx], "accuracy"))

        if val_score > best_val:
            best_val = val_score
            best_test = test_score
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                break

    return best_test


def run_single_condition(
    model_type, graph, text_feat, visual_feat, labels,
    train_idx, val_idx, test_idx,
    n_epochs, lr, wd, dropout, aux_weight, embed_dim,
    n_runs, base_seed,
    apply_degradation_fn=None,
):
    """
    Run a single degradation condition for one model.

    Args:
        apply_degradation_fn: callable that takes (text_feat, visual_feat, graph)
                             and returns (degraded_text, degraded_vis, degraded_graph)
                             If None, no degradation is applied.
    Returns:
        (mean_accuracy, std_accuracy) across n_runs
    """
    text_dim = text_feat.shape[1]
    vis_dim = visual_feat.shape[1]
    n_classes = int(labels.max().item()) + 1
    device = text_feat.device

    scores = []
    for run in range(n_runs):
        th.manual_seed(base_seed + run)
        np.random.seed(base_seed + run)

        # Apply degradation if provided
        if apply_degradation_fn is not None:
            t_feat, v_feat, g = apply_degradation_fn(text_feat, visual_feat, graph)
        else:
            t_feat, v_feat, g = text_feat, visual_feat, graph

        # Build fresh model
        if model_type == "early_mlp":
            model = build_early_mlp(
                make_args(embed_dim=embed_dim, n_hidden=embed_dim, dropout=dropout),
                text_dim, vis_dim, n_classes, device
            )
        elif model_type == "late_gnn":
            model = build_late_gnn(
                make_args(embed_dim=embed_dim, n_hidden=embed_dim, dropout=dropout,
                           late_embed_dim=embed_dim),
                text_dim, vis_dim, n_classes, device
            )
        elif model_type == "supra":
            # Set aux_weight=0: degradation experiment tests SUPRA without
            # auxiliary channel boost, isolating pure topology benefit.
            args_obj = make_args(embed_dim=embed_dim, n_hidden=embed_dim, dropout=dropout,
                                 aux_weight=0.0, mlp_variant="ablate",
                                 n_layers=3)
            model = build_supra(args_obj, text_dim, vis_dim, n_classes, device)
            model.reset_parameters()

        score = train_and_evaluate(
            model, g, t_feat, v_feat, labels,
            train_idx, val_idx, test_idx,
            n_epochs=n_epochs, lr=lr, wd=wd, dropout=dropout,
            model_type=model_type,
        )
        scores.append(score)

        # Cleanup
        del model
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()

    return float(np.mean(scores)), float(np.std(scores))


# -----------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------

def run_noise_experiment(
    graph, text_feat, visual_feat, labels,
    train_idx, val_idx, test_idx,
    noise_ratios, n_runs, base_seed,
    n_epochs, lr, wd, dropout, aux_weight, embed_dim,
) -> Dict:
    """Run feature noise degradation experiment."""
    results = {}
    for ratio in noise_ratios:
        print(f"  [Noise] ratio={ratio}")
        def make_noisy(t, v, g):
            return inject_feature_noise(t, v, ratio, base_seed) + (g,)

        row = {}
        for model_type in ["early_mlp", "late_gnn", "supra"]:
            mean_acc, std_acc = run_single_condition(
                model_type, graph, text_feat, visual_feat, labels,
                train_idx, val_idx, test_idx,
                n_epochs, lr, wd, dropout, aux_weight, embed_dim,
                n_runs, base_seed,
                apply_degradation_fn=make_noisy,
            )
            row[model_type] = (mean_acc, std_acc)
            print(f"    {model_type}: {mean_acc:.4f} ± {std_acc:.4f}")
        results[ratio] = row
    return results


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_degradation(
    noise_results: Dict,
    noise_ratios: List[float],
    save_path: str,
):
    """Plot publication-ready degradation curve with error bands."""
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = 11

    fig, ax = plt.subplots(figsize=(6, 4.5))
    plt.subplots_adjust(bottom=0.18, left=0.18)

    model_configs = {
        "early_mlp": {"color": "#ED7D31", "linestyle": "-",  "marker": "s", "markersize": 5,
                      "label": "Pure MLP"},
        "late_gnn":  {"color": "#4472C4", "linestyle": "--", "marker": "^", "markersize": 5,
                      "label": "MMGCN"},
        "supra":     {"color": "#70AD47", "linestyle": ":",  "marker": "o", "markersize": 5,
                      "label": "SUPRA"},
    }

    for model_type, cfg in model_configs.items():
        means = [noise_results[r][model_type][0] for r in noise_ratios]
        stds = [noise_results[r][model_type][1] for r in noise_ratios]
        ax.plot(noise_ratios, means,
                color=cfg["color"], linestyle=cfg["linestyle"],
                marker=cfg["marker"], markersize=cfg["markersize"],
                linewidth=2.0, label=cfg["label"])
        ax.fill_between(noise_ratios,
                       [m - s for m, s in zip(means, stds)],
                       [m + s for m, s in zip(means, stds)],
                       color=cfg["color"], alpha=0.15)

    ax.set_xlabel(r"Noise Ratio $\alpha$", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(r"Feature Dimension: $\sigma^2_\epsilon$", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(loc="best", framealpha=0.9, fontsize=10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"[Saved] {pdf_path}")


def save_results_csv(noise_results, noise_ratios, save_dir):
    """Save numerical results to CSV."""
    import csv
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, "noise_degradation_results.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ratio", "model", "mean_acc", "std_acc"])
        for ratio in noise_ratios:
            for model in ["early_mlp", "late_gnn", "supra"]:
                mean, std = noise_results[ratio][model]
                w.writerow([ratio, model, f"{mean:.6f}", f"{std:.6f}"])
    print(f"[Saved] {path}")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def make_args(**kwargs):
    """Create a simple namespace with given attributes for model building."""
    import argparse
    defaults = dict(
        embed_dim=256, n_hidden=256, n_layers=3,
        dropout=0.3, lr=0.001, wd=0.0001,
        aux_weight=0.7, mlp_variant="ablate",
        late_embed_dim=256,
    )
    defaults.update(kwargs)
    ns = argparse.Namespace(**defaults)
    return ns


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Controlled Degradation Experiments for SUPRA Theory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--text_feature", type=str, required=True)
    parser.add_argument("--visual_feature", type=str, required=True)
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="Results/degradation")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--aux_weight", type=float, default=0.7)
    parser.add_argument("--n_runs", type=int, default=3,
                        help="Number of runs per condition (for mean/std)")
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--noise_ratios", type=str,
                        default="0.0,0.1,0.3,0.5,0.8,1.2,2.0",
                        help="Comma-separated noise ratios")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    noise_ratios = [float(x) for x in args.noise_ratios.split(",")]

    # Device
    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")

    # Load data
    print(f"\n[{args.data_name}] Loading data...")
    graph, labels, train_idx, val_idx, test_idx = \
        load_data(args.graph_path, train_ratio=0.6, val_ratio=0.2,
                  name=args.data_name, fewshots=False)

    # Undirected + self-loop (matching sweep defaults)
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)
    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    graph = graph.to(device)

    labels = labels.to(device).long()
    train_idx = train_idx.to(device).long()
    val_idx = val_idx.to(device).long()
    test_idx = test_idx.to(device).long()

    text_feat = th.from_numpy(
        np.load(args.text_feature, mmap_mode="r").astype(np.float32)
    ).to(device)
    visual_feat = th.from_numpy(
        np.load(args.visual_feature, mmap_mode="r").astype(np.float32)
    ).to(device)

    print(f"  Nodes: {graph.num_nodes()}, Edges: {graph.num_edges()}")
    print(f"  Text feat: {text_feat.shape}, Visual feat: {visual_feat.shape}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Run experiment
    print(f"\n{'='*60}")
    print("Experiment — Feature Noise Degradation")
    print(f"{'='*60}")
    noise_results = run_noise_experiment(
        graph, text_feat, visual_feat, labels,
        train_idx, val_idx, test_idx,
        noise_ratios, args.n_runs, args.base_seed,
        n_epochs=args.n_epochs, lr=args.lr, wd=args.wd,
        dropout=args.dropout, aux_weight=args.aux_weight,
        embed_dim=args.embed_dim,
    )

    # Save results
    save_results_csv(noise_results, noise_ratios, args.save_dir)

    # Plot
    save_plot = os.path.join(args.save_dir, f"{args.data_name}_degradation.pdf")
    plot_degradation(noise_results, noise_ratios, save_plot)

    print("\nDone.")
