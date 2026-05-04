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

MODEL_CONFIGS = {
    "early_mlp": {"color": "#ED7D31", "linestyle": "-",  "marker": "s", "markersize": 5,
                  "label": "Pure MLP"},
    "late_gnn":  {"color": "#4472C4", "linestyle": "--", "marker": "^", "markersize": 5,
                  "label": "MMGCN"},
    "supra":     {"color": "#70AD47", "linestyle": ":",  "marker": "o", "markersize": 5,
                  "label": "SUPRA"},
}


def _plot_single(ax, noise_results, noise_ratios, title=None, xlabel=True):
    """
    Plot one sub-axis of degradation curve.

    X-axis shows noise ratio from most noisy (left) to clean (right),
    so the reader sees the benefit of higher-quality features.
    """
    # Reverse: left=most noisy, right=clean (original features)
    display_ratios = list(reversed(noise_ratios))

    for model_type, cfg in MODEL_CONFIGS.items():
        # Reverse means/stds to match display_ratios order
        raw_means = [noise_results[r][model_type][0] for r in reversed(noise_ratios)]
        raw_stds = [noise_results[r][model_type][1] for r in reversed(noise_ratios)]
        ax.plot(display_ratios, raw_means,
                color=cfg["color"], linestyle=cfg["linestyle"],
                marker=cfg["marker"], markersize=cfg["markersize"],
                linewidth=2.0, label=cfg["label"])
        ax.fill_between(display_ratios,
                       [m - s for m, s in zip(raw_means, raw_stds)],
                       [m + s for m, s in zip(raw_means, raw_stds)],
                       color=cfg["color"], alpha=0.20)

    if xlabel:
        ax.set_xlabel("Feature Quality  (low → high)", fontsize=11)
    else:
        ax.set_xlabel("")
    ax.set_ylabel("Accuracy", fontsize=11)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.invert_xaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def plot_degradation(
    noise_results: Dict,
    noise_ratios: List[float],
    save_path: str,
    title: str = None,
):
    """
    Plot single-panel publication-ready degradation curve.

    For the main paper (one dataset).
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = 11

    fig, ax = plt.subplots(figsize=(6, 4.5))
    plt.subplots_adjust(bottom=0.18, left=0.18)

    _plot_single(ax, noise_results, noise_ratios, title=title)
    ax.legend(loc="best", framealpha=0.9, fontsize=10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"[Saved] {save_path}")
    plt.close(fig)


def plot_grid_1x4(
    datasets: Dict[str, Dict],
    noise_ratios: List[float] | None,
    save_path: str,
):
    """
    Plot all 4 datasets in a 1×4 horizontal row for single-column paper layout.

    X-axis: left = most noisy (worst features), right = clean (original).
    If noise_ratios is None, extracts ratios from each dataset's results keys.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = 11

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2))
    if n == 1:
        axes = [axes]

    for ax, (name, results) in zip(axes, datasets.items()):
        # Infer per-dataset noise ratios from results keys if not provided
        ds_ratios = noise_ratios if noise_ratios is not None else sorted(results.keys())
        _plot_single(ax, results, ds_ratios, title=name)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=cfg["color"], linestyle=cfg["linestyle"],
                          marker=cfg["marker"], markersize=cfg["markersize"], linewidth=2.0)
               for cfg in MODEL_CONFIGS.values()]
    labels = [cfg["label"] for cfg in MODEL_CONFIGS.values()]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, framealpha=0.9, fontsize=10)

    plt.subplots_adjust(bottom=0.18, top=0.90, wspace=0.30)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"[Saved] {save_path}")
    plt.close(fig)


def plot_multi_dataset(
    datasets: Dict[str, Dict],
    noise_ratios: List[float],
    save_path: str,
):
    """
    Plot multiple datasets in a 1×N row for appendix.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = 11

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2))
    if n == 1:
        axes = [axes]

    plt.subplots_adjust(bottom=0.18, top=0.88, wspace=0.30)

    for ax, (name, results) in zip(axes, datasets.items()):
        _plot_single(ax, results, noise_ratios, title=name)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"[Saved] {save_path}")
    plt.close(fig)


def save_results_csv(noise_results, noise_ratios, save_dir, dataset_name=None):
    """Save numerical results to CSV, named per dataset."""
    import csv
    os.makedirs(save_dir, exist_ok=True)

    prefix = f"{dataset_name}_" if dataset_name else ""
    path = os.path.join(save_dir, f"{prefix}noise_degradation_results.csv")
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
    parser.add_argument("--aux_weight", type=float, default=0.0)
    parser.add_argument("--n_runs", type=int, default=3,
                        help="Number of runs per condition (for mean/std)")
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--noise_ratios", type=str,
                        default="0.0,0.1,0.3,0.5,0.8,1.2,2.0",
                        help="Comma-separated noise ratios")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--appendix_datasets", type=str, default=None,
                        help="Comma-separated list of dataset configs for appendix plot "
                             "in format 'name:text_feature:visual_feature:graph_path[:lr[:n_layers[:noise_ratios]]]'. "
                             "Example: 'Movies:/path1:/path2:/path3:0.001:3' or "
                             "'Reddit-M:/path1:/path2:/path3:0.0005:3:0.0,0.1,0.3,0.5,0.8,1.2,2.0,3.0,5.0'")
    parser.add_argument("--resume", action="store_true",
                        help="Skip dataset if its CSV already exists in save_dir")
    args = parser.parse_args()

    noise_ratios = [float(x) for x in args.noise_ratios.split(",")]

    # Device
    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------
    # Helper: run experiment for one dataset
    # -----------------------------------------------------------------
    def run_for_dataset(data_name, text_feat_path, visual_feat_path, graph_path,
                       lr, n_layers, embed_dim, dropout, save_dir, dataset_noise_ratios):
        # Resume check: skip if CSV already exists
        csv_path = os.path.join(save_dir, f"{data_name}_noise_degradation_results.csv")
        if args.resume and os.path.exists(csv_path):
            import csv
            print(f"  [{data_name}] Skipping (CSV exists: {csv_path})")
            results = {}
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ratio = float(row["ratio"])
                    model = row["model"]
                    if ratio not in results:
                        results[ratio] = {}
                    results[ratio][model] = (float(row["mean_acc"]), float(row["std_acc"]))
            return results

        g, lbls, tr_idx, v_idx, t_idx = load_data(
            graph_path, train_ratio=0.6, val_ratio=0.2,
            name=data_name, fewshots=False)
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)
        g = g.remove_self_loop().add_self_loop()
        g.create_formats_()
        g = g.to(device)
        lbls = lbls.to(device).long()
        tr_idx = tr_idx.to(device).long()
        v_idx = v_idx.to(device).long()
        t_idx = t_idx.to(device).long()

        tf = th.from_numpy(np.load(text_feat_path, mmap_mode="r").astype(np.float32)).to(device)
        vf = th.from_numpy(np.load(visual_feat_path, mmap_mode="r").astype(np.float32)).to(device)

        print(f"  [{data_name}] Nodes={g.num_nodes()}, Edges={g.num_edges()}, "
              f"Train={len(tr_idx)}, Val={len(v_idx)}, Test={len(t_idx)}")

        results = run_noise_experiment(
            g, tf, vf, lbls, tr_idx, v_idx, t_idx,
            dataset_noise_ratios, args.n_runs, args.base_seed,
            n_epochs=args.n_epochs, lr=lr, wd=args.wd,
            dropout=dropout, aux_weight=args.aux_weight, embed_dim=embed_dim,
        )
        # Save per-dataset CSV (named by dataset)
        save_results_csv(results, dataset_noise_ratios, save_dir, dataset_name=data_name)
        return results

    # -----------------------------------------------------------------
    # All datasets for the paper figure (2×2 grid)
    # -----------------------------------------------------------------
    all_datasets = {}

    # Main dataset (Reddit-M)
    print(f"\n[{args.data_name}] Loading data...")
    print(f"  Nodes: ..., Edges: ...")
    paper_results = run_for_dataset(
        args.data_name, args.text_feature, args.visual_feature,
        args.graph_path, args.lr, args.n_layers, args.embed_dim,
        args.dropout, args.save_dir, noise_ratios,
    )
    all_datasets[args.data_name] = paper_results

    # Additional datasets
    if args.appendix_datasets:
        print(f"\n{'='*60}")
        print("Running additional datasets...")
        print(f"{'='*60}")

        for cfg in args.appendix_datasets.split(","):
            cfg = cfg.strip()
            if not cfg:
                continue
            parts = cfg.split(":")
            if len(parts) < 4:
                print(f"[WARN] Skipping malformed appendix config (expected >=4 colon-separated parts, got {len(parts)}): {cfg[:80]}")
                continue
            name = parts[0]
            txt = parts[1]
            vis = parts[2]
            grh = parts[3]

            # Parse optional remaining fields, handling noise_ratios/flags
            # that may have been accidentally appended via bash multiline issues
            ds_lr, ds_nl, ds_noise = 0.001, 3, noise_ratios
            trailing = parts[4:]
            noise_candidates = []
            lr_found, nl_found = False, False
            for i, tok in enumerate(trailing):
                tok = tok.strip()
                if not tok:
                    continue
                # Skip stray flags (e.g. "--resume" appended due to multiline)
                if tok.startswith("--"):
                    continue
                # Comma-separated floats = noise ratios
                if "," in tok:
                    try:
                        noise_candidates = [float(x) for x in tok.split(",")]
                        continue
                    except ValueError:
                        pass
                # Try as lr (positive float)
                if not lr_found:
                    try:
                        v = float(tok)
                        if v > 0:
                            ds_lr = v
                            lr_found = True
                            continue
                    except ValueError:
                        pass
                # Try as n_layers (positive int)
                if not nl_found:
                    try:
                        v = int(tok)
                        if v > 0:
                            ds_nl = v
                            nl_found = True
                            continue
                    except ValueError:
                        pass
                # If we get here and have noise candidates, assign them
                if noise_candidates:
                    ds_noise = noise_candidates
                    noise_candidates = []

            # If noise candidates still unassigned, check if last trailing part looks like noise
            if not noise_candidates and len(trailing) > 0:
                last = trailing[-1].strip()
                if "," in last:
                    try:
                        ds_noise = [float(x) for x in last.split(",")]
                    except ValueError:
                        pass

            print(f"\n[{name}] Loading data...")
            results = run_for_dataset(
                name, txt, vis, grh, ds_lr, ds_nl,
                args.embed_dim, args.dropout, args.save_dir, ds_noise,
            )
            all_datasets[name] = results

    # Paper figure: 1×4 horizontal row of all datasets
    # Extract per-dataset noise ratios from results keys (each dataset may differ)
    grid_path = os.path.join(args.save_dir, "degradation_1x4.pdf")
    plot_grid_1x4(all_datasets, None, grid_path)

    print("\nDone.")
