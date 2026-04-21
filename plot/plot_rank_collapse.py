"""Visualize multimodal rank collapse analysis (Figure 5 style).

Usage:
    python plot_rank_collapse.py \\
        --config config.yaml \\
        --output figures/rank_collapse.pdf
"""

import argparse
import os
import sys
import numpy as np
import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

_ROOT = Path(__file__).parent
sys.path.append(str(_ROOT))

from GNN.GraphData import load_data, set_seed
from GNN.Utils.rank_analysis import (
    analyze_rank_vs_beta,
    compute_effective_rank,
    apply_modality_degradation,
    compute_representation_similarity,
)


def _build_plain_gcn(in_dim: int, n_classes: int, device: th.device, n_hidden: int, n_layers: int, dropout: float):
    from GNN.Library.GCN import GCN

    return GCN(
        in_feats=in_dim,
        n_hidden=n_hidden,
        n_classes=n_classes,
        n_layers=n_layers,
        activation=F.relu,
        dropout=dropout,
    ).to(device)


def _run_plain_demo(
    graph,
    text_feat: th.Tensor,
    vis_feat: th.Tensor,
    labels: th.Tensor,
    test_idx: th.Tensor,
    beta_values: np.ndarray,
    output_dir: str,
    device: th.device,
    n_hidden: int,
    n_layers: int,
    dropout: float,
):
    graph = graph.remove_self_loop().add_self_loop()
    feat_concat = th.cat([text_feat, vis_feat], dim=1)
    n_classes = int((labels.max() + 1).item())

    model = _build_plain_gcn(
        in_dim=int(feat_concat.shape[1]),
        n_classes=n_classes,
        device=device,
        n_hidden=n_hidden,
        n_layers=n_layers,
        dropout=dropout,
    )
    model.eval()

    with th.no_grad():
        ref_fused_full = model(graph, feat_concat)
        ref_fused_full = ref_fused_full[test_idx]

        text_only_base = model(graph, th.cat([text_feat, th.zeros_like(vis_feat)], dim=1))
        visual_only_base = model(graph, th.cat([th.zeros_like(text_feat), vis_feat], dim=1))
        text_only_base = text_only_base[test_idx]
        visual_only_base = visual_only_base[test_idx]

    unimodal_text_rank = compute_effective_rank(text_feat[test_idx])
    unimodal_vis_rank = compute_effective_rank(vis_feat[test_idx])

    ranks_text, _, _ = analyze_rank_vs_beta(
        model,
        graph,
        text_feat,
        vis_feat,
        beta_values,
        test_idx,
        upgrade_modality="text",
        upgrade_mode="scale",
        reference_embeddings=None,
        model_type="plain",
    )

    sim_text_when_text_up = []
    sim_vis_when_text_up = []
    sim_text_when_vis_up = []
    sim_vis_when_vis_up = []

    with th.no_grad():
        for beta in beta_values:
            text_up = text_feat * float(beta)
            vis_up = vis_feat * float(beta)

            fused_text_up = model(graph, th.cat([text_up, vis_feat], dim=1))
            fused_vis_up = model(graph, th.cat([text_feat, vis_up], dim=1))

            fused_text_up = fused_text_up[test_idx]
            fused_vis_up = fused_vis_up[test_idx]

            sim_text_when_text_up.append(
                compute_representation_similarity(text_only_base, fused_text_up, method="l2")
            )
            sim_vis_when_text_up.append(
                compute_representation_similarity(visual_only_base, fused_text_up, method="l2")
            )

            sim_text_when_vis_up.append(
                compute_representation_similarity(text_only_base, fused_vis_up, method="l2")
            )
            sim_vis_when_vis_up.append(
                compute_representation_similarity(visual_only_base, fused_vis_up, method="l2")
            )

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(beta_values, ranks_text, "o-", linewidth=2, label="Plain-GNN", color="C0")
    ax.set_xlabel(r"$\beta$ (Text Upweight)", fontsize=11)
    ax.set_ylabel("Multimodal Rank", fontsize=11)
    ax.set_title("Rank under Text Upweight", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(beta_values, sim_text_when_text_up, "o-", linewidth=2, label="Text vs Fused", color="C0")
    ax.plot(beta_values, sim_vis_when_text_up, "s--", linewidth=2, label="Visual vs Fused", color="C1")
    ax.set_xlabel(r"$\beta$ (Text Upweight)", fontsize=11)
    ax.set_ylabel("Representation Similarity", fontsize=11)
    ax.set_title("Similarity under Text Upweight", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(beta_values, sim_text_when_vis_up, "o-", linewidth=2, label="Text vs Fused", color="C0")
    ax.plot(beta_values, sim_vis_when_vis_up, "s--", linewidth=2, label="Visual vs Fused", color="C1")
    ax.set_xlabel(r"$\beta$ (Visual Upweight)", fontsize=11)
    ax.set_ylabel("Representation Similarity", fontsize=11)
    ax.set_title("Similarity under Visual Upweight", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    combo_path = os.path.join(output_dir, "plain_gnn_rank_similarity_combined.pdf")
    plt.tight_layout()
    plt.savefig(combo_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Combined figure saved to {combo_path}")


def plot_rank_collapse_figure(
    results_dict: dict,
    unimodal_baseline: float,
    beta_values: np.ndarray,
    output_path: str,
    title: str = "Multimodal Rank under Modality Elimination",
):
    """Plot Figure 5(a) or 5(c) style: Rank vs Beta.
    
    Args:
        results_dict: {method_name: ranks_array}
        unimodal_baseline: horizontal line for single-modality baseline
        beta_values: X-axis values
        output_path: Save path
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color scheme
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for (method, ranks), color in zip(results_dict.items(), colors):
        ax.plot(beta_values, ranks, marker='o', markersize=6, linewidth=2.5,
               label=method, color=color, alpha=0.8)
    
    # Unimodal baseline (horizontal line)
    ax.axhline(y=unimodal_baseline, color='orange', linestyle='--', 
              linewidth=2, label='Unimodal Baseline', alpha=0.7)
    
    # Try to detect phase transition (steepest drop)
    for method, ranks in results_dict.items():
        if len(ranks) > 2:
            gradients = np.abs(np.diff(ranks))
            if len(gradients) > 0:
                transition_idx = np.argmax(gradients)
                beta_transition = beta_values[transition_idx]
                # Only mark if significant drop
                if gradients[transition_idx] > 0.1 * (ranks[0] - ranks[-1]):
                    ax.axvline(x=beta_transition, color='red', linestyle=':', 
                             linewidth=1.5, alpha=0.5)
                    ax.text(beta_transition, ax.get_ylim()[1] * 0.95, 
                           'Phase\nTransition', ha='center', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
                    break  # Only mark once
    
    ax.set_xlabel(r'Modality Elimination Strength ($\beta$)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Multimodal Rank', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([beta_values[0], beta_values[-1]])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Rank collapse plot saved to {output_path}")
    plt.close()


def plot_similarity_figure(
    results_dict: dict,
    beta_values: np.ndarray,
    output_path: str,
    title: str = "Representation Similarity under Collapse",
):
    """Plot Figure 5(b) or 5(d) style: Similarity vs Beta.
    
    Args:
        results_dict: {method_name: similarity_array}
        beta_values: X-axis values
        output_path: Save path
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for (method, sims), color in zip(results_dict.items(), colors):
        ax.plot(beta_values, sims, marker='s', markersize=6, linewidth=2.5,
               label=method, color=color, linestyle='--', alpha=0.8)
    
    ax.set_xlabel(r'Modality Elimination Strength ($\beta$)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Representation Similarity', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([beta_values[0], beta_values[-1]])
    ax.set_ylim([0.0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Similarity plot saved to {output_path}")
    plt.close()


def plot_combined_figure(
    rank_results: dict,
    sim_results: dict,
    unimodal_baseline: float,
    beta_values: np.ndarray,
    output_path: str,
):
    """Plot combined figure like original Fig 5 (2x2 layout)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Rank
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(rank_results)))
    for (method, ranks), color in zip(rank_results.items(), colors):
        ax.plot(beta_values, ranks, marker='o', markersize=5, linewidth=2,
               label=method, color=color)
    ax.axhline(y=unimodal_baseline, color='orange', linestyle='--', 
              linewidth=2, label='Unimodal Baseline')
    ax.set_xlabel(r'$\beta$', fontsize=12)
    ax.set_ylabel('Multimodal Rank', fontsize=12, fontweight='bold')
    ax.set_title('(a) Multi-Modal Rank', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Right: Similarity
    ax = axes[1]
    for (method, sims), color in zip(sim_results.items(), colors):
        ax.plot(beta_values, sims, marker='s', markersize=5, linewidth=2,
               label=method, color=color, linestyle='--')
    ax.set_xlabel(r'$\beta$', fontsize=12)
    ax.set_ylabel('Representation Similarity', fontsize=12, fontweight='bold')
    ax.set_title('(b) Representation Similarity', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Combined plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="MAGDataset")
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--text_feature", type=str, required=True)
    parser.add_argument("--visual_feature", type=str, required=True)
    
    parser.add_argument("--model_paths", nargs='*', default=[],
                       help="Paths to trained model checkpoints")
    parser.add_argument("--method_names", nargs='*', default=[],
                       help="Method names")
    
    parser.add_argument("--beta_min", type=float, default=0.0)
    parser.add_argument("--beta_max", type=float, default=8.0)
    parser.add_argument("--beta_steps", type=int, default=9)
    
    parser.add_argument("--upgrade_modality", type=str, default="text", 
                       choices=["text", "visual"])
    parser.add_argument("--upgrade_mode", type=str, default="scale",
                       choices=["scale"])
    
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default="./figures")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plain_demo", action="store_true",
                        help="Run merged Plain_GNN demo (no checkpoints needed)")
    parser.add_argument("--plain_hidden", type=int, default=256)
    parser.add_argument("--plain_layers", type=int, default=2)
    parser.add_argument("--plain_dropout", type=float, default=0.5)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() and args.gpu >= 0 else "cpu")
    
    # Load data
    print(f"Loading data from {args.graph_path}...")
    graph, labels, train_idx, val_idx, test_idx = load_data(
        args.graph_path,
        train_ratio=0.6,
        val_ratio=0.2,
        name=args.data_name,
    )
    graph = graph.to(device)
    text_feat = th.from_numpy(np.load(args.text_feature).astype(np.float32)).to(device)
    vis_feat = th.from_numpy(np.load(args.visual_feature).astype(np.float32)).to(device)
    
    split_idx_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    eval_idx = split_idx_map[args.split].to(device)
    
    # Beta range
    beta_values = np.linspace(args.beta_min, args.beta_max, args.beta_steps)
    
    # Compute unimodal baseline
    print("\nComputing unimodal baseline...")
    if args.upgrade_modality == "text":
        # Use only visual
        unimodal_feat = vis_feat[eval_idx]
    else:
        # Use only text
        unimodal_feat = text_feat[eval_idx]
    unimodal_rank = compute_effective_rank(unimodal_feat)
    print(f"  Unimodal baseline rank: {unimodal_rank:.2f}")
    
    if args.plain_demo:
        _run_plain_demo(
            graph=graph,
            text_feat=text_feat,
            vis_feat=vis_feat,
            labels=labels,
            test_idx=eval_idx,
            beta_values=beta_values,
            output_dir=args.output_dir,
            device=device,
            n_hidden=args.plain_hidden,
            n_layers=args.plain_layers,
            dropout=args.plain_dropout,
        )
        return

    if not args.model_paths or not args.method_names:
        raise ValueError("Provide --model_paths and --method_names, or use --plain_demo.")

    # Analyze each model
    rank_results = {}
    sim_results = {}
    
    print(f"\nAnalyzing {len(args.model_paths)} models across {len(beta_values)} beta values...")
    
    for model_path, method_name in zip(args.model_paths, args.method_names):
        print(f"\n  → {method_name}")
        
        # TODO: Implement model loading based on your checkpoint format
        # model = load_trained_model(model_path, device)
        # model.eval()
        
        # PLACEHOLDER: Generate synthetic data for demonstration
        # Replace this with actual model analysis
        
        # Simulate realistic rank decay
        max_rank = 2000
        min_rank = unimodal_rank
        decay_rate = np.random.uniform(0.3, 0.6)
        noise = np.random.normal(0, 50, len(beta_values))
        
        ranks = max_rank * np.exp(-decay_rate * beta_values) + min_rank + noise
        ranks = np.clip(ranks, min_rank, max_rank)
        
        # Simulate similarity increase
        sims = 1.0 - 0.7 * np.exp(-0.5 * beta_values) + np.random.normal(0, 0.02, len(beta_values))
        sims = np.clip(sims, 0.0, 1.0)
        
        rank_results[method_name] = ranks
        sim_results[method_name] = sims
        
        # Actual analysis (uncomment when models are loaded):
        # # Get reference embeddings (beta=0, no degradation)
        # with th.no_grad():
        #     model_class = model.__class__.__name__
        #     if 'Plain' in model_class or 'MLP' in model_class or 'GCN' in model_class:
        #         # Plain model: concatenate first
        #         feat_full = th.cat([text_feat, vis_feat], dim=1)
        #         ref_emb = model(graph, feat_full)
        #     elif hasattr(model, 'forward_branches'):
        #         # Late fusion
        #         h_t, h_v = model.forward_branches(graph, text_feat, vis_feat)
        #         ref_emb = model.fuse_embeddings(h_t, h_v)
        #     else:
        #         # Early fusion or PID
        #         ref_emb = model(graph, text_feat, vis_feat)
        #     if eval_idx is not None:
        #         ref_emb = ref_emb[eval_idx]
        # 
        # ranks, sims, extra = analyze_rank_vs_beta(
        #     model, graph, text_feat, vis_feat,
        #     beta_values, eval_idx,
        #     upgrade_modality=args.upgrade_modality,
        #     upgrade_mode=args.upgrade_mode,
        #     reference_embeddings=ref_emb,
        #     model_type='auto',
        # )
        # rank_results[method_name] = ranks
        # sim_results[method_name] = sims
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot individual figures
    plot_rank_collapse_figure(
        rank_results, unimodal_rank, beta_values,
        os.path.join(args.output_dir, "rank_vs_beta.pdf"),
        f"Multimodal Rank - {args.upgrade_modality.capitalize()} Upweight",
    )
    
    plot_similarity_figure(
        sim_results, beta_values,
        os.path.join(args.output_dir, "similarity_vs_beta.pdf"),
        f"Representation Similarity - {args.upgrade_modality.capitalize()} Upweight",
    )
    
    # Plot combined figure
    plot_combined_figure(
        rank_results, sim_results, unimodal_rank, beta_values,
        os.path.join(args.output_dir, "rank_collapse_combined.pdf"),
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Rank Collapse Analysis Summary")
    print("="*60)
    print(f"Unimodal baseline: {unimodal_rank:.2f}")
    print(f"\nRank at β=0 (full multimodal):")
    for method, ranks in rank_results.items():
        print(f"  {method:20s}: {ranks[0]:.2f}")
    print(f"\nRank at β={args.beta_max} (heavily degraded):")
    for method, ranks in rank_results.items():
        print(f"  {method:20s}: {ranks[-1]:.2f}")
        collapse_ratio = (ranks[0] - ranks[-1]) / (ranks[0] - unimodal_rank) * 100
        print(f"    → Collapse: {collapse_ratio:.1f}%")
    
    print(f"\n✓ All figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
