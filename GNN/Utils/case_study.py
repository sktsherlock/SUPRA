"""
Case Study Analysis for SUPRA

Provides two types of analysis on a trained SUPRA model:
1. Node-level prediction analysis: identifies nodes where Ut/Uv channels disagree most
2. Dirichlet Energy analysis: quantifies semantic sharpness preservation per channel

Usage:
    python -m GNN.Utils.case_study \
        --checkpoint PATH/to/checkpoint.pt \
        --data_name Movies \
        --graph_path /path/to/MoviesGraph.pt \
        --text_feature /path/to/text.npy \
        --visual_feature /path/to/vis.npy \
        --n_cases 10
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch as th
import numpy as np
import argparse
from typing import List, Dict, Optional

from GNN.SUPRA import SUPRA, _load_mag_context
from GNN.Utils.model_config import str2bool


# =============================================================================
# Dirichlet Energy Analysis
# =============================================================================

def compute_dirichlet_energy_vectorized(features: th.Tensor, adj: th.Tensor) -> float:
    """
    Compute Dirichlet Energy: E = trace(X^T L X) where L = D - A is the Laplacian.

    High Dirichlet Energy = sharp, distinctive features (nodes are well-separated)
    Low Dirichlet Energy = smooth, homogenized features (nodes look similar)

    Args:
        features: (num_nodes, feature_dim)
        adj: (num_nodes, num_nodes) adjacency matrix (sparse or dense)
    Returns:
        Scalar Dirichlet Energy value
    """
    # Convert sparse adjacency to dense if needed
    if adj.is_sparse:
        adj_dense = adj.to_dense()
    else:
        adj_dense = adj

    D = th.diag(adj_dense.sum(dim=1))
    L = D - adj_dense  # Graph Laplacian

    # E = trace(X^T L X) = sum_{i,j} L_ij * <x_i, x_j>
    energy = th.sum(L @ features * features).item()
    return energy / 2  # Divide by 2 for undirected graph (L is symmetric)


def analyze_dirichlet_energy(
    model,
    graph,
    text_feat: th.Tensor,
    visual_feat: th.Tensor,
    labels: th.Tensor,
    dataset_name: str,
) -> Dict:
    """
    Analyze Dirichlet Energy at different stages of the model.

    Compares energy of:
    - Raw concatenated features (baseline)
    - C channel output (shared GNN, expected to smooth)
    - Ut channel output (unique text, expected to preserve energy)
    - Uv channel output (unique visual, expected to preserve energy)
    """
    model.eval()
    # Build adjacency matrix from edge list (avoids DGL sparse library dependency)
    # Use UN-NORMALIZED adjacency for correct Dirichlet Energy (L = D - A must be PSD)
    feat_device = text_feat.device
    srcs, dsts = graph.all_edges()
    num_nodes = graph.num_nodes()
    adj = th.zeros(num_nodes, num_nodes, dtype=th.float32, device=feat_device)
    adj[srcs, dsts] = 1.0
    adj = adj + adj.T  # Make symmetric (undirected), no normalization

    results = {'dataset': dataset_name}

    # 1. Raw concatenated features
    raw_concat = th.cat([text_feat, visual_feat], dim=1)
    results['energy_raw'] = compute_dirichlet_energy_vectorized(raw_concat, adj)

    with th.no_grad():
        # 2. Forward pass through all channels
        out = model.forward_multiple(graph, text_feat, visual_feat)
        logits_C = out.logits_C_0
        logits_Ut = out.logits_Ut_0
        logits_Uv = out.logits_Uv_0

        results['energy_C'] = compute_dirichlet_energy_vectorized(logits_C, adj)
        results['energy_Ut'] = compute_dirichlet_energy_vectorized(logits_Ut, adj)
        results['energy_Uv'] = compute_dirichlet_energy_vectorized(logits_Uv, adj)

    # 3. Compute relative energy preservation (higher = more preserved)
    for channel in ['C', 'Ut', 'Uv']:
        key = f'energy_{channel}'
        drop_key = f'energy_drop_{channel}'
        results[drop_key] = (results['energy_raw'] - results[key]) / (results['energy_raw'] + 1e-10)

    return results


def print_dirichlet_analysis(results: Dict):
    """Print Dirichlet Energy analysis in a readable format."""
    print("\n" + "=" * 70)
    print(f"Dirichlet Energy Analysis — {results['dataset']}")
    print("=" * 70)
    print(f"  Raw concat features:        {results['energy_raw']:.6f}  (baseline)")
    print(f"  C channel (GNN):            {results['energy_C']:.6f}  "
          f"(drop {results['energy_drop_C']*100:+.1f}%)")
    print(f"  Ut channel (unique text):   {results['energy_Ut']:.6f}  "
          f"(drop {results['energy_drop_Ut']*100:+.1f}%)")
    print(f"  Uv channel (unique visual): {results['energy_Uv']:.6f}  "
          f"(drop {results['energy_drop_Uv']*100:+.1f}%)")
    print("-" * 70)
    print("  Interpretation:")
    print("    - Ut/Uv energy drop < C energy drop → unique channels preserve semantics")
    print("    - Small unique channel drop → modality contamination is mitigated")
    print("=" * 70 + "\n")


# =============================================================================
# Node Prediction Case Study
# =============================================================================

def analyze_node_predictions(
    model,
    graph,
    text_feat: th.Tensor,
    visual_feat: th.Tensor,
    labels: th.Tensor,
    train_idx: th.Tensor,
    val_idx: th.Tensor,
    test_idx: th.Tensor,
    n_cases: int = 10,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Analyze per-channel predictions on individual nodes.

    Identifies nodes where Ut and Uv channels disagree most, and shows how
    each channel and the fused prediction perform on those cases.
    """
    model.eval()
    with th.no_grad():
        out = model.forward_multiple(graph, text_feat, visual_feat)
        logits_C = out.logits_C_0
        logits_Ut = out.logits_Ut_0
        logits_Uv = out.logits_Uv_0

    prob_C = th.softmax(logits_C, dim=1)
    prob_Ut = th.softmax(logits_Ut, dim=1)
    prob_Uv = th.softmax(logits_Uv, dim=1)
    prob_final = th.softmax(logits_C + logits_Ut + logits_Uv, dim=1)

    pred_C = logits_C.argmax(dim=1)
    pred_Ut = logits_Ut.argmax(dim=1)
    pred_Uv = logits_Uv.argmax(dim=1)
    pred_final = (logits_C + logits_Ut + logits_Uv).argmax(dim=1)

    # Compute Ut vs Uv probability divergence (L1 distance)
    ut_uv_disagreement = th.abs(prob_Ut - prob_Uv).sum(dim=1)

    # Focus on test set nodes for evaluation
    test_disagreement = ut_uv_disagreement[test_idx]
    test_labels = labels[test_idx]
    test_nodes = test_idx

    # Get top-k most divergent nodes in test set
    top_k = min(n_cases, len(test_idx))
    top_disagreement_vals, top_indices = test_disagreement.topk(top_k)
    # top_indices are local indices into test_disagreement/test_nodes
    # Convert to global node IDs
    global_indices = test_nodes[top_indices]  # (top_k,)

    results = {
        'dataset': None,
        'n_cases': top_k,
        'cases': [],
    }

    for rank in range(top_k):
        actual_idx = global_indices[rank].item()
        node_id = actual_idx
        label = test_labels[top_indices[rank]].item()
        disagreement = top_disagreement_vals[rank].item()

        p_C = pred_C[actual_idx].item()
        p_Ut = pred_Ut[actual_idx].item()
        p_Uv = pred_Uv[actual_idx].item()
        p_final = pred_final[actual_idx].item()

        conf_C = prob_C[actual_idx].max().item()
        conf_Ut = prob_Ut[actual_idx].max().item()
        conf_Uv = prob_Uv[actual_idx].max().item()
        conf_final = prob_final[actual_idx].max().item()

        def fmt_pred(p, label):
            ok = "✓" if p == label else "✗"
            name = class_names[p] if class_names else str(p)
            return f"{name} {ok}", p == label

        label_name = class_names[label] if class_names else str(label)
        c_str, c_ok = fmt_pred(p_C, label)
        ut_str, ut_ok = fmt_pred(p_Ut, label)
        uv_str, uv_ok = fmt_pred(p_Uv, label)
        final_str, final_ok = fmt_pred(p_final, label)

        results['cases'].append({
            'rank': rank + 1,
            'node_id': node_id,
            'label': label_name,
            'disagreement': disagreement,
            'pred_C': c_str,
            'pred_Ut': ut_str,
            'pred_Uv': uv_str,
            'pred_final': final_str,
            'all_correct': c_ok and ut_ok and uv_ok and final_ok,
            'C_only_correct': c_ok and not ut_ok and not uv_ok,
            'Ut_only_correct': ut_ok and not c_ok and not uv_ok,
            'Uv_only_correct': uv_ok and not c_ok and not ut_ok,
            'fusion_helps': final_ok and not (c_ok or ut_ok or uv_ok),
            'confidence_C': conf_C,
            'confidence_Ut': conf_Ut,
            'confidence_Uv': conf_Uv,
            'confidence_final': conf_final,
        })

    return results


def print_node_analysis(results: Dict):
    """Print node case study in a readable format."""
    cases = results['cases']
    n_correct = sum(1 for c in cases if c['all_correct'])
    n_ut_only = sum(1 for c in cases if c['Ut_only_correct'])
    n_uv_only = sum(1 for c in cases if c['Uv_only_correct'])
    n_c_only = sum(1 for c in cases if c['C_only_correct'])
    n_fusion_helps = sum(1 for c in cases if c['fusion_helps'])

    print("\n" + "=" * 70)
    print(f"Node Prediction Case Study (top {results['n_cases']} Ut/Uv divergent nodes)")
    print("=" * 70)
    print(f"  All correct: {n_correct}/{len(cases)}  |  "
          f"Ut-only: {n_ut_only}  |  Uv-only: {n_uv_only}  |  "
          f"C-only: {n_c_only}  |  Fusion helps: {n_fusion_helps}")
    print("-" * 70)

    for c in cases:
        print(f"\n  [Rank {c['rank']}] Node {c['node_id']} | Label: {c['label']} | "
              f"Disagreement: {c['disagreement']:.4f}")
        print(f"    C channel:     {c['pred_C']:10s}  (conf: {c['confidence_C']:.3f})")
        print(f"    Ut channel:    {c['pred_Ut']:10s}  (conf: {c['confidence_Ut']:.3f})")
        print(f"    Uv channel:    {c['pred_Uv']:10s}  (conf: {c['confidence_Uv']:.3f})")
        print(f"    Fusion:        {c['pred_final']:10s}  (conf: {c['confidence_final']:.3f})")

    print("\n" + "-" * 70)
    print("  Legend: ✓ = correct, ✗ = wrong")
    print("  'Fusion helps' = all channels wrong individually but fusion is correct")
    print("=" * 70 + "\n")


# =============================================================================
# Per-Class Accuracy Analysis
# =============================================================================

def analyze_per_class_accuracy(
    model,
    graph,
    text_feat: th.Tensor,
    visual_feat: th.Tensor,
    labels: th.Tensor,
    train_idx: th.Tensor,
    val_idx: th.Tensor,
    test_idx: th.Tensor,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute per-class accuracy for each channel to see which classes
    are dominated by which modality.
    """
    model.eval()
    with th.no_grad():
        out = model.forward_multiple(graph, text_feat, visual_feat)
        logits_C = out.logits_C_0
        logits_Ut = out.logits_Ut_0
        logits_Uv = out.logits_Uv_0

    pred_C = logits_C.argmax(dim=1)
    pred_Ut = logits_Ut.argmax(dim=1)
    pred_Uv = logits_Uv.argmax(dim=1)
    pred_final = (logits_C + logits_Ut + logits_Uv).argmax(dim=1)

    n_classes = int(labels.max().item()) + 1
    results = {
        'n_classes': n_classes,
        'classes': [],
    }

    for cls in range(n_classes):
        mask = (labels == cls)
        test_mask = mask[test_idx]
        test_idx_cls = test_idx[test_mask]

        if len(test_idx_cls) == 0:
            continue

        y = labels[test_idx_cls]
        acc_C = (pred_C[test_idx_cls] == y).float().mean().item()
        acc_Ut = (pred_Ut[test_idx_cls] == y).float().mean().item()
        acc_Uv = (pred_Uv[test_idx_cls] == y).float().mean().item()
        acc_final = (pred_final[test_idx_cls] == y).float().mean().item()

        name = class_names[cls] if class_names else f"Class {cls}"

        # Determine which channel dominates this class
        channel_scores = {'C': acc_C, 'Ut': acc_Ut, 'Uv': acc_Uv}
        best_channel = max(channel_scores, key=channel_scores.get)
        best_acc = channel_scores[best_channel]

        results['classes'].append({
            'class_id': cls,
            'name': name,
            'n_test': len(test_idx_cls),
            'acc_C': acc_C,
            'acc_Ut': acc_Ut,
            'acc_Uv': acc_Uv,
            'acc_final': acc_final,
            'best_channel': best_channel,
            'best_acc': best_acc,
            'fusion_gain': acc_final - max(acc_C, acc_Ut, acc_Uv),
        })

    return results


def print_per_class_analysis(results: Dict):
    """Print per-class accuracy analysis."""
    print("\n" + "=" * 70)
    print(f"Per-Class Accuracy Analysis ({results['n_classes']} classes)")
    print("=" * 70)
    print(f"  {'Class':<20} {'N':>4}  {'C':>6}  {'Ut':>6}  {'Uv':>6}  {'Fusion':>6}  {'Best':>4}  {'Gain':>6}")
    print("  " + "-" * 62)

    for cls in results['classes']:
        print(f"  {cls['name']:<20} {cls['n_test']:>4}  "
              f"{cls['acc_C']*100:>5.1f}% "
              f"{cls['acc_Ut']*100:>5.1f}% "
              f"{cls['acc_Uv']*100:>5.1f}% "
              f"{cls['acc_final']*100:>5.1f}% "
              f"  {cls['best_channel']:<3} "
              f"{cls['fusion_gain']*100:>+5.1f}%")

    print("=" * 70 + "\n")


# =============================================================================
# Main
# =============================================================================

def args_init():
    parser = argparse.ArgumentParser("SUPRA Case Study Analysis",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt file)")
    parser.add_argument("--data_name", type=str, required=True,
                        help="Dataset name (Movies, Grocery, Toys, Reddit-M)")
    parser.add_argument("--graph_path", type=str, required=True,
                        help="Path to graph .pt file")
    parser.add_argument("--text_feature", type=str, required=True,
                        help="Path to text feature .npy file")
    parser.add_argument("--visual_feature", type=str, required=True,
                        help="Path to visual feature .npy file")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--n_cases", type=int, default=10,
                        help="Number of top-disagreement nodes to show")
    parser.add_argument("--undirected", type=str2bool, default=True,
                        help="Treat graph as undirected")
    parser.add_argument("--selfloop", type=str2bool, default=True,
                        help="Add self-loop to graph")
    parser.add_argument("--train_ratio", type=float, default=0.6,
                        help="Training split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Validation split ratio")
    parser.add_argument("--inductive", type=str2bool, default=False,
                        help="Inductive split")
    parser.add_argument("--class_names", type=str, default=None,
                        help="Comma-separated class names (optional)")
    parser.add_argument("--analyze_dirichlet", action="store_true", default=True,
                        help="Run Dirichlet Energy analysis")
    parser.add_argument("--analyze_nodes", action="store_true", default=True,
                        help="Run node case study analysis")
    parser.add_argument("--analyze_per_class", action="store_true", default=True,
                        help="Run per-class accuracy analysis")
    return parser


def main():
    parser = args_init()
    args = parser.parse_args()

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() and args.gpu != -1 else "cpu")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = th.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get('args', None)
    text_in_dim = checkpoint.get('text_in_dim')
    vis_in_dim = checkpoint.get('vis_in_dim')
    embed_dim = checkpoint.get('embed_dim')
    n_classes = checkpoint.get('n_classes')

    print(f"  Model: text_in={text_in_dim}, vis_in={vis_in_dim}, "
          f"embed_dim={embed_dim}, n_classes={n_classes}")

    # Load data
    print(f"Loading data for {args.data_name}...")
    data_args = argparse.Namespace(
        data_name=args.data_name,
        graph_path=args.graph_path,
        text_feature=args.text_feature,
        visual_feature=args.visual_feature,
        undirected=args.undirected,
        selfloop=args.selfloop,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        inductive=args.inductive,
        fewshots=False,
    )
    graph, observe_graph, labels, train_idx, val_idx, test_idx, text_feat, visual_feat, _ = \
        _load_mag_context(data_args, device)

    print(f"  Graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
    print(f"  Features: text {text_feat.shape}, visual {visual_feat.shape}")
    print(f"  Classes: {n_classes}, Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Build model
    model = SUPRA(
        text_in_dim=text_in_dim,
        vis_in_dim=vis_in_dim,
        embed_dim=embed_dim,
        n_classes=n_classes,
        dropout=0.0,  # No dropout needed for analysis
        args=ckpt_args,
        device=device,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded and ready.\n")

    # Parse class names
    class_names = None
    if args.class_names:
        class_names = [s.strip() for s in args.class_names.split(',')]

    # Run analyses
    if args.analyze_dirichlet:
        energy_results = analyze_dirichlet_energy(
            model, graph, text_feat, visual_feat, labels, args.data_name
        )
        print_dirichlet_analysis(energy_results)

    if args.analyze_nodes:
        node_results = analyze_node_predictions(
            model, graph, text_feat, visual_feat, labels,
            train_idx, val_idx, test_idx,
            n_cases=args.n_cases,
            class_names=class_names,
        )
        print_node_analysis(node_results)

    if args.analyze_per_class:
        per_class_results = analyze_per_class_accuracy(
            model, graph, text_feat, visual_feat, labels,
            train_idx, val_idx, test_idx,
            class_names=class_names,
        )
        print_per_class_analysis(per_class_results)


if __name__ == "__main__":
    main()
