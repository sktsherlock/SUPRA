"""
Graph Degradation Utilities for Controlled Degradation Experiments
================================================================

Two types of controlled degradation:
1. Feature noise injection  — validates the feature dimension (sigma_epsilon^2)
2. Edge rewiring            — validates the topology dimension (beta)

Usage:
    from GNN.Utils.graph_degradation import inject_feature_noise, rewire_edges
    noisy_text, noisy_vis = inject_feature_noise(text_feat, visual_feat, ratio=0.5, seed=42)
    rewired_graph = rewire_edges(dgl_graph, ratio=0.3, seed=42)
"""

import random
from typing import Any, Tuple

import dgl
import numpy as np
import torch as th


def inject_feature_noise(
    text_feat: th.Tensor,
    visual_feat: th.Tensor,
    ratio: float,
    seed: int,
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Inject Gaussian noise into features.

    Formula: X_noisy = X + ratio * std(X) * N(0, 1)

    Each feature dimension gets its own noise scaled by that dimension's
    standard deviation across all nodes, matching the empirical distribution.

    Args:
        text_feat:   (N, D_t) text feature tensor
        visual_feat: (N, D_v) visual feature tensor
        ratio:       noise strength multiplier
                     0.0 = no noise, 1.0 = noise std = feature std
        seed:        random seed for reproducibility

    Returns:
        (noisy_text, noisy_visual) tensors with same shape as inputs
    """
    if ratio <= 0.0:
        return text_feat, visual_feat

    # Per-dimension std across all nodes; clamp to avoid degenerate zero-std dims
    text_std = text_feat.std(dim=0, unbiased=False).clamp_min(1e-8)
    vis_std = visual_feat.std(dim=0, unbiased=False).clamp_min(1e-8)

    noise_text = th.randn_like(text_feat) * text_std * ratio
    noise_vis = th.randn_like(visual_feat) * vis_std * ratio

    return text_feat + noise_text, visual_feat + noise_vis


def rewire_edges(
    dgl_graph,
    ratio: float,
    seed: int,
) -> Any:
    """
    Randomly rewire a fraction of edges in the graph.

    For each edge selected for rewiring:
      1. Remove the original edge
      2. Add a new edge between two randomly chosen distinct nodes
         (avoiding self-loops and duplicate edges)

    The total number of edges is preserved (approximately) since we rewire
    rather than add/remove. Node degree distribution is approximately
    maintained for small ratios.

    Args:
        dgl_graph:  DGL graph object
        ratio:       fraction of edges to rewire (0.0 to 1.0)
        seed:        random seed for reproducibility

    Returns:
        A new DGL graph with rewired edges (original graph is not modified)
    """
    if ratio <= 0.0:
        return dgl_graph

    src, dst = dgl_graph.all_edges()
    num_edges = src.numel()
    num_nodes = dgl_graph.num_nodes()
    num_to_rewire = max(1, int(num_edges * ratio))

    # Set Python random and numpy seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    rng_state = th.get_rng_state()
    th.manual_seed(seed)

    # Select edges to rewire — th.manual_seed(seed) already called at caller side;
    # randperm output defaults to CPU so move to graph device for remove_edges compatibility
    rewire_eids = th.randperm(num_edges)[:num_to_rewire].to(src.device)
    old_src = src[rewire_eids].clone()
    old_dst = dst[rewire_eids].clone()

    # Build set of existing edges for fast duplicate avoidance
    existing_edges = set()
    for s, d in zip(src.tolist(), dst.tolist()):
        existing_edges.add((int(s), int(d)))
        existing_edges.add((int(d), int(s)))  # undirected

    new_src_list = []
    new_dst_list = []

    for s, d in zip(old_src.tolist(), old_dst.tolist()):
        # Try to find a valid new target
        attempts = 0
        max_attempts = 50
        new_tgt = d
        while attempts < max_attempts:
            new_tgt = random.randint(0, num_nodes - 1)
            if new_tgt != s and (s, new_tgt) not in existing_edges and (new_tgt, s) not in existing_edges:
                break
            attempts += 1

        # If we couldn't find a unique non-self edge, skip this rewiring
        if attempts < max_attempts:
            new_src_list.append(s)
            new_dst_list.append(new_tgt)
            # Temporarily add to existing set
            existing_edges.add((s, new_tgt))
            existing_edges.add((new_tgt, s))

    th.set_rng_state(rng_state)

    # Build mask of edges to keep (all except the ones being rewired)
    keep_mask = th.ones(num_edges, dtype=th.bool)
    keep_mask[rewire_eids] = False
    keep_eids = keep_mask.nonzero(as_tuple=True)[0]

    # Keep edges that are NOT being rewired
    keep_src = src[keep_eids]
    keep_dst = dst[keep_eids]

    # Rebuild graph from kept edges (avoids clone+remove_edges None issue on CUDA)
    new_graph = dgl.graph((keep_src, keep_dst), num_nodes=num_nodes)
    new_graph.add_edges(keep_dst, keep_src)  # undirected; in-place

    # Add rewired edges
    if new_src_list:
        new_src_t = th.tensor(new_src_list, dtype=src.dtype, device=src.device)
        new_dst_t = th.tensor(new_dst_list, dtype=dst.dtype, device=src.device)
        new_graph.add_edges(new_src_t, new_dst_t)  # in-place
        new_graph.add_edges(new_dst_t, new_src_t)  # undirected; in-place

    new_graph.create_formats_()
    return new_graph
