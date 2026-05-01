"""
NTSFormer Baseline for Multimodal Node Classification.

This is a simplified version of the NTSFormer Teacher Model, adapted for
standard multimodal node classification (without cold-start settings).

Based on: "NTSFormer: Neural Topological Symbolic Reasoning for Cold-Start Multimodal Learning"

Key features:
- Uses SIGN (Scalable Inception-inspired GNN) for multi-hop feature pre-computation
- MultiGroupTransformer for modality fusion
- Removes cold-start specific components for fair comparison
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

try:
    import wandb
except Exception:
    wandb = None

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_ROOT)

from GNN.GraphData import load_data, set_seed
from GNN.Utils.LossFunction import cross_entropy, get_metric
from GNN.Utils.NodeClassification import (
    initialize_early_stopping,
    initialize_optimizer_and_scheduler,
    adjust_learning_rate_if_needed,
    log_progress,
    _make_noisy_feature,
)
from GNN.Utils.model_config import add_common_args
from GNN.Utils.result_logger import build_result_row, update_best_result_csv, append_result_csv


# ==============================================================================
# SIGN Pre-computation (Graph-based Multi-hop Feature Aggregation)
# ==============================================================================

def compute_gcn_weight(g, norm="both"):
    """Compute GCN-style normalized edge weights.

    Args:
        g: DGL graph
        norm: Normalization mode ("both" = symmetric normalization)

    Returns:
        Edge weights of shape [num_edges]
    """
    in_degs = g.in_degrees().float()
    out_degs = g.out_degrees().float()

    if norm == "both":
        src_norm = th.pow(out_degs, -0.5)
        dst_norm = th.pow(in_degs, -0.5)
    elif norm == "right":
        src_norm = th.ones_like(out_degs)
        dst_norm = th.pow(in_degs, -1.0)
    elif norm == "left":
        src_norm = th.pow(out_degs, -1.0)
        dst_norm = th.ones_like(in_degs)
    else:
        raise NotImplementedError(f"norm {norm} not implemented")

    with g.local_scope():
        g.ndata['src_norm'] = src_norm
        g.ndata['dst_norm'] = dst_norm
        g.apply_edges(fn.u_mul_v('src_norm', 'dst_norm', 'gcn_weight'))
        gcn_weight = g.edata['gcn_weight']

    return gcn_weight


def sign_pre_compute(g, x, k, include_input=True, alpha=0.0, norm="both",
                      remove_self_loop=False, device=None):
    """Pre-compute multi-hop neighbor aggregated features (SIGN).

    This computes:
        h^(i) = A^i @ x  (or normalized version)
    for i = 0, 1, ..., k

    The pre-computed features are then used as input to the Transformer.

    Args:
        g: DGL graph
        x: Input features [num_nodes, feature_dim]
        k: Number of hops to propagate
        include_input: If True, include h^(0) = x in the output
        alpha: Residual coefficient for h^(i+1) = (1-alpha)*h^(i+1) + alpha*x
        norm: Normalization mode for GCN weights
        remove_self_loop: Whether to remove self-loops before propagation
        device: Device to use

    Returns:
        List of tensors [h^(0), h^(1), ..., h^(k)], each [num_nodes, feature_dim]
    """
    if remove_self_loop:
        g = dgl.remove_self_loop(g)

    gcn_weight = compute_gcn_weight(g, norm=norm).to(x.dtype)
    g.edata['gcn_weight'] = gcn_weight

    h_list = []
    h = x

    for k_ in range(k):
        # Message passing: aggregate from neighbors
        g.ndata['h'] = h
        g.update_all(fn.u_mul_e('h', 'gcn_weight', 'm'), fn.sum('m', 'h'))
        h = g.ndata.pop('h')

        # Residual connection: h = (1-alpha) * h + alpha * x
        if alpha > 0.0:
            h = (1.0 - alpha) * h + alpha * x

        h_list.append(h)

    if include_input:
        h_list = [x] + h_list

    return h_list


def sign_pre_compute_batched(g, feat_list, k, include_input=True, alpha=0.0,
                              norm="both", remove_self_loop=False, device=None):
    """Pre-compute multi-hop SIGN for multiple modalities at once.

    Runs ONE graph propagation pass that updates ALL modality features
    simultaneously, then splits results back. This is the key optimization
    that matches the official coldgnn behavior.

    Args:
        g: DGL graph
        feat_list: List of feature tensors [(N, d1), (N, d2), ...]
        k: Number of hops
        include_input: Include h^(0) = x in output
        alpha: Residual coefficient
        norm: GCN normalization mode
        remove_self_loop: Remove self loops
        device: Target device

    Returns:
        List of lists: [[h0^0, h0^1, ...], [h1^0, h1^1, ...], ...]
    """
    if remove_self_loop:
        g = dgl.remove_self_loop(g)

    # Stack all features: [N, sum(d_i)]
    stacked = th.cat(feat_list, dim=-1)
    gcn_weight = compute_gcn_weight(g, norm=norm).to(stacked.dtype)
    g.edata['gcn_weight'] = gcn_weight

    # Split point dimensions
    split_dims = [f.shape[-1] for f in feat_list]

    h_list_per_modality = [[] for _ in feat_list]
    h = stacked

    for k_ in range(k):
        g.ndata['h'] = h
        g.update_all(fn.u_mul_e('h', 'gcn_weight', 'm'), fn.sum('m', 'h'))
        h = g.ndata.pop('h')

        if alpha > 0.0:
            h = (1.0 - alpha) * h + alpha * stacked

        # Split back to per-modality
        h_splits = th.split(h, split_dims, dim=-1)
        for mod_idx, mod_h in enumerate(h_splits):
            h_list_per_modality[mod_idx].append(mod_h)

    if include_input:
        for mod_idx, feat in enumerate(feat_list):
            h_list_per_modality[mod_idx] = [feat] + h_list_per_modality[mod_idx]

    return h_list_per_modality


# ==============================================================================
# NTSFormer Core Components
# ==============================================================================

class MyLinear(nn.Linear):
    """Linear layer with Xavier uniform initialization."""
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MyPReLU(nn.Module):
    """Parametric ReLU activation."""
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.alpha = nn.Parameter(th.ones(num_parameters) * init)

    def forward(self, x):
        return F.prelu(x, self.alpha)


class MyMLP(nn.Module):
    """MLP block with optional batch norm and dropout."""
    def __init__(self, in_channels, units_list, activation='prelu', drop_rate=0.0,
                 bn=False, output_activation=None, output_drop_rate=0.0, output_bn=False):
        super().__init__()
        layers = []
        units_list = [in_channels] + units_list

        for i in range(len(units_list) - 1):
            layers.append(MyLinear(units_list[i], units_list[i + 1]))
            if i < len(units_list) - 2:
                if bn:
                    layers.append(nn.BatchNorm1d(units_list[i + 1]))
                layers.append(MyPReLU() if activation == 'prelu' else nn.ReLU())
                layers.append(nn.Dropout(drop_rate))
            else:
                if output_bn:
                    layers.append(nn.BatchNorm1d(units_list[i + 1]))
                if output_activation:
                    layers.append(MyPReLU() if output_activation == 'prelu' else nn.ReLU())
                layers.append(nn.Dropout(output_drop_rate))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class TransformerBlock(nn.Module):
    """Transformer encoder block for multi-hop feature fusion."""
    def __init__(self, channels, num_heads=1, ff_hidden=None, drop_rate=0.1,
                 att_drop_rate=0.0, use_residual=True):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        ff_hidden = ff_hidden or channels * 4

        # Multi-head attention
        self.q_proj = MyLinear(channels, channels)
        self.k_proj = MyLinear(channels, channels)
        self.v_proj = MyLinear(channels, channels)
        self.out_proj = MyLinear(channels, channels)
        self.att_drop = nn.Dropout(att_drop_rate)

        # Feed-forward network
        self.ff = nn.Sequential(
            MyLinear(channels, ff_hidden),
            MyPReLU(),
            nn.Dropout(drop_rate),
            MyLinear(ff_hidden, channels),
        )

        # Layer norm
        self.ln1 = nn.LayerNorm(channels)
        self.ln2 = nn.LayerNorm(channels)
        self.use_residual = use_residual

    def forward(self, x, mask=None):
        # x: [batch, seq_len, channels]
        batch_size, seq_len, _ = x.shape

        # Multi-head attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # Scaled dot-product attention
        scale = np.sqrt(self.channels / self.num_heads)
        att = th.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            att = att.masked_fill(mask == 0, -1e9)

        att = F.softmax(att, dim=-1)
        att = self.att_drop(att)

        # Apply attention to values
        out = th.matmul(att, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)

        # Residual connection + layer norm
        h = (x + out) if self.use_residual else out
        h = self.ln1(h)

        # Feed-forward + residual
        ff_out = self.ff(h)
        out = (h + ff_out) if self.use_residual else ff_out
        out = self.ln2(out)

        return out


class NTSFormerModel(nn.Module):
    """NTSFormer with SIGN pre-computation for multimodal node classification.

    Architecture:
        1. SIGN Pre-compute: Aggregate multi-hop neighbor features for each modality
        2. Multi-hop Fusion: Use Transformer to model cross-hop interactions
        3. Modality Fusion: Combine text and visual representations
    """
    def __init__(
        self,
        text_feat_dim: int,
        vis_feat_dim: int,
        embed_dim: int,
        n_classes: int,
        dropout: float,
        args,
        device: th.device,
        # NTSFormer specific
        num_tf_layers: int = 2,
        num_heads: int = 2,
        sign_k: int = 3,
        sign_alpha: float = 0.0,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = int(embed_dim)
        self.n_classes = int(n_classes)
        self.sign_k = sign_k

        # Input projection for multi-hop features
        # After SIGN: each modality has (sign_k + 1) * original_dim
        multi_hop_text_dim = (sign_k + 1) * text_feat_dim
        multi_hop_vis_dim = (sign_k + 1) * vis_feat_dim

        # Project multi-hop features to embed_dim
        self.text_proj = MyMLP(
            multi_hop_text_dim,
            [embed_dim] * (num_tf_layers - 1) if num_tf_layers > 1 else [],
            activation='prelu',
            drop_rate=dropout,
        )
        self.vis_proj = MyMLP(
            multi_hop_vis_dim,
            [embed_dim] * (num_tf_layers - 1) if num_tf_layers > 1 else [],
            activation='prelu',
            drop_rate=dropout,
        )

        # Modality token for cross-modality reasoning
        self.modality_token = nn.Parameter(th.randn(1, 1, embed_dim) * 0.02)

        # Transformer layers for multi-hop fusion within each modality
        self.hop_transformer = nn.ModuleList([
            TransformerBlock(
                channels=embed_dim,
                num_heads=num_heads,
                drop_rate=dropout,
                att_drop_rate=dropout,
            )
            for _ in range(num_tf_layers)
        ])

        # CLS token for aggregation
        self.cls_token = nn.Parameter(th.randn(1, 1, embed_dim) * 0.02)

        # Final classifier
        self.classifier = MyLinear(embed_dim, n_classes)

        # Dropout
        self.input_dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        # Use named_modules iteration (non-recursive via stack) to avoid deep recursion
        # Skip nn.Module instances in iteration since their reset_parameters()
        # does NOT recurse into children by default - calling it per-module
        # would cause 992-level recursion through the module tree.
        visited = set()
        stack = list(self.modules())
        while stack:
            module = stack.pop()
            if id(module) in visited:
                continue
            visited.add(id(module))
            if hasattr(module, 'reset_parameters') and not isinstance(module, nn.Module):
                module.reset_parameters()

    def forward(self, graph, text_feat: th.Tensor, vis_feat: th.Tensor,
                text_h_list=None, vis_h_list=None) -> th.Tensor:
        """Forward pass.

        Args:
            graph: DGL graph (used for SIGN pre-computation)
            text_feat: Text features [N, text_dim]
            vis_feat: Visual features [N, vis_dim]
            text_h_list: Pre-computed multi-hop text features (optional)
            vis_h_list: Pre-computed multi-hop visual features (optional)

        Returns:
            logits: [N, num_classes]
        """
        # SIGN pre-computation for multi-hop features
        if text_h_list is None:
            text_h_list = sign_pre_compute(
                graph, text_feat, k=self.sign_k, include_input=True,
                alpha=0.0, device=text_feat.device
            )
        if vis_h_list is None:
            vis_h_list = sign_pre_compute(
                graph, vis_feat, k=self.sign_k, include_input=True,
                alpha=0.0, device=vis_feat.device
            )

        # Concatenate multi-hop features
        text_h = th.cat(text_h_list, dim=-1)  # [N, (k+1) * text_dim]
        vis_h = th.cat(vis_h_list, dim=-1)     # [N, (k+1) * vis_dim]

        # Project to embed_dim
        text_h = self.text_proj(text_h)  # [N, embed_dim]
        vis_h = self.vis_proj(vis_h)      # [N, embed_dim]

        # Reshape for transformer: [N, 1, embed_dim]
        text_h = text_h.unsqueeze(1)
        vis_h = vis_h.unsqueeze(1)

        # Stack modalities and add special tokens
        x = th.stack([text_h, vis_h], dim=1)  # [N, 2, 1, embed_dim]
        batch_size, _, _, channels = x.shape
        x = x.view(batch_size, 2, channels)  # [N, 2, embed_dim]

        # Add modality token
        mod_token = self.modality_token.expand(batch_size, -1, -1)
        x = th.cat([mod_token, x], dim=1)  # [N, 3, embed_dim]

        # Add CLS token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = th.cat([x, cls_token], dim=1)  # [N, 4, embed_dim]

        # Apply input dropout
        x = self.input_dropout(x)

        # Apply transformer layers
        for tf_layer in self.hop_transformer:
            x = tf_layer(x)

        # Extract CLS token for classification
        cls_output = x[:, -1]  # Last token is CLS

        # Classification
        logits = self.classifier(cls_output)

        return logits


# ==============================================================================
# Training Functions
# ==============================================================================

def _build_ntsformer_model(args, text_feat_dim, vis_feat_dim, n_classes, device):
    """Build NTSFormer model with given configuration."""
    embed_dim = int(args.n_hidden)

    model = NTSFormerModel(
        text_feat_dim=text_feat_dim,
        vis_feat_dim=vis_feat_dim,
        embed_dim=embed_dim,
        n_classes=n_classes,
        dropout=float(args.dropout),
        args=args,
        device=device,
        num_tf_layers=getattr(args, 'nts_num_tf_layers', 2),
        num_heads=getattr(args, 'nts_num_heads', 2),
        sign_k=getattr(args, 'nts_sign_k', 3),
        sign_alpha=getattr(args, 'nts_sign_alpha', 0.0),
    ).to(device)

    return model


def _pre_compute_sign_features(graph, text_feat, vis_feat, sign_k, device, cache_key=None,
                                force_device=None):
    """Pre-compute SIGN features for text and visual modalities.

    Results are cached to ~/.cache/supra_nts_sign/ to avoid recomputing
    when running the same dataset+features across different hyperparams.

    Uses sign_pre_compute_batched to run ONE graph propagation pass for
    both modalities simultaneously (matching official coldgnn behavior).
    Automatically uses GPU if graph is already on GPU.
    """
    import os

    t0 = time.time()

    # Build cache path from args if not provided
    if cache_key is None:
        # caller must provide cache_key; this branch is only for backwards compat
        raise ValueError("cache_key must be provided")

    cache_dir = os.path.expanduser("~/.cache/supra_nts_sign")
    cache_file = os.path.join(cache_dir, f"sign_{cache_key}.pt")

    if os.path.exists(cache_file):
        print(f"[CACHE HIT] Loading cached SIGN features from {cache_file}")
        cached = th.load(cache_file, map_location=device)
        text_h_list = [h.to(device) for h in cached['text_h_list']]
        vis_h_list = [h.to(device) for h in cached['vis_h_list']]
        print(f"SIGN loaded from cache. Text multi-hop shape: {[h.shape for h in text_h_list]} "
              f"({time.time()-t0:.1f}s)")
        return text_h_list, vis_h_list

    print(f"Pre-computing SIGN features with k={sign_k}... (may take a while on large graphs)")

    # Determine compute device: use GPU if graph is already on GPU
    compute_on_gpu = (force_device == 'gpu') or (
        str(graph.device).startswith('cuda') and force_device != 'cpu'
    )
    compute_device = th.device('cuda') if compute_on_gpu else th.device('cpu')

    if compute_on_gpu:
        print(f"  [GPU] Running SIGN on GPU (graph on {graph.device})")
        graph_dev = graph.to(compute_device)
        text_feat_dev = text_feat.to(compute_device)
        vis_feat_dev = vis_feat.to(compute_device)
    else:
        print(f"  [CPU] Running SIGN on CPU (use --sign_use_gpu to enable GPU)")
        graph_dev = graph.to('cpu')
        text_feat_dev = text_feat.cpu()
        vis_feat_dev = vis_feat.cpu()

    # Use batched SIGN: one graph propagation for both modalities
    # This matches the official coldgnn behavior where concat(text, visual)
    # is processed in a SINGLE update_all() pass.
    t_batched = time.time()
    h_lists = sign_pre_compute_batched(
        graph_dev,
        [text_feat_dev, vis_feat_dev],
        k=sign_k,
        include_input=True,
        alpha=0.0,
        norm="both",
        device=str(compute_device)
    )
    text_h_list, vis_h_list = h_lists[0], h_lists[1]
    print(f"  batched SIGN done ({time.time()-t_batched:.1f}s)  "
          f"text_shape={[h.shape for h in text_h_list]}")

    # Move back to target device
    text_h_list = [h.to(device) for h in text_h_list]
    vis_h_list = [h.to(device) for h in vis_h_list]

    # Save to cache (store on CPU to save GPU memory)
    os.makedirs(cache_dir, exist_ok=True)
    th.save({
        'text_h_list': [h.cpu() for h in text_h_list],
        'vis_h_list': [h.cpu() for h in vis_h_list],
    }, cache_file)
    print(f"SIGN pre-compute done + cached to {cache_file}. "
          f"({time.time()-t0:.1f}s total)")

    return text_h_list, vis_h_list


def _load_mag_context(args, device: th.device):
    """Load multimodal graph data for NTSFormer."""
    graph, labels, train_idx, val_idx, test_idx = load_data(
        args.graph_path, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        name=args.data_name, fewshots=args.fewshots,
    )

    if args.undirected:
        print("The Graph change to the undirected graph")
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)

    observe_graph = copy.deepcopy(graph)

    if args.inductive:
        observe_graph = _make_observe_graph_inductive(graph, val_idx, test_idx)

    if args.selfloop:
        print(f"Total edges before adding self-loop {graph.number_of_edges()}")
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")
        observe_graph = observe_graph.remove_self_loop().add_self_loop()

    graph.create_formats_()
    observe_graph.create_formats_()

    train_idx, val_idx, test_idx = train_idx.to(device), val_idx.to(device), test_idx.to(device)
    labels = labels.to(device)
    graph, observe_graph = graph.to(device), observe_graph.to(device)

    text_feature = th.from_numpy(np.load(args.text_feature).astype(np.float32)).to(device)
    visual_feature = th.from_numpy(np.load(args.visual_feature).astype(np.float32)).to(device)

    return graph, observe_graph, labels, train_idx, val_idx, test_idx, text_feature, visual_feature


def _make_observe_graph_inductive(graph, val_idx, test_idx):
    """Remove edges touching val/test nodes for inductive setting."""
    observe_graph = copy.deepcopy(graph)
    isolated = th.cat((val_idx, test_idx)).to(th.long)
    if isolated.numel() == 0:
        return observe_graph
    isolated = th.unique(isolated)

    n = observe_graph.num_nodes()
    is_isolated = th.zeros((n,), dtype=th.bool)
    is_isolated[isolated] = True

    src, dst = observe_graph.all_edges(order="eid")
    keep = ~(is_isolated[src] | is_isolated[dst])
    if bool(keep.all().item()):
        return observe_graph

    eids = th.arange(observe_graph.num_edges(), dtype=th.int64)
    remove_eids = eids[~keep]
    observe_graph = observe_graph.remove_edges(remove_eids)
    return observe_graph


# ==============================================================================
# Argument Parsing
# ==============================================================================

def args_init():
    parser = argparse.ArgumentParser(
        "NTSFormer: Multimodal Node Classification with SIGN Pre-computation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)

    # NTSFormer specific arguments
    nts = parser.add_argument_group("NTSFormer")
    nts.add_argument("--nts_num_tf_layers", type=int, default=2,
                     help="Number of transformer layers for hop fusion")
    nts.add_argument("--nts_num_heads", type=int, default=2,
                     help="Number of attention heads per transformer layer")
    nts.add_argument("--nts_sign_k", type=int, default=3,
                     help="Number of hops for SIGN pre-computation (k in A^k)")
    nts.add_argument("--nts_sign_alpha", type=float, default=0.0,
                     help="Residual coefficient for SIGN: h=(1-alpha)*Agg(h)+alpha*x")
    nts.add_argument("--sign_use_gpu", action="store_true",
                     help="Run SIGN pre-computation on GPU (faster but uses more GPU memory)")

    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--local_log", type=str, default=None, help="Optional CSV file to upsert local results")

    return parser


# ==============================================================================
# Main Training
# ==============================================================================

def main():
    parser = args_init()
    args = parser.parse_args()

    if args.disable_wandb or wandb is None:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        wandb.init(config=args, reinit=True)

    # Auto-enable degrade metrics when result CSV is requested
    report_drop = bool(getattr(args, "report_drop_modality", False))
    if report_drop or getattr(args, "result_csv", None) or getattr(args, "result_csv_all", None):
        report_drop = True

    # Parse degrade alphas
    raw_alphas = str(getattr(args, "degrade_alphas", "") or "1.0")
    degrade_alphas = []
    for a in raw_alphas.split(","):
        try:
            degrade_alphas.append(float(a.strip()))
        except ValueError:
            degrade_alphas.append(1.0)
    if not degrade_alphas:
        degrade_alphas = [1.0]
    degrade_target = str(getattr(args, "degrade_target", "both")).lower()

    device = th.device("cuda:%d" % args.gpu if th.cuda.is_available() and args.gpu != -1 else "cpu")

    t0 = time.time()
    t_load = t0
    (
        graph, observe_graph, labels, train_idx, val_idx, test_idx,
        text_feat, vis_feat,
    ) = _load_mag_context(args, device)
    t_load = time.time() - t_load

    n_classes = (labels.max() + 1).item()
    print(f"Number of classes: {n_classes}")
    print(f"Text feature dim: {text_feat.shape[1]}, Visual feature dim: {vis_feat.shape[1]}")

    # Pre-compute SIGN features
    sign_k = getattr(args, 'nts_sign_k', 3)
    # Build cache key from dataset + feature paths + sign_k
    import hashlib, os as _os
    text_bn = _os.path.basename(args.text_feature or 'unknown')
    vis_bn = _os.path.basename(args.visual_feature or 'unknown')
    key_str = f"{args.data_name}_{text_bn}_{vis_bn}_k{sign_k}"
    sign_cache_key = hashlib.md5(key_str.encode()).hexdigest()[:12]
    t_precompute = time.time()
    sign_use_gpu = 'gpu' if getattr(args, 'sign_use_gpu', False) else 'cpu'
    text_h_list, vis_h_list = _pre_compute_sign_features(
        observe_graph, text_feat, vis_feat, sign_k, device,
        cache_key=sign_cache_key, force_device=sign_use_gpu
    )
    t_precompute = time.time() - t_precompute
    print(f"[TIME] data_loading={t_load:.2f}s  sign_precompute={t_precompute:.2f}s")

    val_results = []
    test_results = []
    run_degrade_text_results = []
    run_degrade_visual_results = []

    # Efficiency profiling: collect per-run metrics
    efficiency_runs = {
        'peak_memory_MB': [],
        'epoch_times': [],  # list of list: each run has list of epoch times
        'epochs_needed': [],
    }

    # Build one model to count params (before the run loop)
    _model_for_count = _build_ntsformer_model(
        args,
        text_feat_dim=text_feat.shape[1],
        vis_feat_dim=vis_feat.shape[1],
        n_classes=n_classes,
        device=device,
    )
    n_params = sum(p.numel() for p in _model_for_count.parameters() if p.requires_grad)
    n_params_M = n_params / 1e6
    del _model_for_count

    for run in range(args.n_runs):
        set_seed(args.seed + run)

        t_run_start = time.time()
        t_model_init = time.time()
        model = _build_ntsformer_model(
            args,
            text_feat_dim=text_feat.shape[1],
            vis_feat_dim=vis_feat.shape[1],
            n_classes=n_classes,
            device=device,
        )
        model.reset_parameters()
        t_model_init = time.time() - t_model_init

        stopper = initialize_early_stopping(args)
        optimizer, lr_scheduler = initialize_optimizer_and_scheduler(args, model)

        # Peak memory tracking (reset after model init and optimizer creation)
        peak_memory_mb = 0.0
        if th.cuda.is_available():
            th.cuda.reset_peak_memory_stats(device)
            th.cuda.empty_cache()

        train_step_times = []
        eval_step_times = []
        degrade_times = []
        epoch_times = []  # track full epoch time (train + eval)

        best_val_score = -1.0
        final_test_result = 0.0
        best_test_degrade = None
        epochs_needed = args.n_epochs  # will be updated if early stop triggered
        if report_drop:
            best_test_degrade = {alpha: (None, None) for alpha in degrade_alphas}

        for epoch in range(1, args.n_epochs + 1):
            tic = time.time()

            adjust_learning_rate_if_needed(args, optimizer, epoch)

            model.train()
            optimizer.zero_grad()

            t_fwd_start = time.time()
            logits = model(observe_graph, text_feat, vis_feat, text_h_list, vis_h_list)
            loss = cross_entropy(logits[train_idx], labels[train_idx],
                                label_smoothing=args.label_smoothing)
            loss.backward()
            optimizer.step()
            train_step_times.append(time.time() - t_fwd_start)

            if epoch % args.eval_steps == 0:
                t_eval_start = time.time()
                model.eval()
                with th.no_grad():
                    logits = model(graph, text_feat, vis_feat, text_h_list, vis_h_list)

                val_pred = th.argmax(logits[val_idx], dim=1)
                val_result = get_metric(val_pred, labels[val_idx], args.metric, average=args.average)

                test_pred = th.argmax(logits[test_idx], dim=1)
                test_result = get_metric(test_pred, labels[test_idx], args.metric, average=args.average)

                lr_scheduler.step(float(loss.detach().item()))
                eval_step_times.append(time.time() - t_eval_start)

                toc = time.time()
                total_time = toc - tic
                epoch_times.append(total_time)  # track full epoch time for efficiency profiling

                if val_result > best_val_score:
                    best_val_score = float(val_result)
                    final_test_result = float(test_result)
                    best_test_degrade = None
                    if report_drop:
                        t_degrade_start = time.time()
                        best_test_degrade = {}
                        for alpha in degrade_alphas:
                            dt_val, dv_val = None, None
                            target = degrade_target
                            # Degrade text: recompute SIGN from noisy text
                            if target in ("text", "both"):
                                noisy_text = _make_noisy_feature(text_feat, train_idx, float(alpha))
                                noisy_text_h_list = sign_pre_compute(
                                    graph, noisy_text, k=sign_k, include_input=True,
                                    alpha=0.0, device=text_feat.device
                                )
                                logits_dt = model(
                                    graph, noisy_text, vis_feat,
                                    text_h_list=noisy_text_h_list, vis_h_list=vis_h_list
                                )
                                pred_dt = th.argmax(logits_dt[test_idx], dim=1)
                                dt_val = get_metric(pred_dt, labels[test_idx], args.metric, average=args.average)
                                dt_val = float(np.asarray(dt_val).mean())
                            # Degrade visual: recompute SIGN from noisy visual
                            if target in ("visual", "both"):
                                noisy_vis = _make_noisy_feature(vis_feat, train_idx, float(alpha))
                                noisy_vis_h_list = sign_pre_compute(
                                    graph, noisy_vis, k=sign_k, include_input=True,
                                    alpha=0.0, device=vis_feat.device
                                )
                                logits_dv = model(
                                    graph, text_feat, noisy_vis,
                                    text_h_list=text_h_list, vis_h_list=noisy_vis_h_list
                                )
                                pred_dv = th.argmax(logits_dv[test_idx], dim=1)
                                dv_val = get_metric(pred_dv, labels[test_idx], args.metric, average=args.average)
                                dv_val = float(np.asarray(dv_val).mean())
                            best_test_degrade[alpha] = (dt_val, dv_val)
                        degrade_times.append(time.time() - t_degrade_start)

                if stopper and stopper.step(val_result):
                    epochs_needed = epoch
                    break

                log_progress(
                    args, epoch, run + 1, total_time,
                    loss, loss, loss,
                    val_result, val_result, test_result,
                    best_val_score, final_test_result,
                )

        # Record peak memory after training
        if th.cuda.is_available():
            peak_memory_mb = th.cuda.max_memory_allocated(device) / 1048576.0

        print(f"Run {run+1}: Best Val {args.metric}={best_val_score:.4f}, Final Test {args.metric}={final_test_result:.4f}")
        if th.cuda.is_available():
            print(f"  [MEMORY] peak={peak_memory_mb:.2f} MB")
        t_run_total = time.time() - t_run_start
        t_train_total = sum(train_step_times)
        t_eval_total = sum(eval_step_times)
        t_degrade_total = sum(degrade_times)
        t_other = t_run_total - t_model_init - t_train_total - t_eval_total - t_degrade_total
        avg_train = np.mean(train_step_times) if train_step_times else 0
        avg_eval = np.mean(eval_step_times) if eval_step_times else 0
        print(f"  [TIME] total={t_run_total:.1f}s  model_init={t_model_init:.2f}s  "
              f"train_total={t_train_total:.1f}s(avg={avg_train:.3f}s×{len(train_step_times)}ep)  "
              f"eval_total={t_eval_total:.1f}s(avg={avg_eval:.3f}s×{len(eval_step_times)}ep)  "
              f"degrade={t_degrade_total:.2f}s  other={t_other:.1f}s")
        if report_drop and best_test_degrade:
            alpha0 = degrade_alphas[0]
            dt, dv = best_test_degrade[alpha0]
            print(f"  Degrade (alpha={alpha0}): degrade_text={dt}, degrade_visual={dv}")
            run_degrade_text_results.append(dt)
            run_degrade_visual_results.append(dv)
        val_results.append(best_val_score)
        test_results.append(final_test_result)

        # Collect efficiency profiling data
        efficiency_runs['peak_memory_MB'].append(peak_memory_mb)
        efficiency_runs['epoch_times'].append(epoch_times)
        efficiency_runs['epochs_needed'].append(epochs_needed)

        if wandb is not None and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
            wandb.log({f'Val_{args.metric}': best_val_score, f'Test_{args.metric}': final_test_result})

    print(f"\nRunned {args.n_runs} times")
    print(f"Average val {args.metric}: {np.mean(val_results):.4f} ± {np.std(val_results):.4f}")
    print(f"Average test {args.metric}: {np.mean(test_results):.4f} ± {np.std(test_results):.4f}")

    # Efficiency profiling summary
    all_epoch_times = [t for run_times in efficiency_runs['epoch_times'] for t in run_times]
    avg_epoch_time = float(np.mean(all_epoch_times)) if all_epoch_times else 0
    std_epoch_time = float(np.std(all_epoch_times)) if all_epoch_times else 0
    avg_epochs_needed = float(np.mean(efficiency_runs['epochs_needed']))
    std_epochs_needed = float(np.std(efficiency_runs['epochs_needed']))
    avg_peak_memory = float(np.mean(efficiency_runs['peak_memory_MB']))
    std_peak_memory = float(np.std(efficiency_runs['peak_memory_MB']))
    avg_total_time = avg_epochs_needed * avg_epoch_time
    std_total_time = float(np.std([sum(run_times) for run_times in efficiency_runs['epoch_times']]))

    print(f"\n{'='*60}")
    print(f"Efficiency Profile: NTSFormer on {args.data_name}")
    print(f"{'='*60}")
    print(f"  Parameters:       {n_params_M:.3f} M")
    print(f"  Peak Memory:     {avg_peak_memory:.2f} ± {std_peak_memory:.2f} MB")
    print(f"  Total Time(est): {avg_total_time:.2f} ± {std_total_time:.2f} s  ({avg_total_time/60:.1f} min)")
    print(f"  Avg Epoch:        {avg_epoch_time:.4f} ± {std_epoch_time:.4f} s/epoch")
    print(f"  Epochs Needed:    {avg_epochs_needed:.1f} ± {std_epochs_needed:.1f}")
    print(f"{'='*60}")
    if report_drop:
        alpha0 = degrade_alphas[0]
        print(f"Best test degrade (alpha={alpha0}): text={np.mean(run_degrade_text_results):.4f}, visual={np.mean(run_degrade_visual_results):.4f}")

    if wandb is not None and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
        wandb.log({
            f'Mean_Val_{args.metric}': np.mean(val_results),
            f'Mean_Test_{args.metric}': np.mean(test_results),
        })

    if getattr(args, 'result_csv', None) or getattr(args, 'result_csv_all', None):
        test_mean = float(np.mean(test_results))
        test_std = float(np.std(test_results))
        degrade_text = float(np.mean(run_degrade_text_results)) if run_degrade_text_results else None
        degrade_visual = float(np.mean(run_degrade_visual_results)) if run_degrade_visual_results else None
        degrade_text_std = float(np.std(run_degrade_text_results)) if run_degrade_text_results else None
        degrade_visual_std = float(np.std(run_degrade_visual_results)) if run_degrade_visual_results else None
        row = build_result_row(args=args, method="NTSFormer", full_metric=test_mean,
                               degrade_text=degrade_text, degrade_visual=degrade_visual,
                               extra={
                                   "full_std": test_std,
                                   "degrade_text_std": degrade_text_std,
                                   "degrade_visual_std": degrade_visual_std,
                                   "nts_sign_k": getattr(args, "nts_sign_k", ""),
                                   "nts_sign_alpha": getattr(args, "nts_sign_alpha", ""),
                                   "nts_num_tf_layers": getattr(args, "nts_num_tf_layers", ""),
                                   "nts_num_heads": getattr(args, "nts_num_heads", ""),
                               })
        key_fields = ["dataset", "method", "metric", "n_hidden", "inductive", "fewshots"]
        if getattr(args, 'result_csv', None):
            update_best_result_csv(args.result_csv, row, key_fields=key_fields, score_field="full")
        if getattr(args, 'result_csv_all', None):
            append_result_csv(args.result_csv_all, row)


if __name__ == "__main__":
    main()
