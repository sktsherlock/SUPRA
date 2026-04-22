"""SUPRA: Unified Multimodal Learning with Spectral Orthogonalization."""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

# Allow running as a script
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(_ROOT)

from GNN.GraphData import load_data, set_seed
from GNN.Utils.LossFunction import cross_entropy, get_metric
from GNN.Utils.NodeClassification import (
    initialize_early_stopping,
    initialize_optimizer_and_scheduler,
    adjust_learning_rate_if_needed,
    log_results_to_wandb,
    log_progress,
    print_final_results,
    _as_scalar_float,
)
from GNN.Utils.model_config import (
    add_common_args,
    add_sage_args,
    add_gat_args,
    add_revgat_args,
    add_sgc_args,
    add_appnp_args,
)
from GNN.Utils.result_logger import build_result_row, update_best_result_csv, append_result_csv
from GNN.Baselines.Early_GNN import _make_observe_graph_inductive


# ==============================================================================
# Newton-Schulz Spectral Orthogonalization
# Prevents low-rank collapse in shared representations by orthogonalizing
# gradient matrices during backpropagation.
# ==============================================================================
class SpectralOrthogonalizer:
    """Newton-Schulz iteration for spectral normalization of gradient matrices.

    Applies iterative orthogonalization: X = a*X + b*(X@A) + c*(X@A@A)
    where A = X.T @ X, to prevent rank deficiency in shared channels.

    Supports dynamic orthogonalization strength (alpha):
        grad = (1 - alpha) * grad_original + alpha * grad_orthogonalized

    Alpha can be:
        - Fixed value (e.g., 0.5)
        - Scheduled decay (high early, low later)
        - Rank-adaptive (based on singular value decay rate)
    """
    def __init__(self, ns_steps: int = 5, alpha: float = 1.0, rank_adaptive: bool = False):
        self.ns_steps = ns_steps
        self.alpha = alpha  # Orthogonalization strength (0=off, 1=full)
        self.rank_adaptive = rank_adaptive
        self._current_step = 0

    def set_alpha(self, alpha: float):
        """Set the orthogonalization strength alpha."""
        self.alpha = max(0.0, min(1.0, alpha))

    def set_step(self, step: int):
        """Set current training step for adaptive scheduling."""
        self._current_step = step

    def _compute_rank_ratio(self, X: th.Tensor) -> float:
        """Compute stable rank ratio: sum(s^2) / max(s^2).

        Low ratio indicates potential low-rank tendency.
        """
        try:
            s = th.linalg.svd(X, compute_uv=False)
            ss = s * s
            total = ss.sum()
            max_s = ss[0] if len(ss) > 0 else 1.0
            return (total / max_s).item() if max_s > 0 else 1.0
        except Exception:
            return 1.0

    def _adaptive_alpha(self, grad: th.Tensor) -> float:
        """Compute adaptive alpha based on gradient rank analysis."""
        if not self.rank_adaptive:
            return self.alpha

        try:
            r, c = grad.shape
            X = grad.float()
            if r < c:
                X = X.T

            s = th.linalg.svd(X, compute_uv=False)
            if len(s) < 2:
                return self.alpha

            # Compute singular value decay rate
            # Large decay (s[0] >> s[-1]) suggests low-rank prone -> higher alpha
            ratio = (s[0] / (s[-1] + 1e-7)).item()

            # Map ratio to alpha: higher ratio -> stronger orthogonalization
            # Clamp to [alpha, 1.0]
            adaptive = min(1.0, self.alpha * (1 + 0.1 * ratio))
            return max(self.alpha, adaptive)
        except Exception:
            return self.alpha

    def __call__(self, param_id: int, grad: th.Tensor) -> th.Tensor:
        if len(grad.shape) != 2:
            return grad

        r, c = grad.shape
        if r == 0 or c == 0:
            return grad

        # If alpha is 0, skip orthogonalization
        if self.alpha <= 0.0:
            return grad

        # Adaptive alpha based on rank analysis
        alpha = self._adaptive_alpha(grad)

        dtype, device = grad.dtype, grad.device
        X = grad.float()

        transpose = False
        if r < c:
            X = X.T
            r, c = c, r
            transpose = True

        X = X / (X.norm() + 1e-7)
        # Newton-Schulz coefficients for 5-step iteration
        a, b, c_coef = 3.4445, -4.7750, 2.0315

        for _ in range(self.ns_steps):
            A = X.T @ X
            B = A @ A
            X = a * X + b * (X @ A) + c_coef * (X @ B)

        if transpose:
            X = X.T

        scale = max(r, c) ** 0.5
        ortho_grad = X * scale

        # Mix original gradient with orthogonalized gradient
        if alpha < 1.0:
            mixed = (1 - alpha) * grad + alpha * ortho_grad.to(dtype)
        else:
            mixed = ortho_grad.to(dtype)

        return mixed


# ==============================================================================


class ModalityEncoder(nn.Module):
    """Encodes raw modality features (text/visual) into embedding space.

    In SUPRA, modality encoders serve dual purposes:
    1. Project raw modality features to a common embedding space
    2. Act as 'unique' branches that capture modality-specific information
       without being contaminated by co-modal signals
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.proj.reset_parameters()

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.proj(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


@dataclass
class ForwardOutputs:
    """Outputs from a single forward pass with all branch logits and embeddings."""
    logits_final: th.Tensor
    logits_C: th.Tensor
    logits_Ut: th.Tensor
    logits_Uv: th.Tensor
    emb_C: Optional[th.Tensor] = None
    emb_Ut: Optional[th.Tensor] = None
    emb_Uv: Optional[th.Tensor] = None


@dataclass
class ForwardMultiOutputs:
    """Container for multiple forward pass outputs (training/eval)."""
    logits_final_0: th.Tensor
    logits_C_0: th.Tensor
    logits_Ut_0: th.Tensor
    logits_Uv_0: th.Tensor
    emb_C_0: Optional[th.Tensor] = None
    emb_Ut_0: Optional[th.Tensor] = None
    emb_Uv_0: Optional[th.Tensor] = None


def _build_gnn_backbone(args, in_dim: int, out_dim: int, device: th.device, *, n_layers_override: Optional[int] = None, n_hidden_override: Optional[int] = None):
    name = str(getattr(args, "model_name", "GCN"))
    n_layers = int(n_layers_override) if n_layers_override is not None else int(args.n_layers)
    n_hidden = int(n_hidden_override) if n_hidden_override is not None else int(args.n_hidden)

    if name == "GCN":
        from GNN.Library.GCN import GCN
        return GCN(in_dim, n_hidden, out_dim, n_layers, F.relu, args.dropout).to(device)

    if name == "SAGE":
        from GNN.Library.GraphSAGE import GraphSAGE
        return GraphSAGE(
            in_dim, n_hidden, out_dim, n_layers, F.relu, args.dropout,
            aggregator_type=getattr(args, "aggregator", "mean"),
        ).to(device)

    if name == "GAT":
        from GNN.Library.GAT import GAT
        return GAT(
            in_dim, out_dim, n_hidden, n_layers, args.n_heads, F.relu,
            args.dropout, args.attn_drop, args.edge_drop, not getattr(args, "no_attn_dst", True),
        ).to(device)

    if name == "RevGAT":
        from GNN.Library.RevGAT.model import RevGAT
        return RevGAT(
            in_dim, out_dim, n_hidden, n_layers, args.n_heads, F.relu,
            dropout=args.dropout, attn_drop=args.attn_drop, edge_drop=args.edge_drop,
            use_attn_dst=False, use_symmetric_norm=getattr(args, "use_symmetric_norm", True),
        ).to(device)

    raise ValueError(f"Unsupported --model_name: {name}")


class SUPRA(nn.Module):
    """
    SUPRA: Unified multimodal learning with Shared-Unique decomposition
    and spectral orthogonalization to prevent representation collapse.

    Architecture:
        - Shared channel (C): captures modality-invariant representations
          via concat(text, visual) + message passing on graph
          Subject to spectral orthogonalization to prevent low-rank collapse
        - Unique channels (Ut/Uv): modality-specific encoders that project
          raw features to embedding space WITHOUT co-modal contamination
          (no message passing on graph, just linear projection + activation)

    The unique channels are realized by the ModalityEncoders themselves,
    which naturally capture modality-specific patterns during training.
    """
    def __init__(
        self, *,
        text_in_dim: int,
        vis_in_dim: int,
        embed_dim: int,
        n_classes: int,
        dropout: float,
        args,
        device: th.device,
        shared_depth: int,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = int(embed_dim)
        self.n_classes = int(n_classes)
        self.shared_depth = int(shared_depth)

        # Modality encoders: project raw features to shared embedding space
        # These serve as the "unique" branches, capturing modality-specific
        # information without co-modal signal contamination
        self.enc_t = ModalityEncoder(int(text_in_dim), self.embed_dim, float(dropout))
        self.enc_v = ModalityEncoder(int(vis_in_dim), self.embed_dim, float(dropout))

        def _make_mp_layers(num_layers: int) -> nn.ModuleList:
            # First layer: projects from concat(e_t, e_v) = 2*embed_dim down to embed_dim
            # Subsequent layers: process embed_dim features sequentially
            layers = []
            first_in_dim = self.embed_dim * 2
            for i in range(int(num_layers)):
                in_dim = first_in_dim if i == 0 else self.embed_dim
                layers.append(
                    _build_gnn_backbone(args, in_dim, self.embed_dim, device, n_layers_override=1, n_hidden_override=self.embed_dim)
                )
            return nn.ModuleList(layers)

        # Shared message passing layers (deeper propagation)
        self.mp_C = _make_mp_layers(self.shared_depth)

        # Prediction heads for shared and unique channels
        self.head_C = nn.Linear(self.embed_dim, self.n_classes)
        self.head_Ut = nn.Linear(self.embed_dim, self.n_classes)
        self.head_Uv = nn.Linear(self.embed_dim, self.n_classes)

        # Attach spectral orthogonalization hook to shared channel
        self.spectral_orthogonalizer = SpectralOrthogonalizer(ns_steps=5, alpha=getattr(args, "ortho_alpha", 1.0))
        self._register_spectral_hooks()

    def _register_spectral_hooks(self):
        """Register backward hooks on shared channel to prevent low-rank collapse."""
        shared_modules = [self.mp_C, self.head_C]
        for module in shared_modules:
            for name, param in module.named_parameters():
                if 'weight' in name and len(param.shape) == 2:
                    param.register_hook(
                        lambda grad, pid=id(param): self.spectral_orthogonalizer(pid, grad)
                    )

    def reset_parameters(self):
        for module in [self.enc_t, self.enc_v, self.head_C, self.head_Ut, self.head_Uv]:
            if hasattr(module, "reset_parameters"): module.reset_parameters()
        for layer in self.mp_C:
            if hasattr(layer, "reset_parameters"): layer.reset_parameters()

    def _init_channels(self, e_t: th.Tensor, e_v: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Initialize shared and unique channel representations.

        - Unique channels: raw encoded features from modality encoders
          (no co-modal contamination, no graph message passing)
        - Shared channel: concatenated text+visual representation (no MLP projection)
        """
        u_t = e_t  # Unique text: just the encoded feature
        u_v = e_v  # Unique visual: just the encoded feature
        c = th.cat([e_t, e_v], dim=1)  # Shared: concatenated representation (direct concat)
        return c, u_t, u_v

    def _run_layers(self, graph, x: th.Tensor, mp_layers) -> th.Tensor:
        h = x
        for layer in mp_layers:
            h = layer(graph, h)
        return h

    def _encode_with_drop(self, text_feat: th.Tensor, vis_feat: th.Tensor, present_t: th.Tensor, present_v: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        e_t = self.enc_t(text_feat)
        e_v = self.enc_v(vis_feat)
        if present_t is not None:
            e_t = e_t * present_t.to(device=e_t.device, dtype=e_t.dtype).unsqueeze(1)
        if present_v is not None:
            e_v = e_v * present_v.to(device=e_v.device, dtype=e_v.dtype).unsqueeze(1)
        return e_t, e_v

    def _forward_from_encoded(self, graph, e_t: th.Tensor, e_v: th.Tensor) -> ForwardOutputs:
        h_C, h_Ut, h_Uv = self._init_channels(e_t, e_v)

        # Shared uses deeper propagation; unique channels skip graph propagation
        h_C = self._run_layers(graph, h_C, self.mp_C)
        # Unique channels (Ut, Uv) do NOT go through graph message passing

        logits_C = self.head_C(h_C)
        logits_Ut = self.head_Ut(h_Ut)
        logits_Uv = self.head_Uv(h_Uv)

        # Late fusion: average logits from all channels
        logits_final = (logits_C + logits_Ut + logits_Uv) / 3.0

        return ForwardOutputs(
            logits_final=logits_final,
            logits_C=logits_C, logits_Ut=logits_Ut, logits_Uv=logits_Uv,
            emb_C=h_C, emb_Ut=h_Ut, emb_Uv=h_Uv,
        )

    def forward(self, graph, text_feat: th.Tensor, vis_feat: th.Tensor) -> th.Tensor:
        present_t = (text_feat.abs().sum(dim=1) > 0)
        present_v = (vis_feat.abs().sum(dim=1) > 0)
        e_t, e_v = self._encode_with_drop(text_feat, vis_feat, present_t, present_v)
        return self._forward_from_encoded(graph, e_t, e_v).logits_final

    def forward_multiple(self, graph, text_feat: th.Tensor, vis_feat: th.Tensor, *, stochastic: bool = False, profile_mem: bool = False) -> ForwardMultiOutputs:
        present_t = (text_feat.abs().sum(dim=1) > 0)
        present_v = (vis_feat.abs().sum(dim=1) > 0)
        e_t0, e_v0 = self._encode_with_drop(text_feat, vis_feat, present_t, present_v)
        out0 = self._forward_from_encoded(graph, e_t0, e_v0)
        return ForwardMultiOutputs(
            logits_final_0=out0.logits_final,
            logits_C_0=out0.logits_C, logits_Ut_0=out0.logits_Ut, logits_Uv_0=out0.logits_Uv,
            emb_C_0=out0.emb_C, emb_Ut_0=out0.emb_Ut, emb_Uv_0=out0.emb_Uv,
        )


def _compute_losses(*, out: ForwardMultiOutputs, labels: th.Tensor, train_idx: th.Tensor, args, graph) -> Tuple[th.Tensor, Dict[str, float]]:
    """Compute task loss on fused logits (shared-unique decomposition with spectral orthogonalization)."""
    idx = train_idx
    ls = float(getattr(args, "label_smoothing", 0.0))

    # Main loss: average of C, Ut, Uv logits
    logits_final = (out.logits_C_0 + out.logits_Ut_0 + out.logits_Uv_0) / 3.0
    total_task_loss = cross_entropy(logits_final[idx], labels[idx], label_smoothing=ls)

    logs = {
        "loss/task": float(total_task_loss.detach().cpu().item()),
    }

    # Auxiliary loss: each branch predicts independently
    if getattr(args, "use_aux_loss", False):
        loss_C = cross_entropy(out.logits_C_0[idx], labels[idx], label_smoothing=ls)
        loss_Ut = cross_entropy(out.logits_Ut_0[idx], labels[idx], label_smoothing=ls)
        loss_Uv = cross_entropy(out.logits_Uv_0[idx], labels[idx], label_smoothing=ls)
        aux_weight = 0.5
        total_task_loss = total_task_loss + aux_weight * (loss_C + loss_Ut + loss_Uv)
        logs.update({
            "loss/aux_C": float(loss_C.detach().cpu().item()),
            "loss/aux_Ut": float(loss_Ut.detach().cpu().item()),
            "loss/aux_Uv": float(loss_Uv.detach().cpu().item()),
        })

    return total_task_loss, logs


def _load_mag_context(args, device: th.device):
    graph, labels, train_idx, val_idx, test_idx = load_data(
        args.graph_path, train_ratio=args.train_ratio, val_ratio=args.val_ratio, name=args.data_name, fewshots=args.fewshots,
    )
    if args.undirected:
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)
    observe_graph = copy.deepcopy(graph)
    if args.inductive:
        observe_graph = _make_observe_graph_inductive(graph, val_idx, test_idx)
    if args.selfloop:
        graph = graph.remove_self_loop().add_self_loop()
        observe_graph = observe_graph.remove_self_loop().add_self_loop()

    graph.create_formats_()
    observe_graph.create_formats_()

    train_idx, val_idx, test_idx = train_idx.to(device), val_idx.to(device), test_idx.to(device)
    labels = labels.to(device)
    graph, observe_graph = graph.to(device), observe_graph.to(device)
    text_feature = th.from_numpy(np.load(args.text_feature).astype(np.float32)).to(device)
    visual_feature = th.from_numpy(np.load(args.visual_feature).astype(np.float32)).to(device)
    return graph, observe_graph, labels, train_idx, val_idx, test_idx, text_feature, visual_feature, int((labels.max() + 1).item())


def args_init():
    parser = argparse.ArgumentParser("SUPRA: Unified Multimodal Learning", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)
    add_sage_args(parser)
    add_gat_args(parser)
    add_revgat_args(parser)
    add_sgc_args(parser)
    add_appnp_args(parser)
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")

    # SUPRA model arguments
    supra = parser.add_argument_group("SUPRA")
    supra.add_argument("--embed_dim", type=int, default=None, help="Embedding dimension for SUPRA channels")
    supra.add_argument("--shared_depth", type=int, default=None, help="Propagation depth for shared channel")
    supra.add_argument("--ortho_alpha", type=float, default=1.0, help="Spectral orthogonalization strength (0=disable)")
    supra.add_argument("--use_aux_loss", action="store_true", help="Enable auxiliary loss on each branch (C, Ut, Uv)")

    return parser


def main():
    parser = args_init()
    args = parser.parse_args()
    if args.disable_wandb or wandb is None: os.environ["WANDB_DISABLED"] = "true"

    device = th.device("cuda:%d" % args.gpu if th.cuda.is_available() and args.gpu != -1 else "cpu")

    (
        graph, observe_graph, labels, train_idx, val_idx, test_idx,
        text_feat, vis_feat, n_classes,
    ) = _load_mag_context(args, device)

    val_results, test_results = [], []
    select_metric, select_average = args.metric, args.average
    embed_dim = int(args.embed_dim) if args.embed_dim is not None else int(args.n_hidden)
    shared_depth = int(args.shared_depth) if args.shared_depth is not None else int(args.n_layers)

    for run in range(args.n_runs):
        set_seed(args.seed + run)
        model = SUPRA(
            text_in_dim=int(text_feat.shape[1]), vis_in_dim=int(vis_feat.shape[1]),
            embed_dim=embed_dim, n_classes=n_classes, dropout=float(args.dropout),
            args=args, device=device, shared_depth=shared_depth,
        ).to(device)
        model.reset_parameters()

        stopper = initialize_early_stopping(args)
        optimizer, lr_scheduler = initialize_optimizer_and_scheduler(args, model)

        best_val_score, final_test_result, best_val_result, total_time = -1.0, 0.0, -1.0, 0.0

        for epoch in range(1, args.n_epochs + 1):
            tic = time.time()
            adjust_learning_rate_if_needed(args, optimizer, epoch)

            model.train()
            optimizer.zero_grad()
            out = model.forward_multiple(observe_graph, text_feat, vis_feat)

            # Pure joint loss with spectral orthogonalization
            total_loss, loss_logs = _compute_losses(out=out, labels=labels, train_idx=train_idx, args=args, graph=observe_graph)
            total_loss.backward()
            optimizer.step()

            if epoch % args.eval_steps == 0:
                model.eval()
                with th.no_grad():
                    logits_e = model(graph, text_feat, vis_feat)

                val_score = get_metric(th.argmax(logits_e[val_idx], dim=1), labels[val_idx], select_metric, average=select_average)
                test_score = get_metric(th.argmax(logits_e[test_idx], dim=1), labels[test_idx], select_metric, average=select_average)

                if val_score > best_val_score:
                    best_val_score = float(val_score)
                    best_val_result = float(val_score)
                    final_test_result = float(test_score)
                if stopper and stopper.step(val_score): break

                total_time += time.time() - tic

        print(f"Run {run+1} Final Test Score: {final_test_result:.4f}")
        val_results.append(best_val_result)
        test_results.append(final_test_result)

    def _mean_std(values): return float(np.mean(values)), float(np.std(values))
    def _fmt_pct(values):
        mean, std = _mean_std(values)
        return f"{mean * 100.0:.3f} +/- {std * 100.0:.3f}%"

    print(f"\nRunned {args.n_runs} times")
    print(f"Average val {args.metric}: {_fmt_pct(val_results)}")
    print(f"Average test {args.metric}: {_fmt_pct(test_results)}")

    # Save results to CSV if requested
    if getattr(args, 'result_csv', None):
        test_mean = float(np.mean(test_results))
        test_std = float(np.std(test_results))
        method_name = getattr(args, 'result_tag', None) or "SUPRA"
        row = build_result_row(args=args, method=method_name, full_metric=test_mean, extra={"full_std": test_std})
        key_fields = ["dataset", "method", "backbone", "metric", "single_modality", "inductive", "fewshots"]
        update_best_result_csv(args.result_csv, row, key_fields=key_fields, score_field="full")

if __name__ == "__main__":
    main()
