"""SUPRA: Unified Multimodal Learning with Shared-Unique Channel Decomposition."""
from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

# Allow running as a script
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(_ROOT)

from GNN.GraphData import load_data, set_seed
from GNN.Utils.LossFunction import cross_entropy, get_metric
from GNN.Utils.NodeClassification import (
    initialize_early_stopping,
    initialize_optimizer_and_scheduler,
    adjust_learning_rate_if_needed,
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
from GNN.Utils.NodeClassification import (
    _compute_degrade_metrics_mag,
    _as_scalar_float,
)


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


class DeepResidualModalityEncoder(nn.Module):
    """Deep Residual MLP modality encoder with LayerNorm, GELU, and skip connections.

    Reference: Similar philosophy to GCNII's identity mapping — high-dimensional
    raw features should be continuously supervised across multiple transformation layers.
    Each residual block performs: x_{l+1} = x_l + F(x_l) where F is MLP with LN+GELU+Dropout.

    Architecture for high-dimensional inputs (e.g. 4096d Llama features):
        in_dim -> hidden_dim (bottleneck) -> ... (residual blocks) -> out_dim

    Args:
        in_dim: Input feature dimension
        out_dim: Output embedding dimension (default 256)
        dropout: Dropout rate
        n_layers: Number of residual blocks (n_layers >= 2 to have at least 1 block)
        hidden_dim: Intermediate hidden dimension (default 1024, designed for 4096d input)
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float, n_layers: int = 3,
                 hidden_dim: int = 1024):
        super().__init__()
        self.n_layers = n_layers

        # First projection: in_dim -> hidden_dim (compress high-dim to hidden space)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Residual blocks in hidden space
        self.blocks = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ))

        # Final projection: hidden_dim -> out_dim
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.input_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = x + block(x)
            x = self.act(x)
        x = self.output_proj(x)
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
    ):
        super().__init__()
        self.args = args
        self.embed_dim = int(embed_dim)
        self.n_classes = int(n_classes)

        # Modality encoders: project raw features to shared embedding space
        # These serve as the "unique" branches, capturing modality-specific
        # information without co-modal signal contamination
        modality_encoder = str(getattr(args, 'modality_encoder', 'linear'))
        enc_n_layers = int(getattr(args, 'enc_n_layers', 3))
        enc_hidden_dim = int(getattr(args, 'enc_hidden_dim', 1024))

        if modality_encoder == 'deep':
            self.enc_t = DeepResidualModalityEncoder(
                int(text_in_dim), self.embed_dim, float(dropout),
                n_layers=enc_n_layers, hidden_dim=enc_hidden_dim)
            self.enc_v = DeepResidualModalityEncoder(
                int(vis_in_dim), self.embed_dim, float(dropout),
                n_layers=enc_n_layers, hidden_dim=enc_hidden_dim)
        else:
            self.enc_t = ModalityEncoder(int(text_in_dim), self.embed_dim, float(dropout))
            self.enc_v = ModalityEncoder(int(vis_in_dim), self.embed_dim, float(dropout))

        def _make_mp_layers(num_layers: int, mlp_variant: str = "full") -> nn.ModuleList:
            # MLP projection before GNN:
            # full: Linear→ReLU→LN→Linear (2 layers with nonlinear activation)
            # ablate: no projection - concat features go directly to GNN
            # Then n_layers-1 GNN layers (each processes embed_dim -> embed_dim)
            # n_layers=1: projection or direct GNN
            # n_layers=2: projection + 1 GNN
            # n_layers=3: projection + 2 GNN
            layers = []
            first_in_dim = self.embed_dim * 2
            if mlp_variant == "full":
                proj = nn.Sequential(
                    nn.Linear(first_in_dim, self.embed_dim),
                    nn.ReLU(),
                    nn.LayerNorm(self.embed_dim),
                    nn.Linear(self.embed_dim, self.embed_dim),
                )
                layers.append(proj)
                gnn_start_in_dim = self.embed_dim
            else:  # ablate - no projection, concat goes directly to GNN
                gnn_start_in_dim = first_in_dim  # 2*embed_dim

            # Add GNN layers (n_layers includes the projection if present)
            # For "ablate": n_layers means n_layers GNN layers (no projection)
            # For "full": n_layers-1 GNN layers after projection
            gnn_layers_count = int(num_layers) if mlp_variant == "ablate" else max(0, int(num_layers) - 1)
            for _ in range(gnn_layers_count):
                layers.append(
                    _build_gnn_backbone(args, gnn_start_in_dim, self.embed_dim, device, n_layers_override=1, n_hidden_override=self.embed_dim)
                )
                gnn_start_in_dim = self.embed_dim  # subsequent GNN layers use embed_dim
            return nn.ModuleList(layers)

        # Shared message passing layers (n_layers from args, mlp_variant from args)
        mlp_variant = str(getattr(args, "mlp_variant", "full"))
        self.mp_C = _make_mp_layers(int(args.n_layers), mlp_variant=mlp_variant)

        # Prediction heads for shared and unique channels
        self.head_C = nn.Linear(self.embed_dim, self.n_classes)
        self.head_Ut = nn.Linear(self.embed_dim, self.n_classes)
        self.head_Uv = nn.Linear(self.embed_dim, self.n_classes)

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
        if not mp_layers:
            return h

        for layer in mp_layers:
            # Projection MLP / tensor-only layers
            if isinstance(layer, nn.Sequential):
                h = layer(h)
                continue

            # Graph-based layers (GCN/SAGE/GAT/RevGAT, etc.)
            try:
                h = layer(graph, h)
            except TypeError as e:
                # Fallback for tensor-only layers that don't accept (graph, h)
                msg = str(e)
                if (
                    "positional argument" in msg
                    or "positional arguments" in msg
                    or "required positional" in msg
                    or "missing" in msg
                ):
                    h = layer(h)
                else:
                    raise

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

        # Average fusion of three channels
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
    """Compute task loss on fused logits (shared-unique decomposition with optional gating)."""
    idx = train_idx
    ls = float(getattr(args, "label_smoothing", 0.0))

    # [Group 2] Force C-only forward: logits_final = logits_C
    ablate_bypass = getattr(args, "ablate_bypass", False)
    if ablate_bypass:
        logits_final = out.logits_C_0
        total_task_loss = cross_entropy(logits_final[idx], labels[idx], label_smoothing=ls)
        logs = {"loss/task": float(total_task_loss.detach().cpu().item())}
        return total_task_loss, logs

    # Normal SUPRA: logits_final includes all channels
    logits_final = out.logits_final_0
    total_task_loss = cross_entropy(logits_final[idx], labels[idx], label_smoothing=ls)

    logs = {
        "loss/task": float(total_task_loss.detach().cpu().item()),
    }

    # Auxiliary loss: Ut and Uv channels get independent gradients
    # (not C, because C already learns from logits_final loss)
    aux_weight = float(getattr(args, "aux_weight", 0.0))
    if aux_weight > 0:
        loss_Ut = cross_entropy(out.logits_Ut_0[idx], labels[idx], label_smoothing=ls)
        loss_Uv = cross_entropy(out.logits_Uv_0[idx], labels[idx], label_smoothing=ls)
        total_task_loss = total_task_loss + aux_weight * (loss_Ut + loss_Uv)
        logs.update({
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
    supra.add_argument("--aux_weight", type=float, default=0.0, help="Auxiliary loss weight for Ut/Uv channels (0=disable)")
    supra.add_argument("--mlp_variant", type=str, default="ablate", choices=["full", "ablate"], help="MLP before GNN: full=Linear→ReLU→LN→Linear, ablate=concat only (no projection)")
    supra.add_argument("--modality_encoder", type=str, default="linear",
                        choices=["linear", "deep"],
                        help="Modality encoder architecture: linear=single Linear+ReLU, deep=Deep Residual MLP (LN+GELU+Residual)")
    supra.add_argument("--enc_n_layers", type=int, default=3,
                        help="Number of residual blocks for deep modality encoder (only used when modality_encoder=deep)")
    supra.add_argument("--enc_hidden_dim", type=int, default=1024,
                        help="Hidden dimension for deep residual modality encoder (default 1024, designed for high-dim 4096d LLM features)")
    supra.add_argument("--use_gate", action="store_true", help="Enable learnable channel gate for adaptive fusion")
    supra.add_argument("--ablate_bypass", action="store_true",
                        help="[Group 2] Ablate bypass branches: force logits_final=logits_C only, "
                             "removing Ut/Uv channels to isolate shared GNN gradient dynamics")
    parser.add_argument("--save_checkpoint", type=str, default=None,
                        help="Path to save best model checkpoint after training")
    parser.add_argument("--export_predictions", type=str, default=None,
                        help="Path to save test predictions as torch.Tensor (argmax, shape=[N_test])")
    parser.add_argument("--analyze_gradients", action="store_true",
                        help="Enable gradient SVD analysis during training")
    parser.add_argument("--gradient_csv", type=str, default=None,
                        help="Path to save gradient analysis CSV")

    return parser


def oracle_gate_analysis(logits_Ut, logits_Uv, logits_C, labels, train_idx, val_idx, test_idx):
    """
    Offline analysis: theoretical upper bound of oracle gate.

    For each node, we choose the channel that gives the correct prediction.
    Priority: Ut > Uv > C > average
    """
    results = {}

    for split_name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        y = labels[idx]
        p_Ut = th.argmax(logits_Ut[idx], dim=1)
        p_Uv = th.argmax(logits_Uv[idx], dim=1)
        p_C = th.argmax(logits_C[idx], dim=1)
        p_avg = th.argmax((logits_Ut[idx] + logits_Uv[idx] + logits_C[idx]) / 3, dim=1)

        # Individual channel accuracy
        acc_Ut = (p_Ut == y).float().mean().item()
        acc_Uv = (p_Uv == y).float().mean().item()
        acc_C = (p_C == y).float().mean().item()
        acc_avg = (p_avg == y).float().mean().item()

        # Oracle gate: choose the channel that predicts correctly for each node
        correct_Ut = (p_Ut == y)
        correct_Uv = (p_Uv == y)
        correct_C = (p_C == y)

        # Priority: Ut > Uv > C > avg
        oracle_pred = th.where(correct_Ut, p_Ut,
                       th.where(correct_Uv, p_Uv,
                       th.where(correct_C, p_C, p_avg)))
        acc_oracle = (oracle_pred == y).float().mean().item()

        # Classification statistics
        n_total = len(y)
        n_all_correct = (correct_Ut & correct_Uv & correct_C).sum().item()
        n_Ut_only = (correct_Ut & ~correct_Uv & ~correct_C).sum().item()
        n_Uv_only = (~correct_Ut & correct_Uv & ~correct_C).sum().item()
        n_C_only = (~correct_Ut & ~correct_Uv & correct_C).sum().item()
        n_Ut_Uv = (correct_Ut & correct_Uv & ~correct_C).sum().item()
        n_none = (~correct_Ut & ~correct_Uv & ~correct_C).sum().item()

        results[split_name] = {
            "acc_Ut": acc_Ut,
            "acc_Uv": acc_Uv,
            "acc_C": acc_C,
            "acc_avg": acc_avg,
            "acc_oracle": acc_oracle,
            "n_total": n_total,
            "n_all_correct": n_all_correct,
            "n_Ut_only": n_Ut_only,
            "n_Uv_only": n_Uv_only,
            "n_C_only": n_C_only,
            "n_Ut_Uv": n_Ut_Uv,
            "n_none": n_none,
        }

        # Pretty print
        print(f"\n{'='*60}")
        print(f"  Oracle Gate Analysis - {split_name} Set")
        print(f"{'='*60}")
        print(f"  Individual Channel Accuracy:")
        print(f"    Ut (text only, no GNN):     {acc_Ut*100:.2f}%")
        print(f"    Uv (visual only, no GNN):   {acc_Uv*100:.2f}%")
        print(f"    C  (GNN aggregated):         {acc_C*100:.2f}%")
        print(f"    Average (current):           {acc_avg*100:.2f}%")
        print(f"  Oracle Gate (ideal):          {acc_oracle*100:.2f}%")
        print(f"  {'='*60}")
        print(f"  Channel Contribution Statistics:")
        print(f"    Total nodes:                {n_total}")
        print(f"    All three correct:          {n_all_correct} ({n_all_correct/n_total*100:.1f}%)")
        print(f"    Only Ut correct:            {n_Ut_only} ({n_Ut_only/n_total*100:.1f}%)")
        print(f"    Only Uv correct:            {n_Uv_only} ({n_Uv_only/n_total*100:.1f}%)")
        print(f"    Only C correct:             {n_C_only} ({n_C_only/n_total*100:.1f}%)")
        print(f"    Ut+Uv correct, C wrong:     {n_Ut_Uv} ({n_Ut_Uv/n_total*100:.1f}%)")
        print(f"    All three wrong:            {n_none} ({n_none/n_total*100:.1f}%)")
        print(f"  {'='*60}")
        print(f"  Key Insights:")
        print(f"    C-only rescue nodes:       {n_C_only} ({n_C_only/n_total*100:.1f}% of total)")
        print(f"    Oracle improvement over avg: {(acc_oracle-acc_avg)*100:.2f}%")
        print(f"  {'='*60}")

    return results


def main():
    parser = args_init()
    args = parser.parse_args()
    if args.disable_wandb or wandb is None:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        wandb.init(config=args, reinit=True)

    device = th.device("cuda:%d" % args.gpu if th.cuda.is_available() and args.gpu != -1 else "cpu")

    (
        graph, observe_graph, labels, train_idx, val_idx, test_idx,
        text_feat, vis_feat, n_classes,
    ) = _load_mag_context(args, device)

    val_results, test_results = [], []
    select_metric, select_average = args.metric, args.average
    embed_dim = int(args.embed_dim) if args.embed_dim is not None else int(args.n_hidden)

    # Degrade experiment settings
    report_drop = getattr(args, 'report_drop_modality', False)
    degrade_target = str(getattr(args, 'degrade_target', 'both'))
    # Parse degrade_alphas (already defined in add_common_args)
    raw_alphas = getattr(args, 'degrade_alphas', '')
    if raw_alphas is None or str(raw_alphas).strip() == "":
        degrade_alphas = [1.0]
    else:
        import re
        parts = re.split(r'[\s,]+', str(raw_alphas).strip())
        degrade_alphas = [float(p) for p in parts if p]
    best_degrade_metrics = {}  # alpha -> (degrade_text, degrade_vis) from best model
    # Collect degrade metrics across all runs for mean/std computation
    run_degrade_text_results = []
    run_degrade_visual_results = []

    # Store best logits from all channels for oracle gate analysis
    best_logits_Ut_all, best_logits_Uv_all, best_logits_C_all = None, None, None
    best_labels_all = None

    # Global best model state across all runs (for checkpoint saving)
    global_best_model_state = None
    global_best_val_score = -1.0
    global_best_test_logits = None

    # Gradient analyzer setup (initialized once, persists across runs)
    gradient_analyzer = None
    if getattr(args, 'analyze_gradients', False):
        from GNN.Utils.gradient_analyzer import GradientAnalyzer
        analyzer_layer_names = ['enc_t.proj', 'enc_v.proj', 'mp_C', 'head_C', 'head_Ut', 'head_Uv']

    # Efficiency profiling: collect per-run metrics
    efficiency_runs = {
        'peak_memory_MB': [],
        'epoch_times': [],
        'epochs_needed': [],
    }

    # Build one model to count params (before the run loop)
    _model_for_count = SUPRA(
        text_in_dim=int(text_feat.shape[1]), vis_in_dim=int(vis_feat.shape[1]),
        embed_dim=embed_dim, n_classes=n_classes, dropout=float(args.dropout),
        args=args, device=device,
    ).to(device)
    n_params = sum(p.numel() for p in _model_for_count.parameters() if p.requires_grad)
    n_params_M = n_params / 1e6
    del _model_for_count

    for run in range(args.n_runs):
        set_seed(args.seed + run)
        model = SUPRA(
            text_in_dim=int(text_feat.shape[1]), vis_in_dim=int(vis_feat.shape[1]),
            embed_dim=embed_dim, n_classes=n_classes, dropout=float(args.dropout),
            args=args, device=device,
        ).to(device)
        model.reset_parameters()

        if getattr(args, 'analyze_gradients', False):
            analyzer_layer_names = ['enc_t.proj', 'enc_v.proj', 'mp_C', 'head_C', 'head_Ut', 'head_Uv']
            # Re-attach to new model instance
            gradient_analyzer = GradientAnalyzer(model, layer_names=analyzer_layer_names)
            gradient_analyzer.attach()

        # Per-epoch gradient L2 norm history (for starvation verification)
        grad_history = {'enc_t': [], 'enc_v': [], 'gnn': []}

        stopper = initialize_early_stopping(args)
        optimizer, lr_scheduler = initialize_optimizer_and_scheduler(args, model)

        # Peak memory tracking (reset after model init and optimizer creation)
        peak_memory_mb = 0.0
        if th.cuda.is_available():
            th.cuda.reset_peak_memory_stats(device)
            th.cuda.empty_cache()

        best_val_score, final_test_result, best_val_result, total_time = -1.0, 0.0, -1.0, 0.0
        run_best_logits = None
        epochs_needed = args.n_epochs  # will be updated if early stop triggered

        for epoch in range(1, args.n_epochs + 1):
            tic = time.time()
            adjust_learning_rate_if_needed(args, optimizer, epoch)

            model.train()
            optimizer.zero_grad()
            out = model.forward_multiple(observe_graph, text_feat, vis_feat)

            # Pure joint loss with spectral orthogonalization
            total_loss, loss_logs = _compute_losses(out=out, labels=labels, train_idx=train_idx, args=args, graph=observe_graph)
            total_loss.backward()

            # Record gradient L2 norms for gradient starvation verification
            if getattr(args, 'analyze_gradients', False):
                def _grad_norm_sq(m):
                    return sum(
                        p.grad.float().norm(2).pow(2).item()
                        for p in m.parameters()
                        if p.grad is not None
                    )
                grad_history['enc_t'].append(_grad_norm_sq(model.enc_t) ** 0.5)
                grad_history['enc_v'].append(_grad_norm_sq(model.enc_v) ** 0.5)
                gnn_norm_sq = sum(_grad_norm_sq(layer) for layer in model.mp_C)
                grad_history['gnn'].append(gnn_norm_sq ** 0.5)

            optimizer.step()

            if epoch % args.eval_steps == 0:
                model.eval()
                with th.no_grad():
                    out_full = model.forward_multiple(graph, text_feat, vis_feat)
                    logits_e = out_full.logits_final_0

                val_score = get_metric(th.argmax(logits_e[val_idx], dim=1), labels[val_idx], select_metric, average=select_average)
                test_score = get_metric(th.argmax(logits_e[test_idx], dim=1), labels[test_idx], select_metric, average=select_average)

                if val_score > best_val_score:
                    best_val_score = float(val_score)
                    best_val_result = float(val_score)
                    final_test_result = float(test_score)
                    # Store logits from all channels for oracle analysis
                    run_best_logits = {
                        'Ut': out_full.logits_Ut_0.detach().clone(),
                        'Uv': out_full.logits_Uv_0.detach().clone(),
                        'C': out_full.logits_C_0.detach().clone(),
                    }
                    # Save best model state for checkpoint
                    best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    # Update global best model if this is the best across all runs
                    if val_score > global_best_val_score:
                        global_best_val_score = float(val_score)
                        global_best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                        global_best_test_logits = logits_e[test_idx].detach().clone()
                    # Compute degrade metrics if enabled
                    if report_drop and degrade_alphas:
                        for alpha in degrade_alphas:
                            dt, dv = _compute_degrade_metrics_mag(
                                model=model,
                                graph=graph,
                                text_feat=text_feat,
                                visual_feat=vis_feat,
                                labels=labels,
                                idx=test_idx,
                                metric=select_metric,
                                average=select_average,
                                train_idx=train_idx,
                                degrade_alpha=float(alpha),
                                degrade_target=degrade_target,
                            )
                            best_degrade_metrics[alpha] = (_as_scalar_float(dt), _as_scalar_float(dv))
                if stopper and stopper.step(val_score):
                    epochs_needed = epoch
                    break

                total_time += time.time() - tic  # total_time = time for eval_steps epochs

        # Compute avg epoch time: total_time covers epochs_needed epochs
        if epochs_needed > 0:
            avg_epoch_time = total_time / epochs_needed
        else:
            avg_epoch_time = 0.0

        # Record peak memory after training
        if th.cuda.is_available():
            th.cuda.synchronize()
            peak_memory_mb = th.cuda.max_memory_allocated(device) / 1048576.0

        print(f"Run {run+1} Final Test Score: {final_test_result:.4f}")
        val_results.append(best_val_result)
        test_results.append(final_test_result)

        # Collect efficiency profiling data
        efficiency_runs['peak_memory_MB'].append(peak_memory_mb)
        efficiency_runs['epoch_times'].append([avg_epoch_time] * epochs_needed)  # list of per-epoch times
        efficiency_runs['epochs_needed'].append(epochs_needed)

        # Collect degrade metrics from this run
        if report_drop and best_degrade_metrics:
            # Use alpha=1.0 or first alpha as the primary
            primary_alpha = 1.0 if 1.0 in best_degrade_metrics else next(iter(best_degrade_metrics))
            dt, dv = best_degrade_metrics[primary_alpha]
            run_degrade_text_results.append(dt)
            run_degrade_visual_results.append(dv)

        if wandb is not None and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
            wandb.log({f'Val_{args.metric}': best_val_result, f'Test_{args.metric}': final_test_result})

        # Use last run's best logits for oracle analysis
        if run_best_logits is not None:
            best_logits_Ut_all = run_best_logits['Ut']
            best_logits_Uv_all = run_best_logits['Uv']
            best_logits_C_all = run_best_logits['C']
            best_labels_all = labels.clone()

        # Save per-epoch gradient L2 norm for this run (gradient starvation verification)
        if getattr(args, 'analyze_gradients', False) and getattr(args, 'gradient_csv', None):
            import csv
            grad_csv_path = args.gradient_csv.replace('.csv', f'_l2_norm_run{run+1}.csv')
            os.makedirs(os.path.dirname(grad_csv_path), exist_ok=True)
            with open(grad_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'enc_t', 'enc_v', 'gnn'])
                for epoch_idx, (et, ev, g) in enumerate(
                    zip(grad_history['enc_t'], grad_history['enc_v'], grad_history['gnn']), start=1
                ):
                    writer.writerow([epoch_idx, et, ev, g])
            print(f"[Run {run+1}] Gradient L2 norm saved to {grad_csv_path}")

        # Clean up GPU memory between runs to prevent allocator cache accumulation
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()

    def _mean_std(values): return float(np.mean(values)), float(np.std(values))
    def _fmt_pct(values):
        mean, std = _mean_std(values)
        return f"{mean * 100.0:.3f} +/- {std * 100.0:.3f}%"

    print(f"\nRunned {args.n_runs} times")
    print(f"Average val {args.metric}: {_fmt_pct(val_results)}")
    print(f"Average test {args.metric}: {_fmt_pct(test_results)}")

    if report_drop and best_degrade_metrics:
        print(f"\nModality Degradation Results (alpha=1.0):")
        for alpha, (dt, dv) in best_degrade_metrics.items():
            if alpha == 1.0:
                print(f"  degrade_text {args.metric}: {dt*100:.3f}%")
                print(f"  degrade_visual {args.metric}: {dv*100:.3f}%")

    if wandb is not None and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
        wandb.log({f'Mean_Val_{args.metric}': float(np.mean(val_results)), f'Mean_Test_{args.metric}': float(np.mean(test_results))})

    # Oracle Gate Analysis: theoretical upper bound of selective fusion
    if best_logits_Ut_all is not None and best_labels_all is not None:
        print("\n" + "="*60)
        print("  ORACLE GATE ANALYSIS")
        print("="*60)
        oracle_gate_analysis(
            logits_Ut=best_logits_Ut_all,
            logits_Uv=best_logits_Uv_all,
            logits_C=best_logits_C_all,
            labels=best_labels_all,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )

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
    print(f"Efficiency Profile: SUPRA on {args.data_name}")
    print(f"{'='*60}")
    print(f"  Parameters:       {n_params_M:.3f} M")
    print(f"  Peak Memory:     {avg_peak_memory:.2f} ± {std_peak_memory:.2f} MB")
    print(f"  Total Time(est): {avg_total_time:.2f} ± {std_total_time:.2f} s  ({avg_total_time/60:.1f} min)")
    print(f"  Avg Epoch:        {avg_epoch_time:.4f} ± {std_epoch_time:.4f} s/epoch")
    print(f"  Epochs Needed:    {avg_epochs_needed:.1f} ± {std_epochs_needed:.1f}")
    print(f"{'='*60}")

    # Save results to CSV if requested
    if getattr(args, 'result_csv', None) or getattr(args, 'result_csv_all', None):
        test_mean = float(np.mean(test_results))
        test_std = float(np.std(test_results))
        method_name = getattr(args, 'result_tag', None) or "SUPRA"

        # Extract degrade metrics for direct parameters
        degrade_text_value = None
        degrade_visual_value = None
        extra: Dict[str, object] = {"full_std": test_std}

        if report_drop and run_degrade_text_results:
            degrade_text_value = float(np.mean(run_degrade_text_results))
            degrade_visual_value = float(np.mean(run_degrade_visual_results))
            extra["degrade_text_std"] = float(np.std(run_degrade_text_results))
            extra["degrade_visual_std"] = float(np.std(run_degrade_visual_results))

        row = build_result_row(
            args=args,
            method=method_name,
            full_metric=test_mean,
            degrade_text=degrade_text_value,
            degrade_visual=degrade_visual_value,
            extra=extra,
        )
        key_fields = ["dataset", "method", "backbone", "metric", "single_modality", "inductive", "fewshots"]
        if getattr(args, 'result_csv', None):
            update_best_result_csv(args.result_csv, row, key_fields=key_fields, score_field="full")
        if getattr(args, 'result_csv_all', None):
            append_result_csv(args.result_csv_all, row)

    # Write gradient analysis CSV if requested
    if gradient_analyzer is not None and getattr(args, 'gradient_csv', None):
        import csv
        summary = gradient_analyzer.get_summary()
        all_stats = gradient_analyzer.get_all_stats()
        gradient_analyzer.detach()
        with open(args.gradient_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['run', 'layer', 'stable_rank_mean', 'stable_rank_std',
                             'cond_num_mean', 'cond_num_std', 'ortho_score_mean', 'ortho_score_std', 'n_samples'])
            for run_idx in range(args.n_runs):
                for layer_name, stats in summary.items():
                    writer.writerow([
                        run_idx + 1, layer_name,
                        stats['stable_rank_mean'], stats['stable_rank_std'],
                        stats['cond_num_mean'], stats['cond_num_std'],
                        stats['ortho_score_mean'], stats['ortho_score_std'],
                        stats['n_samples'],
                    ])
        print(f"Gradient analysis saved to {args.gradient_csv}")

    # Save model checkpoint if requested
    if getattr(args, 'save_checkpoint', None) and global_best_model_state is not None:
        th.save({
            'model_state_dict': global_best_model_state,
            'args': args,
            'text_in_dim': int(text_feat.shape[1]),
            'vis_in_dim': int(vis_feat.shape[1]),
            'embed_dim': embed_dim,
            'n_classes': n_classes,
        }, args.save_checkpoint)
        print(f"Checkpoint saved to {args.save_checkpoint}")

    # Export predictions if requested
    if getattr(args, 'export_predictions', None) and global_best_test_logits is not None:
        pred_path = str(args.export_predictions)
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        th.save(th.argmax(global_best_test_logits, dim=1), pred_path)
        print(f"[Export] Test predictions → {pred_path}")

if __name__ == "__main__":
    main()
