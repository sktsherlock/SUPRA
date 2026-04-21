"""Polysemanticity / collision analysis for MAG baselines and SUPRA.

Implements two analysis axes:
1) Modality-level collision: compare neuron/dimension activation patterns under text-only vs visual-only inputs.
2) Neighbor/topology-level collision: compare representations on full graph vs self-loop-only graph, plus degree analysis.

The script supports:
- Late_GNN (per-modality GNN branches + late fusion)
- SUPRA (Shared/Unique streams with asymmetric propagation)

Outputs:
- Console summary
- CSV files under --out_dir

Example:
python analyze_polysemanticity_mag.py \
  --graph_path ... --data_name ... --text_feature ... --visual_feature ... \
  --model_name GCN --n_layers 2 --n_hidden 256 --dropout 0.2 --selfloop \
  --late_ckpt path/to/late_state.pt --supra_ckpt path/to/supra_ckpt.pt \
  --top_ratio 0.1 --degree_bins 0,1,2,4,8,16,32,64,128

Notes:
- For strict "only one modality" semantics, enable --strict_zero, which masks out the absent modality branch
  after encoding/propagation to avoid bias terms leaking non-zero activations.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
except Exception as e:  # pragma: no cover
    dgl = None

from GNN.GraphData import load_data
from GNN.Utils.LossFunction import cross_entropy, get_metric
from GNN.Utils.model_config import (
    add_common_args,
    add_sage_args,
    add_gat_args,
    add_revgat_args,
    add_sgc_args,
    add_appnp_args,
)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t", "on"):
        return True
    if s in ("false", "0", "no", "n", "f", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


@dataclass
class MagContext:
    graph: Any
    observe_graph: Any
    labels: th.Tensor
    train_idx: th.Tensor
    val_idx: th.Tensor
    test_idx: th.Tensor
    text_feat: th.Tensor
    vis_feat: th.Tensor
    n_classes: int


def _ensure_dir(path: str) -> None:
    if path is None:
        return
    os.makedirs(path, exist_ok=True)


def _resolve_ckpt_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    p = str(path)
    if p.strip() == "":
        return path
    if os.path.exists(p):
        return p
    root, ext = os.path.splitext(p)
    cand = f"{root}_run1{ext}" if ext else f"{p}_run1"
    if os.path.exists(cand):
        print(f"[Warn] ckpt not found: {p}; fallback to: {cand}")
        return cand
    return p


def _as_list_of_ints(raw: str) -> List[int]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def _make_self_loop_graph(graph) -> Any:
    if dgl is None:
        raise RuntimeError("dgl is required for this analysis")
    n = int(graph.num_nodes())
    ids = th.arange(n, device=graph.device)
    g = dgl.graph((ids, ids), num_nodes=n, device=graph.device)
    g.create_formats_()
    return g


def _importance_mean_abs(h: th.Tensor, idx: th.Tensor) -> th.Tensor:
    if idx is None or int(idx.numel()) == 0:
        return h.abs().mean(dim=0)
    return h[idx].abs().mean(dim=0)


def _topk_set(scores: th.Tensor, top_ratio: float, eps: float = 1e-12) -> set[int]:
    # If all scores are (near) zero, "top-k" is arbitrary and should not be treated as meaningful.
    # Returning an empty set makes IoU semantics consistent under strict_zero (absent modality => no active dims).
    if scores.numel() == 0:
        return set()
    if float(scores.detach().abs().max().cpu().item()) <= float(eps):
        return set()

    d = int(scores.numel())
    k = int(np.ceil(float(top_ratio) * float(d)))
    k = max(1, min(d, k))
    _, top_idx = th.topk(scores, k=k, largest=True)
    return set(int(x) for x in top_idx.detach().cpu().tolist())


def _iou(a: set[int], b: set[int]) -> float:
    if len(a) == 0 and len(b) == 0:
        return 1.0
    u = a | b
    if len(u) == 0:
        return 0.0
    return float(len(a & b)) / float(len(u))


def _cosine_per_node(a: th.Tensor, b: th.Tensor, eps: float = 1e-12) -> th.Tensor:
    # a,b: [N, d]
    an = a.norm(dim=1).clamp_min(eps)
    bn = b.norm(dim=1).clamp_min(eps)
    return (a * b).sum(dim=1) / (an * bn)


def _l2_ratio_per_node(num: th.Tensor, den: th.Tensor, eps: float = 1e-12) -> th.Tensor:
    # ratio_i = ||num_i||_2 / (||den_i||_2 + eps)
    nn = num.norm(dim=1)
    dn = den.norm(dim=1).clamp_min(eps)
    return nn / dn


def _bin_mean(x: np.ndarray, y: np.ndarray, bins: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    bins = list(int(b) for b in bins)
    if len(bins) < 2:
        raise ValueError("degree_bins must have at least 2 entries")

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (x >= lo) & (x < hi)
        if not np.any(mask):
            rows.append({"deg_lo": lo, "deg_hi": hi, "n": 0, "mean": float("nan"), "std": float("nan")})
            continue
        v = y[mask]
        rows.append({
            "deg_lo": lo,
            "deg_hi": hi,
            "n": int(v.size),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
        })
    return rows


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    if not rows:
        return

    def _round_obj(v: Any, digits: Optional[int]) -> Any:
        if digits is None:
            return v
        if isinstance(v, (np.floating, float)):
            fv = float(v)
            if not np.isfinite(fv):
                return fv
            return round(fv, int(digits))
        if isinstance(v, (np.integer, int)):
            return int(v)
        return v

    digits: Optional[int] = getattr(_write_csv, "_float_digits", None)  # type: ignore[attr-defined]
    if digits is not None:
        rows = [{k: _round_obj(v, digits) for k, v in r.items()} for r in rows]

    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _load_mag_context(args, device: th.device) -> MagContext:
    # Mirror SUPRA/Late_GNN loaders.
    graph, labels, train_idx, val_idx, test_idx = load_data(
        args.graph_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        name=args.data_name,
        fewshots=args.fewshots,
    )

    if args.undirected:
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)

    observe_graph = graph
    if args.inductive:
        # Keep consistent with MAG scripts: remove edges connected to val/test nodes.
        from GNN.Baselines.Early_GNN import Early_GNN as mag_base

        observe_graph = mag_base._make_observe_graph_inductive(graph, val_idx, test_idx)

    if args.selfloop:
        graph = graph.remove_self_loop().add_self_loop()
        observe_graph = observe_graph.remove_self_loop().add_self_loop()

    graph.create_formats_()
    observe_graph.create_formats_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)
    graph = graph.to(device)
    observe_graph = observe_graph.to(device)

    text_feat = th.from_numpy(np.load(args.text_feature).astype(np.float32)).to(device)
    vis_feat = th.from_numpy(np.load(args.visual_feature).astype(np.float32)).to(device)
    n_classes = int((labels.max() + 1).item())

    return MagContext(
        graph=graph,
        observe_graph=observe_graph,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        text_feat=text_feat,
        vis_feat=vis_feat,
        n_classes=n_classes,
    )


# ------------------------- Model builders -------------------------

def _build_late_gnn(args, ctx: MagContext, device: th.device) -> nn.Module:
    from GNN.Baselines.Late_GNN import Late_GNN as late
    from GNN.Baselines.Early_GNN import Early_GNN as mag_base

    proj_dim = int(args.mm_proj_dim) if args.mm_proj_dim is not None else int(args.n_hidden)
    embed_dim = int(args.late_embed_dim) if args.late_embed_dim is not None else int(args.n_hidden)

    text_encoder = mag_base.ModalityEncoder(int(ctx.text_feat.shape[1]), proj_dim, float(args.dropout)).to(device)
    visual_encoder = mag_base.ModalityEncoder(int(ctx.vis_feat.shape[1]), proj_dim, float(args.dropout)).to(device)
    text_gnn = mag_base._build_gnn_backbone(args, proj_dim, embed_dim, device)
    vis_gnn = mag_base._build_gnn_backbone(args, proj_dim, embed_dim, device)
    classifier = nn.Linear(embed_dim, ctx.n_classes).to(device)

    model = late.LateFusionMAG(
        text_encoder,
        visual_encoder,
        text_gnn,
        vis_gnn,
        classifier,
        modality_dropout=0.0,
    ).to(device)
    model.reset_parameters()
    return model


def _build_early_gnn(args, ctx: MagContext, device: th.device) -> nn.Module:
    from GNN.Baselines.Early_GNN import Early_GNN as early

    proj_dim = int(args.mm_proj_dim) if args.mm_proj_dim is not None else int(args.n_hidden)
    text_encoder = early.ModalityEncoder(int(ctx.text_feat.shape[1]), proj_dim, float(args.dropout)).to(device)
    visual_encoder = early.ModalityEncoder(int(ctx.vis_feat.shape[1]), proj_dim, float(args.dropout)).to(device)

    if str(getattr(args, "early_fuse", "concat")).lower() == "sum":
        gnn_in_dim = proj_dim
    else:
        gnn_in_dim = proj_dim * 2

    # Use embed_dim as the backbone output dim to probe hidden, then attach a classifier.
    embed_dim = int(args.early_embed_dim) if args.early_embed_dim is not None else int(args.n_hidden)
    gnn = early._build_gnn_backbone(args, gnn_in_dim, embed_dim, device)
    model = early.SimpleMAGGNN(
        text_encoder=text_encoder,
        visual_encoder=visual_encoder,
        gnn=gnn,
        early_fuse=str(getattr(args, "early_fuse", "concat")),
        modality_dropout=0.0,
    ).to(device)

    # Wrap with a linear classifier so gradients are meaningful.
    head = nn.Linear(embed_dim, ctx.n_classes).to(device)
    if hasattr(early, "_SeparateHeadWrapper"):
        wrapper = early._SeparateHeadWrapper(model, head)  # type: ignore[attr-defined]
    else:
        wrapper = nn.Sequential(model, head)
        wrapper._poly_base = model  # type: ignore[attr-defined]
        wrapper._poly_head = wrapper[1]  # type: ignore[attr-defined]
    return wrapper


def _build_supra(args, ctx: MagContext, device: th.device) -> nn.Module:
    from GNN import SUPRA

    embed_dim = int(args.pid_embed_dim) if args.pid_embed_dim is not None else int(args.n_hidden)
    pid_L = int(args.pid_L) if args.pid_L is not None else int(args.n_layers)

    model = SUPRA.PIDMAGNN(
        text_in_dim=int(ctx.text_feat.shape[1]),
        vis_in_dim=int(ctx.vis_feat.shape[1]),
        embed_dim=embed_dim,
        n_classes=ctx.n_classes,
        dropout=float(args.pid_dropout),
        args=args,
        device=device,
        pid_L=pid_L,
        pid_lu=int(args.pid_lu),
    ).to(device)
    model.reset_parameters()
    return model


def _build_tri_gnn(args, ctx: MagContext, device: th.device) -> nn.Module:
    # Tri_GNN module no longer exists - this function is disabled
    raise NotImplementedError("Tri_GNN has been removed from the codebase")
    from GNN.Baselines.Early_GNN import Early_GNN as mag_base

    proj_dim = int(args.mm_proj_dim) if args.mm_proj_dim is not None else int(args.n_hidden)
    embed_dim = int(getattr(args, "tri_embed_dim", None) or args.n_hidden)

    text_encoder = mag_base.ModalityEncoder(int(ctx.text_feat.shape[1]), proj_dim, float(args.dropout)).to(device)
    visual_encoder = mag_base.ModalityEncoder(int(ctx.vis_feat.shape[1]), proj_dim, float(args.dropout)).to(device)
    early_gnn = mag_base._build_gnn_backbone(args, 2 * proj_dim, embed_dim, device)
    text_gnn = mag_base._build_gnn_backbone(args, proj_dim, embed_dim, device)
    vis_gnn = mag_base._build_gnn_backbone(args, proj_dim, embed_dim, device)
    classifier = nn.Linear(embed_dim, ctx.n_classes).to(device)

    model = tri.TriFusionMAG(
        text_encoder,
        visual_encoder,
        early_gnn,
        text_gnn,
        vis_gnn,
        classifier,
        modality_dropout=0.0,
    ).to(device)
    model.reset_parameters()
    return model


def _load_ckpt(path: str) -> Dict[str, Any]:
    obj = th.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj
    # Allow raw state_dict.
    return {"state_dict": obj}


# ------------------------- Representation extractors -------------------------

def _zero_like_feat(feat: th.Tensor) -> th.Tensor:
    return th.zeros_like(feat)


def _late_reprs(
    model,
    graph,
    text_feat: th.Tensor,
    vis_feat: th.Tensor,
    *,
    strict_zero: bool,
) -> Dict[str, th.Tensor]:
    # Returns: text_h, vis_h, fused
    text_z = model.text_encoder(text_feat)
    vis_z = model.visual_encoder(vis_feat)

    if strict_zero:
        present_t = (text_feat.abs().sum(dim=1) > 0).to(device=text_feat.device)
        present_v = (vis_feat.abs().sum(dim=1) > 0).to(device=vis_feat.device)
        text_z = text_z * present_t.to(dtype=text_z.dtype).unsqueeze(1)
        vis_z = vis_z * present_v.to(dtype=vis_z.dtype).unsqueeze(1)

    text_h = model.text_gnn(graph, text_z)
    vis_h = model.visual_gnn(graph, vis_z)

    if strict_zero:
        present_t = (text_feat.abs().sum(dim=1) > 0).to(device=text_feat.device)
        present_v = (vis_feat.abs().sum(dim=1) > 0).to(device=vis_feat.device)
        text_h = text_h * present_t.to(dtype=text_h.dtype).unsqueeze(1)
        vis_h = vis_h * present_v.to(dtype=vis_h.dtype).unsqueeze(1)

    fused = model.fuse_embeddings(text_h, vis_h)
    return {"late/text_h": text_h, "late/vis_h": vis_h, "late/fused": fused}


def _early_reprs(
    wrapper,
    graph,
    text_feat: th.Tensor,
    vis_feat: th.Tensor,
    *,
    strict_zero: bool,
) -> Dict[str, th.Tensor]:
    base = wrapper._poly_base  # type: ignore[attr-defined]

    drop_text = False
    drop_visual = False
    if strict_zero:
        # In analysis we use all-zeros matrices to represent "drop a modality".
        drop_text = float(text_feat.abs().sum().detach().cpu().item()) == 0.0
        drop_visual = float(vis_feat.abs().sum().detach().cpu().item()) == 0.0
    h = base.forward_with_drop(graph, text_feat, vis_feat, drop_text=drop_text, drop_visual=drop_visual)
    return {"early/h": h}


def _supra_reprs(
    model,
    graph,
    text_feat: th.Tensor,
    vis_feat: th.Tensor,
    *,
    strict_zero: bool,
) -> Dict[str, th.Tensor]:
    # Use forward_multiple to obtain embeddings.
    model.eval()
    with th.no_grad():
        out = model.forward_multiple(graph, text_feat, vis_feat, stochastic=False, profile_mem=False)

    emb_C = out.emb_C_0
    emb_Ut = out.emb_Ut_0
    emb_Uv = out.emb_Uv_0

    if strict_zero:
        present_t = (text_feat.abs().sum(dim=1) > 0).to(device=text_feat.device)
        present_v = (vis_feat.abs().sum(dim=1) > 0).to(device=vis_feat.device)
        if emb_Ut is not None:
            emb_Ut = emb_Ut * present_t.to(dtype=emb_Ut.dtype).unsqueeze(1)
        if emb_Uv is not None:
            emb_Uv = emb_Uv * present_v.to(dtype=emb_Uv.dtype).unsqueeze(1)

    reps: Dict[str, th.Tensor] = {}
    if emb_C is not None:
        reps["supra/C"] = emb_C
    if emb_Ut is not None:
        reps["supra/Ut"] = emb_Ut
    if emb_Uv is not None:
        reps["supra/Uv"] = emb_Uv
    reps["supra/logits_final"] = out.logits_final_0
    return reps


def _tri_reprs(
    model,
    graph,
    text_feat: th.Tensor,
    vis_feat: th.Tensor,
    *,
    strict_zero: bool,
) -> Dict[str, th.Tensor]:
    # Extract branch embeddings and fused embedding.
    text_z = model.text_encoder(text_feat)
    vis_z = model.visual_encoder(vis_feat)

    if strict_zero:
        present_t = (text_feat.abs().sum(dim=1) > 0).to(device=text_feat.device)
        present_v = (vis_feat.abs().sum(dim=1) > 0).to(device=vis_feat.device)
        text_z = text_z * present_t.to(dtype=text_z.dtype).unsqueeze(1)
        vis_z = vis_z * present_v.to(dtype=vis_z.dtype).unsqueeze(1)

    early_feat = th.cat([text_z, vis_z], dim=1)
    early_h = model.early_gnn(graph, early_feat)
    text_h = model.text_gnn(graph, text_z)
    vis_h = model.visual_gnn(graph, vis_z)

    if strict_zero:
        present_t = (text_feat.abs().sum(dim=1) > 0).to(device=text_feat.device)
        present_v = (vis_feat.abs().sum(dim=1) > 0).to(device=vis_feat.device)
        # For branch-specific semantics.
        text_h = text_h * present_t.to(dtype=text_h.dtype).unsqueeze(1)
        vis_h = vis_h * present_v.to(dtype=vis_h.dtype).unsqueeze(1)

    fused = model.fuse_embeddings(early_h, text_h, vis_h)
    return {
        "tri/early_h": early_h,
        "tri/text_h": text_h,
        "tri/vis_h": vis_h,
        "tri/fused": fused,
    }


# ------------------------- Analyses -------------------------

def _modality_iou(
    *,
    get_reprs,
    model,
    graph,
    text_feat: th.Tensor,
    vis_feat: th.Tensor,
    idx: th.Tensor,
    top_ratio: float,
    strict_zero: bool,
    tags: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # Text-only and Visual-only inputs.
    text_only = (text_feat, _zero_like_feat(vis_feat))
    vis_only = (_zero_like_feat(text_feat), vis_feat)

    with th.no_grad():
        reps_t = get_reprs(model, graph, text_only[0], text_only[1], strict_zero=strict_zero)
        reps_v = get_reprs(model, graph, vis_only[0], vis_only[1], strict_zero=strict_zero)

    for tag in tags:
        if tag not in reps_t or tag not in reps_v:
            continue
        ht = reps_t[tag]
        hv = reps_v[tag]
        if ht.dim() != 2 or hv.dim() != 2:
            continue
        st = _importance_mean_abs(ht, idx)
        sv = _importance_mean_abs(hv, idx)
        at = _topk_set(st, top_ratio)
        av = _topk_set(sv, top_ratio)
        rows.append({
            "space": tag,
            "strict_zero": int(bool(strict_zero)),
            "top_ratio": float(top_ratio),
            "dim": int(ht.shape[1]),
            "iou_text_vs_visual": _iou(at, av),
        })

    return rows


def _neighbor_overlap_and_degree(
    *,
    get_reprs,
    model,
    graph_full,
    graph_self,
    text_feat: th.Tensor,
    vis_feat: th.Tensor,
    idx: th.Tensor,
    top_ratio: float,
    strict_zero: bool,
    degree_bins: Sequence[int],
    tags: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    overlap_rows: List[Dict[str, Any]] = []
    degree_rows: List[Dict[str, Any]] = []
    overlap_by_deg_rows: List[Dict[str, Any]] = []
    energy_by_deg_rows: List[Dict[str, Any]] = []

    with th.no_grad():
        reps_full = get_reprs(model, graph_full, text_feat, vis_feat, strict_zero=strict_zero)
        reps_self = get_reprs(model, graph_self, text_feat, vis_feat, strict_zero=strict_zero)

    deg = graph_full.in_degrees().detach().cpu().numpy().astype(np.int64)

    for tag in tags:
        if tag not in reps_full or tag not in reps_self:
            continue
        hf = reps_full[tag]
        hs = reps_self[tag]
        if hf.dim() != 2 or hs.dim() != 2:
            continue
        # Neighbor-effect proxy.
        hn = hf - hs

        s_self = _importance_mean_abs(hs, idx)
        s_nei = _importance_mean_abs(hn, idx)
        a_self = _topk_set(s_self, top_ratio)
        a_nei = _topk_set(s_nei, top_ratio)

        overlap_rows.append({
            "space": tag,
            "strict_zero": int(bool(strict_zero)),
            "top_ratio": float(top_ratio),
            "dim": int(hf.shape[1]),
            "iou_self_vs_neighbor": _iou(a_self, a_nei),
        })

        # Degree-binned IoU(self, neighbor) using bin-specific top-k dims.
        bins = list(int(b) for b in degree_bins)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (deg >= lo) & (deg < hi)
            n_bin = int(np.sum(mask))
            if n_bin <= 0:
                overlap_by_deg_rows.append({
                    "space": tag,
                    "strict_zero": int(bool(strict_zero)),
                    "top_ratio": float(top_ratio),
                    "dim": int(hf.shape[1]),
                    "deg_lo": int(lo),
                    "deg_hi": int(hi),
                    "n": 0,
                    "iou_self_vs_neighbor": float("nan"),
                })
                continue

            idx_bin_np = np.where(mask)[0].astype(np.int64)
            idx_bin = th.from_numpy(idx_bin_np).to(device=hs.device)
            s_self_bin = hs[idx_bin].abs().mean(dim=0)
            s_nei_bin = hn[idx_bin].abs().mean(dim=0)
            a_self_bin = _topk_set(s_self_bin, top_ratio)
            a_nei_bin = _topk_set(s_nei_bin, top_ratio)
            overlap_by_deg_rows.append({
                "space": tag,
                "strict_zero": int(bool(strict_zero)),
                "top_ratio": float(top_ratio),
                "dim": int(hf.shape[1]),
                "deg_lo": int(lo),
                "deg_hi": int(hi),
                "n": n_bin,
                "iou_self_vs_neighbor": _iou(a_self_bin, a_nei_bin),
            })

        # Degree-retention curve: cosine(sim(full, self)) per node.
        cos = _cosine_per_node(hf.detach(), hs.detach()).detach().cpu().numpy().astype(np.float32)
        binned = _bin_mean(deg, cos, degree_bins)
        for r in binned:
            degree_rows.append({
                "space": tag,
                "strict_zero": int(bool(strict_zero)),
                "metric": "cos(full,self)",
                **r,
            })

        # Degree curve: neighbor-energy ratio ||h_full - h_self|| / ||h_self||.
        ratio = _l2_ratio_per_node(hn.detach(), hs.detach()).detach().cpu().numpy().astype(np.float32)
        binned_ratio = _bin_mean(deg, ratio, degree_bins)
        for r in binned_ratio:
            energy_by_deg_rows.append({
                "space": tag,
                "strict_zero": int(bool(strict_zero)),
                "metric": "||neighbor||/||self||",
                **r,
            })

        # Add an overall Pearson correlation with log(deg+1)
        x = np.log1p(deg.astype(np.float32))
        y = cos
        if np.isfinite(y).all() and y.size > 1:
            xm = float(np.mean(x))
            ym = float(np.mean(y))
            denom = float(np.std(x) * np.std(y) + 1e-12)
            corr = float(np.mean((x - xm) * (y - ym)) / denom)
        else:
            corr = float("nan")
        overlap_rows[-1]["pearson_logdeg_vs_retention"] = corr

    return overlap_rows, degree_rows, overlap_by_deg_rows, energy_by_deg_rows


def _eval_logits(
    *,
    logits: th.Tensor,
    labels: th.Tensor,
    idx: th.Tensor,
    metric: str,
    average: Optional[str],
) -> float:
    if idx is None or int(idx.numel()) == 0:
        return float("nan")
    pred = logits[idx].argmax(dim=1)
    return float(get_metric(pred, labels[idx], metric, average=average))


def _grad_flow(
    *,
    mode: str,
    model_name: str,
    model,
    ctx: MagContext,
    graph,
    strict_zero: bool,
    label_smoothing: float,
) -> Dict[str, float]:
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    if model_name == "late":
        if mode == "both":
            text_feat, vis_feat = ctx.text_feat, ctx.vis_feat
        elif mode == "text":
            text_feat, vis_feat = ctx.text_feat, _zero_like_feat(ctx.vis_feat)
        else:
            text_feat, vis_feat = _zero_like_feat(ctx.text_feat), ctx.vis_feat

        reps = _late_reprs(model, graph, text_feat, vis_feat, strict_zero=strict_zero)
        fused = reps["late/fused"]
        logits = model.classifier(fused)
        loss = cross_entropy(logits[ctx.train_idx], ctx.labels[ctx.train_idx], label_smoothing=float(label_smoothing))
        loss.backward()

        def _group_norm(mods: Sequence[nn.Module]) -> float:
            acc = 0.0
            for m in mods:
                for p in m.parameters(recurse=True):
                    if p.grad is None:
                        continue
                    acc += float(p.grad.detach().norm().cpu().item())
            return acc

        g_text = _group_norm([model.text_encoder, model.text_gnn])
        g_vis = _group_norm([model.visual_encoder, model.visual_gnn])
        g_head = _group_norm([model.classifier])

        return {
            "loss": float(loss.detach().cpu().item()),
            "grad/text": g_text,
            "grad/visual": g_vis,
            "grad/head": g_head,
            "grad/ratio_text_over_visual": float(g_text / (g_vis + 1e-12)),
            "grad/ratio_visual_over_text": float(g_vis / (g_text + 1e-12)),
        }

    raise ValueError(f"Unknown model_name for grad flow: {model_name}")


def _grad_flow_early(
    *,
    mode: str,
    model,
    ctx: MagContext,
    graph,
    strict_zero: bool,
    label_smoothing: float,
) -> Dict[str, float]:
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    base = model._poly_base  # type: ignore[attr-defined]
    head = model._poly_head  # type: ignore[attr-defined]

    if mode == "both":
        text_feat, vis_feat = ctx.text_feat, ctx.vis_feat
        drop_text = False
        drop_visual = False
    elif mode == "text":
        text_feat, vis_feat = ctx.text_feat, _zero_like_feat(ctx.vis_feat)
        drop_text = False
        drop_visual = bool(strict_zero)
    else:
        text_feat, vis_feat = _zero_like_feat(ctx.text_feat), ctx.vis_feat
        drop_text = bool(strict_zero)
        drop_visual = False

    h = base.forward_with_drop(graph, text_feat, vis_feat, drop_text=drop_text, drop_visual=drop_visual)
    logits = head(h)
    loss = cross_entropy(logits[ctx.train_idx], ctx.labels[ctx.train_idx], label_smoothing=float(label_smoothing))
    loss.backward()

    def _group_norm(mods: Sequence[nn.Module]) -> float:
        acc = 0.0
        for m in mods:
            for p in m.parameters(recurse=True):
                if p.grad is None:
                    continue
                acc += float(p.grad.detach().norm().cpu().item())
        return acc

    g_text = _group_norm([base.text_encoder])
    g_vis = _group_norm([base.visual_encoder])
    g_backbone = _group_norm([base.gnn])
    g_head = _group_norm([head])

    return {
        "loss": float(loss.detach().cpu().item()),
        "grad/text": g_text,
        "grad/visual": g_vis,
        "grad/backbone": g_backbone,
        "grad/head": g_head,
        "grad/ratio_text_over_visual": float(g_text / (g_vis + 1e-12)),
        "grad/ratio_visual_over_text": float(g_vis / (g_text + 1e-12)),
    }


def _grad_flow_supra(
    *,
    model,
    ctx: MagContext,
    graph,
    strict_zero: bool,
    mode: str,
    label_smoothing: float,
) -> Dict[str, float]:
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    if mode == "both":
        text_feat, vis_feat = ctx.text_feat, ctx.vis_feat
    elif mode == "text":
        text_feat, vis_feat = ctx.text_feat, _zero_like_feat(ctx.vis_feat)
    else:
        text_feat, vis_feat = _zero_like_feat(ctx.text_feat), ctx.vis_feat

    out = model.forward_multiple(graph, text_feat, vis_feat, stochastic=False, profile_mem=False)
    loss = cross_entropy(out.logits_final_0[ctx.train_idx], ctx.labels[ctx.train_idx], label_smoothing=float(label_smoothing))
    loss.backward()

    def _group_norm(mods: Sequence[nn.Module]) -> float:
        acc = 0.0
        for m in mods:
            for p in m.parameters(recurse=True):
                if p.grad is None:
                    continue
                acc += float(p.grad.detach().norm().cpu().item())
        return acc

    g_ut = _group_norm([model.enc_t, model.mp_Ut, model.head_Ut])
    g_uv = _group_norm([model.enc_v, model.mp_Uv, model.head_Uv])
    g_c = _group_norm([model.mp_C, model.head_C])

    return {
        "loss": float(loss.detach().cpu().item()),
        "grad/Ut": g_ut,
        "grad/Uv": g_uv,
        "grad/C": g_c,
        "grad/ratio_Ut_over_Uv": float(g_ut / (g_uv + 1e-12)),
        "grad/ratio_Uv_over_Ut": float(g_uv / (g_ut + 1e-12)),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        "MAG polysemanticity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data/backbone args
    add_common_args(p)
    add_sage_args(p)
    add_gat_args(p)
    add_revgat_args(p)
    add_sgc_args(p)
    add_appnp_args(p)

    # Compatibility: training scripts accept this flag; analysis ignores it.
    p.add_argument("--disable_wandb", action="store_true", help="Ignored (kept for CLI compatibility)")

    # Checkpoints
    p.add_argument("--late_ckpt", type=str, default=None, help="Path to Late_GNN state_dict or ckpt dict.")
    p.add_argument("--supra_ckpt", type=str, default=None, help="Path to SUPRA ckpt (dict with state_dict).")
    p.add_argument("--early_ckpt", type=str, default=None, help="Optional: Early_GNN state_dict.")
    p.add_argument("--tri_ckpt", type=str, default=None, help="Optional: Tri_GNN state_dict.")

    # Late/Early model dims
    p.add_argument("--late_embed_dim", type=int, default=None)
    p.add_argument("--mm_proj_dim", type=int, default=None)
    p.add_argument("--early_fuse", type=str, default="concat", choices=["concat", "sum"])
    p.add_argument("--early_embed_dim", type=int, default=None)
    p.add_argument("--tri_embed_dim", type=int, default=None)

    # SUPRA params (minimal subset)
    p.add_argument("--pid_embed_dim", type=int, default=None)
    p.add_argument("--pid_dropout", type=float, default=0.2)
    p.add_argument("--pid_L", type=int, default=None)
    p.add_argument("--pid_lu", type=int, default=0)
    p.add_argument("--ablation_remove_unique_mlp", action="store_true", default=True)
    p.add_argument("--ablation_remove_shared_mlp", action="store_true")

    # Analysis controls
    p.add_argument("--out_dir", type=str, default="Polysemanticity", help="Output folder for CSVs")
    p.add_argument("--top_ratio", type=float, default=0.1, help="Top neuron ratio (e.g., 0.1 for top-10%)")
    p.add_argument(
        "--degree_bins",
        type=str,
        default="0,1,2,4,8,16,32,64,128,1000000000",
        help="Comma-separated degree bin edges (inclusive-exclusive).",
    )
    p.add_argument(
        "--spaces",
        type=str,
        default="late/fused,supra/Ut,supra/Uv,supra/C",
        help="Comma-separated representation spaces to analyze.",
    )
    p.add_argument(
        "--float_digits",
        type=int,
        default=6,
        help="Round float outputs in CSVs to this many digits (use -1 to disable rounding).",
    )
    p.add_argument("--strict_zero", action="store_true", help="Mask absent modality to exact zeros (avoid bias leakage)")
    p.add_argument(
        "--allow_random_ckpt",
        type=_str2bool,
        default=False,
        help="If true, allow running analysis without providing ckpt paths (uses random weights).",
    )

    # Extra outputs for rebuttal-friendly analysis.
    p.add_argument(
        "--write_pred_eval",
        action="store_true",
        help="If set, write per-model/logit prediction metrics (e.g., per-channel accuracy) to CSV.",
    )

    return p


def main():
    args = build_parser().parse_args()

    float_digits = int(getattr(args, "float_digits", 6))
    if float_digits >= 0:
        setattr(_write_csv, "_float_digits", float_digits)  # type: ignore[attr-defined]
    else:
        setattr(_write_csv, "_float_digits", None)  # type: ignore[attr-defined]

    args.late_ckpt = _resolve_ckpt_path(getattr(args, "late_ckpt", None))
    args.supra_ckpt = _resolve_ckpt_path(getattr(args, "supra_ckpt", None))
    args.early_ckpt = _resolve_ckpt_path(getattr(args, "early_ckpt", None))
    args.tri_ckpt = _resolve_ckpt_path(getattr(args, "tri_ckpt", None))

    device = th.device("cuda:%d" % args.gpu if th.cuda.is_available() and args.gpu != -1 else "cpu")
    ctx = _load_mag_context(args, device)

    if dgl is None:
        raise RuntimeError("dgl is required")

    out_dir = str(args.out_dir)
    _ensure_dir(out_dir)

    spaces = [s.strip() for s in str(args.spaces).split(",") if s.strip()]
    degree_bins = _as_list_of_ints(args.degree_bins)

    graph_self = _make_self_loop_graph(ctx.graph)

    # Build models (only if needed)
    models: Dict[str, nn.Module] = {}
    get_reprs_fns: Dict[str, Any] = {}

    want_late = any(s.startswith("late/") for s in spaces)
    want_supra = any(s.startswith("supra/") for s in spaces)
    want_early = any(s.startswith("early/") for s in spaces)
    want_tri = any(s.startswith("tri/") for s in spaces)

    if (want_late or want_supra or want_early or want_tri) and (dgl is None):
        raise RuntimeError("dgl is required")

    if want_late and (not args.allow_random_ckpt):
        if args.late_ckpt is None or str(args.late_ckpt).strip() == "":
            raise ValueError("late spaces requested but --late_ckpt not provided; set --allow_random_ckpt true to override")
        if not os.path.exists(str(args.late_ckpt)):
            raise FileNotFoundError(f"late_ckpt not found: {args.late_ckpt}")

    if want_supra and (not args.allow_random_ckpt):
        if args.supra_ckpt is None or str(args.supra_ckpt).strip() == "":
            raise ValueError("supra spaces requested but --supra_ckpt not provided; set --allow_random_ckpt true to override")
        if not os.path.exists(str(args.supra_ckpt)):
            raise FileNotFoundError(f"supra_ckpt not found: {args.supra_ckpt}")

    if want_early and (not args.allow_random_ckpt):
        if args.early_ckpt is None or str(args.early_ckpt).strip() == "":
            raise ValueError("early spaces requested but --early_ckpt not provided; set --allow_random_ckpt true to override")
        if not os.path.exists(str(args.early_ckpt)):
            raise FileNotFoundError(f"early_ckpt not found: {args.early_ckpt}")

    if want_tri and (not args.allow_random_ckpt):
        if args.tri_ckpt is None or str(args.tri_ckpt).strip() == "":
            raise ValueError("tri spaces requested but --tri_ckpt not provided; set --allow_random_ckpt true to override")
        if not os.path.exists(str(args.tri_ckpt)):
            raise FileNotFoundError(f"tri_ckpt not found: {args.tri_ckpt}")

    if want_late:
        late_model = _build_late_gnn(args, ctx, device)
        if args.late_ckpt:
            ck = _load_ckpt(args.late_ckpt)
            late_model.load_state_dict(ck["state_dict"], strict=False)
        late_model.eval()
        models["late"] = late_model
        get_reprs_fns["late"] = _late_reprs

    if want_supra:
        supra_model = _build_supra(args, ctx, device)
        if args.supra_ckpt:
            ck = _load_ckpt(args.supra_ckpt)
            supra_model.load_state_dict(ck["state_dict"], strict=False)
        supra_model.eval()
        models["supra"] = supra_model
        get_reprs_fns["supra"] = _supra_reprs

    if want_early:
        early_model = _build_early_gnn(args, ctx, device)
        if args.early_ckpt:
            ck = _load_ckpt(args.early_ckpt)
            early_model.load_state_dict(ck["state_dict"], strict=False)
        early_model.eval()
        models["early"] = early_model
        get_reprs_fns["early"] = _early_reprs

    if want_tri:
        tri_model = _build_tri_gnn(args, ctx, device)
        if args.tri_ckpt:
            ck = _load_ckpt(args.tri_ckpt)
            tri_model.load_state_dict(ck["state_dict"], strict=False)
        tri_model.eval()
        models["tri"] = tri_model
        get_reprs_fns["tri"] = _tri_reprs

    # Modality IoU
    mod_rows: List[Dict[str, Any]] = []
    neigh_rows: List[Dict[str, Any]] = []
    deg_rows: List[Dict[str, Any]] = []
    neigh_deg_rows: List[Dict[str, Any]] = []
    neigh_energy_rows: List[Dict[str, Any]] = []

    top_ratio = float(args.top_ratio)
    strict_zero = bool(args.strict_zero)

    # Choose idx set for stats.
    idx = ctx.train_idx

    # Late modality IoU on its requested spaces.
    if "late" in models:
        tags = [s for s in spaces if s.startswith("late/")]
        mod_rows += _modality_iou(
            get_reprs=_late_reprs,
            model=models["late"],
            graph=ctx.graph,
            text_feat=ctx.text_feat,
            vis_feat=ctx.vis_feat,
            idx=idx,
            top_ratio=top_ratio,
            strict_zero=strict_zero,
            tags=tags,
        )
        o, drows, odeg, en_deg = _neighbor_overlap_and_degree(
            get_reprs=_late_reprs,
            model=models["late"],
            graph_full=ctx.graph,
            graph_self=graph_self,
            text_feat=ctx.text_feat,
            vis_feat=ctx.vis_feat,
            idx=idx,
            top_ratio=top_ratio,
            strict_zero=strict_zero,
            degree_bins=degree_bins,
            tags=tags,
        )
        neigh_rows += o
        deg_rows += drows
        neigh_deg_rows += odeg
        neigh_energy_rows += en_deg

    if "supra" in models:
        tags = [s for s in spaces if s.startswith("supra/")]
        mod_rows += _modality_iou(
            get_reprs=_supra_reprs,
            model=models["supra"],
            graph=ctx.graph,
            text_feat=ctx.text_feat,
            vis_feat=ctx.vis_feat,
            idx=idx,
            top_ratio=top_ratio,
            strict_zero=strict_zero,
            tags=tags,
        )
        o, drows, odeg, en_deg = _neighbor_overlap_and_degree(
            get_reprs=_supra_reprs,
            model=models["supra"],
            graph_full=ctx.graph,
            graph_self=graph_self,
            text_feat=ctx.text_feat,
            vis_feat=ctx.vis_feat,
            idx=idx,
            top_ratio=top_ratio,
            strict_zero=strict_zero,
            degree_bins=degree_bins,
            tags=tags,
        )
        neigh_rows += o
        deg_rows += drows
        neigh_deg_rows += odeg
        neigh_energy_rows += en_deg

    if "early" in models:
        tags = [s for s in spaces if s.startswith("early/")]
        mod_rows += _modality_iou(
            get_reprs=_early_reprs,
            model=models["early"],
            graph=ctx.graph,
            text_feat=ctx.text_feat,
            vis_feat=ctx.vis_feat,
            idx=idx,
            top_ratio=top_ratio,
            strict_zero=strict_zero,
            tags=tags,
        )
        o, drows, odeg, en_deg = _neighbor_overlap_and_degree(
            get_reprs=_early_reprs,
            model=models["early"],
            graph_full=ctx.graph,
            graph_self=graph_self,
            text_feat=ctx.text_feat,
            vis_feat=ctx.vis_feat,
            idx=idx,
            top_ratio=top_ratio,
            strict_zero=strict_zero,
            degree_bins=degree_bins,
            tags=tags,
        )
        neigh_rows += o
        deg_rows += drows
        neigh_deg_rows += odeg
        neigh_energy_rows += en_deg

    if "tri" in models:
        tags = [s for s in spaces if s.startswith("tri/")]
        mod_rows += _modality_iou(
            get_reprs=_tri_reprs,
            model=models["tri"],
            graph=ctx.graph,
            text_feat=ctx.text_feat,
            vis_feat=ctx.vis_feat,
            idx=idx,
            top_ratio=top_ratio,
            strict_zero=strict_zero,
            tags=tags,
        )
        o, drows, odeg, en_deg = _neighbor_overlap_and_degree(
            get_reprs=_tri_reprs,
            model=models["tri"],
            graph_full=ctx.graph,
            graph_self=graph_self,
            text_feat=ctx.text_feat,
            vis_feat=ctx.vis_feat,
            idx=idx,
            top_ratio=top_ratio,
            strict_zero=strict_zero,
            degree_bins=degree_bins,
            tags=tags,
        )
        neigh_rows += o
        deg_rows += drows
        neigh_deg_rows += odeg
        neigh_energy_rows += en_deg

    _write_csv(os.path.join(out_dir, "modality_iou.csv"), mod_rows)
    _write_csv(os.path.join(out_dir, "neighbor_overlap.csv"), neigh_rows)
    _write_csv(os.path.join(out_dir, "degree_retention.csv"), deg_rows)
    _write_csv(os.path.join(out_dir, "neighbor_overlap_by_degree.csv"), neigh_deg_rows)
    _write_csv(os.path.join(out_dir, "neighbor_energy_by_degree.csv"), neigh_energy_rows)

    # Gradient flow (optional, only for models we built)
    grad_rows: List[Dict[str, Any]] = []

    if "late" in models:
        for mode in ("both", "text", "visual"):
            stats = _grad_flow(
                mode=mode,
                model_name="late",
                model=models["late"],
                ctx=ctx,
                graph=ctx.observe_graph,
                strict_zero=strict_zero,
                label_smoothing=float(getattr(args, "label_smoothing", 0.0)),
            )
            grad_rows.append({"model": "late", "mode": mode, "strict_zero": int(strict_zero), **stats})

    if "supra" in models:
        for mode in ("both", "text", "visual"):
            stats = _grad_flow_supra(
                model=models["supra"],
                ctx=ctx,
                graph=ctx.observe_graph,
                strict_zero=strict_zero,
                mode=mode,
                label_smoothing=float(getattr(args, "label_smoothing", 0.0)),
            )
            grad_rows.append({"model": "supra", "mode": mode, "strict_zero": int(strict_zero), **stats})

    if "early" in models:
        for mode in ("both", "text", "visual"):
            stats = _grad_flow_early(
                mode=mode,
                model=models["early"],
                ctx=ctx,
                graph=ctx.observe_graph,
                strict_zero=strict_zero,
                label_smoothing=float(getattr(args, "label_smoothing", 0.0)),
            )
            grad_rows.append({"model": "early", "mode": mode, "strict_zero": int(strict_zero), **stats})

    if "tri" in models:
        for mode in ("both", "text", "visual"):
            # Tri: report per-branch grad norms.
            m = models["tri"]
            m.train()
            for p in m.parameters():
                if p.grad is not None:
                    p.grad = None

            if mode == "both":
                text_feat, vis_feat = ctx.text_feat, ctx.vis_feat
            elif mode == "text":
                text_feat, vis_feat = ctx.text_feat, _zero_like_feat(ctx.vis_feat)
            else:
                text_feat, vis_feat = _zero_like_feat(ctx.text_feat), ctx.vis_feat

            reps = _tri_reprs(m, ctx.observe_graph, text_feat, vis_feat, strict_zero=strict_zero)
            fused = reps["tri/fused"]
            logits = m.classifier(fused)
            loss = cross_entropy(logits[ctx.train_idx], ctx.labels[ctx.train_idx], label_smoothing=float(getattr(args, "label_smoothing", 0.0)))
            loss.backward()

            def _group_norm(mods: Sequence[nn.Module]) -> float:
                acc = 0.0
                for module in mods:
                    for p in module.parameters(recurse=True):
                        if p.grad is None:
                            continue
                        acc += float(p.grad.detach().norm().cpu().item())
                return acc

            g_text = _group_norm([m.text_encoder, m.text_gnn])
            g_vis = _group_norm([m.visual_encoder, m.visual_gnn])
            g_early = _group_norm([m.early_gnn])
            g_head = _group_norm([m.classifier])

            grad_rows.append({
                "model": "tri",
                "mode": mode,
                "strict_zero": int(strict_zero),
                "loss": float(loss.detach().cpu().item()),
                "grad/text": g_text,
                "grad/visual": g_vis,
                "grad/early": g_early,
                "grad/head": g_head,
                "grad/ratio_text_over_visual": float(g_text / (g_vis + 1e-12)),
                "grad/ratio_visual_over_text": float(g_vis / (g_text + 1e-12)),
            })

    _write_csv(os.path.join(out_dir, "grad_flow.csv"), grad_rows)

    # Prediction metrics (optional): show channels are predictive (esp. SUPRA C/Ut/Uv).
    if bool(getattr(args, "write_pred_eval", False)):
        pred_rows: List[Dict[str, Any]] = []
        metric = str(getattr(args, "metric", "accuracy"))
        average = getattr(args, "average", None)

        split = "test"
        split_idx = ctx.test_idx

        if "late" in models:
            m = models["late"]
            with th.no_grad():
                reps_full = _late_reprs(m, ctx.graph, ctx.text_feat, ctx.vis_feat, strict_zero=strict_zero)
                reps_self = _late_reprs(m, graph_self, ctx.text_feat, ctx.vis_feat, strict_zero=strict_zero)
                logits_full = m.classifier(reps_full["late/fused"])
                logits_self = m.classifier(reps_self["late/fused"])
            pred_rows += [
                {
                    "model": "late",
                    "space": "late/fused",
                    "graph": "full",
                    "split": split,
                    "metric": metric,
                    "value": _eval_logits(logits=logits_full, labels=ctx.labels, idx=split_idx, metric=metric, average=average),
                },
                {
                    "model": "late",
                    "space": "late/fused",
                    "graph": "self",
                    "split": split,
                    "metric": metric,
                    "value": _eval_logits(logits=logits_self, labels=ctx.labels, idx=split_idx, metric=metric, average=average),
                },
            ]

        if "early" in models:
            w = models["early"]
            base = w._poly_base  # type: ignore[attr-defined]
            head = w._poly_head  # type: ignore[attr-defined]
            with th.no_grad():
                h_full = base.forward_with_drop(ctx.graph, ctx.text_feat, ctx.vis_feat, drop_text=False, drop_visual=False)
                h_self = base.forward_with_drop(graph_self, ctx.text_feat, ctx.vis_feat, drop_text=False, drop_visual=False)
                logits_full = head(h_full)
                logits_self = head(h_self)
            pred_rows += [
                {
                    "model": "early",
                    "space": "early/h",
                    "graph": "full",
                    "split": split,
                    "metric": metric,
                    "value": _eval_logits(logits=logits_full, labels=ctx.labels, idx=split_idx, metric=metric, average=average),
                },
                {
                    "model": "early",
                    "space": "early/h",
                    "graph": "self",
                    "split": split,
                    "metric": metric,
                    "value": _eval_logits(logits=logits_self, labels=ctx.labels, idx=split_idx, metric=metric, average=average),
                },
            ]

        if "tri" in models:
            m = models["tri"]
            with th.no_grad():
                reps_full = _tri_reprs(m, ctx.graph, ctx.text_feat, ctx.vis_feat, strict_zero=strict_zero)
                reps_self = _tri_reprs(m, graph_self, ctx.text_feat, ctx.vis_feat, strict_zero=strict_zero)
                logits_full = m.classifier(reps_full["tri/fused"])
                logits_self = m.classifier(reps_self["tri/fused"])
            pred_rows += [
                {
                    "model": "tri",
                    "space": "tri/fused",
                    "graph": "full",
                    "split": split,
                    "metric": metric,
                    "value": _eval_logits(logits=logits_full, labels=ctx.labels, idx=split_idx, metric=metric, average=average),
                },
                {
                    "model": "tri",
                    "space": "tri/fused",
                    "graph": "self",
                    "split": split,
                    "metric": metric,
                    "value": _eval_logits(logits=logits_self, labels=ctx.labels, idx=split_idx, metric=metric, average=average),
                },
            ]

        if "supra" in models:
            m = models["supra"]
            with th.no_grad():
                out_full = m.forward_multiple(ctx.graph, ctx.text_feat, ctx.vis_feat, stochastic=False, profile_mem=False)
                out_self = m.forward_multiple(graph_self, ctx.text_feat, ctx.vis_feat, stochastic=False, profile_mem=False)

            for graph_name, out in (("full", out_full), ("self", out_self)):
                pred_rows.append({
                    "model": "supra",
                    "space": "supra/logits_final",
                    "graph": graph_name,
                    "split": split,
                    "metric": metric,
                    "value": _eval_logits(logits=out.logits_final_0, labels=ctx.labels, idx=split_idx, metric=metric, average=average),
                })
                pred_rows.append({
                    "model": "supra",
                    "space": "supra/logits_C",
                    "graph": graph_name,
                    "split": split,
                    "metric": metric,
                    "value": _eval_logits(logits=out.logits_C_0, labels=ctx.labels, idx=split_idx, metric=metric, average=average),
                })
                pred_rows.append({
                    "model": "supra",
                    "space": "supra/logits_Ut",
                    "graph": graph_name,
                    "split": split,
                    "metric": metric,
                    "value": _eval_logits(logits=out.logits_Ut_0, labels=ctx.labels, idx=split_idx, metric=metric, average=average),
                })
                pred_rows.append({
                    "model": "supra",
                    "space": "supra/logits_Uv",
                    "graph": graph_name,
                    "split": split,
                    "metric": metric,
                    "value": _eval_logits(logits=out.logits_Uv_0, labels=ctx.labels, idx=split_idx, metric=metric, average=average),
                })

        _write_csv(os.path.join(out_dir, "pred_eval.csv"), pred_rows)

    # Console summary
    def _best(rows, key):
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        vals = [float(v) for v in vals if np.isfinite(float(v))]
        return (min(vals), max(vals)) if vals else (float("nan"), float("nan"))

    mi_min, mi_max = _best(mod_rows, "iou_text_vs_visual")
    ni_min, ni_max = _best(neigh_rows, "iou_self_vs_neighbor")
    print("[Polysemanticity] wrote CSVs to:", out_dir)
    print(f"[Modality IoU] min/max: {mi_min:.4f}/{mi_max:.4f}")
    print(f"[Neighbor IoU]  min/max: {ni_min:.4f}/{ni_max:.4f}")


if __name__ == "__main__":
    main()
