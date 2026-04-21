import math
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch as th
except Exception:  # pragma: no cover
    th = None  # type: ignore


def _to_1d_label_tensor(labels):
    """Return labels as a 1D torch tensor on CPU when possible."""
    if th is None:
        raise RuntimeError("PyTorch is required to compute graph statistics.")

    if isinstance(labels, np.ndarray):
        t = th.from_numpy(labels)
    elif isinstance(labels, th.Tensor):
        t = labels
    else:
        # common case: list
        t = th.tensor(labels)

    if t.dim() > 1:
        # e.g. shape (N,1)
        t = t.view(-1)

    return t.detach().to("cpu")


def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def compute_basic_counts(graph, labels) -> Dict[str, Any]:
    """Compute num_nodes / num_edges / num_classes / class_counts.

    - `num_edges` is the directed edge count as stored in the DGLGraph.
    - `num_classes` is computed from observed labels (excluding negative labels).
    """
    y = _to_1d_label_tensor(labels)

    num_nodes = _safe_int(graph.num_nodes())
    num_edges = _safe_int(graph.num_edges())

    y_np = y.numpy()
    valid = y_np[y_np >= 0]
    if valid.size == 0:
        num_classes = 0
        class_counts = {}
    else:
        class_counts_counter = Counter(valid.tolist())
        # normalize keys to int for nicer printing
        class_counts = {int(k): int(v) for k, v in class_counts_counter.items()}
        num_classes = int(max(class_counts.keys())) + 1

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_classes": num_classes,
        "class_counts": class_counts,
        "num_labeled_nodes": int(valid.size),
    }


def compute_edge_homogeneity(
    graph,
    labels,
    *,
    ignore_negative_labels: bool = True,
    chunk_size: int = 2_000_000,
) -> Optional[float]:
    """Compute edge homogeneity (aka edge homophily ratio).

    Definition (default): fraction of edges (u,v) with label[u] == label[v].

    If `ignore_negative_labels=True`, edges where either endpoint has label < 0
    are excluded from the denominator.

    Returns None if there are no valid edges to evaluate.
    """
    if th is None:
        raise RuntimeError("PyTorch is required to compute graph statistics.")

    y = _to_1d_label_tensor(labels)
    src, dst = graph.edges()
    src = src.to("cpu")
    dst = dst.to("cpu")

    total = src.numel()
    if total == 0:
        return None

    same_count = 0
    valid_count = 0

    # chunk to avoid peak memory on very large graphs
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        s = src[start:end]
        d = dst[start:end]
        ys = y[s]
        yd = y[d]

        if ignore_negative_labels:
            mask = (ys >= 0) & (yd >= 0)
            if mask.any():
                same_count += int((ys[mask] == yd[mask]).sum().item())
                valid_count += int(mask.sum().item())
        else:
            same_count += int((ys == yd).sum().item())
            valid_count += int((end - start))

    if valid_count == 0:
        return None

    return float(same_count) / float(valid_count)


def compute_graph_statistics(graph, labels) -> Dict[str, Any]:
    """One-shot summary: nodes/edges/classes/class distribution/homogeneity."""
    out: Dict[str, Any] = {}
    out.update(compute_basic_counts(graph, labels))

    hom = compute_edge_homogeneity(graph, labels, ignore_negative_labels=True)
    out["edge_homogeneity"] = hom

    # also provide log-friendly strings
    if hom is None or (isinstance(hom, float) and (math.isnan(hom) or math.isinf(hom))):
        out["edge_homogeneity_str"] = "NA"
    else:
        out["edge_homogeneity_str"] = f"{hom:.6f}"

    return out


def format_graph_statistics(stats: Dict[str, Any], *, max_classes_to_show: int = 20) -> str:
    num_nodes = stats.get("num_nodes")
    num_edges = stats.get("num_edges")
    num_classes = stats.get("num_classes")
    num_labeled_nodes = stats.get("num_labeled_nodes")
    edge_hom = stats.get("edge_homogeneity_str", "NA")

    class_counts = stats.get("class_counts") or {}
    # stable display: sort by class id
    items = sorted(class_counts.items(), key=lambda kv: kv[0])
    shown = items[:max_classes_to_show]
    more = len(items) - len(shown)

    class_counts_str = ", ".join(f"{k}:{v}" for k, v in shown)
    if more > 0:
        class_counts_str = f"{class_counts_str} ... (+{more} classes)"

    lines = [
        f"num_nodes: {num_nodes}",
        f"num_edges: {num_edges}",
        f"num_classes: {num_classes}",
        f"num_labeled_nodes: {num_labeled_nodes}",
        f"edge_homogeneity: {edge_hom}",
    ]

    if class_counts_str:
        lines.append(f"class_counts: {class_counts_str}")

    return "\n".join(lines)
