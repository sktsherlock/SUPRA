"""Convert polysemanticity analysis CSVs to rebuttal-friendly Markdown tables.

Typical workflow (single setting):
  python analyze_polysemanticity_mag.py ... --out_dir <analysis_dir> --write_pred_eval \
    --degree_bins "0,2,8,32,1000000000"
  python tools/polysemanticity_to_markdown.py --analysis_dir <analysis_dir>

This script is intentionally lightweight (no pandas) and prints Markdown to stdout.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_int(x: Any) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _fmt(x: Any, digits: int = 3) -> str:
    v = _to_float(x)
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def _ensure_list(raw: str) -> List[str]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    return parts


def _bin_label(lo: int, hi: int) -> str:
    if hi >= 1_000_000_000:
        return f"{lo}+"
    return f"{lo}-{hi}"


def _unique_bins(rows: Sequence[Dict[str, str]]) -> List[Tuple[int, int]]:
    bins = set()
    for r in rows:
        if "deg_lo" not in r or "deg_hi" not in r:
            continue
        bins.add((_to_int(r.get("deg_lo")), _to_int(r.get("deg_hi"))))
    return sorted(bins, key=lambda t: (t[0], t[1]))


def _filter_spaces(rows: Sequence[Dict[str, str]], spaces: Sequence[str]) -> List[Dict[str, str]]:
    if not spaces:
        return list(rows)
    sset = set(spaces)
    return [r for r in rows if str(r.get("space", "")) in sset]


def _pivot_degree_table(
    *,
    rows: Sequence[Dict[str, str]],
    spaces: Sequence[str],
    value_key: str,
    metric_key: Optional[str] = None,
    metric_value: Optional[str] = None,
    digits: int = 3,
) -> str:
    if metric_key is not None and metric_value is not None:
        rows = [r for r in rows if str(r.get(metric_key, "")) == str(metric_value)]

    rows = _filter_spaces(rows, spaces)
    if not rows:
        return "(no data)"

    bins = _unique_bins(rows)
    bin_labels = [_bin_label(lo, hi) for lo, hi in bins]

    # space -> (deg_lo,deg_hi) -> value
    table: Dict[str, Dict[Tuple[int, int], str]] = {}
    for r in rows:
        space = str(r.get("space", ""))
        lo = _to_int(r.get("deg_lo"))
        hi = _to_int(r.get("deg_hi"))
        table.setdefault(space, {})[(lo, hi)] = _fmt(r.get(value_key), digits=digits)

    # Order rows by provided spaces if any.
    ordered_spaces = list(spaces) if spaces else sorted(table.keys())

    # Build Markdown.
    header = "| space | " + " | ".join(bin_labels) + " |\n"
    sep = "|---|" + "|".join(["---:"] * len(bin_labels)) + "|\n"
    body_lines: List[str] = []
    for s in ordered_spaces:
        if s not in table:
            continue
        vals = [table[s].get(b, "nan") for b in bins]
        body_lines.append("| " + s + " | " + " | ".join(vals) + " |")
    return header + sep + "\n".join(body_lines)


def _neighbor_global_table(rows: Sequence[Dict[str, str]], spaces: Sequence[str], digits: int = 3) -> str:
    rows = _filter_spaces(rows, spaces)
    if not rows:
        return "(no data)"

    # Keep the first row per space (analysis typically has a single strict_zero/top_ratio).
    first: Dict[str, Dict[str, str]] = {}
    for r in rows:
        s = str(r.get("space", ""))
        if s and s not in first:
            first[s] = r

    ordered_spaces = list(spaces) if spaces else sorted(first.keys())

    header = "| space | IoU(self,neighbor) | Pearson(logdeg, retention) |\n"
    sep = "|---|---:|---:|\n"
    body_lines: List[str] = []
    for s in ordered_spaces:
        r = first.get(s)
        if not r:
            continue
        body_lines.append(
            "| "
            + s
            + " | "
            + _fmt(r.get("iou_self_vs_neighbor"), digits=digits)
            + " | "
            + _fmt(r.get("pearson_logdeg_vs_retention"), digits=digits)
            + " |"
        )
    return header + sep + "\n".join(body_lines)


def _pred_eval_table(rows: Sequence[Dict[str, str]], digits: int = 3) -> str:
    if not rows:
        return "(no data; run analyze_polysemanticity_mag.py with --write_pred_eval)"

    # Order: model, then space, then graph.
    rows2 = sorted(
        rows,
        key=lambda r: (
            str(r.get("model", "")),
            str(r.get("space", "")),
            str(r.get("graph", "")),
        ),
    )

    header = "| model | space | graph | split | metric | value |\n"
    sep = "|---|---|---|---|---|---:|\n"
    body_lines = [
        "| {model} | {space} | {graph} | {split} | {metric} | {value} |".format(
            model=str(r.get("model", "")),
            space=str(r.get("space", "")),
            graph=str(r.get("graph", "")),
            split=str(r.get("split", "")),
            metric=str(r.get("metric", "")),
            value=_fmt(r.get("value"), digits=digits),
        )
        for r in rows2
    ]
    return header + sep + "\n".join(body_lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        "polysemanticity_to_markdown",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--analysis_dir", type=str, required=True, help="Folder containing analysis CSVs")
    p.add_argument(
        "--spaces",
        type=str,
        default="late/fused,early/h,tri/fused,supra/C,supra/Ut,supra/Uv",
        help="Comma-separated spaces to display (order preserved)",
    )
    p.add_argument("--digits", type=int, default=3, help="Float digits in markdown")
    p.add_argument("--out_md", type=str, default=None, help="If set, write markdown to this path instead of stdout")
    return p


def main() -> None:
    args = build_parser().parse_args()

    analysis_dir = str(args.analysis_dir)
    spaces = _ensure_list(args.spaces)
    digits = int(args.digits)

    neighbor_overlap = _read_csv(os.path.join(analysis_dir, "neighbor_overlap.csv"))
    degree_retention = _read_csv(os.path.join(analysis_dir, "degree_retention.csv"))
    overlap_by_degree = _read_csv(os.path.join(analysis_dir, "neighbor_overlap_by_degree.csv"))
    energy_by_degree = _read_csv(os.path.join(analysis_dir, "neighbor_energy_by_degree.csv"))
    pred_eval = _read_csv(os.path.join(analysis_dir, "pred_eval.csv"))

    md_parts: List[str] = []

    md_parts.append("### Global Neighbor Interference (self vs neighbor)\n")
    md_parts.append(_neighbor_global_table(neighbor_overlap, spaces, digits=digits))

    md_parts.append("\n\n### Degree-binned Retention cos(full,self)\n")
    md_parts.append(
        _pivot_degree_table(
            rows=degree_retention,
            spaces=spaces,
            value_key="mean",
            metric_key="metric",
            metric_value="cos(full,self)",
            digits=digits,
        )
    )

    md_parts.append("\n\n### Degree-binned Neighbor Overlap IoU(self,neighbor)\n")
    md_parts.append(
        _pivot_degree_table(
            rows=overlap_by_degree,
            spaces=spaces,
            value_key="iou_self_vs_neighbor",
            metric_key=None,
            metric_value=None,
            digits=digits,
        )
    )

    md_parts.append("\n\n### Degree-binned Neighbor Energy Ratio ||neighbor||/||self||\n")
    md_parts.append(
        _pivot_degree_table(
            rows=energy_by_degree,
            spaces=spaces,
            value_key="mean",
            metric_key="metric",
            metric_value="||neighbor||/||self||",
            digits=digits,
        )
    )

    md_parts.append("\n\n### Prediction Metrics (channels are predictive)\n")
    md_parts.append(_pred_eval_table(pred_eval, digits=digits))

    out = "\n".join(md_parts).strip() + "\n"

    if args.out_md:
        out_path = str(args.out_md)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
