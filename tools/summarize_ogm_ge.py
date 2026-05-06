#!/usr/bin/env python3
"""
Summarize OGM-GE Experiment Results
===================================

Reads Results/ogm_ge/*.csv files and prints comparison tables for:
  1. Main performance (Accuracy / F1-macro)
  2. Modality contamination resilience (degrade_text / degrade_visual)

Usage:
    python tools/summarize_ogm_ge.py
    python tools/summarize_ogm_ge.py --dir Results/ogm_ge
    python tools/summarize_ogm_ge.py --f1
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Filename patterns
# ---------------------------------------------------------------------------
# grocery_g1_baseline.csv         → group g1, Baseline(C)
# grocery_g2_aux.csv              → group g2, SUPRA(aux)
# grocery_g3_ogm.csv             → group g3, OGM-GE
# grocery_g4_ogm_aux.csv         → group g4, OGM+aux
# *_f1macro.csv                  → F1-macro metric
# *_all.csv                       → per-run raw results (for mean±std)

FILE_PATTERNS = {
    "grocery_g1_baseline":  ("g1", "Baseline(C)"),
    "grocery_g2_aux":       ("g2", "SUPRA(aux)"),
    "grocery_g3_ogm":       ("g3", "OGM-GE"),
    "grocery_g4_ogm_aux":   ("g4", "OGM+aux"),
}


def parse_filename(fname):
    """Parse OGM-GE result filename. Returns (group_key, group_label, metric) or None."""
    base = fname.replace("_all.csv", "").replace(".csv", "")

    is_f1 = "_f1macro" in base
    base = base.replace("_f1macro", "")

    for pattern, (gkey, glabel) in FILE_PATTERNS.items():
        if base == pattern:
            return (gkey, glabel, "F1-macro" if is_f1 else "Accuracy")
    return None


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------

def _parse_val(s):
    """Parse '75.67' or '75.67 ± 0.16' → float 75.67."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    if "±" in s or "+/-" in s:
        parts = re.split(r"[±+/-]+", s)
        try:
            return float(parts[0].strip())
        except:
            return None
    try:
        return float(s)
    except:
        return None


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def read_best_csv(path):
    """Read best result from aggregated *_best.csv or plain result CSV.

    Returns dict with keys: mean, std, degrade_text, degrade_visual, degrade_text_std, degrade_visual_std.
    Values are already in percentage form (e.g. 75.50, not 0.7550).
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None

        first_row = rows[0]
        # Determine metric column
        metric_col = None
        if "full" in first_row:
            metric_col = "full"
        elif "val_metric" in first_row:
            metric_col = "val_metric"

        if not metric_col:
            return None

        # Use best (max) row
        best_row = max(rows, key=lambda r: float(_parse_val(r.get(metric_col)) or 0))
        mean_val = _parse_val(best_row.get(metric_col))
        std_val = _parse_val(best_row.get("full_std")) or 0.0

        dt = _parse_val(best_row.get("degrade_text")) or None
        dv = _parse_val(best_row.get("degrade_visual")) or None
        dt_std = _parse_val(best_row.get("degrade_text_std")) or 0.0
        dv_std = _parse_val(best_row.get("degrade_visual_std")) or 0.0

        return {
            "mean": mean_val,
            "std": std_val,
            "degrade_text": dt,
            "degrade_visual": dv,
            "degrade_text_std": dt_std,
            "degrade_visual_std": dv_std,
        }
    except Exception as e:
        print(f"  [WARN] {path}: {e}", file=sys.stderr)
        return None


def read_all_runs(path):
    """Read all runs from *_all.csv and compute mean±std.

    Returns dict with: mean, std, degrade_text, degrade_visual (mean of all runs).
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None

        metric_col = None
        if "full" in rows[0]:
            metric_col = "full"

        if not metric_col:
            return None

        run_values = []
        dt_runs, dv_runs = [], []
        for row in rows:
            val = _parse_val(row.get(metric_col))
            if val is not None:
                run_values.append(val)
            dt = _parse_val(row.get("degrade_text"))
            dv = _parse_val(row.get("degrade_visual"))
            if dt is not None:
                dt_runs.append(dt)
            if dv is not None:
                dv_runs.append(dv)

        if not run_values:
            return None

        import statistics
        mean_val = statistics.mean(run_values)
        std_val = statistics.stdev(run_values) if len(run_values) > 1 else 0.0
        dt_mean = statistics.mean(dt_runs) if dt_runs else None
        dv_mean = statistics.mean(dv_runs) if dv_runs else None
        dt_std = statistics.stdev(dt_runs) if len(dt_runs) > 1 else 0.0
        dv_std = statistics.stdev(dv_runs) if len(dv_runs) > 1 else 0.0

        return {
            "mean": mean_val,
            "std": std_val,
            "degrade_text": dt_mean,
            "degrade_visual": dv_mean,
            "degrade_text_std": dt_std,
            "degrade_visual_std": dv_std,
        }
    except Exception as e:
        print(f"  [WARN] {path}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(ogm_dir, metric_type="Accuracy"):
    """Returns dict[group_key] -> result dict."""
    results = defaultdict(dict)

    if not os.path.isdir(ogm_dir):
        print(f"[ERROR] {ogm_dir} not found", file=sys.stderr)
        return results

    for fname in sorted(os.listdir(ogm_dir)):
        if not fname.endswith(".csv"):
            continue

        parsed = parse_filename(fname)
        if parsed is None:
            continue

        gkey, glabel, fname_metric = parsed
        if fname_metric != metric_type:
            continue

        # Determine if this is a per-run file
        is_all = "_all" in fname
        if is_all:
            csv_path = os.path.join(ogm_dir, fname)
            data = read_all_runs(csv_path)
        else:
            # Try to find corresponding _all file for aggregation
            base = fname.replace(".csv", "")
            all_path = os.path.join(ogm_dir, base + "_all.csv")
            if os.path.exists(all_path):
                data = read_all_runs(all_path)
            else:
                data = read_best_csv(os.path.join(ogm_dir, fname))

        if data is not None:
            results[gkey] = data

    return results


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _col(entry):
    """Format mean±std for display. Values are already percentages."""
    if entry is None:
        return "    —    "
    v = entry.get("mean", 0)
    s = entry.get("std", 0)
    return f"{v:.2f}±{s:.2f}"


def _d(entry):
    """Format single value."""
    if entry is None:
        return "    —    "
    return f"{entry:.2f}"


def _delta(a_entry, b_entry):
    """Delta of b - a, formatted with sign."""
    if a_entry is None or b_entry is None:
        return "    —    "
    av = a_entry.get("mean", 0)
    bv = b_entry.get("mean", 0)
    d = bv - av
    return f"{'+' if d >= 0 else ''}{d:.2f}"


def _drop(entry, deg_entry):
    """Performance drop when modality is degraded. Formatted as negative (worse is more negative)."""
    if entry is None or deg_entry is None:
        return "    —    "
    full = entry.get("mean", 0)
    deg = deg_entry.get("mean", 0) if isinstance(deg_entry, dict) else deg_entry
    d = deg - full
    return f"{d:+.2f}"


# ---------------------------------------------------------------------------
# Print tables
# ---------------------------------------------------------------------------

def print_results_table(results, metric):
    groups = [
        ("g1", "Baseline(C)"),
        ("g2", "SUPRA(aux)"),
        ("g3", "OGM-GE"),
        ("g4", "OGM+aux"),
    ]
    sep = "=" * 90

    print(f"\n{sep}")
    print(f"  OGM-GE Comparison — {metric}  |  Dataset: Grocery")
    print(sep)

    # Header
    hdr = f"  {'Config':<16}" + f" {'Full':>12}" + f" {'Txt Degraded':>14}" + f" {'Vis Degraded':>14}" + f" {'Drop(Text)':>12}" + f" {'Drop(Vis)':>12}"
    print(hdr)
    print(f"  {'─'*16}" + "─"*13*4 + "─"*13*2)

    for gkey, glabel in groups:
        entry = results.get(gkey)
        dt = {"mean": entry.get("degrade_text"), "std": entry.get("degrade_text_std")} if entry and entry.get("degrade_text") is not None else None
        dv = {"mean": entry.get("degrade_visual"), "std": entry.get("degrade_visual_std")} if entry and entry.get("degrade_visual") is not None else None

        row = f"  {glabel:<16}"
        row += f" {_col(entry):>12}"
        row += f" {_col(dt):>14}"
        row += f" {_col(dv):>14}"
        row += f" {_drop(entry, dt):>12}"
        row += f" {_drop(entry, dv):>12}"
        print(row)

    # Delta vs Baseline(C)
    print(f"\n  ## Delta vs Baseline(C) — Full Accuracy")
    print(f"  {'─'*16}" + "─"*13*4)
    baseline = results.get("g1")
    for gkey, glabel in groups[1:]:
        entry = results.get(gkey)
        row = f"  {glabel:<16}"
        row += f" {_delta(baseline, entry):>12}"
        dt_b = {"mean": baseline.get("degrade_text")} if baseline and baseline.get("degrade_text") else None
        dt_e = {"mean": entry.get("degrade_text")} if entry and entry.get("degrade_text") else None
        dv_b = {"mean": baseline.get("degrade_visual")} if baseline and baseline.get("degrade_visual") else None
        dv_e = {"mean": entry.get("degrade_visual")} if entry and entry.get("degrade_visual") else None
        row += f" {_delta(dt_b, dt_e):>14}"
        row += f" {_delta(dv_b, dv_e):>14}"
        row += f" {'    —    ':>12}"  # no delta for drop columns
        row += f" {'    —    ':>12}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="Results/ogm_ge")
    parser.add_argument("--f1", action="store_true", help="Show F1-macro instead of Accuracy")
    args = parser.parse_args()

    ogm_dir = args.dir
    metric = "F1-macro" if args.f1 else "Accuracy"

    results = load_results(ogm_dir, metric)
    if not results:
        print(f"[ERROR] No results found in {ogm_dir}", file=sys.stderr)
        print(f"  Expected files: grocery_g1_baseline.csv, grocery_g2_aux.csv, grocery_g3_ogm.csv, grocery_g4_ogm_aux.csv", file=sys.stderr)
        print(f"  (and their _all.csv variants for per-run aggregation)", file=sys.stderr)
        sys.exit(1)

    print_results_table(results, metric)
    print(f"\n  Source: {os.path.abspath(ogm_dir)}/")
    print("=" * 90)


if __name__ == "__main__":
    main()
