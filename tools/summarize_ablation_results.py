#!/usr/bin/env python3
"""
Consolidate GNN Backbone Ablation Results
=========================================

Reads all CSV files from Results/ablation/ and prints comparison tables.

Usage:
    python tools/summarize_ablation_results.py
    python tools/summarize_ablation_results.py --f1
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict


def parse_filename(fname):
    """Parse filename like 'early_gnn_gat_reddit-m.csv' or 'supra_gat_reddit-m_aux0_f1macro.csv'."""
    base = fname.replace(".csv", "")

    is_f1 = "_f1macro" in base
    base = base.replace("_f1macro", "")

    if base.startswith("early_gnn_"):
        arch = "early_gnn"
        rest = base.replace("early_gnn_", "")
    elif base.startswith("supra_"):
        arch = "supra"
        rest = base.replace("supra_", "")
    else:
        return None

    gnn_raw = rest.split("_")[0]
    # Normalize gnn name to match table headers: jknet->JKNet, gat->GAT, sage->SAGE
    gnn_map = {"jknet": "JKNet", "gat": "GAT", "sage": "SAGE", "gcn": "GCN"}
    gnn = gnn_map.get(gnn_raw.lower(), gnn_raw.upper())

    # Match aux with various padding: aux0.0, aux00, aux0, aux05, aux5
    aux_match = re.search(r"_aux([\d.]+)", rest)
    if aux_match:
        aux = aux_match.group(1)
        # Normalize: "0.0" → "0.0", "00" → "0.0", "05" → "0.5", "5" → "0.5"
        if "." not in aux:
            if len(aux) == 2:
                aux = aux[0] + "." + aux[1:]  # "00"→"0.0", "05"→"0.5"
            elif len(aux) == 1:
                aux = "0." + aux   # "0"→"0.0", "5"→"0.5"
        dataset = rest[len(gnn.lower()) + 1: aux_match.start()]
    else:
        aux = None
        dataset = "_".join(rest.split("_")[1:])

    return {"arch": arch, "gnn": gnn, "dataset": dataset, "aux": aux, "is_f1": is_f1}


def read_best_result(path):
    """Return (mean, std) from the row with highest accuracy/f1."""
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None

        # Detect column names: try full/full_std first (our format),
        # then fall back to val_metric column names
        metric_col = None
        std_col = None
        first_row = rows[0]

        # Our format: full / full_std
        if "full" in first_row:
            metric_col = "full"
            std_col = "full_std" if "full_std" in first_row else None
        # Fallback: try val_metric variants
        if not metric_col:
            for k in first_row:
                kl = k.lower()
                if "val_metric" in kl and "std" not in kl and metric_col is None:
                    metric_col = k
                if "val_metric" in kl and "std" in kl and std_col is None:
                    std_col = k
        # Last resort: mean_acc
        if not metric_col:
            for k in ["mean_acc", "val_metric"]:
                if k in first_row:
                    metric_col = k
                    break

        if not metric_col:
            return None

        best = max(rows, key=lambda r: float(r.get(metric_col, 0) or 0))
        # Parse "75.67 ± 0.16" style strings
        def parse_val(s):
            if s is None:
                return 0.0, 0.0
            s = str(s).strip()
            if "±" in s or "+/-" in s:
                parts = re.split(r"[±+/-]+", s)
                mean = float(parts[0].strip())
                std = float(parts[1].strip()) if len(parts) > 1 else 0.0
                return mean, std
            return float(s), 0.0

        mean_val, std_val = parse_val(best.get(metric_col))
        if std_col:
            _, std_val = parse_val(best.get(std_col))

        return (mean_val, std_val)
    except Exception as e:
        print(f"  [WARN] {path}: {e}", file=sys.stderr)
        return None


def load_all_results(ablation_dir, metric_type="accuracy"):
    """Returns results[gnn][dataset][cfg_key] = (mean, std)."""
    is_f1 = metric_type == "f1"
    results = defaultdict(lambda: defaultdict(dict))

    if not os.path.isdir(ablation_dir):
        print(f"[ERROR] {ablation_dir} not found", file=sys.stderr)
        return results

    for fname in sorted(os.listdir(ablation_dir)):
        if not fname.endswith(".csv"):
            continue
        p = parse_filename(fname)
        if p is None or p["is_f1"] != is_f1:
            continue
        data = read_best_result(os.path.join(ablation_dir, fname))
        if data is None:
            continue
        key = p["arch"]
        if p["aux"] is not None:
            key = f"supra_aux{p['aux']}"
        results[p["gnn"]][p["dataset"]][key] = data

    return results


def _col(entry):
    return f"{entry[0]:.4f}±{entry[1]:.3f}" if entry else "      —      "


def _delta(entry_a, entry_b):
    if entry_a and entry_b:
        d = entry_b[0] - entry_a[0]
        return f"{'+' if d >= 0 else ''}{d:.4f}"
    return "      —      "


def print_results(results, metric_type="accuracy", ablation_dir="Results/ablation"):
    datasets = ["Reddit-M", "Movies", "Grocery", "Toys"]
    gnns = ["GAT", "SAGE", "JKNet"]
    configs = [
        ("Early_GNN",       "early_gnn"),
        ("SUPRA(aux=0)",    "supra_aux0.0"),
        ("SUPRA(aux=0.5)",  "supra_aux0.5"),
    ]

    label = "F1-macro" if metric_type == "f1" else "Accuracy"
    sep = "=" * 96
    print(f"\n{sep}")
    print(f"  GNN Backbone Ablation — {label}")
    print(sep)

    for gnn in gnns:
        print(f"\n  ## {gnn}")
        hdr = f"  {'Model':<18}" + "".join(f" {ds:>16}" for ds in datasets)
        print(hdr)
        print(f"  {'─'*18}" + "─"*17*4)
        for label_text, cfg_key in configs:
            row = f"  {label_text:<18}"
            for ds in datasets:
                row += f" {_col(results[gnn][ds].get(cfg_key)):>16}"
            print(row)

    print(f"\n  ## Delta: SUPRA(aux=0) − Early_GNN")
    print(f"  {'─'*18}" + "─"*17*4)
    for gnn in gnns:
        row = f"  {gnn:<18}"
        for ds in datasets:
            early = results[gnn][ds].get("early_gnn")
            supra0 = results[gnn][ds].get("supra_aux0.0")
            row += f" {_delta(early, supra0):>16}"
        print(row)

    print(f"\n  ## Delta: SUPRA(aux=0.5) − Early_GNN")
    print(f"  {'─'*18}" + "─"*17*4)
    for gnn in gnns:
        row = f"  {gnn:<18}"
        for ds in datasets:
            early = results[gnn][ds].get("early_gnn")
            supra5 = results[gnn][ds].get("supra_aux0.5")
            row += f" {_delta(early, supra5):>16}"
        print(row)

    print(f"\n  Results: {os.path.abspath(os.path.dirname(ablation_dir.rstrip('/')))}/ablation/")
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1", action="store_true", help="Show F1-macro (accuracy shown by default)")
    parser.add_argument("--dir", default="Results/ablation")
    args = parser.parse_args()

    ablation_dir = args.dir

    acc = load_all_results(ablation_dir, "accuracy")
    print_results(acc, "accuracy", ablation_dir)

    if args.f1:
        f1 = load_all_results(ablation_dir, "f1")
        print_results(f1, "f1", ablation_dir)


if __name__ == "__main__":
    main()
