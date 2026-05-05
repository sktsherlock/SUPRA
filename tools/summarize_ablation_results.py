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

    gnn = rest.split("_")[0].upper()

    aux_match = re.search(r"_aux([\d.]+)", rest)
    if aux_match:
        aux = aux_match.group(1)
        dataset = rest[len(gnn.lower()) + 1: aux_match.start()]
    else:
        aux = None
        dataset = "_".join(rest.split("_")[1:])

    return {"arch": arch, "gnn": gnn, "dataset": dataset, "aux": aux, "is_f1": is_f1}


def read_best_result(path):
    """Return (mean, std) from the row with highest val_metric."""
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        metric_col = None
        std_col = None
        for row in rows:
            for k in row:
                kl = k.lower()
                if "val_metric" in kl and "std" not in kl and metric_col is None:
                    metric_col = k
                if "val_metric" in kl and "std" in kl and std_col is None:
                    std_col = k
            if metric_col:
                break
        if not metric_col:
            for row in rows:
                for k in ["mean_acc", "val_metric", "mean_val_metric"]:
                    if k in row:
                        metric_col = k
                        break
                if metric_col:
                    break
        best = max(rows, key=lambda r: float(r.get(metric_col, 0) or 0))
        mean_val = float(best.get(metric_col, 0) or 0)
        std_val = float(best.get(std_col, 0) or 0) if std_col else 0.0
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
