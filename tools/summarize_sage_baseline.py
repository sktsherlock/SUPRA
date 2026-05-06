#!/usr/bin/env python3
"""
Summarize SAGE Baseline Results
===============================

Reads Results/0506/*_best_all.csv files and prints a comparison table
of Early_GNN-SAGE vs Late_GNN-SAGE across all datasets and feature groups.

Usage:
    python tools/summarize_sage_baseline.py
    python tools/summarize_sage_baseline.py --dir Results/0506
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict


def parse_best_filename(fname):
    """Parse filename like 'early_sage_Movies_default_L2_lr0.0005_best_all.csv'."""
    base = fname.replace("_best_all.csv", "").replace("_all.csv", "")
    base = base.replace("_f1macro", "")  # strip metric suffix too

    if base.startswith("early_sage_"):
        model = "Early_GNN"
        rest = base[len("early_sage_"):]
    elif base.startswith("late_sage_"):
        model = "Late_GNN"
        rest = base[len("late_sage_"):]
    else:
        return None

    # Split off hyperparam suffix: *_L#_lr#.#  (e.g. _L2_lr0.0005)
    # Use rsplit('_L', 1) to split at the _ before layer number
    if "_L" not in rest:
        return None
    prefix, hyp = rest.rsplit("_L", 1)
    hyp = "L" + hyp  # restore: "L2_lr0.0005"

    # prefix = "Grocery_default" or "Reddit-M_clip_roberta"
    if prefix.endswith("_clip_roberta"):
        dataset_raw = prefix[:-len("_clip_roberta")]
        fg = "clip_roberta"
    elif prefix.endswith("_default"):
        dataset_raw = prefix[:-len("_default")]
        fg = "default"
    else:
        return None

    # Normalize dataset name
    DS_MAP = {"Reddit-M": "RedditM", "Movies": "Movies", "Grocery": "Grocery", "Toys": "Toys"}
    dataset = DS_MAP.get(dataset_raw, dataset_raw.replace("-", "").replace("_", ""))

    return {"model": model, "dataset": dataset, "fg": fg}


def read_all_runs(path):
    """Read all runs from *_all.csv and return (mean, std)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None

        # Find 'full' column
        first_row = rows[0]
        metric_col = None
        for k in first_row:
            if k == "full":
                metric_col = k
                break

        if not metric_col:
            return None

        def parse_val(s):
            if s is None:
                return None
            s = str(s).strip()
            if not s:
                return None
            if "±" in s or "+/-" in s:
                parts = re.split(r"[±+/-]+", s)
                return float(parts[0].strip())
            try:
                return float(s)
            except:
                return None

        run_values = []
        for row in rows:
            val = parse_val(row.get(metric_col))
            if val is not None:
                run_values.append(val)

        if not run_values:
            return None

        mean_val = sum(run_values) / len(run_values)
        if len(run_values) > 1:
            std_val = (sum((v - mean_val) ** 2 for v in run_values) / (len(run_values) - 1)) ** 0.5
        else:
            std_val = 0.0
        return (mean_val, std_val)
    except Exception as e:
        print(f"  [WARN] {path}: {e}", file=sys.stderr)
        return None


def load_results(results_dir):
    """Returns results[model][dataset][fg][metric] = (mean, std).
    Reads all *_all.csv files and selects the best across hyperparameter configs."""
    # by_key[(model, dataset, fg, metric)] = list of (mean, std, fname)
    by_key = defaultdict(list)

    if not os.path.isdir(results_dir):
        print(f"[ERROR] {results_dir} not found", file=sys.stderr)
        return defaultdict(lambda: defaultdict(dict))

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith("_all.csv"):
            continue

        p = parse_best_filename(fname)
        if p is None:
            continue

        is_f1 = "_f1macro" in fname
        metric = "F1-macro" if is_f1 else "Accuracy"
        key = (p["model"], p["dataset"], p["fg"], metric)

        all_path = os.path.join(results_dir, fname)
        data = read_all_runs(all_path)
        if data is not None:
            by_key[key].append((data[0], data[1], fname))

    # Select best for each (model, dataset, fg, metric)
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for key, entries in by_key.items():
        model, dataset, fg, metric = key
        best = max(entries, key=lambda e: e[0])
        results[model][dataset][fg][metric] = (best[0], best[1])

    return results


def _col(entry):
    # Values are stored as percentage (e.g. 75.67), no division needed
    return f"{entry[0]:.2f}±{entry[1]:.2f}" if entry else "   —   "


def _delta(a, b):
    if a and b:
        d = b[0] - a[0]
        return f"{'+' if d >= 0 else ''}{d:.2f}"
    return "   —   "


def print_results(results, metric):
    datasets = ["Movies", "Grocery", "Toys", "RedditM"]
    fgs = ["default", "clip_roberta"]

    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  SAGE Baseline — {metric} (Early_GNN vs Late_GNN)")
    print(sep)

    for fg in fgs:
        fg_label = "Llama" if fg == "default" else "RoBERTa+CLIP"
        print(f"\n  ## Feature Group: {fg_label} ({fg})")
        hdr = f"  {'Model':<12}" + "".join(f" {ds:>14}" for ds in datasets)
        print(hdr)
        print(f"  {'─'*12}" + "─"*15*4)
        for model in ["Early_GNN", "Late_GNN"]:
            row = f"  {model:<12}"
            for ds in datasets:
                entry = results[model][ds][fg].get(metric)
                row += f" {_col(entry):>14}"
            print(row)

    # Delta row
    print(f"\n  ## Delta: Late_GNN - Early_GNN")
    print(f"  {'─'*12}" + "─"*15*4)
    for fg in fgs:
        fg_label = "Llama" if fg == "default" else "RoBERTa+CLIP"
        row = f"  {fg_label:<12}"
        for ds in datasets:
            early = results["Early_GNN"][ds][fg].get(metric)
            late = results["Late_GNN"][ds][fg].get(metric)
            row += f" {_delta(early, late):>14}"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="Results/0506")
    args = parser.parse_args()

    results_dir = args.dir

    for metric in ["Accuracy", "F1-macro"]:
        print_results(load_results(results_dir), metric)

    print(f"\n  Source: {os.path.abspath(results_dir)}/")
    print("=" * 90)


if __name__ == "__main__":
    main()
