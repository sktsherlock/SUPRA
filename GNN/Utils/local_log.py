import csv
import math
import os

import numpy as np


def format_float(x, ndigits: int = 5) -> str:
    return f"{float(x):.{ndigits}f}"


def format_runs(values, ndigits: int = 5) -> str:
    return ";".join(format_float(v, ndigits=ndigits) for v in values)


def build_metric_fields(prefix: str, values, ndigits: int = 5) -> dict:
    values = list(values)
    mean = float(np.mean(values))
    std = float(np.std(values))
    var = float(np.var(values))

    return {
        f"{prefix}_mean": format_float(mean, ndigits=ndigits),
        f"{prefix}_std": format_float(std, ndigits=ndigits),
        f"{prefix}_var": format_float(var, ndigits=ndigits),
        f"{prefix}_summary": f"{format_float(mean, ndigits=ndigits)} ± {format_float(std, ndigits=ndigits)}",
        f"{prefix}_summary_pct": f"{mean * 100.0:.2f} ± {std * 100.0:.2f}",
        f"{prefix}_runs": format_runs(values, ndigits=ndigits),
    }


def is_missing_number(value) -> bool:
    if value is None:
        return True
    s = str(value).strip()
    if s == "":
        return True
    try:
        x = float(s)
    except ValueError:
        return True
    return math.isnan(x)


def row_has_complete_results(row: dict, required_fields=("val_mean", "test_mean")) -> bool:
    return all(not is_missing_number(row.get(k)) for k in required_fields)


def make_target_from_args(args, key_fields) -> dict:
    return {k: str(getattr(args, k, "")) for k in key_fields}


def already_logged(path: str, key_fields, target: dict, required_fields=("val_mean", "test_mean")) -> bool:
    if not path or not os.path.exists(path):
        return False
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if all(row.get(k, "") == target.get(k, "") for k in key_fields) and row_has_complete_results(
                row, required_fields=required_fields
            ):
                return True
    return False


def upsert_row(path: str, key_fields, row: dict, required_fields=("val_mean", "test_mean")) -> None:
    if not path:
        return

    rows = []
    existing_fieldnames = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            rows = list(reader)

    desired_fieldnames = list(dict.fromkeys(existing_fieldnames + list(row.keys())))
    if not desired_fieldnames:
        desired_fieldnames = list(row.keys())

    for r in rows:
        for fn in desired_fieldnames:
            r.setdefault(fn, "")

    target = {k: str(row.get(k, "")) for k in key_fields}

    update_idx = None
    match_indices = [i for i, r in enumerate(rows) if all(r.get(k, "") == target.get(k, "") for k in key_fields)]
    for i in match_indices:
        if not row_has_complete_results(rows[i], required_fields=required_fields):
            update_idx = i
            break

    if update_idx is None and match_indices:
        update_idx = match_indices[0]

    if update_idx is None:
        rows.append(row)
    else:
        rows[update_idx].update(row)

    tmp_path = f"{path}.tmp"
    with open(tmp_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=desired_fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({fn: r.get(fn, "") for fn in desired_fieldnames})
    os.replace(tmp_path, path)

