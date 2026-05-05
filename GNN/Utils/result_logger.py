import csv
import os
from typing import Dict, Iterable, Optional


def _as_str(x):
    if x is None:
        return ""
    return str(x)


def _as_float(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def _fmt_percent(x) -> Optional[float]:
    if x is None:
        return None
    return round(float(x) * 100.0, 3)


def _fmt_pm(mean: Optional[float], std: Optional[float], *, decimals: int = 2) -> str:
    if mean is None or std is None:
        return ""
    return f"{mean * 100.0:.{decimals}f} ± {std * 100.0:.{decimals}f}"




def build_result_row(
    *,
    args,
    method: str,
    full_metric: float,
    degrade_text: Optional[float] = None,
    degrade_visual: Optional[float] = None,
    extra: Optional[Dict[str, object]] = None,
    run: Optional[int] = None,
) -> Dict[str, object]:
    full_std = None
    degrade_text_std = None
    degrade_visual_std = None
    overall_hmean = None
    overall_hmean_std = None
    if extra:
        full_std = _as_float(extra.get("full_std"))
        degrade_text_std = _as_float(extra.get("degrade_text_std"))
        degrade_visual_std = _as_float(extra.get("degrade_visual_std"))
        overall_hmean = _as_float(extra.get("overall_hmean"))
        overall_hmean_std = _as_float(extra.get("overall_hmean_std"))

    row = {
        "dataset": getattr(args, "data_name", ""),
        "method": method,
        "backbone": getattr(args, "model_name", ""),
        "n_layers": getattr(args, "n_layers", ""),
        "metric": getattr(args, "metric", ""),
        "full": _fmt_percent(full_metric),
        "degrade_text": "" if degrade_text is None else _fmt_percent(degrade_text),
        "degrade_visual": "" if degrade_visual is None else _fmt_percent(degrade_visual),
        "full_pm": _fmt_pm(full_metric, full_std),
        "degrade_text_pm": _fmt_pm(degrade_text, degrade_text_std),
        "degrade_visual_pm": _fmt_pm(degrade_visual, degrade_visual_std),
        "overall_hmean_pm": _fmt_pm(overall_hmean, overall_hmean_std),
        "single_modality": getattr(args, "single_modality", "none"),
        "inductive": getattr(args, "inductive", False),
        "fewshots": getattr(args, "fewshots", 0),
        "text_feature": os.path.basename(getattr(args, "text_feature", "") or ""),
        "visual_feature": os.path.basename(getattr(args, "visual_feature", "") or ""),
        "lr": getattr(args, "lr", ""),
        "wd": getattr(args, "wd", ""),
        "dropout": getattr(args, "dropout", ""),
        "n_hidden": getattr(args, "n_hidden", ""),
        "embed_dim": getattr(args, "embed_dim", getattr(args, "n_hidden", "")),
        "aux_weight": getattr(args, "aux_weight", ""),
        "mlp_variant": getattr(args, "mlp_variant", ""),
        # Deep residual modality encoder parameters
        "modality_encoder": getattr(args, "modality_encoder", ""),
        "enc_n_layers": getattr(args, "enc_n_layers", ""),
        "enc_hidden_dim": getattr(args, "enc_hidden_dim", ""),
    }
    if extra:
        for k, v in extra.items():
            if k in {"full_std", "degrade_text_std", "degrade_visual_std", "overall_hmean", "overall_hmean_std"}:
                row[k] = _fmt_percent(v)
            else:
                row[k] = v
    if run is not None:
        row["run"] = run
    return row


def update_best_result_csv(
    path: str,
    row: Dict[str, object],
    *,
    key_fields: Iterable[str],
    score_field: str = "full",
) -> None:
    if not path:
        return

    path = str(path).strip()
    if not path:
        return

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    existing = []
    header = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames or [])
            for r in reader:
                existing.append(r)

    # Ensure header contains all row keys.
    for k in row.keys():
        if k not in header:
            header.append(k)

    def _key(r):
        return tuple(_as_str(r.get(k, "")) for k in key_fields)

    row_key = _key(row)
    replaced = False
    for i, r in enumerate(existing):
        if _key(r) == row_key:
            old_score = _as_float(r.get(score_field))
            new_score = _as_float(row.get(score_field))
            if new_score is not None and (old_score is None or new_score > old_score):
                existing[i] = {h: _as_str(row.get(h, "")) for h in header}
            replaced = True
            break

    if not replaced:
        existing.append({h: _as_str(row.get(h, "")) for h in header})

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in existing:
            writer.writerow({h: r.get(h, "") for h in header})


def append_result_csv(path: str, row: Dict[str, object]) -> None:
    if not path:
        return

    path = str(path).strip()
    if not path:
        return

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    existing = []
    header = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames or [])
            for r in reader:
                existing.append(r)

    for k in row.keys():
        if k not in header:
            header.append(k)

    existing.append({h: _as_str(row.get(h, "")) for h in header})

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in existing:
            writer.writerow({h: r.get(h, "") for h in header})
