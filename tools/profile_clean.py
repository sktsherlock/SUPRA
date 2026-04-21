#!/usr/bin/env python
"""Profile MAG experiments: parameter count, GPU peak memory (Torch Internal), and runtime."""
from __future__ import annotations
import argparse
import json
import os
import re
import shlex
import subprocess
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch as th

# Add repo root to path
_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(_ROOT))

from GNN.GraphData import load_data
from GNN.Baselines.Early_GNN import Early_GNN as mag_early
from GNN.Baselines.Late_GNN import Late_GNN as mag_late
# Tri_GNN and PID_new no longer exist in the current codebase
# from GNN.Library.MAG import Tri_GNN as mag_tri
# from GNN.Library.MAG import PID_new as mag_pid

FEATURE_MAP = {
    "Movies|default": ("TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy", "ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy"),
    "Grocery|default": ("TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy", "ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy"),
    "Toys|default": ("TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy", "ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy"),
    "Reddit-M|default": ("TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy", "ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy"),
    "Reddit-S|default": ("TextFeature/RedditS_Llama_3.2_11B_Vision_Instruct_100_mean.npy", "ImageFeature/RedditS_Llama-3.2-11B-Vision-Instruct_visual.npy"),
}

def _expand_data_root_in_cmd(cmd: str, data_root: Path) -> str:
    root_native = str(data_root)
    root_posix = data_root.as_posix()
    out = cmd.replace("${DATA_ROOT}", root_posix).replace("$DATA_ROOT", root_posix).replace("%DATA_ROOT%", root_native)
    return out

@dataclass
class ResolvedData:
    graph_path: Path
    text_path: Path
    visual_path: Path
    text_dim: int
    visual_dim: int
    n_classes: int

def _resolve_paths(entry: Dict[str, Any], data_root: Path) -> ResolvedData:
    dataset = entry["dataset"]
    key = f"{dataset}|{entry.get('feature_group', 'default')}"
    text_rel, visual_rel = FEATURE_MAP.get(key, (None, None))
    ds_prefix = dataset.replace("-", "")
    graph_path = Path(entry.get("graph_path") or data_root / dataset / f"{ds_prefix}Graph.pt")
    
    text_path_str = entry.get("text_feature")
    text_path = Path(text_path_str) if text_path_str else (data_root / dataset / text_rel if text_rel else Path("MISSING"))
    visual_path_str = entry.get("visual_feature")
    visual_path = Path(visual_path_str) if visual_path_str else (data_root / dataset / visual_rel if visual_rel else Path("MISSING"))

    text_dim = int(np.load(text_path, mmap_mode="r").shape[1])
    visual_dim = int(np.load(visual_path, mmap_mode="r").shape[1])
    _, labels, *_ = load_data(str(graph_path), train_ratio=0.6, val_ratio=0.2, name=dataset, fewshots=None)
    return ResolvedData(graph_path, text_path, visual_path, text_dim, visual_dim, int((labels.max() + 1).item()))

def _make_args(entry: Dict[str, Any]) -> argparse.Namespace:
    args = argparse.Namespace()
    for k, v in entry.items(): setattr(args, k, v)
    if not hasattr(args, "model_name"): args.model_name = entry.get("backbone", "GCN")
    args.n_layers = int(entry.get("n_layers", 2 if str(args.model_name).upper() == "GAT" else 3))
    args.n_hidden = int(entry.get("n_hidden", 256))
    args.dropout = float(entry.get("dropout", 0.2))
    args.n_heads = int(entry.get("n_heads", 3))
    return args

def _count_params(model: th.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

def _build_and_count(entry: Dict[str, Any], resolved: ResolvedData) -> Tuple[int, int]:
    method = entry.get("method", "early_gnn")
    device = th.device("cpu")
    args = _make_args(entry)
    t_enc = mag_early.ModalityEncoder(resolved.text_dim, args.n_hidden, args.dropout)
    v_enc = mag_early.ModalityEncoder(resolved.visual_dim, args.n_hidden, args.dropout)
    head_params = _count_params(t_enc) + _count_params(v_enc)

    if method in ("pid_magnn", "pid_ogm", "pid_ogm_new"):
        embed_dim = int(entry.get("pid_embed_dim") or args.n_hidden)
        model = mag_pid.PIDMAGNN(
            text_in_dim=resolved.text_dim, vis_in_dim=resolved.visual_dim, embed_dim=embed_dim,
            n_classes=resolved.n_classes, dropout=args.dropout, args=args, device=device,
            pid_L=int(entry.get("pid_L", args.n_layers)), pid_lu=int(entry.get("pid_lu", 0))
        )
        total_p = _count_params(model)
        enc_p = _count_params(mag_early.ModalityEncoder(resolved.text_dim, embed_dim, args.dropout)) + \
                _count_params(mag_early.ModalityEncoder(resolved.visual_dim, embed_dim, args.dropout))
        return total_p, total_p - enc_p
    return 0, 0 # Simplified for other models to save space as PID is the main concern here

def _run_command(cmd: str, gpu_id: int, log_path: Optional[Path], data_root: Path) -> Tuple[float, Optional[float], Optional[float], int, int]:
    start = time.time()
    avg_epoch_time = None
    peak_mb = None
    total_epochs = 0
    
    env = os.environ.copy()
    env.setdefault("DATA_ROOT", str(data_root))
    if gpu_id >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = _expand_data_root_in_cmd(cmd, data_root)
    
    # Extract script running parts
    parts = shlex.split(cmd)
    script_path = None
    sys_argv = []
    for i, p in enumerate(parts):
        if p.endswith('.py'):
            script_path = p
            sys_argv = parts[i:]
            break
            
    if not script_path:
        raise ValueError("Could not find python script in cmd")

    wrapper_hook = f"""
import sys
import runpy
import torch

sys.argv = {repr(sys_argv)}
try:
    runpy.run_path({repr(script_path)}, run_name="__main__")
except SystemExit:
    pass
except BaseException as e:
    import traceback
    traceback.print_exc()

if torch.cuda.is_available():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\\n[PROFILER_TORCH_PEAK] {{torch.cuda.max_memory_allocated(dev) / 1048576:.2f}}")
"""
    
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(wrapper_hook)
        temp_script = f.name

    log_file = log_path.open("w", encoding="utf-8") if log_path else None

    proc = subprocess.Popen(
        [sys.executable, temp_script],
        cwd=str(_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, 
        text=True,
        env=env,
        bufsize=1, 
    )

    epoch_pat = re.compile(r"Average epoch time:\s+([0-9.]+)")
    peak_pat = re.compile(r"\[PROFILER_TORCH_PEAK\]\s+([0-9.]+)")
    epoch_count_pat = re.compile(r"Epoch[^\d]*(\d+)")

    last_epoch = 0
    cumulative_epochs = 0

    if proc.stdout:
        for line in proc.stdout:
            if log_file: log_file.write(line)
            epoch_match = epoch_pat.search(line)
            if epoch_match:
                try: avg_epoch_time = float(epoch_match.group(1))
                except Exception: pass
            
            peak_match = peak_pat.search(line)
            if peak_match:
                try: peak_mb = float(peak_match.group(1))
                except Exception: pass

            ec_match = epoch_count_pat.search(line)
            if ec_match:
                try: 
                    ep = int(ec_match.group(1))
                    if ep < last_epoch:
                        # A new run started, so accumulate the max epoch reached by the previous run
                        cumulative_epochs += last_epoch
                    last_epoch = ep
                except Exception: pass

    proc.wait()
    if log_file: log_file.close()
    try: os.remove(temp_script)
    except: pass

    total_epochs = cumulative_epochs + last_epoch
    end = time.time()
    return end - start, peak_mb, avg_epoch_time, proc.returncode, total_epochs

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=Path(os.environ.get("DATA_ROOT", "/hyperai/input/input0/MAGB_Dataset")))
    parser.add_argument("--mode", choices=["params", "run", "all"], default="all")
    parser.add_argument("--out_csv", type=Path, default=Path("profile_torch_peak.csv"))
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    rows = []

    for idx, entry in enumerate(cfg, start=1):
        if args.verbose: print(f"Processing {entry.get('name', '')}...")
        
        target_layers = entry.get("n_layers", 2 if str(entry.get("backbone")).upper() == "GAT" else 3)
        if entry.get("method") in ("pid_magnn", "pid_ogm", "pid_ogm_new") and "pid_L" in entry:
            target_layers = entry["pid_L"]

        if "cmd" in entry:
            c = entry["cmd"]
            c = re.sub(r"--n-layers\s+\d+", f"--n-layers {target_layers}", c)
            c = re.sub(r"--n_layers\s+\d+", f"--n-layers {target_layers}", c)
            if entry.get("method") in ("pid_magnn", "pid_ogm", "pid_ogm_new"):
                c = re.sub(r"--pid_L\s+\d+", f"--pid_L {target_layers}", c)
            entry["cmd"] = c

        resolved = _resolve_paths(entry, args.data_root)
        p_tot, p_no_head = None, None
        if args.mode in ("params", "all"):
            try: p_tot, p_no_head = _build_and_count(entry, resolved)
            except Exception: pass

        wall, peak_mb, avg_ep, code, tot_eps = None, None, None, None, None
        if args.mode in ("run", "all") and "cmd" in entry:
            log_p = Path(entry["log_path"]) if entry.get("log_path") else (args.log_dir / f"{entry['name']}.log" if args.log_dir else None)
            wall, peak_mb, avg_ep, code, tot_eps = _run_command(entry["cmd"], int(entry.get("gpu_id", 0)), log_p, args.data_root)
            if args.verbose: print(f"  -> Done. Peak Torch: {peak_mb} MB")

        calc_avg_ep = (wall / tot_eps) if (wall and tot_eps and tot_eps > 0) else None

        rows.append({
            "name": entry.get("name"), "dataset": entry.get("dataset"), "method": entry.get("method"),
            "backbone": entry.get("backbone"), "param_count_total": p_tot,
            "wall_time_sec": wall, "total_epochs": tot_eps, "avg_time_per_epoch": calc_avg_ep, "torch_peak_mb": peak_mb, "pure_train_time_20_epochs": (avg_ep * 20 if avg_ep else None)
        })

    header = ["name", "dataset", "method", "backbone", "param_count_total", "wall_time_sec", "total_epochs", "avg_time_per_epoch", "torch_peak_mb", "pure_train_time_20_epochs"]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join("" if v is None else (f"{v:.6f}" if isinstance(v, float) else str(v)) for h, v in row.items() if h in header) + "\n")

if __name__ == "__main__":
    main()
