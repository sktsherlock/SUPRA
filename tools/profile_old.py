#!/usr/bin/env python
"""Profile MAG experiments: parameter count, GPU peak memory, and runtime.
UPDATED v3: 
- Fixes --log_every format bug.
- Calculates 'pure_train_time_10_epochs' (derived from internal logs to exclude data loading).
"""
from __future__ import annotations
import argparse
import json
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch as th
try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None

# Add repo root to path
_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(_ROOT))

from GNN.GraphData import load_data  # noqa: E402
from GNN.Baselines.Early_GNN import Early_GNN as mag_early  # noqa: E402
from GNN.Baselines.Late_GNN import Late_GNN as mag_late  # noqa: E402
# The following modules no longer exist in the current codebase:
# from GNN.Library.MAG import Tri_GNN as mag_tri  # noqa: E402
# from GNN.Library.MAG import OGM_GNN as mag_ogm  # noqa: E402
# from GNN.Library.MAG import PID_new as mag_pid  # noqa: E402
# from GNN.Library.MAG import PID_OGM as mag_pid_ogm  # noqa: E402
# from GNN.Library.MAG import PID_OGM_new as mag_pid_ogm_new  # noqa: E402

FEATURE_MAP = {
    "Movies|default": (
        "TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy",
        "ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy",
    ),
    "Grocery|default": (
        "TextFeature/Grocery_Llama_3.2_11B_Vision_Instruct_256_mean.npy",
        "ImageFeature/Grocery_Llama-3.2-11B-Vision-Instruct_visual.npy",
    ),
    "Toys|default": (
        "TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy",
        "ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy",
    ),
    "Reddit-M|default": (
        "TextFeature/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy",
        "ImageFeature/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy",
    ),
    "Reddit-S|default": (
        "TextFeature/RedditS_Llama_3.2_11B_Vision_Instruct_100_mean.npy",
        "ImageFeature/RedditS_Llama-3.2-11B-Vision-Instruct_visual.npy",
    ),
}

def _expand_data_root_in_cmd(cmd: str, data_root: Path) -> str:
    root_native = str(data_root)
    root_posix = data_root.as_posix()
    out = cmd
    out = out.replace("${DATA_ROOT}", root_posix)
    out = out.replace("$DATA_ROOT", root_posix)
    out = out.replace("%DATA_ROOT%", root_native)
    return out

def _parse_suite_script(path: Path) -> Tuple[Optional[Path], Dict[str, Tuple[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Suite script not found: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    data_root = None
    data_root_pat = re.compile(r"^\s*DATA_ROOT=\$\{DATA_ROOT:-([^}]+)\}", re.MULTILINE)
    m = data_root_pat.search(text)
    if m:
        data_root = Path(m.group(1).strip())

    map_out: Dict[str, Tuple[str, str]] = {}
    text_pat = re.compile(r"TEXT_FEATURE_BY_DS_GROUP\[\"([^\"]+)\"\]\s*=\s*'([^']+)'")
    vis_pat = re.compile(r"VIS_FEATURE_BY_DS_GROUP\[\"([^\"]+)\"\]\s*=\s*'([^']+)'")

    text_map: Dict[str, str] = {m.group(1): m.group(2) for m in text_pat.finditer(text)}
    vis_map: Dict[str, str] = {m.group(1): m.group(2) for m in vis_pat.finditer(text)}

    for key, text_rel in text_map.items():
        vis_rel = vis_map.get(key)
        if vis_rel:
            map_out[key] = (text_rel, vis_rel)
    return data_root, map_out

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
    feature_group = entry.get("feature_group", "default")
    key = f"{dataset}|{feature_group}"
    
    text_rel, visual_rel = FEATURE_MAP.get(key, (None, None))
    ds_prefix = dataset.replace("-", "")
    graph_path = Path(entry.get("graph_path") or data_root / dataset / f"{ds_prefix}Graph.pt")
    
    text_path_str = entry.get("text_feature")
    if not text_path_str and text_rel:
        text_path = data_root / dataset / text_rel
    else:
        text_path = Path(text_path_str) if text_path_str else Path("MISSING_TEXT")

    visual_path_str = entry.get("visual_feature")
    if not visual_path_str and visual_rel:
        visual_path = data_root / dataset / visual_rel
    else:
        visual_path = Path(visual_path_str) if visual_path_str else Path("MISSING_VISUAL")

    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph: {graph_path}")
    if not text_path.exists():
         raise FileNotFoundError(f"Missing text feature: {text_path}")
    if not visual_path.exists():
         raise FileNotFoundError(f"Missing visual feature: {visual_path}")

    text_dim = int(np.load(text_path, mmap_mode="r").shape[1])
    visual_dim = int(np.load(visual_path, mmap_mode="r").shape[1])

    _, labels, *_ = load_data(
        str(graph_path),
        train_ratio=0.6,
        val_ratio=0.2,
        name=dataset,
        fewshots=None,
    )
    n_classes = int((labels.max() + 1).item())

    return ResolvedData(
        graph_path=graph_path,
        text_path=text_path,
        visual_path=visual_path,
        text_dim=text_dim,
        visual_dim=visual_dim,
        n_classes=n_classes,
    )

def _make_args(entry: Dict[str, Any]) -> argparse.Namespace:
    args = argparse.Namespace()
    for k, v in entry.items():
        setattr(args, k, v)
    
    if not hasattr(args, "model_name"): args.model_name = entry.get("backbone", "GCN")
    
    # Set default layers according to model type (SAGE=3, GAT=2, others=3)
    default_n_layers = 3
    if str(args.model_name).upper() == "GAT":
        default_n_layers = 2
    
    if not hasattr(args, "n_layers"): args.n_layers = int(entry.get("n_layers", default_n_layers))
    if not hasattr(args, "n_hidden"): args.n_hidden = int(entry.get("n_hidden", 256))
    if not hasattr(args, "dropout"): args.dropout = float(entry.get("dropout", 0.2))
    if not hasattr(args, "aggregator"): args.aggregator = entry.get("aggregator", "mean")
    if not hasattr(args, "n_heads"): args.n_heads = int(entry.get("n_heads", 3))
    if not hasattr(args, "attn_drop"): args.attn_drop = float(entry.get("attn_drop", 0.0))
    if not hasattr(args, "edge_drop"): args.edge_drop = float(entry.get("edge_drop", 0.0))
    if not hasattr(args, "no_attn_dst"): args.no_attn_dst = bool(entry.get("no_attn_dst", True))
    if not hasattr(args, "k"): args.k = int(entry.get("k", 2))
    if not hasattr(args, "bias"): args.bias = bool(entry.get("bias", True))
    if not hasattr(args, "input_dropout"): args.input_dropout = float(entry.get("input_dropout", 0.5))
    if not hasattr(args, "alpha"): args.alpha = float(entry.get("alpha", 0.1))
    if not hasattr(args, "k_ps"): args.k_ps = int(entry.get("k_ps", 10))
    return args

def _count_params(model: th.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))

def _build_and_count(entry: Dict[str, Any], resolved: ResolvedData) -> Tuple[int, int]:
    method = entry.get("method", "early_gnn")
    device = th.device("cpu")
    args = _make_args(entry)
    n_classes = resolved.n_classes

    # Count functions for common blocks
    t_enc = mag_early.ModalityEncoder(resolved.text_dim, args.n_hidden, args.dropout)
    v_enc = mag_early.ModalityEncoder(resolved.visual_dim, args.n_hidden, args.dropout)
    head_params = _count_params(t_enc) + _count_params(v_enc)

    if method == "pid_magnn":
        embed_dim = int(entry.get("pid_embed_dim") or args.n_hidden)
        pid_L = int(entry.get("pid_L") if entry.get("pid_L") is not None else args.n_layers)
        pid_lu = int(entry.get("pid_lu", 0)) 
        
        model = mag_pid.PIDMAGNN(
            text_in_dim=resolved.text_dim,
            vis_in_dim=resolved.visual_dim,
            embed_dim=embed_dim,
            n_classes=n_classes,
            dropout=args.dropout,
            args=args,
            device=device,
            pid_L=pid_L,
            pid_lu=pid_lu,
        )
        total_params = _count_params(model)
        
        enc_t_params = _count_params(mag_early.ModalityEncoder(resolved.text_dim, embed_dim, args.dropout))
        enc_v_params = _count_params(mag_early.ModalityEncoder(resolved.visual_dim, embed_dim, args.dropout))
        params_no_head = total_params - (enc_t_params + enc_v_params)
        return total_params, params_no_head

    elif method == "pid_ogm":
        embed_dim = int(entry.get("pid_embed_dim") or args.n_hidden)
        pid_L = int(entry.get("pid_L") if entry.get("pid_L") is not None else args.n_layers)
        pid_lu = int(entry.get("pid_lu", 0)) 
        
        model = mag_pid_ogm.PIDMAGNN(
            text_in_dim=resolved.text_dim,
            vis_in_dim=resolved.visual_dim,
            embed_dim=embed_dim,
            n_classes=n_classes,
            dropout=args.dropout,
            args=args,
            device=device,
            pid_L=pid_L,
            pid_lu=pid_lu,
        )
        total_params = _count_params(model)
        
        enc_t_params = _count_params(mag_early.ModalityEncoder(resolved.text_dim, embed_dim, args.dropout))
        enc_v_params = _count_params(mag_early.ModalityEncoder(resolved.visual_dim, embed_dim, args.dropout))
        params_no_head = total_params - (enc_t_params + enc_v_params)
        return total_params, params_no_head

    elif method == "pid_ogm_new":
        embed_dim = int(entry.get("pid_embed_dim") or args.n_hidden)
        pid_L = int(entry.get("pid_L") if entry.get("pid_L") is not None else args.n_layers)
        pid_lu = int(entry.get("pid_lu", 0)) 
        
        model = mag_pid_ogm_new.PIDMAGNN(
            text_in_dim=resolved.text_dim,
            vis_in_dim=resolved.visual_dim,
            embed_dim=embed_dim,
            n_classes=n_classes,
            dropout=args.dropout,
            args=args,
            device=device,
            pid_L=pid_L,
            pid_lu=pid_lu,
        )
        total_params = _count_params(model)
        
        enc_t_params = _count_params(mag_early.ModalityEncoder(resolved.text_dim, embed_dim, args.dropout))
        enc_v_params = _count_params(mag_early.ModalityEncoder(resolved.visual_dim, embed_dim, args.dropout))
        params_no_head = total_params - (enc_t_params + enc_v_params)
        return total_params, params_no_head

    elif method == "early_gnn":
        gnn_in_dim = args.n_hidden * 2 if entry.get("early_fuse", "concat") == "concat" else args.n_hidden
        gnn = mag_early._build_gnn_backbone(args, gnn_in_dim, n_classes, device)
        model = mag_early.SimpleMAGGNN(t_enc, v_enc, gnn, early_fuse=entry.get("early_fuse", "concat"))
        t_p = _count_params(model)
        return t_p, t_p - head_params

    elif method == "late_gnn":
        t_gnn = mag_early._build_gnn_backbone(args, args.n_hidden, args.n_hidden, device)
        v_gnn = mag_early._build_gnn_backbone(args, args.n_hidden, args.n_hidden, device)
        clf = mag_pid._make_mlp(args.n_hidden * 2, args.n_hidden, n_classes, args.dropout)
        model = mag_late.LateFusionMAG(t_enc, v_enc, t_gnn, v_gnn, clf)
        t_p = _count_params(model)
        return t_p, t_p - head_params

    elif method == "tri_gnn":
        e_gnn = mag_early._build_gnn_backbone(args, args.n_hidden * 2, args.n_hidden, device)
        t_gnn = mag_early._build_gnn_backbone(args, args.n_hidden, args.n_hidden, device)
        v_gnn = mag_early._build_gnn_backbone(args, args.n_hidden, args.n_hidden, device)
        clf = mag_pid._make_mlp(args.n_hidden * 3, args.n_hidden, n_classes, args.dropout)
        model = mag_tri.TriFusionMAG(t_enc, v_enc, e_gnn, t_gnn, v_gnn, clf)
        t_p = _count_params(model)
        return t_p, t_p - head_params

    elif method in ["ntsformer", "ntsformer_mini"]:
        import sys
        _nts_path = str(_ROOT / "GNN" / "Library" / "MAG" / "NTS")
        if _nts_path not in sys.path:
            sys.path.append(_nts_path)
            
        from coldgnn.configs.coldgnn_default_config import load_coldgnn_default_config
        from coldgnn.layers.ntsformer import NTSFormer
        
        try:
            config = load_coldgnn_default_config(entry.get("dataset"), use_pre_train=False, config_name=None, use_echoless_feat=True)
        except Exception:
            config = load_coldgnn_default_config("magb-movies", use_pre_train=False, config_name=None, use_echoless_feat=True)
            
        input_shape = [[-1, config.pre_k + 1, resolved.text_dim], [-1, config.pre_k + 1, resolved.visual_dim]]
        model = NTSFormer(
            feat_proj_units_list=config.feat_proj_units_list,
            att_group_units_list=[],
            global_units_list=config.global_units_list + [n_classes],
            merge_mode="concat",
            input_shape=input_shape,
            input_drop_rate=config.input_drop_rate,
            group_drop_rate=config.group_drop_rate,
            global_drop_rate=config.global_drop_rate,
            group_output_drop_rate=config.group_output_drop_rate,
            global_input_drop_rate=config.global_input_drop_rate,
            ff_drop_rate=config.global_drop_rate,
            att_drop_rate=config.att_drop_rate,
            bn=config.bn,
            activation="prelu",
            num_heads=config.num_heads,
            num_tf_layers=config.num_tf_layers,
            feat_proj_residual=config.feat_proj_residual,
            group_encoder_mode=config.group_encoder_mode,
            ff_units_list=config.ff_units_list,
            sample_neighbors=True,
            pre_k=config.pre_k,
            num_routed_experts=config.num_routed_experts,
            num_shared_experts=config.num_shared_experts,
            split_text_visual=True,
            use_gl_stu=False,
            drop_modality=False,
            use_input_feat_moe=False,
            use_dual_teacher=False,
            metrics_dict={},
        )
        t_p = _count_params(model)
        
        # Calculate NTSFormer internal projection head parameters to subtract
        head_params = 0
        if hasattr(model, 'multi_group_fusion'):
            # The actual multi-modal projection layers are held inside the multi_group_fusion module
            mgf = model.multi_group_fusion
            if hasattr(mgf, 't_input_feat_fcs'):
                for fc in mgf.t_input_feat_fcs:
                    head_params += _count_params(fc)
            if hasattr(mgf, 'v_input_feat_fcs'):
                for fc in mgf.v_input_feat_fcs:
                    head_params += _count_params(fc)
            
            # Subtracted projection heads: t_input_feat_fcs and v_input_feat_fcs which do the initial dimensionality reduction.
        
        return t_p, t_p - head_params

    raise ValueError(f"Unknown method or not supported in this profile version: {method}")

def _nvml_init():
    if pynvml is None:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False

def _monitor_gpu(gpu_id: int, stop_event: threading.Event, out_q: queue.Queue, interval: float = 0.5):
    peak = 0
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        while not stop_event.is_set():
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            peak = max(peak, int(info.used))
            time.sleep(interval)
    except Exception:
        pass
    out_q.put(peak)

def _run_command(
    cmd: str, gpu_id: int, log_path: Optional[Path], interval: float, data_root: Path
) -> Tuple[float, Optional[float], Optional[float], int]:
    stop_event = threading.Event()
    out_q: queue.Queue = queue.Queue()
    peak_mb: Optional[float] = None
    monitor_thread: Optional[threading.Thread] = None

    if gpu_id >= 0 and _nvml_init():
        monitor_thread = threading.Thread(
            target=_monitor_gpu, args=(gpu_id, stop_event, out_q, interval), daemon=True
        )
        monitor_thread.start()

    start = time.time()
    avg_epoch_time = None
    epoch_pat = re.compile(r"Average epoch time:\s+([0-9.]+)")

    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("DGLBACKEND", "pytorch")
    env.setdefault("DATA_ROOT", str(data_root))
    if gpu_id >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = _expand_data_root_in_cmd(cmd, data_root)

    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=str(_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, 
        text=True,
        env=env,
        bufsize=1, 
    )

    if proc.stdout:
        for line in proc.stdout:
            if log_file:
                log_file.write(line)
            match = epoch_pat.search(line)
            if match:
                try:
                    avg_epoch_time = float(match.group(1))
                except Exception:
                    pass

    proc.wait()
    return_code = int(proc.returncode or 0)

    if log_file:
        log_file.flush()
        log_file.close()

    end = time.time()
    stop_event.set()
    if monitor_thread is not None:
        monitor_thread.join(timeout=5)
        try:
            peak_bytes = out_q.get_nowait()
            peak_mb = peak_bytes / (1024 * 1024)
        except Exception:
            peak_mb = None

    return end - start, peak_mb, avg_epoch_time, return_code

def main() -> None:
    parser = argparse.ArgumentParser(description="Profile MAG experiments.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config list.")
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--suite_script", type=Path, default=None)
    parser.add_argument("--mode", choices=["params", "run", "all"], default="all")
    parser.add_argument("--out_csv", type=Path, default=Path("profile_results.csv"))
    parser.add_argument("--poll_interval", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")

    args = parser.parse_args()

    if args.suite_script:
        suite_root, suite_map = _parse_suite_script(args.suite_script)
        if suite_map:
            FEATURE_MAP.update(suite_map)
        if args.data_root is None and suite_root is not None:
            args.data_root = suite_root

    if args.data_root is None:
        args.data_root = Path(os.environ.get("DATA_ROOT", "/hyperai/input/input0/MAGB_Dataset"))

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    
    rows: List[Dict[str, Any]] = []
    total = len(cfg)

    for idx, entry in enumerate(cfg, start=1):
        name = entry.get("name", "")
        method = entry.get("method", "")
        backbone = entry.get("backbone", "")

        # --- Force Layer Counts (User Request) ---
        # GAT -> 2 layers, SAGE (and others) -> 3 layers
        # target_layers = 3
        # if str(backbone).upper() == "GAT":
        #     target_layers = 2
        
        # # 1. Update entry dict (for parameter counting)
        # entry["n_layers"] = target_layers
        # if method == "pid_magnn" and "pid_L" in entry:
        #     entry["pid_L"] = target_layers
        target_layers = entry.get("n_layers")
        if method in ["pid_magnn", "pid_ogm", "pid_ogm_new"] and "pid_L" in entry:
            target_layers = entry["pid_L"]

        # 2. Update cmd string (for execution)
        if "cmd" in entry:
            c = entry["cmd"]
            # Replace --n-layers X
            c = re.sub(r"--n-layers\s+\d+", f"--n-layers {target_layers}", c)
            # Replace --n_layers X (just in case)
            c = re.sub(r"--n_layers\s+\d+", f"--n-layers {target_layers}", c)
            
            # For PID-MAGNN, also replace --pid_L X
            if method in ["pid_magnn", "pid_ogm", "pid_ogm_new"]:
                c = re.sub(r"--pid_L\s+\d+", f"--pid_L {target_layers}", c)
            
            entry["cmd"] = c
        # ----------------------------------------
        
        if args.verbose:
            print(f"[{idx}/{total}] Processing {name}...", flush=True)

        resolved = _resolve_paths(entry, args.data_root)
        param_count_total = None
        param_count_no_head = None
        
        if args.mode in ("params", "all"):
            try:
                res = _build_and_count(entry, resolved)
                if isinstance(res, tuple):
                    param_count_total, param_count_no_head = res
                else:
                    # Should not reach here with new logic, but safe fallback
                    param_count_no_head = res
                
                if args.verbose:
                    if param_count_total is not None:
                        print(f"  -> Params (Total): {param_count_total:,}")
                    print(f"  -> Params (No Head): {param_count_no_head:,}")
            except Exception as e:
                print(f"  [Error] Parameter counting failed for {name}: {e}")
                import traceback
                traceback.print_exc()
                param_count_total = "N/A"
                param_count_no_head = "N/A"

        wall_time = None
        peak_mb = None
        avg_epoch = None
        exit_code = None
        
        if args.mode in ("run", "all"):
            cmd = entry.get("cmd")
            gpu_id = int(entry.get("gpu_id", 0))
            log_path = entry.get("log_path")
            if log_path:
                log_path = Path(log_path)
            elif args.log_dir:
                log_path = args.log_dir / f"{name}.log"
            
            if args.verbose:
                print(f"  -> Running command on GPU {gpu_id}...")
            
            wall_time, peak_mb, avg_epoch, exit_code = _run_command(
                cmd, gpu_id, log_path, args.poll_interval, args.data_root
            )
            
            if args.strict and exit_code != 0:
                print(f"  [Error] Run failed with code {exit_code}")
            elif args.verbose:
                 print(f"  -> Done. Wall: {wall_time:.2f}s, Peak GPU: {peak_mb} MB, Avg Epoch: {avg_epoch}")

        # Modified output: Calculate pure train time for 20 epochs based on internal average
        # This excludes data loading time which 'wall_time_sec' includes
        pure_train_time_20_epochs = avg_epoch * 20 if avg_epoch is not None else None

        rows.append({
            "name": name,
            "dataset": entry.get("dataset"),
            "method": method,
            "backbone": backbone,
            "param_count_total": param_count_total,
            "param_count_no_head": param_count_no_head,
            "wall_time_sec": wall_time,
            "peak_gpu_mb": peak_mb,
            "pure_train_time_20_epochs": pure_train_time_20_epochs,
            "exit_code": exit_code,
        })

    # Write CSV
    header = [
        "name", "dataset", "method", "backbone", 
        "param_count_total", "param_count_no_head", "wall_time_sec", "peak_gpu_mb", "pure_train_time_20_epochs", "exit_code"
    ]

    def _fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(_fmt(row.get(h)) for h in header) + "\n")

if __name__ == "__main__":
    main()