"""Gradient analysis via SVD of gradient matrices for rank/training dynamics."""
import torch as th
import torch.nn as nn
from typing import Dict, Optional, List
from collections import defaultdict
import numpy as np


class GradientAnalyzer:
    """Attaches to a model and collects gradient statistics during training.

    Computes per-layer:
    - Stable rank: sum(s^2) / max(s^2) where s are singular values
    - Condition number: max(s) / min(s)
    - Orthogonality score: ||X @ X.T - I||_F / ||X||_F
    """

    def __init__(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        self.model = model
        self.layer_names = layer_names or []
        self.hooks = []
        self.grad_stats = defaultdict(list)
        self._activated = False

    def _compute_svd_metrics(self, grad: th.Tensor) -> Dict[str, float]:
        if grad.dim() != 2 or grad.numel() == 0:
            return {}
        X = grad.float()
        if X.shape[0] < X.shape[1]:
            X = X.T
        try:
            s = th.linalg.svd(X, compute_uv=False)
            ss = s * s
            stable_rank = (ss.sum() / ss[0].clamp_min(1e-10)).item()
            cond_num = (s[0] / s[-1].clamp_min(1e-10)).item()
            XtX = X @ X.T
            d = XtX.shape[0]
            I = th.eye(d, device=XtX.device, dtype=XtX.dtype)
            ortho_score = (XtX - I).norm() / (X.norm() + 1e-10)
            return {
                'stable_rank': stable_rank,
                'cond_num': cond_num,
                'ortho_score': ortho_score.item(),
                'singular_val_0': s[0].item(),
                'singular_val_last': s[-1].item(),
                'rank_ratio': (s > 1e-6).sum().item() / len(s),
            }
        except Exception:
            return {}

    def _register_hook(self, name: str, param: nn.Parameter):
        if not param.requires_grad:
            return
        # register_hook on Parameter registers a backward hook (receives gradient)
        handle = param.register_hook(lambda grad: self._hook_fn(name, grad))
        self.hooks.append(handle)

    def _hook_fn(self, name: str, grad: th.Tensor):
        if grad is None or not isinstance(grad, th.Tensor):
            return
        metrics = self._compute_svd_metrics(grad)
        if metrics:
            self.grad_stats[name].append(metrics)

    def attach(self):
        """Attach hooks to model parameters."""
        if self._activated:
            return
        self._activated = True
        for name, param in self.model.named_parameters():
            if not self.layer_names or any(ln in name for ln in self.layer_names):
                if param.dim() >= 2:
                    self._register_hook(name, param)

    def detach(self):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self._activated = False

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get average metrics across all backward passes."""
        summary = {}
        for name, stats in self.grad_stats.items():
            if not stats:
                continue
            summary[name] = {
                'stable_rank_mean': float(np.mean([s['stable_rank'] for s in stats])),
                'stable_rank_std': float(np.std([s['stable_rank'] for s in stats])),
                'cond_num_mean': float(np.mean([s['cond_num'] for s in stats])),
                'cond_num_std': float(np.std([s['cond_num'] for s in stats])),
                'ortho_score_mean': float(np.mean([s['ortho_score'] for s in stats])),
                'ortho_score_std': float(np.std([s['ortho_score'] for s in stats])),
                'n_samples': len(stats),
            }
        return summary

    def get_all_stats(self) -> Dict[str, List[Dict[str, float]]]:
        """Get all collected stats per layer."""
        return dict(self.grad_stats)

    def reset(self):
        """Clear collected statistics."""
        self.grad_stats.clear()
