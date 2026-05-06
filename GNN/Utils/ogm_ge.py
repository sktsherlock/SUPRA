"""
OGM-GE: On-the-fly Gradient Modulation with Gaussian Enhancement
================================================================

Implements gradient modulation for SUPRA's shared-unique channel architecture.
Unlike the original OGM-GE (which compares modalities to each other), this version
uses the C channel (GNN-aggregated shared representation) as the anchor:

    - If Ut >> C: Ut is bypassing GNN, hogging optimization → suppress Ut gradients
    - If Ut << C: Ut is under-optimized → do not suppress (let it learn)
    - Same logic for Uv vs C
    - C channel itself is NEVER modulated (anchor/reference)

Reference: "Balanced Multimodal Learning via On-the-fly Gradient Modulation"
(Peng et al., CVPR 2022)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F


def compute_ogm_coefficients(
    logits_Ut: torch.Tensor,
    logits_Uv: torch.Tensor,
    logits_C: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> Tuple[float, float]:
    """
    Compute OGM modulation coefficients for Ut and Uv channels.

    Uses softmax confidence on correct class as the dominance metric.
    Compares each unique channel's confidence against the C channel anchor.

    Args:
        logits_Ut: Shape [N, C] — Ut channel logits (text, no GNN)
        logits_Uv: Shape [N, C] — Uv channel logits (visual, no GNN)
        logits_C:  Shape [N, C] — C channel logits (GNN-aggregated)
        labels:   Shape [N] — ground truth class indices
        alpha:    Modulation strength (higher = more aggressive suppression)

    Returns:
        (coeff_t, coeff_v): Coefficient for Ut and Uv gradient modulation.
            coeff = 1.0 (no change) when channel is weaker than or equal to C
            coeff = 1 - tanh(alpha * ratio) when channel is stronger than C
    """
    softmax = F.softmax(logits_Ut, dim=1)
    target_softmax_Ut = softmax[torch.arange(len(labels), device=labels.device), labels]

    softmax = F.softmax(logits_Uv, dim=1)
    target_softmax_Uv = softmax[torch.arange(len(labels), device=labels.device), labels]

    softmax = F.softmax(logits_C, dim=1)
    target_softmax_C = softmax[torch.arange(len(labels), device=labels.device), labels]

    # Confidence ratio: how much stronger is this channel vs C?
    # score > score_C → ratio > 1 → over-confident → should suppress
    score_t = target_softmax_Ut.sum()
    score_v = target_softmax_Uv.sum()
    score_c = target_softmax_C.sum()

    ratio_t = score_t / (score_c + 1e-8)
    ratio_v = score_v / (score_c + 1e-8)

    # OGM formula (same as original paper, Eq. 10):
    #   k = 1 - tanh(alpha * rho)   if rho > 1
    #   k = 1                        otherwise
    # rho is the dominance ratio (score_channel / score_C)

    def _coeff(ratio: float) -> float:
        if ratio > 1.0:
            return 1.0 - torch.tanh(torch.tensor(alpha * (ratio - 1.0), device=logits_Ut.device)).item()
        return 1.0

    coeff_t = _coeff(ratio_t.item())
    coeff_v = _coeff(ratio_v.item())

    return coeff_t, coeff_v


def apply_ogm_ge(
    model,
    coeff_t: float,
    coeff_v: float,
    use_ge: bool = False,
) -> Dict[str, float]:
    """
    Apply OGM-GE gradient modulation to SUPRA model parameters.

    Modulates gradients for:
        - enc_t parameters (feeds into Ut, C channels)
        - enc_v parameters (feeds into Uv, C channels)
        - head_Ut parameters (Ut prediction head)
        - head_Uv parameters (Uv prediction head)

    C channel parameters (mp_C, head_C) are NEVER modulated.

    With GE (Gaussian Enhancement):
        grad *= coeff + noise * std(grad)

    Without GE:
        grad *= coeff

    Args:
        model:   SUPRA model instance
        coeff_t: Modulation coefficient for text branch (0.0-1.0, 1.0=no change)
        coeff_v: Modulation coefficient for visual branch (0.0-1.0, 1.0=no change)
        use_ge:  Whether to add Gaussian noise (GE component)

    Returns:
        Dict of modulation info for logging
    """
    info = {"coeff_t": coeff_t, "coeff_v": coeff_v, "use_ge": use_ge}

    # Modulate enc_t parameters (feed into Ut and C)
    if coeff_t < 1.0:
        for name, param in model.named_parameters():
            if "enc_t" in name and param.grad is not None:
                if use_ge:
                    noise_std = param.grad.std().item() + 1e-8
                    param.grad.data = (
                        param.grad.data * coeff_t
                        + torch.zeros_like(param.grad.data).normal_(0, noise_std)
                    )
                else:
                    param.grad.data *= coeff_t

    # Modulate head_Ut parameters (Ut prediction head)
    if coeff_t < 1.0:
        for name, param in model.named_parameters():
            if "head_Ut" in name and param.grad is not None:
                if use_ge:
                    noise_std = param.grad.std().item() + 1e-8
                    param.grad.data = (
                        param.grad.data * coeff_t
                        + torch.zeros_like(param.grad.data).normal_(0, noise_std)
                    )
                else:
                    param.grad.data *= coeff_t

    # Modulate enc_v parameters (feeds into Uv and C)
    if coeff_v < 1.0:
        for name, param in model.named_parameters():
            if "enc_v" in name and param.grad is not None:
                if use_ge:
                    noise_std = param.grad.std().item() + 1e-8
                    param.grad.data = (
                        param.grad.data * coeff_v
                        + torch.zeros_like(param.grad.data).normal_(0, noise_std)
                    )
                else:
                    param.grad.data *= coeff_v

    # Modulate head_Uv parameters (Uv prediction head)
    if coeff_v < 1.0:
        for name, param in model.named_parameters():
            if "head_Uv" in name and param.grad is not None:
                if use_ge:
                    noise_std = param.grad.std().item() + 1e-8
                    param.grad.data = (
                        param.grad.data * coeff_v
                        + torch.zeros_like(param.grad.data).normal_(0, noise_std)
                    )
                else:
                    param.grad.data *= coeff_v

    return info
