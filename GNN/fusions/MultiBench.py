# GNN/fusions/MultiBench.py
# Minimal placeholder for fusion modules used by model_config.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TensorFusion(nn.Module):
    """Element-wise addition fusion (fallback for concat-based methods)."""
    def __init__(self):
        super().__init__()

    def forward(self, tensors):
        # tensors: list of [batch, dim] tensors
        return torch.stack(tensors, dim=0).sum(dim=0)


class LowRankTensorFusion(nn.Module):
    """Low-rank tensor fusion approximation."""
    def __init__(self, input_dims, output_dim, rank=16):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.rank = rank

        total_dim = sum(input_dims)
        self.U = nn.Parameter(torch.randn(total_dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, output_dim) * 0.01)

    def forward(self, tensors):
        # Concat and project
        x = torch.cat(tensors, dim=-1)
        return x @ self.U @ self.V


def get_fusion(fusion_type, input_dims=None, output_dim=None, rank=16):
    if fusion_type == "tensor":
        return TensorFusion()
    elif fusion_type == "lowrank":
        return LowRankTensorFusion(input_dims, output_dim, rank)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")