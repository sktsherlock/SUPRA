import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GNN.SUPRA import SUPRA
from GNN.Baselines.Early_GNN import Early_GNN
from GNN.Baselines.Late_GNN import LateFusionMAG

__all__ = ["SUPRA", "Early_GNN", "LateFusionMAG"]

