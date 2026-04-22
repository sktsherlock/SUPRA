"""GCNII: Graph Convolutional Networks with Initial Residual and Identity Mapping.
Based on "Simple and Deep Graph Convolutional Networks" (ICML 2020).

Adapted from PyG implementation to DGL.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn


class GraphConvolution(nn.Module):
    """GCNII Graph Convolution layer with initial residual and identity mapping."""

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super().__init__()
        self.variant = variant
        if variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.weight = nn.Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj, h0, lamda, alpha, l):
        """Forward pass.

        Args:
            inputs: Node features [N, in_features]
            adj: DGL graph
            h0: Initial features (x_0) [N, in_features]
            lamda: Damping factor
            alpha: Residual coefficient
            l: Current layer index (1-based)
        """
        theta = math.log(lamda / l + 1)

        # DGL graph: use message passing API for normalized aggregation
        g = adj
        if not g.is_homogeneous:
            raise ValueError("GCNII only supports homogeneous graphs")

        # Store features in local graph (avoids modifying the original graph)
        g = g.local_var()
        g.ndata['h'] = inputs
        g.ndata['h0'] = h0

        # Symmetric normalization: D^{-1/2} A D^{-1/2} for undirected graphs
        deg = g.in_degrees().float().clamp_min(1)
        deg_inv_sqrt = deg.pow(-0.5).view(-1, 1)  # [N, 1] for broadcasting with [E, D]
        g.ndata['deg_norm'] = deg_inv_sqrt
        # Message: src_h * deg_inv_sqrt(src) -> [E, D] * [E, 1] broadcasts to [E, D]
        g.update_all(
            lambda edges: {'m': edges.src['h'] * edges.src['deg_norm']},
            dgl.function.sum('m', 'h_acc')
        )
        # Receive: multiply by deg_inv_sqrt(dst) for symmetric normalization: [N, D] * [N, 1] -> [N, D]
        h_agg = g.ndata['h_acc'] * g.ndata['deg_norm']

        # Clean up temp features
        g.ndata.pop('h0')
        g.ndata.pop('deg_norm')

        hi = h_agg

        if self.variant:
            support = torch.cat([hi, h0], dim=1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support

        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + inputs
        return output


class GCNII(nn.Module):
    """GCNII for node classification.

    Args:
        nfeat: Input feature dimension
        nlayers: Number of hidden layers
        nhidden: Hidden dimension
        nclass: Number of output classes
        dropout: Dropout rate
        lamda: Damping factor for initial residual
        alpha: Coefficient for initial residual
        variant: Whether to use variant (with concatenated initial features)
    """

    def __init__(
        self,
        nfeat,
        nlayers,
        nhidden,
        nclass,
        dropout,
        lamda=0.5,
        alpha=0.5,
        variant=False,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(
                GraphConvolution(nhidden, nhidden, variant=variant)
            )
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        """Forward pass.

        Args:
            x: Input features [N, nfeat]
            adj: DGL graph or sparse adjacency matrix
        """
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)

        for i, conv in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                conv(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1)
            )

        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)


class GCNIIppi(nn.Module):
    """GCNII with sigmoid output for link prediction tasks."""

    def __init__(
        self,
        nfeat,
        nlayers,
        nhidden,
        nclass,
        dropout,
        lamda=0.5,
        alpha=0.5,
        variant=False,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(
                GraphConvolution(
                    nhidden, nhidden, variant=variant, residual=True
                )
            )
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)

        for i, conv in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                conv(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1)
            )

        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.sigmoid(self.fcs[-1](layer_inner))
        return layer_inner
