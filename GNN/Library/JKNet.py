"""JKNet: Jump Knowledge Network for multi-scale representation learning.
Based on "Representation Learning on Graphs with Jumping Knowledge Networks" (ICML 2018).

JKNet aggregates representations from all layers using configurable strategies:
- concat: concatenate all layer outputs (JKNetConcat)
- max: max pooling across layers (JKNetMaxpool)
- last: use only the last layer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn


class JKNet(nn.Module):
    """JKNet for node classification.

    The jumping knowledge aggregation is applied before the final classification layer.

    Args:
        in_feats: Input feature dimension
        n_hidden: Hidden dimension for all layers
        n_classes: Number of output classes
        n_layers: Number of GNN layers
        dropout: Dropout rate
        aggr: Aggregation strategy for jumping knowledge ('concat', 'max', 'last')
    """

    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
        dropout,
        aggr="concat",
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.aggr = aggr

        # Build each graph convolution layer
        # First layer: in_feats -> n_hidden
        # Subsequent layers: n_hidden -> n_hidden
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            in_dim = n_hidden if i > 0 else in_feats
            self.convs.append(
                dglnn.GraphConv(in_dim, n_hidden, "both")
            )

        self.dropout = nn.Dropout(dropout)

        # Calculate aggregation output dimension
        if aggr == "concat":
            self.aggr_dim = n_layers * n_hidden
        elif aggr == "max":
            self.aggr_dim = n_hidden
        else:  # last
            self.aggr_dim = n_hidden

        # Final classifier
        self.classifier = nn.Linear(self.aggr_dim, n_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, graph, feat):
        """Forward pass.

        Args:
            graph: DGL graph
            feat: Input features [N, in_feats]

        Returns:
            Output logits [N, n_classes]
        """
        x = feat
        layer_outputs = []

        for i in range(self.n_layers):
            # Graph convolution
            x = self.convs[i](graph, x)
            # ReLU activation
            x = F.relu(x)
            # Dropout
            x = self.dropout(x)
            # Save output for jumping knowledge
            layer_outputs.append(x)

        # Jumping knowledge aggregation
        if self.aggr == "last":
            h_aggr = layer_outputs[-1]
        elif self.aggr == "max":
            stack = torch.stack(layer_outputs, dim=0)  # [n_layers, N, hidden]
            h_aggr = torch.max(stack, dim=0)[0]
        else:  # concat
            h_aggr = torch.cat(layer_outputs, dim=1)  # [N, n_layers * hidden]

        return self.classifier(h_aggr)
