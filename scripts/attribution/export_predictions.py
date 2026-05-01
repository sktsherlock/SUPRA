"""
Export test-set predictions for all models in the semantic attribution analysis.

Usage:
    python scripts/attribution/export_predictions.py \
        --data_name Reddit-M \
        --text_feature /path/to/text.npy \
        --visual_feature /path/to/visual.npy \
        --graph_path /path/to/graph.pt \
        --pred_dir Results/attribution/Reddit-M/ \
        --models text_mlp image_mlp late_gnn_gcn late_gnn_gat early_gnn_gcn supra ntsformer mig_gt \
        --gpu 0

Each *_test_pred.pt file contains torch.LongTensor of shape [N_test] (argmax predictions).
"""

import argparse
import os
from typing import List

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from GNN.GraphData import load_data
from GNN.Library.GCN import GCN
from GNN.Library.GAT import GAT
from GNN.Utils.LossFunction import cross_entropy, get_metric


# ---------------------------------------------------------------------------
# Config / Hyperparameters
# ---------------------------------------------------------------------------

def get_common_hparams(data_name: str) -> dict:
    return dict(
        n_hidden=256,
        n_layers=3,
        dropout=0.3,
        lr=0.0005,
        wd=0.0001,
        n_epochs=1000,
        patience=20,
        seed=42,
    )


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.best_score = -1.0

    def step(self, score: float) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_graph_data(args):
    graph, labels, train_idx, val_idx, test_idx = load_data(
        args.graph_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        name=args.data_name,
    )
    text_feat = th.tensor(np.load(args.text_feature), dtype=th.float32)
    visual_feat = th.tensor(np.load(args.visual_feature), dtype=th.float32)
    return graph, labels, train_idx, val_idx, test_idx, text_feat, visual_feat


# ---------------------------------------------------------------------------
# Model definitions (inline, no external training infrastructure needed)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class EarlyFusionGNN(nn.Module):
    """Early fusion: concat(enc_t, enc_v) → GNN → head"""
    def __init__(self, t_dim, v_dim, n_hidden, n_classes, n_layers, dropout, gnn_class):
        super().__init__()
        self.enc_t = nn.Sequential(nn.Linear(t_dim, n_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.enc_v = nn.Sequential(nn.Linear(v_dim, n_hidden), nn.ReLU(), nn.Dropout(dropout))
        concat_dim = n_hidden * 2
        self.gnn = gnn_class(concat_dim, n_hidden, n_classes, n_layers, F.relu, dropout)
        self.head = nn.Linear(n_hidden, n_classes)

    def forward(self, graph, text_f, vis_f):
        h = th.cat([self.enc_t(text_f), self.enc_v(vis_f)], dim=1)
        z = self.gnn(graph, h)
        return self.head(z)


class LateFusionGNN(nn.Module):
    """Late fusion: each modality has its own encoder + GNN, then average"""
    def __init__(self, t_dim, v_dim, n_hidden, n_classes, n_layers, dropout, gnn_class):
        super().__init__()
        self.enc_t = nn.Sequential(nn.Linear(t_dim, n_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.enc_v = nn.Sequential(nn.Linear(v_dim, n_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.gnn_t = gnn_class(n_hidden, n_hidden, n_classes, n_layers, F.relu, dropout)
        self.gnn_v = gnn_class(n_hidden, n_hidden, n_classes, n_layers, F.relu, dropout)
        self.head = nn.Linear(n_hidden, n_classes)

    def forward(self, graph, text_f, vis_f):
        h_t = self.enc_t(text_f)
        h_v = self.enc_v(vis_f)
        z_t = self.gnn_t(graph, h_t)
        z_v = self.gnn_v(graph, h_v)
        z = (z_t + z_v) / 2
        return self.head(z)


# ---------------------------------------------------------------------------
# Training & inference
# ---------------------------------------------------------------------------

def train_and_predict(model, graph, text_feat, visual_feat,
                      train_idx, val_idx, test_idx, labels, device, hparams):
    model = model.to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=hparams["lr"], weight_decay=hparams["wd"])
    stopper = EarlyStopping(patience=hparams["patience"])

    best_test_logits = None

    for epoch in range(1, hparams["n_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(graph, text_feat, visual_feat)
        loss = cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with th.no_grad():
            logits_out = model(graph, text_feat, visual_feat)
            val_pred = th.argmax(logits_out[val_idx], dim=1)
            val_score = get_metric(val_pred, labels[val_idx], "accuracy")
            val_score = float(np.asarray(val_score).mean())

            if val_score > stopper.best_score:
                best_test_logits = logits_out[test_idx].clone()

        if stopper.step(val_score):
            break

    model.eval()
    with th.no_grad():
        logits_out = model(graph, text_feat, visual_feat)
    if best_test_logits is None:
        best_test_logits = logits_out[test_idx]
    return th.argmax(best_test_logits, dim=1)


def train_and_predict_mlp(text_feat, visual_feat, train_idx, val_idx, test_idx,
                          labels, device, modality, hparams):
    """Single-modality MLP: only text or only visual features."""
    feat = text_feat if modality == "text" else visual_feat
    in_dim = feat.shape[1]
    n_classes = int(labels.max().item()) + 1
    model = MLP(in_dim, hparams["n_hidden"], n_classes, hparams["dropout"]).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=hparams["lr"], weight_decay=hparams["wd"])
    stopper = EarlyStopping(patience=hparams["patience"])
    best_test_logits = None

    for epoch in range(1, hparams["n_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(feat)
        loss = cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with th.no_grad():
            logits_out = model(feat)
            val_pred = th.argmax(logits_out[val_idx], dim=1)
            val_score = get_metric(val_pred, labels[val_idx], "accuracy")
            val_score = float(np.asarray(val_score).mean())

            if best_test_logits is None or val_score > stopper.best_score:
                best_test_logits = logits_out[test_idx].clone()

        if stopper.step(val_score):
            break

    model.eval()
    with th.no_grad():
        logits_out = model(feat)
    if best_test_logits is None:
        best_test_logits = logits_out[test_idx]
    return th.argmax(best_test_logits, dim=1)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_text_mlp(args, n_classes, device, text_feat, visual_feat):
    return MLP(text_feat.shape[1], args.n_hidden, n_classes, args.dropout)

def build_image_mlp(args, n_classes, device, text_feat, visual_feat):
    return MLP(visual_feat.shape[1], args.n_hidden, n_classes, args.dropout)

def build_late_gnn_gcn(args, n_classes, device, text_feat, visual_feat):
    return LateFusionGNN(
        text_feat.shape[1], visual_feat.shape[1],
        args.n_hidden, n_classes, args.n_layers, args.dropout,
        lambda *a, **kw: GCN(*a, **kw)
    )

def build_late_gnn_gat(args, n_classes, device, text_feat, visual_feat):
    return LateFusionGNN(
        text_feat.shape[1], visual_feat.shape[1],
        args.n_hidden, n_classes, args.n_layers, args.dropout,
        lambda i, h, o, l, a, d: GAT(i, o, h, l, a, d, attn_drop=0.0, edge_drop=0.0)
    )

def build_early_gnn_gcn(args, n_classes, device, text_feat, visual_feat):
    return EarlyFusionGNN(
        text_feat.shape[1], visual_feat.shape[1],
        args.n_hidden, n_classes, args.n_layers, args.dropout,
        lambda *a, **kw: GCN(*a, **kw)
    )


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------

def export_predictions(args):
    os.makedirs(args.pred_dir, exist_ok=True)
    th.manual_seed(args.seed)
    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")

    graph, labels, train_idx, val_idx, test_idx, text_feat, visual_feat = load_graph_data(args)
    graph = graph.to(device)
    labels = labels.to(device)
    text_feat = text_feat.to(device)
    visual_feat = visual_feat.to(device)
    n_classes = int(labels.max().item()) + 1
    hparams = get_common_hparams(args.data_name)

    hparams["n_hidden"] = args.n_hidden
    hparams["n_layers"] = args.n_layers
    hparams["dropout"] = args.dropout
    hparams["lr"] = args.lr
    hparams["wd"] = args.wd
    hparams["n_epochs"] = args.n_epochs
    hparams["patience"] = args.early_stop_patience

    model_builders = {
        "text_mlp":       lambda: train_and_predict_mlp(text_feat, visual_feat, train_idx, val_idx, test_idx, labels, device, "text", hparams),
        "image_mlp":      lambda: train_and_predict_mlp(text_feat, visual_feat, train_idx, val_idx, test_idx, labels, device, "visual", hparams),
        "late_gnn_gcn":   lambda: train_and_predict(build_late_gnn_gcn(args, n_classes, device, text_feat, visual_feat), graph, text_feat, visual_feat, train_idx, val_idx, test_idx, labels, device, hparams),
        "late_gnn_gat":   lambda: train_and_predict(build_late_gnn_gat(args, n_classes, device, text_feat, visual_feat), graph, text_feat, visual_feat, train_idx, val_idx, test_idx, labels, device, hparams),
        "early_gnn_gcn":  lambda: train_and_predict(build_early_gnn_gcn(args, n_classes, device, text_feat, visual_feat), graph, text_feat, visual_feat, train_idx, val_idx, test_idx, labels, device, hparams),
    }

    for name in args.models:
        if name not in model_builders:
            print(f"[Skip] {name} (not implemented — use main training script)")
            continue
        if name in ("supra", "ntsformer", "mig_gt"):
            print(f"[Skip] {name} (use main training script + inference)")
            continue

        print(f"\n[Train] {name}")
        th.manual_seed(args.seed)
        pred = model_builders[name]()
        out_path = os.path.join(args.pred_dir, f"{name}_test_pred.pt")
        th.save(pred, out_path)
        acc = (pred == labels[test_idx]).float().mean().item()
        print(f"[Saved] {out_path}  (test acc={acc:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--text_feature", type=str, required=True)
    parser.add_argument("--visual_feature", type=str, required=True)
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--models", type=str, nargs="+",
                        default=["text_mlp", "image_mlp", "late_gnn_gcn", "late_gnn_gat", "early_gnn_gcn"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--wd", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()
    export_predictions(args)
