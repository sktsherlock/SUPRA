"""
Semantic Attribution Analysis for Multimodal GNN Comparison
===========================================================

分类测试节点为4个互斥集合（基于单模态MLP预测）：
  - Shared:     Text MLP 和 Image MLP 都正确
  - T-Unique:  仅 Text MLP 正确
  - V-Unique:  仅 Image MLP 正确
  - Hard:      Text MLP 和 Image MLP 都错误

然后测量每个模型在每个集合上的绝对准确率贡献：
  contribution = (该集合中预测正确的节点数) / (总节点数)

用法:
  python -m GNN.Utils.semantic_attribution \
      --data_name Reddit-M \
      --text_feature /path/to/text.npy \
      --visual_feature /path/to/visual.npy \
      --graph_path /path/to/graph.pt \
      --result_csv Results/attribution/reddit_m.csv \
      --save_plot Results/attribution/reddit_m.png
"""

import argparse
import os
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from GNN.GraphData import load_data, set_seed
from GNN.Library.GCN import GCN
from GNN.Library.GAT import GAT
from GNN.Library.GraphSAGE import GraphSAGE
from GNN.Utils.LossFunction import cross_entropy, get_metric


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _as_scalar(x) -> float:
    if isinstance(x, th.Tensor):
        return x.item()
    if isinstance(x, (int, float)):
        return float(x)
    return float(np.asarray(x).mean())


def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)


# ---------------------------------------------------------------------------
# Core attribution logic
# ---------------------------------------------------------------------------

def classify_nodes(
    preds_text: th.Tensor,
    preds_image: th.Tensor,
    labels: th.Tensor,
) -> Dict[str, th.Tensor]:
    """
    根据单模态MLP预测将测试节点分类为4个互斥集合。
    """
    correct_t = preds_text == labels
    correct_v = preds_image == labels

    shared   = correct_t & correct_v
    t_unique = correct_t & (~correct_v)
    v_unique = (~correct_t) & correct_v
    hard     = ~correct_t & ~correct_v

    return {
        "shared":   shared,
        "t_unique": t_unique,
        "v_unique": v_unique,
        "hard":     hard,
    }


def compute_contributions(
    preds: th.Tensor,
    node_sets: Dict[str, th.Tensor],
    labels: th.Tensor,
    total_nodes: int,
) -> Dict[str, float]:
    """
    计算每个节点集合上的绝对准确率贡献（占总节点数的比例）。
    """
    correct = preds == labels
    return {
        "shared":   (correct & node_sets["shared"]).sum().item() / total_nodes,
        "t_unique": (correct & node_sets["t_unique"]).sum().item() / total_nodes,
        "v_unique": (correct & node_sets["v_unique"]).sum().item() / total_nodes,
        "hard":     (correct & node_sets["hard"]).sum().item() / total_nodes,
        "total":    correct.sum().item() / total_nodes,
    }


# ---------------------------------------------------------------------------
# Unified training loop
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.best_score = -1.0
        self.early_stop = False

    def step(self, score: float) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_model(
    model: nn.Module,
    graph,
    text_feat: th.Tensor,
    visual_feat: th.Tensor,
    train_idx: th.Tensor,
    val_idx: th.Tensor,
    test_idx: th.Tensor,
    labels: th.Tensor,
    device: th.device,
    n_epochs: int = 1000,
    lr: float = 0.0005,
    wd: float = 0.0001,
    patience: int = 20,
    label_smoothing: float = 0.1,
) -> th.Tensor:
    """
    统一训练函数，返回测试集 logits。
    """
    model = model.to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    stopper = EarlyStopping(patience=patience)

    best_test_logits = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward: 不同模型有不同的 forward 签名
        if isinstance(model, (GCN, GAT, GraphSAGE)):
            # Bimodal GNN (Early_GNN / Late_GNN 风格): concat → GNN
            x = th.cat([text_feat, visual_feat], dim=1)
            logits = model(graph, x)
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        loss = cross_entropy(logits[train_idx], labels[train_idx],
                           label_smoothing=label_smoothing)
        loss.backward()
        optimizer.step()

        # Evaluate
        if epoch % 1 == 0:
            model.eval()
            with th.no_grad():
                logits = model(graph, x)
            val_pred = th.argmax(logits[val_idx], dim=1)
            val_score = get_metric(val_pred, labels[val_idx], "accuracy")
            val_score = float(_as_scalar(val_score))

            if val_score > best_test_logits:
                best_test_logits = logits[test_idx].clone()

            if stopper.step(val_score):
                break

    model.eval()
    with th.no_grad():
        logits = model(graph, x)
    if best_test_logits is None:
        best_test_logits = logits[test_idx]
    return best_test_logits


def train_single_modality_mlp(
    text_feat: th.Tensor,
    visual_feat: th.Tensor,
    train_idx: th.Tensor,
    val_idx: th.Tensor,
    test_idx: th.Tensor,
    labels: th.Tensor,
    device: th.device,
    modality: str = "text",  # "text" or "visual"
    n_hidden: int = 256,
    n_epochs: int = 1000,
    lr: float = 0.0005,
    wd: float = 0.0001,
    patience: int = 20,
    label_smoothing: float = 0.1,
) -> th.Tensor:
    """
    训练单模态MLP（Text MLP 或 Image MLP），返回测试集 logits。
    """
    in_dim = text_feat.shape[1] if modality == "text" else visual_feat.shape[1]
    feat = text_feat if modality == "text" else visual_feat

    class MLP(nn.Module):
        def __init__(self, in_d, hidden, out):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_d, hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, out),
            )
        def forward(self, x):
            return self.net(x)

    model = MLP(in_dim, n_hidden, int(labels.max().item()) + 1).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    stopper = EarlyStopping(patience=patience)
    best_test_logits = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(feat)
        loss = cross_entropy(logits[train_idx], labels[train_idx],
                           label_smoothing=label_smoothing)
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            model.eval()
            with th.no_grad():
                logits_out = model(feat)
            val_pred = th.argmax(logits_out[val_idx], dim=1)
            val_score = get_metric(val_pred, labels[val_idx], "accuracy")
            val_score = float(_as_scalar(val_score))

            if best_test_logits is None or val_score > stopper.best_score:
                best_test_logits = logits_out[test_idx].clone()

            if stopper.step(val_score):
                break

    model.eval()
    with th.no_grad():
        logits_out = model(feat)
    if best_test_logits is None:
        best_test_logits = logits_out[test_idx]
    return best_test_logits


# ---------------------------------------------------------------------------
# Late_GNN (GCN / GAT backbone)
# ---------------------------------------------------------------------------

def build_late_gnn(
    text_feat: th.Tensor,
    visual_feat: th.Tensor,
    n_classes: int,
    backbone: str = "GCN",  # "GCN" or "GAT"
    n_hidden: int = 256,
    n_layers: int = 3,
    dropout: float = 0.3,
    device: th.device = None,
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """
    构建 Late_GNN: 两个独立单模态Encoder + 两个独立GNN + 融合头。
    返回 (text_encoder, visual_encoder, fusion_model)
    融合模型包含两个GNN分支，forward返回融合logits。
    """
    class LateFusionGNN(nn.Module):
        def __init__(self, text_enc, vis_enc, gnn_t, gnn_v, head):
            super().__init__()
            self.text_encoder = text_enc
            self.visual_encoder = vis_enc
            self.gnn_t = gnn_t
            self.gnn_v = gnn_v
            self.head = head

        def forward(self, graph, text_f, vis_f):
            h_t = self.text_encoder(text_f)
            h_v = self.visual_encoder(vis_f)
            z_t = self.gnn_t(graph, h_t)
            z_v = self.gnn_v(graph, h_v)
            z = (z_t + z_v) / 2
            return self.head(z)

    t_dim = text_feat.shape[1]
    v_dim = visual_feat.shape[1]

    # 模态编码器
    text_enc = nn.Sequential(
        nn.Linear(t_dim, n_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(n_hidden, n_hidden),
    )
    vis_enc = nn.Sequential(
        nn.Linear(v_dim, n_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(n_hidden, n_hidden),
    )

    # 两个独立 GNN
    if backbone == "GCN":
        gnn_t = GCN(n_hidden, n_hidden, n_classes, n_layers, F.relu, dropout).to(device)
        gnn_v = GCN(n_hidden, n_hidden, n_classes, n_layers, F.relu, dropout).to(device)
    elif backbone == "GAT":
        gnn_t = GAT(n_hidden, n_classes, n_hidden, n_layers, F.relu, dropout,
                    attn_drop=0.0, edge_drop=0.0).to(device)
        gnn_v = GAT(n_hidden, n_classes, n_hidden, n_layers, F.relu, dropout,
                    attn_drop=0.0, edge_drop=0.0).to(device)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    # 融合头
    head = nn.Linear(n_hidden, n_classes)

    return text_enc, vis_enc, LateFusionGNN(text_enc, vis_enc, gnn_t, gnn_v, head)


# ---------------------------------------------------------------------------
# Early_GNN (GCN backbone, concat fusion)
# ---------------------------------------------------------------------------

def build_early_gnn(
    text_feat: th.Tensor,
    visual_feat: th.Tensor,
    n_classes: int,
    backbone: str = "GCN",
    n_hidden: int = 256,
    n_layers: int = 3,
    dropout: float = 0.3,
    device: th.device = None,
) -> nn.Module:
    """
    Early_GNN: concat(enc_t, enc_v) → GNN → head
    """

    class EarlyFusionGNN(nn.Module):
        def __init__(self, enc_t, enc_v, gnn, head):
            super().__init__()
            self.enc_t = enc_t
            self.enc_v = enc_v
            self.gnn = gnn
            self.head = head

        def forward(self, graph, text_f, vis_f):
            h = th.cat([self.enc_t(text_f), self.enc_v(vis_f)], dim=1)
            z = self.gnn(graph, h)
            return self.head(z)

    t_dim = text_feat.shape[1]
    v_dim = visual_feat.shape[1]

    enc_t = nn.Sequential(
        nn.Linear(t_dim, n_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
    )
    enc_v = nn.Sequential(
        nn.Linear(v_dim, n_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
    )

    concat_dim = n_hidden * 2
    if backbone == "GCN":
        gnn = GCN(concat_dim, n_hidden, n_classes, n_layers, F.relu, dropout).to(device)
    elif backbone == "GAT":
        gnn = GAT(concat_dim, n_classes, n_hidden, n_layers, F.relu, dropout,
                   attn_drop=0.0, edge_drop=0.0).to(device)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    head = nn.Linear(n_hidden, n_classes)
    return EarlyFusionGNN(enc_t, enc_v, gnn, head)


# ---------------------------------------------------------------------------
# Main attribution analysis
# ---------------------------------------------------------------------------

def run_attribution_analysis(args):
    set_seed(args.seed)
    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load data
    graph, labels, train_idx, val_idx, test_idx = load_data(
        args.graph_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        name=args.data_name,
        fewshots=getattr(args, "fewshots", None),
    )
    graph = graph.to(device)
    labels = labels.to(device)

    text_feat = th.as_tensor(np.load(args.text_feature)).float().to(device)
    visual_feat = th.as_tensor(np.load(args.visual_feature)).float().to(device)

    n_classes = int(labels.max().item()) + 1
    n_test = test_idx.shape[0]
    print(f"[Data] {args.data_name}: {n_test} test nodes, {n_classes} classes")

    # Hyperparameters
    n_hidden = getattr(args, "n_hidden", 256)
    n_layers = getattr(args, "n_layers", 3)
    dropout = getattr(args, "dropout", 0.3)
    lr = getattr(args, "lr", 0.0005)
    wd = getattr(args, "wd", 0.0001)
    n_epochs = getattr(args, "n_epochs", 1000)
    patience = getattr(args, "early_stop_patience", 20)

    # Step 1: Text MLP and Image MLP
    print("\n[1/7] Training Text MLP...")
    preds_text = train_single_modality_mlp(
        text_feat, visual_feat, train_idx, val_idx, test_idx, labels, device,
        modality="text", n_hidden=n_hidden, n_epochs=n_epochs, lr=lr, wd=wd, patience=patience,
    )
    preds_text = th.argmax(preds_text, dim=1)

    print("[2/7] Training Image MLP...")
    preds_image = train_single_modality_mlp(
        text_feat, visual_feat, train_idx, val_idx, test_idx, labels, device,
        modality="visual", n_hidden=n_hidden, n_epochs=n_epochs, lr=lr, wd=wd, patience=patience,
    )
    preds_image = th.argmax(preds_image, dim=1)

    # Step 2: Classify nodes
    labels_test = labels[test_idx]
    node_sets = classify_nodes(preds_text, preds_image, labels_test)
    set_sizes = {k: v.sum().item() / n_test for k, v in node_sets.items()}
    print(f"\n[Node Sets] Shared={set_sizes['shared']:.1%}  "
          f"T-Unique={set_sizes['t_unique']:.1%}  "
          f"V-Unique={set_sizes['v_unique']:.1%}  "
          f"Hard={set_sizes['hard']:.1%}")

    # Step 3: Train multimodal models
    results = {}

    def collect(name: str, preds: th.Tensor):
        contrib = compute_contributions(preds, node_sets, labels_test, n_test)
        results[name] = contrib
        print(f"  {name:20s}: total={contrib['total']:.4f}  "
              f"(shared={contrib['shared']:.4f}, t={contrib['t_unique']:.4f}, "
              f"v={contrib['v_unique']:.4f}, hard={contrib['hard']:.4f})")

    collect("Text MLP", preds_text)
    collect("Image MLP", preds_image)

    # Late_GNN-GCN
    print("\n[3/7] Training Late_GNN-GCN...")
    _, _, late_gcn = build_late_gnn(
        text_feat, visual_feat, n_classes, backbone="GCN",
        n_hidden=n_hidden, n_layers=n_layers, dropout=dropout, device=device,
    )
    logits_late_gcn = train_model(
        late_gcn, graph, text_feat, visual_feat,
        train_idx, val_idx, test_idx, labels, device,
        n_epochs=n_epochs, lr=lr, wd=wd, patience=patience,
    )
    collect("Late_GNN-GCN", th.argmax(logits_late_gcn, dim=1))

    # Late_GNN-GAT
    print("\n[4/7] Training Late_GNN-GAT...")
    _, _, late_gat = build_late_gnn(
        text_feat, visual_feat, n_classes, backbone="GAT",
        n_hidden=n_hidden, n_layers=n_layers, dropout=dropout, device=device,
    )
    logits_late_gat = train_model(
        late_gat, graph, text_feat, visual_feat,
        train_idx, val_idx, test_idx, labels, device,
        n_epochs=n_epochs, lr=lr, wd=wd, patience=patience,
    )
    collect("Late_GNN-GAT", th.argmax(logits_late_gat, dim=1))

    # Early_GNN-GCN (for reference — same arch as MMGCN)
    print("\n[5/7] Training Early_GNN-GCN...")
    early_gcn = build_early_gnn(
        text_feat, visual_feat, n_classes, backbone="GCN",
        n_hidden=n_hidden, n_layers=n_layers, dropout=dropout, device=device,
    )
    logits_early_gcn = train_model(
        early_gcn, graph, text_feat, visual_feat,
        train_idx, val_idx, test_idx, labels, device,
        n_epochs=n_epochs, lr=lr, wd=wd, patience=patience,
    )
    collect("Early_GNN-GCN", th.argmax(logits_early_gcn, dim=1))

    # Note: NTSFormer, MIG_GT, and SUPRA require the full training infrastructure
    # which is complex. For a fair comparison in the attribution analysis,
    # we recommend using the Late_GNN and Early_GNN models which share the same
    # architecture patterns. The key comparison for the paper is:
    #   - Text MLP / Image MLP: single modality baselines
    #   - Late_GNN-GCN / Late_GNN-GAT: late fusion multimodal GNN
    #   - Early_GNN-GCN: early fusion multimodal GNN (MMGCN-style)
    #   - SUPRA: our three-channel architecture (run separately and add results)

    print("\n" + "=" * 80)
    print(f"{'Model':<20} {'Shared':>9} {'T-Unique':>9} {'V-Unique':>9} {'Hard':>9} {'Total':>9}")
    print("=" * 80)
    for name, contrib in results.items():
        print(f"{name:<20} {contrib['shared']:>9.4f} {contrib['t_unique']:>9.4f} "
              f"{contrib['v_unique']:>9.4f} {contrib['hard']:>9.4f} {contrib['total']:>9.4f}")
    print("=" * 80)

    # Save CSV
    if getattr(args, "result_csv", None):
        import csv
        os.makedirs(os.path.dirname(args.result_csv), exist_ok=True)
        with open(args.result_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Shared", "T-Unique", "V-Unique", "Hard", "Total"])
            for name, contrib in results.items():
                writer.writerow([name,
                    f"{contrib['shared']:.4f}", f"{contrib['t_unique']:.4f}",
                    f"{contrib['v_unique']:.4f}", f"{contrib['hard']:.4f}",
                    f"{contrib['total']:.4f}"])
        print(f"[Saved] {args.result_csv}")

    # Plot
    if getattr(args, "plot", True):
        plot_stacked_bar(results, set_sizes, args.data_name,
                         getattr(args, "save_plot", None))

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_stacked_bar(
    results: Dict[str, Dict[str, float]],
    set_sizes: Dict[str, float],
    data_name: str,
    save_path: Optional[str] = None,
):
    """
    绘制堆叠柱状图。
    X轴: 模型
    Y轴: 绝对准确率贡献（各堆叠块之和 = 模型总准确率）
    颜色: Shared(灰) / T-Unique(蓝) / V-Unique(橙) / Hard(绿)
    """
    models = list(results.keys())
    n_models = len(models)

    colors = {
        "shared":   "#A0A0A0",
        "t_unique": "#4C72B0",
        "v_unique": "#DD8452",
        "hard":     "#55A868",
    }
    labels_text = {
        "shared":   "Shared Semantics",
        "t_unique": "Text-Unique",
        "v_unique": "Visual-Unique",
        "hard":     "Hard / Synergy",
    }

    bar_width = 0.6
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.6), 6))

    bottom = np.zeros(n_models)
    for set_key in ["shared", "t_unique", "v_unique", "hard"]:
        heights = [results[m][set_key] for m in models]
        ax.bar(x, heights, bar_width, bottom=bottom,
               label=labels_text[set_key], color=colors[set_key],
               edgecolor="white", linewidth=0.5)
        bottom += np.array(heights)

    # 顶部标注总准确率
    for i, m in enumerate(models):
        ax.text(i, bottom[i] + 0.005, f"{results[m]['total']:.1%}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Absolute Accuracy Contribution", fontsize=11)
    ax.set_title(f"Semantic Attribution Analysis — {data_name}", fontsize=13, fontweight="bold")
    ax.set_ylim(0, min(bottom.max() * 1.15, 1.0))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] Plot → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Attribution Analysis")
    parser.add_argument("--data_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--text_feature", type=str, required=True, help="Path to text features (.npy)")
    parser.add_argument("--visual_feature", type=str, required=True, help="Path to visual features (.npy)")
    parser.add_argument("--graph_path", type=str, required=True, help="Path to graph (.pt)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--wd", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--result_csv", type=str, default=None, help="Save CSV results to this path")
    parser.add_argument("--save_plot", type=str, default=None, help="Save plot to this path")
    parser.add_argument("--plot", type=str2bool, default=True)

    args = parser.parse_args()
    run_attribution_analysis(args)
