# 双Y轴单曲线，展示epoch变化时秩与准确率。

import argparse
import os
from typing import List

import numpy as np
import torch as th
import matplotlib.pyplot as plt
from tqdm import tqdm

from GNN.GraphData import load_data, set_seed
from GNN.Utils.LossFunction import cross_entropy, get_metric
from GNN.Utils.rank_analysis import compute_effective_rank

from GNN.Baselines.Early_GNN import Early_GNN as mag_base
from GNN.Baselines.Early_GNN import ModalityEncoder, SimpleMAGGNN, _make_observe_graph_inductive
from GNN.Baselines.Late_GNN import LateFusionMAG


def _extract_gnn_embedding(gnn, graph, feat: th.Tensor) -> th.Tensor:
    """Return penultimate embeddings when possible; fallback to logits."""
    if hasattr(gnn, "convs") and hasattr(gnn, "n_layers") and int(gnn.n_layers) > 1:
        h = feat
        for i in range(int(gnn.n_layers) - 1):
            h = gnn.convs[i](graph, h)
            if hasattr(gnn, "norms") and i < len(gnn.norms):
                h = gnn.norms[i](h)
            if hasattr(gnn, "activation"):
                h = gnn.activation(h)
            if hasattr(gnn, "dropout"):
                h = gnn.dropout(h)
        return h
    if hasattr(gnn, "linears") and hasattr(gnn, "n_layers") and int(gnn.n_layers) > 1:
        h = feat
        for i in range(int(gnn.n_layers) - 1):
            h = gnn.linears[i](h)
            if hasattr(gnn, "norms") and i < len(gnn.norms):
                h = gnn.norms[i](h)
            if hasattr(gnn, "activation"):
                h = gnn.activation(h)
            if hasattr(gnn, "dropout"):
                h = gnn.dropout(h)
        return h
    return gnn(graph, feat)


def _get_eval_idx(args, train_idx, val_idx, test_idx) -> th.Tensor:
    if args.eval_split == "train":
        return train_idx
    if args.eval_split == "val":
        return val_idx
    return test_idx


def _build_plain(args, in_dim: int, n_classes: int, device: th.device):
    model = mag_base._build_gnn_backbone(args, in_dim, n_classes, device)
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()
    return model


def _build_early(args, text_dim: int, vis_dim: int, n_classes: int, device: th.device):
    proj_dim = int(args.mm_proj_dim) if args.mm_proj_dim is not None else int(args.n_hidden)
    text_encoder = ModalityEncoder(text_dim, proj_dim, args.dropout).to(device)
    visual_encoder = ModalityEncoder(vis_dim, proj_dim, args.dropout).to(device)
    gnn = mag_base._build_gnn_backbone(args, 2 * proj_dim, n_classes, device)
    model = SimpleMAGGNN(text_encoder, visual_encoder, gnn, modality_dropout=float(args.modality_dropout or 0.0))
    model = model.to(device)
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()
    return model


def _build_late(args, text_dim: int, vis_dim: int, n_classes: int, device: th.device):
    embed_dim = int(args.late_embed_dim) if args.late_embed_dim is not None else int(args.n_hidden)
    text_gnn = mag_base._build_gnn_backbone(args, text_dim, embed_dim, device)
    vis_gnn = mag_base._build_gnn_backbone(args, vis_dim, embed_dim, device)
    classifier = th.nn.Linear(embed_dim, n_classes).to(device)
    model = LateFusionMAG(text_gnn, vis_gnn, classifier, modality_dropout=float(args.modality_dropout or 0.0))
    model = model.to(device)
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()
    return model


def _compute_rank_plain(model, graph, feat, eval_idx: th.Tensor) -> float:
    emb = _extract_gnn_embedding(model, graph, feat)
    emb = emb[eval_idx]
    return compute_effective_rank(emb)


def _compute_rank_early(model, graph, text_feat, vis_feat, eval_idx: th.Tensor) -> float:
    text_h = model.text_encoder(text_feat)
    vis_h = model.visual_encoder(vis_feat)
    feat = th.cat([text_h, vis_h], dim=1)
    emb = _extract_gnn_embedding(model.gnn, graph, feat)
    emb = emb[eval_idx]
    return compute_effective_rank(emb)


def _compute_rank_late(model, graph, text_feat, vis_feat, eval_idx: th.Tensor) -> float:
    text_h, vis_h = model.forward_branches(graph, text_feat, vis_feat)
    fused = model.fuse_embeddings(text_h, vis_h)
    fused = fused[eval_idx]
    return compute_effective_rank(fused)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["plain", "early", "late"])
    parser.add_argument("--data_name", type=str, default="MAGDataset")
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--text_feature", type=str, required=True)
    parser.add_argument("--visual_feature", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="GCN", choices=["GCN", "SAGE"])
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--metric", type=str, default="accuracy")
    parser.add_argument("--average", type=str, default="macro")

    parser.add_argument("--mm_proj_dim", type=int, default=None)
    parser.add_argument("--late_embed_dim", type=int, default=None)
    parser.add_argument("--modality_dropout", type=float, default=0.0)

    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--selfloop", action="store_true")
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--inductive", action="store_true")

    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default="./figures")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() and args.gpu >= 0 else "cpu")

    graph, labels, train_idx, val_idx, test_idx = load_data(
        args.graph_path, train_ratio=args.train_ratio, val_ratio=args.val_ratio, name=args.data_name
    )

    if args.undirected:
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)

    observe_graph = graph
    if args.inductive:
        observe_graph = _make_observe_graph_inductive(graph, val_idx, test_idx)

    if args.selfloop:
        graph = graph.remove_self_loop().add_self_loop()
        observe_graph = observe_graph.remove_self_loop().add_self_loop()

    graph.create_formats_()
    observe_graph.create_formats_()

    graph = graph.to(device)
    observe_graph = observe_graph.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

    text_feat = th.from_numpy(np.load(args.text_feature).astype(np.float32)).to(device)
    vis_feat = th.from_numpy(np.load(args.visual_feature).astype(np.float32)).to(device)

    n_classes = int((labels.max() + 1).item())

    if args.model_type == "plain":
        feat = th.cat([text_feat, vis_feat], dim=1)
        model = _build_plain(args, feat.shape[1], n_classes, device)
    elif args.model_type == "early":
        model = _build_early(args, text_feat.shape[1], vis_feat.shape[1], n_classes, device)
    else:
        model = _build_late(args, text_feat.shape[1], vis_feat.shape[1], n_classes, device)

    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    eval_idx = _get_eval_idx(args, train_idx, val_idx, test_idx)

    epochs: List[int] = []
    ranks: List[float] = []
    metrics: List[float] = []

    iterator = range(1, args.n_epochs + 1)
    for epoch in tqdm(iterator, desc="Training"):
        model.train()
        optimizer.zero_grad()

        if args.model_type == "plain":
            logits = model(observe_graph, feat)
        else:
            logits = model(observe_graph, text_feat, vis_feat)

        loss = cross_entropy(logits[train_idx], labels[train_idx], label_smoothing=args.label_smoothing)
        loss.backward()
        optimizer.step()

        if epoch % args.eval_steps == 0 or epoch == args.n_epochs:
            model.eval()
            with th.no_grad():
                if args.model_type == "plain":
                    rank = _compute_rank_plain(model, graph, feat, eval_idx)
                    logits_eval = model(graph, feat)
                elif args.model_type == "early":
                    rank = _compute_rank_early(model, graph, text_feat, vis_feat, eval_idx)
                    logits_eval = model(graph, text_feat, vis_feat)
                else:
                    rank = _compute_rank_late(model, graph, text_feat, vis_feat, eval_idx)
                    logits_eval = model(graph, text_feat, vis_feat)

                metric_val = get_metric(
                    th.argmax(logits_eval[eval_idx], dim=1),
                    labels[eval_idx],
                    args.metric,
                    average=args.average,
                )

            epochs.append(epoch)
            ranks.append(float(rank))
            metrics.append(float(metric_val))
            print(f"[Epoch {epoch:03d}] {args.metric}: {float(metric_val):.4f} | Rank: {float(rank):.2f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = args.output_name
    if not out_name:
        out_name = f"rank_over_training_{args.model_type}.pdf"
    out_path = os.path.join(args.output_dir, out_name)

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    ax1.plot(epochs, ranks, "o-", linewidth=2, color="C0", label="Effective Rank")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Effective Rank", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, metrics, "s--", linewidth=1.5, color="C1", label=args.metric)
    ax2.set_ylabel(args.metric, color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    title = f"{args.model_type.upper()} rank during training ({args.eval_split})"
    ax1.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved plot to {out_path}")

    if metrics:
        print(f"Final {args.metric}: {metrics[-1]:.4f}")

    if args.save_csv:
        csv_path = os.path.splitext(out_path)[0] + ".csv"
        data = np.column_stack([np.array(epochs), np.array(ranks), np.array(metrics)])
        header = "epoch,rank,metric"
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
        print(f"✓ Saved csv to {csv_path}")


if __name__ == "__main__":
    main()
