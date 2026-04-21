"""Test OGM-GNN: train per beta and plot rank/similarity curves (no wandb).

- Trains unimodal references by zeroing the other modality.
- For each beta, re-trains OGM-GNN with text/visual upweighting.
- Plots: rank (text/visual upweight) + similarity under text/visual upweight.
放大某一模态的特征输入（例如文本或视觉），观察模型性能、表示相似性和多模态秩的变化趋势。
"""

import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from GNN.GraphData import load_data, set_seed
from GNN.Baselines.Early_GNN import Early_GNN as mag_base
# OGM_GNN module no longer exists
# from GNN.Library.MAG.OGM_GNN import OGM_GNN, train_epoch, evaluate
from GNN.Utils.rank_analysis import compute_effective_rank, linear_cka


def _build_args(args):
    return SimpleNamespace(
        # common
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        dropout=args.dropout,
        model_name=args.model_name,
        lr=args.lr,
        wd=args.wd,
        min_lr=args.min_lr,
        label_smoothing=args.label_smoothing,
        metric=args.metric,
        average=args.average,
        log_every=args.log_every,
        # ogm
        backend=args.backend,
        ogm_modulation=args.ogm_modulation,
        ogm_modulation_starts=args.ogm_modulation_starts,
        ogm_modulation_ends=args.ogm_modulation_ends,
        ogm_alpha=args.ogm_alpha,
        ogm_projection_dim=args.ogm_projection_dim,
    )


def _train_one_ogm(
    args_ns,
    graph,
    observe_graph,
    text_feat,
    visual_feat,
    labels,
    train_idx,
    val_idx,
    test_idx,
    device,
    seed_offset,
):
    set_seed(int(args_ns.seed) + int(seed_offset))
    model = OGM_GNN(args_ns, text_feat.shape[1], visual_feat.shape[1], int(labels.max().item()) + 1, device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args_ns.lr, weight_decay=args_ns.wd)
    best_val = -1.0
    best_state = None

    for epoch in range(1, args_ns.n_epochs + 1):
        train_epoch(args_ns, epoch, model, observe_graph, text_feat, visual_feat, labels, train_idx, optimizer)
        if epoch % args_ns.eval_steps == 0:
            train_result, val_result, test_result, _, _, _ = evaluate(
                model, graph, text_feat, visual_feat, labels, train_idx, val_idx, test_idx, args_ns
            )
            if float(val_result) > best_val:
                best_val = float(val_result)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        _, t, v = model(graph, text_feat, visual_feat)
        fused = 0.5 * (t + v)
        fused = fused[test_idx]

    return fused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="MAGDataset")
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--text_feature", type=str, required=True)
    parser.add_argument("--visual_feature", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="GCN", choices=["GCN", "SAGE"])
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--metric", type=str, default="accuracy")
    parser.add_argument("--average", type=str, default="macro")
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument("--backend", type=str, default="gnn", choices=["gnn", "mlp"])
    parser.add_argument("--ogm_modulation", type=str, default="Normal", choices=["Normal", "OGM", "OGM_GE"])
    parser.add_argument("--ogm_modulation_starts", type=int, default=0)
    parser.add_argument("--ogm_modulation_ends", type=int, default=50)
    parser.add_argument("--ogm_alpha", type=float, default=1.0)
    parser.add_argument("--ogm_projection_dim", type=int, default=512)

    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--selfloop", action="store_true")
    parser.add_argument("--undirected", action="store_true")

    parser.add_argument("--beta_min", type=float, default=0.0)
    parser.add_argument("--beta_max", type=float, default=8.0)
    parser.add_argument("--beta_steps", type=int, default=9)

    parser.add_argument("--output_dir", type=str, default="./figures")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    graph, labels, train_idx, val_idx, test_idx = load_data(
        args.graph_path, train_ratio=args.train_ratio, val_ratio=args.val_ratio, name=args.data_name
    )

    if args.undirected:
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()

    observe_graph = graph

    graph = graph.to(device)
    observe_graph = observe_graph.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

    text_feat = torch.from_numpy(np.load(args.text_feature).astype(np.float32)).to(device)
    visual_feat = torch.from_numpy(np.load(args.visual_feature).astype(np.float32)).to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    args_ns = _build_args(args)
    args_ns.seed = args.seed
    args_ns.n_epochs = args.n_epochs
    args_ns.eval_steps = args.eval_steps

    print(">>> Phase 0: Training Unimodal References (zeroing the other modality)")
    ref_text = _train_one_ogm(
        args_ns,
        graph,
        observe_graph,
        text_feat,
        torch.zeros_like(visual_feat),
        labels,
        train_idx,
        val_idx,
        test_idx,
        device,
        seed_offset=100,
    )
    ref_vis = _train_one_ogm(
        args_ns,
        graph,
        observe_graph,
        torch.zeros_like(text_feat),
        visual_feat,
        labels,
        train_idx,
        val_idx,
        test_idx,
        device,
        seed_offset=200,
    )

    beta_values = np.linspace(args.beta_min, args.beta_max, args.beta_steps)
    data_text_up = {"rank": [], "sim_text": [], "sim_vis": []}
    data_vis_up = {"rank": [], "sim_text": [], "sim_vis": []}

    print(">>> Phase 1: Text Upweight Sweep")
    for i, beta in enumerate(tqdm(beta_values, desc="Text Upweight")):
        feat_text = text_feat * float(beta)
        emb = _train_one_ogm(
            args_ns,
            graph,
            observe_graph,
            feat_text,
            visual_feat,
            labels,
            train_idx,
            val_idx,
            test_idx,
            device,
            seed_offset=1000 + i,
        )
        data_text_up["rank"].append(compute_effective_rank(emb))
        data_text_up["sim_text"].append(linear_cka(emb, ref_text))
        data_text_up["sim_vis"].append(linear_cka(emb, ref_vis))

    print(">>> Phase 2: Visual Upweight Sweep")
    for i, beta in enumerate(tqdm(beta_values, desc="Visual Upweight")):
        feat_vis = visual_feat * float(beta)
        emb = _train_one_ogm(
            args_ns,
            graph,
            observe_graph,
            text_feat,
            feat_vis,
            labels,
            train_idx,
            val_idx,
            test_idx,
            device,
            seed_offset=2000 + i,
        )
        data_vis_up["rank"].append(compute_effective_rank(emb))
        data_vis_up["sim_text"].append(linear_cka(emb, ref_text))
        data_vis_up["sim_vis"].append(linear_cka(emb, ref_vis))

    print(">>> Plotting")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(beta_values, data_text_up["rank"], "o-", linewidth=2, label="Text Upweight", color="C0")
    ax.plot(beta_values, data_vis_up["rank"], "s--", linewidth=2, label="Visual Upweight", color="C1")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Multimodal Rank")
    ax.set_title("Rank under Text/Visual Upweight")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(beta_values, data_text_up["sim_text"], "o-", linewidth=2, label="Text vs Fused", color="C0")
    ax.plot(beta_values, data_text_up["sim_vis"], "s--", linewidth=2, label="Visual vs Fused", color="C1")
    ax.set_xlabel(r"$\beta$ (Text Upweight)")
    ax.set_ylabel("Representation Similarity (CKA)")
    ax.set_title("Similarity under Text Upweight")
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(beta_values, data_vis_up["sim_text"], "o-", linewidth=2, label="Text vs Fused", color="C0")
    ax.plot(beta_values, data_vis_up["sim_vis"], "s--", linewidth=2, label="Visual vs Fused", color="C1")
    ax.set_xlabel(r"$\beta$ (Visual Upweight)")
    ax.set_ylabel("Representation Similarity (CKA)")
    ax.set_title("Similarity under Visual Upweight")
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(args.output_dir, "test_ogm_rank_dynamics.pdf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved to {out_path}")


if __name__ == "__main__":
    main()
