#!/usr/bin/env python
"""
Efficiency profiling for SUPRA, NTSFormer, MIG_GT, and Late_GNN (GCN/GAT).

Measures:
1. Number of parameters (M)
2. Peak GPU memory during training (MB)
3. Estimated total training time + average per epoch (s)

Usage:
    python tools/profile_efficiency.py \
        --model SUPRA --data_name Movies \
        --text_feature /path/Movies_roberta_base_512_mean.npy \
        --visual_feature /path/Movies_openai_clip-vit-large-patch14.npy \
        --graph_path /path/MoviesGraph.pt \
        --embed_dim 256 --n_layers 2 --n_hidden 256 \
        --gpu 0 --n_epochs 1000

    python tools/profile_efficiency.py \
        --model Late_GNN_GCN --data_name Reddit-M \
        --text_feature /path/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
        --visual_feature /path/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
        --graph_path /path/RedditMGraph.pt \
        --n_hidden 256 --n_layers 2 --dropout 0.3 --lr 0.001 --wd 1e-4 \
        --gpu 0 --n_epochs 1000

    python tools/profile_efficiency.py \
        --model Late_GNN_GAT --data_name Reddit-M \
        --text_feature /path/RedditM_Llama_3.2_11B_Vision_Instruct_100_mean.npy \
        --visual_feature /path/RedditM_Llama-3.2-11B-Vision-Instruct_visual.npy \
        --graph_path /path/RedditMGraph.pt \
        --n_hidden 256 --n_layers 2 --dropout 0.3 --lr 0.001 --wd 1e-4 \
        --n_heads 4 --attn_drop 0.0 --gpu 0 --n_epochs 1000
"""
import argparse
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch as th
import torch.nn as nn
import numpy as np

from GNN.SUPRA import SUPRA
from GNN.Baselines.NTSFormer import NTSFormerModel, sign_pre_compute_batched
from GNN.Baselines.MIG_GT import MIGGT_NodeClassifier
from GNN.Baselines import Late_GNN as late_gnn_module
from GNN.Baselines import Early_GNN as mag_base
from GNN.GraphData import load_data
from GNN.Utils.LossFunction import cross_entropy


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_supra(args, text_dim, vis_dim, n_classes, device):
    model = SUPRA(
        text_in_dim=text_dim,
        vis_in_dim=vis_dim,
        embed_dim=int(args.embed_dim),
        n_classes=n_classes,
        dropout=float(args.dropout),
        args=args,
        device=device,
    ).to(device)
    return model


def build_ntsformer(args, text_dim, vis_dim, n_classes, device):
    model = NTSFormerModel(
        text_feat_dim=text_dim,
        vis_feat_dim=vis_dim,
        embed_dim=int(args.n_hidden),
        n_classes=n_classes,
        dropout=float(args.dropout),
        args=args,
        device=device,
    ).to(device)
    return model


def build_miggt(args, text_dim, vis_dim, n_classes, device):
    model = MIGGT_NodeClassifier(
        args, text_dim, vis_dim, n_classes
    ).to(device)
    return model


def build_late_gnn(args, text_dim, vis_dim, n_classes, device, backbone):
    """Late_GNN with GCN or GAT backbone."""
    embed_dim = int(args.n_hidden)
    proj_dim = embed_dim

    # Build encoders
    text_encoder = mag_base.ModalityEncoder(text_dim, proj_dim, float(args.dropout)).to(device)
    visual_encoder = mag_base.ModalityEncoder(vis_dim, proj_dim, float(args.dropout)).to(device)

    # Build GNN backbones
    text_gnn = mag_base._build_gnn_backbone(args, proj_dim, embed_dim, device)
    vis_gnn = mag_base._build_gnn_backbone(args, proj_dim, embed_dim, device)

    classifier = nn.Linear(2 * embed_dim, n_classes).to(device)
    model = late_gnn_module.LateFusionMAG(
        text_encoder,
        visual_encoder,
        text_gnn,
        vis_gnn,
        classifier,
        use_mlp_before_fusion=False,
        use_no_encoder=False,
    )
    model.reset_parameters()
    return model


def train_one_epoch(model, graph, text_feat, vis_feat, labels, train_idx, optimizer, label_smoothing, model_type, **kwargs):
    """Train one full epoch and return loss."""
    model.train()
    optimizer.zero_grad()
    if model_type == "SUPRA":
        out = model(graph, text_feat, vis_feat)
    elif model_type == "NTSFormer":
        text_h_list = kwargs.get("text_h_list")
        vis_h_list = kwargs.get("vis_h_list")
        out = model(graph, text_feat, vis_feat, text_h_list, vis_h_list)
    elif model_type == "MIG_GT":
        logits, _, _ = model(graph, text_feat, vis_feat)
        out = logits
    elif model_type in ("Late_GNN_GCN", "Late_GNN_GAT"):
        text_h, vis_h = model.forward_branches(graph, text_feat, vis_feat)
        fused = model.fuse_embeddings(text_h, vis_h)
        out = model.classifier(fused)
    loss = cross_entropy(out[train_idx], labels[train_idx], label_smoothing=label_smoothing)
    loss.backward()
    optimizer.step()
    return loss.item()


@th.no_grad()
def infer(model, graph, text_feat, vis_feat, labels, val_idx, model_type, **kwargs):
    model.eval()
    if model_type == "SUPRA":
        out = model(graph, text_feat, vis_feat)
    elif model_type == "NTSFormer":
        text_h_list = kwargs.get("text_h_list")
        vis_h_list = kwargs.get("vis_h_list")
        out = model(graph, text_feat, vis_feat, text_h_list, vis_h_list)
    elif model_type == "MIG_GT":
        out, _, _ = model(graph, text_feat, vis_feat)
    elif model_type in ("Late_GNN_GCN", "Late_GNN_GAT"):
        text_h, vis_h = model.forward_branches(graph, text_feat, vis_feat)
        fused = model.fuse_embeddings(text_h, vis_h)
        out = model.classifier(fused)
    val_pred = out[val_idx].argmax(dim=1)
    val_true = labels[val_idx]
    acc = (val_pred == val_true).float().mean().item()
    return acc


def profile(model_type: str, args, text_dim, vis_dim, n_classes, device):
    n_profile_epochs = int(getattr(args, "n_profile_epochs", 10))
    n_total_epochs = int(args.n_epochs)
    early_stop_patience = int(getattr(args, "early_stop_patience", 40))
    label_smoothing = float(getattr(args, "label_smoothing", 0.1))

    # Build model
    if model_type == "SUPRA":
        model = build_supra(args, text_dim, vis_dim, n_classes, device)
    elif model_type == "NTSFormer":
        model = build_ntsformer(args, text_dim, vis_dim, n_classes, device)
    elif model_type == "MIG_GT":
        model = build_miggt(args, text_dim, vis_dim, n_classes, device)
    elif model_type in ("Late_GNN_GCN", "Late_GNN_GAT"):
        model = build_late_gnn(args, text_dim, vis_dim, n_classes, device, backbone=model_type.replace("Late_GNN_", ""))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    n_params = count_params(model)

    # Load data
    graph, labels, train_idx, val_idx, test_idx = \
        load_data(args.graph_path, train_ratio=float(args.train_ratio),
                  val_ratio=float(args.val_ratio), name=args.data_name, fewshots=False)

    # Apply undirected and selfloop (matching sweep script defaults)
    undirected = getattr(args, "undirected", True)
    selfloop = getattr(args, "selfloop", True)
    if undirected:
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)
    if selfloop:
        graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    graph = graph.to(device)

    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)

    # For Late_GNN: use observe_graph (same as graph in non-inductive mode)
    observe_graph = graph

    # Load features from numpy files
    text_feat = th.from_numpy(np.load(args.text_feature, mmap_mode="r").astype(np.float32)).to(device)
    visual_feat = th.from_numpy(np.load(args.visual_feature, mmap_mode="r").astype(np.float32)).to(device)

    # Pre-compute for NTSFormer
    text_h_list, vis_h_list = None, None
    if model_type == "NTSFormer":
        text_h_list_dev = text_feat
        vis_h_list_dev = visual_feat
        text_h_list, vis_h_list = sign_pre_compute_batched(
            graph, [text_h_list_dev, vis_h_list_dev],
            k=int(getattr(args, "nts_sign_k", 2)),
            include_input=True, alpha=0.0, device=device
        )

    optimizer = th.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.wd))

    # Reset peak memory
    if th.cuda.is_available():
        th.cuda.reset_peak_memory_stats(device)

    # Profile: run n_profile_epochs to measure per-epoch time and peak memory
    epoch_times = []
    for epoch in range(1, n_profile_epochs + 1):
        tic = time.time()
        loss = train_one_epoch(
            model, graph, text_feat, visual_feat, labels, train_idx,
            optimizer, label_smoothing, model_type,
            text_h_list=text_h_list, vis_h_list=vis_h_list
        )
        toc = time.time()
        epoch_times.append(toc - tic)

    avg_epoch_time = np.mean(epoch_times)
    peak_memory_mb = 0.0
    if th.cuda.is_available():
        peak_memory_mb = th.cuda.max_memory_allocated(device) / 1048576.0

    # Estimate total training time based on early stopping
    # Run a few more epochs with validation to estimate early stop point
    patience_counter = 0
    best_val = -1.0
    epochs_needed = n_profile_epochs

    for epoch in range(n_profile_epochs + 1, n_total_epochs + 1):
        tic = time.time()
        loss = train_one_epoch(
            model, graph, text_feat, visual_feat, labels, train_idx,
            optimizer, label_smoothing, model_type,
            text_h_list=text_h_list, vis_h_list=vis_h_list
        )
        val_acc = infer(model, graph, text_feat, visual_feat, labels, val_idx, model_type,
                        text_h_list=text_h_list, vis_h_list=vis_h_list)
        toc = time.time()
        epoch_time = toc - tic
        epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(epoch_times)
        epochs_needed = epoch

        if val_acc > best_val:
            best_val = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break

    total_time = epochs_needed * avg_epoch_time

    return {
        "model": model_type,
        "dataset": args.data_name,
        "params_M": n_params / 1e6,
        "peak_memory_MB": peak_memory_mb,
        "total_time_s": total_time,
        "avg_epoch_time_s": avg_epoch_time,
        "epochs_needed": epochs_needed,
        "n_profile_epochs": n_profile_epochs,
    }


def main():
    parser = argparse.ArgumentParser("Efficiency Profiler")
    parser.add_argument("--model", type=str, required=True,
                        choices=["SUPRA", "NTSFormer", "MIG_GT", "Late_GNN_GCN", "Late_GNN_GAT"])
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--text_feature", type=str, required=True)
    parser.add_argument("--visual_feature", type=str, required=True)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--n_profile_epochs", type=int, default=10,
                        help="Epochs to run for profiling (per-epoch time + peak memory)")
    parser.add_argument("--early_stop_patience", type=int, default=40)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    # GAT parameters
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--attn_drop", type=float, default=0.0)
    parser.add_argument("--edge_drop", type=float, default=0.0)
    # NTSFormer specific
    parser.add_argument("--nts_sign_k", type=int, default=2)
    # MIG_GT specific
    parser.add_argument("--k_t", type=int, default=3)
    parser.add_argument("--k_v", type=int, default=2)
    parser.add_argument("--mgdcf_alpha", type=float, default=0.1)
    parser.add_argument("--mgdcf_beta", type=float, default=0.9)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--tur_weight", type=float, default=1.0)
    # Late_GNN specific
    parser.add_argument("--late_embed_dim", type=int, default=None)
    parser.add_argument("--mm_proj_dim", type=int, default=None)
    args = parser.parse_args()

    # Set model_name for Late_GNN backbone selection
    if args.model == "Late_GNN_GCN":
        args.model_name = "GCN"
    elif args.model == "Late_GNN_GAT":
        args.model_name = "GAT"

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Device: {device}")

    text_dim = int(np.load(args.text_feature, mmap_mode="r").shape[1])
    vis_dim = int(np.load(args.visual_feature, mmap_mode="r").shape[1])

    print(f"Building {args.model} (text_dim={text_dim}, vis_dim={vis_dim})...")
    _, labels, _, _, _ = \
        load_data(args.graph_path, train_ratio=args.train_ratio,
                  val_ratio=args.val_ratio, name=args.data_name, fewshots=False)
    n_classes = int(labels.max().item()) + 1

    result = profile(args.model, args, text_dim, vis_dim, n_classes, device)

    print(f"\n{'='*50}")
    print(f"Profile Result: {args.model} on {args.data_name}")
    print(f"{'='*50}")
    print(f"  Parameters:       {result['params_M']:.3f} M")
    print(f"  Peak Memory:     {result['peak_memory_MB']:.2f} MB")
    print(f"  Total Time(est): {result['total_time_s']:.2f} s  ({result['total_time_s']/60:.1f} min)")
    print(f"  Avg Epoch:        {result['avg_epoch_time_s']:.4f} s/epoch")
    print(f"  Epochs Needed:    {result['epochs_needed']}  (early_stop={args.early_stop_patience}, profile={args.n_profile_epochs}ep)")


if __name__ == "__main__":
    main()
