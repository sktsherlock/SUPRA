# 训练一个多模态 GNN，并遍历不同的 GNN 层数（例如 1到5层），计算模型学到的节点嵌入表示
# 计算聚合特征、文本特征、视觉特征之间的余弦距离

import argparse
import os
from typing import List, Dict

import numpy as np
import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from GNN.GraphData import load_data, set_seed
from GNN.Utils.LossFunction import cross_entropy, get_metric
from GNN.Utils.rank_analysis import compute_effective_rank
from GNN.Utils.NodeClassification import _make_noisy_feature

from GNN.Baselines.Early_GNN import Early_GNN as mag_base
from GNN.Baselines.Early_GNN import ModalityEncoder, SimpleMAGGNN, _make_observe_graph_inductive
from GNN.Baselines.Late_GNN import LateFusionMAG

def _extract_gnn_embedding(gnn, graph, feat: th.Tensor) -> th.Tensor:
    """Return penultimate embeddings when possible; fallback to logits."""
    # Logic to handle different GNN structures to get embeddings
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


def _build_plain(args, in_dim: int, n_classes: int, device: th.device, n_layers: int):
    original_n_layers = args.n_layers
    args.n_layers = n_layers
    
    model = mag_base._build_gnn_backbone(args, in_dim, n_classes, device)
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()
    
    args.n_layers = original_n_layers
    return model


def _build_early(args, text_dim: int, vis_dim: int, n_classes: int, device: th.device, n_layers: int):
    original_n_layers = args.n_layers
    args.n_layers = n_layers

    proj_dim = int(args.mm_proj_dim) if args.mm_proj_dim is not None else int(args.n_hidden)
    text_encoder = ModalityEncoder(text_dim, proj_dim, args.dropout).to(device)
    visual_encoder = ModalityEncoder(vis_dim, proj_dim, args.dropout).to(device)
    gnn = mag_base._build_gnn_backbone(args, 2 * proj_dim, n_classes, device)
    model = SimpleMAGGNN(text_encoder, visual_encoder, gnn, modality_dropout=float(args.modality_dropout or 0.0))
    model = model.to(device)
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()

    args.n_layers = original_n_layers
    return model


def _build_late(args, text_dim: int, vis_dim: int, n_classes: int, device: th.device, n_layers: int):
    original_n_layers = args.n_layers
    args.n_layers = n_layers

    embed_dim = int(args.late_embed_dim) if args.late_embed_dim is not None else int(args.n_hidden)
    text_gnn = mag_base._build_gnn_backbone(args, text_dim, embed_dim, device)
    vis_gnn = mag_base._build_gnn_backbone(args, vis_dim, embed_dim, device)
    classifier = th.nn.Linear(embed_dim, n_classes).to(device)
    model = LateFusionMAG(text_gnn, vis_gnn, classifier, modality_dropout=float(args.modality_dropout or 0.0))
    model = model.to(device)
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()


    args.n_layers = original_n_layers
    return model

def _get_embedding_generic(args, model, graph, text_feat, vis_feat):
    """Generic wrapper to get representation before final classifier."""
    if args.model_type == "plain":
        feat = th.cat([text_feat, vis_feat], dim=1)
        # Using helper to get penultimate layer
        return _extract_gnn_embedding(model, graph, feat)
    elif args.model_type == "early":
        text_h = model.text_encoder(text_feat)
        vis_h = model.visual_encoder(vis_feat)
        feat = th.cat([text_h, vis_h], dim=1)
        return _extract_gnn_embedding(model.gnn, graph, feat)
    else: # late
        text_h, vis_h = model.forward_branches(graph, text_feat, vis_feat)
        return model.fuse_embeddings(text_h, vis_h)


def _compute_geometric_gap(args, model, graph, text_feat, vis_feat, eval_idx):
    """
    Computes Geometric Gap (Cosine Distance) between Full Representation and Unimodal Representations.
    Unimodal Representation is defined as the output when one modality is ZEROED OUT.
    
    Args:
        text_feat: Current text feature (could be clean or noisy)
        vis_feat: Current visual feature (could be clean or noisy)
    Returns:
        dist_to_text: Dist(H(T,V), H(T,0))
        dist_to_vis:  Dist(H(T,V), H(0,V))
    """
    # 1. Full Representation
    emb_full = _get_embedding_generic(args, model, graph, text_feat, vis_feat)[eval_idx]
    
    # 2. Text-Only Reference (Vis Zeroed)
    zeros_vis = th.zeros_like(vis_feat)
    emb_text_ref = _get_embedding_generic(args, model, graph, text_feat, zeros_vis)[eval_idx]
    
    # 3. Vis-Only Reference (Text Zeroed)
    zeros_text = th.zeros_like(text_feat)
    emb_vis_ref = _get_embedding_generic(args, model, graph, zeros_text, vis_feat)[eval_idx]
    
    # Normalize
    emb_full_norm = F.normalize(emb_full, p=2, dim=1)
    emb_text_ref_norm = F.normalize(emb_text_ref, p=2, dim=1)
    emb_vis_ref_norm = F.normalize(emb_vis_ref, p=2, dim=1)
    
    # Distances
    dist_to_text = 1.0 - (emb_full_norm * emb_text_ref_norm).sum(dim=1).mean().item()
    dist_to_vis = 1.0 - (emb_full_norm * emb_vis_ref_norm).sum(dim=1).mean().item()
    
    return dist_to_text, dist_to_vis

def _evaluate_all_settings(args, model, graph, text_feat_clean, vis_feat_clean, train_idx, eval_idx):
    """
    Evaluates the model under 3 settings:
    1. Clean (Original)
    2. Noisy Text (Text degraded, Vis clean)
    3. Noisy Vis (Text clean, Vis degraded)
    
    Computes Geometric Gaps for each.
    """
    results = {}
    
    # Prepare Noisy Features
    # Note: Using args.degrade_alpha for noise level.
    # If alpha=1.0, it's fully noisy/shuffled.
    noisy_text_feat = _make_noisy_feature(text_feat_clean, train_idx, float(args.degrade_alpha))
    noisy_vis_feat = _make_noisy_feature(vis_feat_clean, train_idx, float(args.degrade_alpha))
    
    # --- Setting 1: Clean ---
    d_clean_txt, d_clean_vis = _compute_geometric_gap(args, model, graph, text_feat_clean, vis_feat_clean, eval_idx)
    results["clean_dist_text"] = d_clean_txt
    results["clean_dist_vis"] = d_clean_vis
    
    # --- Setting 2: Noisy Text ---
    # Input: (NoisyText, CleanVis)
    # Comparison: H(NT, CV) vs H(NT, 0) and H(0, CV)
    d_nt_txt, d_nt_vis = _compute_geometric_gap(args, model, graph, noisy_text_feat, vis_feat_clean, eval_idx)
    results["noisy_text_dist_text"] = d_nt_txt
    results["noisy_text_dist_vis"] = d_nt_vis
    
    # --- Setting 3: Noisy Vis ---
    # Input: (CleanText, NoisyVis)
    d_nv_txt, d_nv_vis = _compute_geometric_gap(args, model, graph, text_feat_clean, noisy_vis_feat, eval_idx)
    results["noisy_vis_dist_text"] = d_nv_txt
    results["noisy_vis_dist_vis"] = d_nv_vis
    
    return results

def train_one_layer_config(args, n_layers, device, data_pack):
    """
    Trains a model with specific n_layers and returns the FINAL metrics.
    """
    (graph, observe_graph, text_feat, vis_feat, labels, train_idx, eval_idx, n_classes) = data_pack
    
    print(f"\n>>> Training with n_layers={n_layers} ...")
    
    # Build Model
    if args.model_type == "plain":
        feat = th.cat([text_feat, vis_feat], dim=1)
        model = _build_plain(args, feat.shape[1], n_classes, device, n_layers)
    elif args.model_type == "early":
        model = _build_early(args, text_feat.shape[1], vis_feat.shape[1], n_classes, device, n_layers)
    else: # late
        model = _build_late(args, text_feat.shape[1], vis_feat.shape[1], n_classes, device, n_layers)

    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    iterator = range(1, args.n_epochs + 1)
    
    for epoch in tqdm(iterator, leave=False):
        model.train()
        optimizer.zero_grad()

        if args.model_type == "plain":
            logits = model(observe_graph, feat)
        else:
            logits = model(observe_graph, text_feat, vis_feat)

        loss = cross_entropy(logits[train_idx], labels[train_idx], label_smoothing=args.label_smoothing)
        loss.backward()
        optimizer.step()
        
    # Final Eval
    model.eval()
    with th.no_grad():
        results = _evaluate_all_settings(args, model, graph, text_feat, vis_feat, train_idx, eval_idx)
    
    return results



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["plain", "early", "late"])
    parser.add_argument("--data_name", type=str, default="MAGDataset")
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--text_feature", type=str, required=True)
    parser.add_argument("--visual_feature", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="GCN", choices=["GCN", "SAGE"])
    parser.add_argument("--n_hidden", type=int, default=256)
    # n_layers will be scanned
    parser.add_argument("--layers_scan", type=str, default="1,2,3,4,5", help="Comma separated list of layers to scan")
    
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--metric", type=str, default="accuracy")
    parser.add_argument("--average", type=str, default="macro")
    parser.add_argument("--degrade_alpha", type=float, default=1.0)

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
    
    # dummy n_layers for build functions args (will be overridden)
    parser.add_argument("--n_layers", type=int, default=2) 

    args = parser.parse_args()

    set_seed(args.seed)
    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() and args.gpu >= 0 else "cpu")
    
    # Load Data Once
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
    
    eval_idx = _get_eval_idx(args, train_idx, val_idx, test_idx)
    
    data_pack = (graph, observe_graph, text_feat, vis_feat, labels, train_idx, eval_idx, n_classes)

    # Parse Layers
    layers_to_scan = [int(x) for x in args.layers_scan.split(",")]
    
    # Results containers
    # { n_layer: { full: ..., deg_text: ..., deg_vis: ... } }
    scan_results = {}
    
    for L in layers_to_scan:
        res = train_one_layer_config(args, L, device, data_pack)
        scan_results[L] = res
        print(f"  -> Layers={L} | Res={res}")

    # --- Plotting ---
    os.makedirs(args.output_dir, exist_ok=True)
    out_name = args.output_name
    if not out_name:
        out_name = f"layer_scan_{args.model_type}.pdf"
    out_path = os.path.join(args.output_dir, out_name)

    layers_arr = np.array(sorted(scan_results.keys()))
    
    # Results keys: clean_dist_text, clean_dist_vis, noisy_text_dist_text, ...
    
    clean_dist_text = np.array([scan_results[l]['clean_dist_text'] for l in layers_arr])
    clean_dist_vis = np.array([scan_results[l]['clean_dist_vis'] for l in layers_arr])
    
    nt_dist_text = np.array([scan_results[l]['noisy_text_dist_text'] for l in layers_arr])
    nt_dist_vis = np.array([scan_results[l]['noisy_text_dist_vis'] for l in layers_arr])
    
    nv_dist_text = np.array([scan_results[l]['noisy_vis_dist_text'] for l in layers_arr])
    nv_dist_vis = np.array([scan_results[l]['noisy_vis_dist_vis'] for l in layers_arr])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Distance to Text Reference
    # (Checking if Text Dominates)
    ax = axes[0]
    ax.plot(layers_arr, clean_dist_text, "o-", label="Clean Inputs", color="black", linewidth=2)
    ax.plot(layers_arr, nt_dist_text, "s--", label="Noisy Text (Weak Text)", color="green")
    ax.plot(layers_arr, nv_dist_text, "^--", label="Noisy Vis (Weak Vis)", color="orange")
    
    ax.set_xlabel("GNN Layers")
    ax.set_ylabel("Cos Dist(H_full, H_text_only)")
    ax.set_title("Dependency on Text Modality")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Subplot 2: Distance to Vis Reference
    ax = axes[1]
    ax.plot(layers_arr, clean_dist_vis, "o-", label="Clean Inputs", color="black", linewidth=2)
    ax.plot(layers_arr, nt_dist_vis, "s--", label="Noisy Text (Weak Text)", color="green")
    ax.plot(layers_arr, nv_dist_vis, "^--", label="Noisy Vis (Weak Vis)", color="orange")
    
    ax.set_xlabel("GNN Layers")
    ax.set_ylabel("Cos Dist(H_full, H_vis_only)")
    ax.set_title("Dependency on Visual Modality")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle(f"Modality Collapse Analysis: {args.model_type.upper()} on {args.data_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"✓ Saved plot to {out_path}")
    
    if args.save_csv:
        csv_path = os.path.splitext(out_path)[0] + ".csv"
        # Collecting columns
        header = "layers,clean_dt,clean_dv,nt_dt,nt_dv,nv_dt,nv_dv"
        data = np.column_stack([
            layers_arr, 
            clean_dist_text, clean_dist_vis,
            nt_dist_text, nt_dist_vis,
            nv_dist_text, nv_dist_vis
        ])
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
        print(f"✓ Saved csv to {csv_path}")

if __name__ == "__main__":
    main()
