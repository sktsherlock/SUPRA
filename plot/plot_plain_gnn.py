# 放大某一模态的特征输入（例如文本或视觉），观察模型性能、表示相似性和多模态秩的变化趋势。

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# 请确保这些路径与你的项目结构匹配
from GNN.GraphData import load_data, set_seed
from GNN.Baselines.Early_GNN import Early_GNN as mag_base
from GNN.Utils.rank_analysis import compute_effective_rank, linear_cka

# ----------------------------------------------------------------------
# 辅助函数：训练单个模型并返回 Embedding
# ----------------------------------------------------------------------
def extract_embedding(model, graph, feat):
    """Return penultimate embeddings when possible; fallback to logits."""
    if hasattr(model, "convs") and hasattr(model, "n_layers") and int(model.n_layers) > 1:
        h = feat
        for i in range(int(model.n_layers) - 1):
            h = model.convs[i](graph, h)
            if hasattr(model, "norms") and i < len(model.norms):
                h = model.norms[i](h)
            if hasattr(model, "activation"):
                h = model.activation(h)
            if hasattr(model, "dropout"):
                h = model.dropout(h)
        return h
    return model(graph, feat)


def train_one_model(args, graph, feat, labels, train_idx, val_idx, test_idx, device, n_classes, silent=True, seed_offset=0):
    """
    从头初始化并训练一个 Plain GNN 模型。
    """
    # 1. 构建模型参数
    model_args = SimpleNamespace(
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        dropout=args.dropout,
        aggregator="mean",
        attn_drop=0.0,
        edge_drop=0.0,
        n_heads=1,
        no_attn_dst=True,
        use_symmetric_norm=True,
        model_name=args.model_name
    )
    
    # 2. 初始化模型 (Random Init)
    set_seed(int(args.seed) + int(seed_offset))
    input_dim = feat.shape[1]
    model = mag_base._build_gnn_backbone(model_args, input_dim, n_classes, device)
    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # 3. 训练循环 (快速版)
    # 为了画平滑曲线，建议 Epochs 稍微多一点，或者保证收敛
    best_val_acc = -1.0
    best_state = None
    
    iterator = range(args.n_epochs)
    if not silent:
        iterator = tqdm(iterator, desc="Training Epochs", leave=False)

    for epoch in iterator:
        model.train()
        optimizer.zero_grad()
        logits = model(graph, feat)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            with th.no_grad():
                val_logits = model(graph, feat)
                val_acc = (val_logits[val_idx].argmax(dim=1) == labels[val_idx]).float().mean().item()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print(f"   [Epoch {epoch:03d}] New Best Val Acc: {best_val_acc:.4f}")
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    
    # 4. 加载最佳状态
    if best_state is not None:
        model.load_state_dict(best_state)
    
    model.eval()
    with th.no_grad():
        # 获取最终表示 (Plain GNN 用倒数第二层或者 Logits 都可以，这里用输出)
        final_emb = extract_embedding(model, graph, feat)[test_idx]
        
    return final_emb

# ----------------------------------------------------------------------
# 主程序
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    # 数据参数
    parser.add_argument("--data_name", type=str, default="MAGDataset")
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--text_feature", type=str, required=True)
    parser.add_argument("--visual_feature", type=str, required=True)
    # 模型参数
    parser.add_argument("--model_name", type=str, default="GCN", choices=["GCN", "SAGE"])
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--n_epochs", type=int, default=80) 
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--selfloop", action="store_true")
    parser.add_argument("--undirected", action="store_true")
    # Beta 扫描参数
    parser.add_argument("--beta_min", type=float, default=0.0)
    parser.add_argument("--beta_max", type=float, default=8.0)
    parser.add_argument("--beta_steps", type=int, default=9) # 推荐多一点点，曲线更滑，比如 9 或 17
    # 输出
    parser.add_argument("--output_dir", type=str, default="./figures")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. 环境设置
    set_seed(args.seed)
    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() and args.gpu >= 0 else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. 数据加载
    print(">>> Loading Data...")
    graph, labels, train_idx, val_idx, test_idx = load_data(
        args.graph_path, train_ratio=args.train_ratio, val_ratio=args.val_ratio, name=args.data_name
    )
    if args.undirected:
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)
    graph = graph.remove_self_loop().add_self_loop()

    graph.create_formats_()
    
    graph = graph.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

    text_feat = th.from_numpy(np.load(args.text_feature).astype(np.float32)).to(device)
    vis_feat = th.from_numpy(np.load(args.visual_feature).astype(np.float32)).to(device)
    n_classes = int((labels.max() + 1).item())

    # 3. 训练 Reference (单模态基准)
    print("\n>>> Phase 0: Training Unimodal Baselines (References)...")
    
    # Text Only
    print("   [1/2] Training Text-Only Reference...")
    ref_emb_text = train_one_model(
        args, graph, text_feat, labels, train_idx, val_idx, test_idx, device, n_classes, seed_offset=100
    )
    rank_base_text = compute_effective_rank(ref_emb_text)
    
    # Visual Only
    print("   [2/2] Training Visual-Only Reference...")
    ref_emb_vis = train_one_model(
        args, graph, vis_feat, labels, train_idx, val_idx, test_idx, device, n_classes, seed_offset=200
    )
    rank_base_vis = compute_effective_rank(ref_emb_vis)
    
    print(f"   Baseline Ranks -> Text: {rank_base_text:.2f} | Visual: {rank_base_vis:.2f}")

    # =================================================================================
    # 4. 双重扫描 (Double Beta Sweep)
    # =================================================================================
    
    beta_values = np.linspace(args.beta_min, args.beta_max, args.beta_steps)
    
    # 存储结果
    # 实验 A: Text Upweighting (Visual constant)
    data_text_up = {"rank": [], "sim_text": [], "sim_vis": []}
    # 实验 B: Visual Upweighting (Text constant)
    data_vis_up = {"rank": [], "sim_text": [], "sim_vis": []}

    print(f"\n>>> Phase 1: Text Upweighting Sweep (0 -> {args.beta_max})...")
    for i, beta in enumerate(tqdm(beta_values, desc="Text Upweight")):
        # 构造输入: Text * Beta, Visual * 1
        feat_in = th.cat([text_feat * beta, vis_feat], dim=1)
        
        # 训练
        emb_fused = train_one_model(
            args, graph, feat_in, labels, train_idx, val_idx, test_idx, device, n_classes, seed_offset=1000 + i
        )
        
        # 记录
        data_text_up["rank"].append(compute_effective_rank(emb_fused))
        data_text_up["sim_text"].append(linear_cka(emb_fused, ref_emb_text))
        data_text_up["sim_vis"].append(linear_cka(emb_fused, ref_emb_vis))

    print(f"\n>>> Phase 2: Visual Upweighting Sweep (0 -> {args.beta_max})...")
    for i, beta in enumerate(tqdm(beta_values, desc="Visual Upweight")):
        # 构造输入: Text * 1, Visual * Beta
        feat_in = th.cat([text_feat, vis_feat * beta], dim=1)
        
        # 训练
        emb_fused = train_one_model(
            args, graph, feat_in, labels, train_idx, val_idx, test_idx, device, n_classes, seed_offset=2000 + i
        )
        
        # 记录
        data_vis_up["rank"].append(compute_effective_rank(emb_fused))
        data_vis_up["sim_text"].append(linear_cka(emb_fused, ref_emb_text))
        data_vis_up["sim_vis"].append(linear_cka(emb_fused, ref_emb_vis))

    # =================================================================================
    # 5. 画图 (完全还原你的样式)
    # =================================================================================
    print("\n>>> Plotting...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: Rank Comparison (Left) ---
    ax = axes[0]
    # 线条 1: 当放大 Text 时，Rank 怎么变
    ax.plot(beta_values, data_text_up["rank"], "o-", linewidth=2, label="Text Upweight", color="C0")
    # 线条 2: 当放大 Visual 时，Rank 怎么变
    ax.plot(beta_values, data_vis_up["rank"], "s--", linewidth=2, label="Visual Upweight", color="C1")
    
    # 可以选择性画 Baseline 虚线
    # ax.axhline(y=rank_base_text, color='C0', linestyle=':', alpha=0.5)
    # ax.axhline(y=rank_base_vis, color='C1', linestyle=':', alpha=0.5)
    
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Multimodal Rank")
    ax.set_title("Rank under Text/Visual Upweight")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Similarity under Text Upweight (Middle) ---
    ax = axes[1]
    # 线条 1: 像不像 Text?
    ax.plot(beta_values, data_text_up["sim_text"], "o-", linewidth=2, label="Text vs Fused", color="C0")
    # 线条 2: 像不像 Visual?
    ax.plot(beta_values, data_text_up["sim_vis"], "s--", linewidth=2, label="Visual vs Fused", color="C1")
    
    ax.set_xlabel(r"$\beta$ (Text Upweight)")
    ax.set_ylabel("Representation Similarity (CKA)")
    ax.set_title("Similarity under Text Upweight")
    ax.set_ylim([0, 1.05]) # CKA 归一化在 0-1 之间
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Similarity under Visual Upweight (Right) ---
    ax = axes[2]
    # 线条 1: 像不像 Text?
    ax.plot(beta_values, data_vis_up["sim_text"], "o-", linewidth=2, label="Text vs Fused", color="C0")
    # 线条 2: 像不像 Visual?
    ax.plot(beta_values, data_vis_up["sim_vis"], "s--", linewidth=2, label="Visual vs Fused", color="C1")
    
    ax.set_xlabel(r"$\beta$ (Visual Upweight)")
    ax.set_ylabel("Representation Similarity (CKA)")
    ax.set_title("Similarity under Visual Upweight")
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(args.output_dir, "rank_dynamics.pdf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved styled plot to {out_path}")

if __name__ == "__main__":
    main()