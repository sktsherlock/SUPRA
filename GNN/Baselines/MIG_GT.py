import argparse
import os
import sys
import time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(_ROOT))

from GNN.GraphData import load_data, set_seed
from GNN.Utils.LossFunction import cross_entropy, get_metric, EarlyStopping
from GNN.Utils.NodeClassification import _as_scalar_float
from GNN.Utils.model_config import add_common_args
from GNN.Utils.result_logger import build_result_row, update_best_result_csv, append_result_csv

class _LocalMyMLP(nn.Module):
    """Local fallback for MyMLP (prelu MLP with optional batch norm)."""
    def __init__(self, in_dim, hidden_dims, activation="prelu", drop_rate=0.0, bn=True,
                 output_activation=None, output_drop_rate=0.0, output_bn=False):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if bn:
                layers.append(nn.BatchNorm1d(h_dim))
            if activation == "prelu":
                layers.append(nn.PReLU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            if drop_rate > 0:
                layers.append(nn.Dropout(drop_rate))
            prev_dim = h_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _LocalMGDCF(nn.Module):
    """Local fallback for MGDCF (Multi-hop Graph Diffusion Convolution)."""
    def __init__(self, k_hops, alpha, beta, drop_rate=0.0, edge_drop=0.0, att_drop=0.0):
        super().__init__()
        self.k_hops = k_hops
        self.alpha = alpha
        self.beta = beta
        self.dropout = nn.Dropout(drop_rate)
        self.edge_drop = edge_drop

    def forward(self, g, x):
        # Multi-hop graph diffusion: weighted neighborhood aggregation
        h = x
        g = g.local_var()
        g.ndata['h'] = h
        out = self.beta * x
        hop_weight = self.alpha
        for _ in range(self.k_hops):
            g.update_all(lambda edges: {'msg': edges.src['h']}, dgl.function.mean('msg', 'h'))
            h = g.ndata['h']
            out = out + hop_weight * h
            hop_weight = hop_weight * self.alpha
        g.ndata.clear()
        return self.dropout(out)


class _LocalTransformer(nn.Module):
    """Local fallback for Transformer (self-attention with optional residual)."""
    def __init__(self, in_units, att_units, out_units, ff_units_list=None,
                 att_residual=True, ff_residual=False, att_h_drop_rate=0.0, drop_rate=0.0, ln=False):
        super().__init__()
        self.att_residual = att_residual
        self.ln = nn.LayerNorm(in_units) if ln else None
        self.att = nn.MultiheadAttention(in_units, 1, dropout=att_h_drop_rate, batch_first=True)
        self.dropout = nn.Dropout(drop_rate)
        # Project to output
        self.out_proj = nn.Linear(in_units, out_units)

    def forward(self, q, k, att_mask=None):
        # Self-attention: q=k=memory
        att_out, _ = self.att(q, k, k, att_mask)
        if self.ln is not None:
            att_out = self.ln(att_out)
        out = self.dropout(att_out)
        if self.att_residual:
            out = out + q
        return self.out_proj(out)


try:
    from mig_gt.layers.common import MyMLP
    from mig_gt.layers.mgdcf import MGDCF
    from mig_gt.layers.mirf_gt import Transformer
except ModuleNotFoundError:
    MyMLP = _LocalMyMLP
    MGDCF = _LocalMGDCF
    Transformer = _LocalTransformer


class MIGGT_NodeClassifier(nn.Module):
    def __init__(self, args, t_in_dim, v_in_dim, num_classes):
        super().__init__()
        self.num_samples = args.num_samples
        hidden_dim = args.n_hidden
        
        # 1. 原版 MMMGDCF 的 MLP 封装 (包含 input dropout 和 output bn)
        self.t_mlp = nn.Sequential(
            nn.Dropout(args.dropout),
            MyMLP(t_in_dim, [hidden_dim], activation="prelu", drop_rate=args.dropout, bn=True, output_activation="prelu", output_drop_rate=0.0, output_bn=True)
        )
        self.v_mlp = nn.Sequential(
            nn.Dropout(args.dropout),
            MyMLP(v_in_dim, [hidden_dim], activation="prelu", drop_rate=args.dropout, bn=True, output_activation="prelu", output_drop_rate=0.0, output_bn=True)
        )

        # 2. 调用原版 MGDCF
        self.t_mgdcf = MGDCF(args.k_t, args.mgdcf_alpha, args.mgdcf_beta, args.dropout, args.edge_drop, args.dropout)
        self.v_mgdcf = MGDCF(args.k_v, args.mgdcf_alpha, args.mgdcf_beta, args.dropout, args.edge_drop, args.dropout)

        self.z_dropout = nn.Dropout(args.dropout)
        
        # 3. 调用原版 Transformer (参数完全对齐原版 self.z_transformer)
        self.z_transformer = Transformer(
            in_units=hidden_dim, 
            att_units=4, 
            out_units=hidden_dim, 
            ff_units_list=[], 
            att_residual=True, 
            ff_residual=False, 
            att_h_drop_rate=0.0, 
            drop_rate=args.dropout, 
            ln=False
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, g, text_feat, visual_feat):
        encoded_t = self.t_mlp(text_feat)
        encoded_v = self.v_mlp(visual_feat)

        t_h = self.t_mgdcf(g, encoded_t)
        v_h = self.v_mgdcf(g, encoded_v)

        # 原版融合方式： element-wise add
        combined_h = self.z_dropout(t_h + v_h)

        num_nodes = combined_h.size(0)
        
        # Sampling-based Global Transformer
        if self.training and self.num_samples > 0:
            memory_index = th.randint(0, num_nodes, [num_nodes, self.num_samples], device=combined_h.device)
            memory = combined_h[memory_index]
            memory = th.cat([combined_h.unsqueeze(1), memory], dim=1) # [N, C+1, D]
            
            z_memory_h = self.z_transformer(memory, memory)
            final_h = z_memory_h[:, 0]
        else:
            if self.num_samples > 0:
                memory_index = th.randint(0, num_nodes, [num_nodes, self.num_samples], device=combined_h.device)
                memory = combined_h[memory_index]
                memory = th.cat([combined_h.unsqueeze(1), memory], dim=1)
                z_memory_h = self.z_transformer(memory, memory)
                final_h = z_memory_h[:, 0]
            else:
                z_memory_h = None
                final_h = combined_h

        logits = self.classifier(final_h)
        return logits, final_h, z_memory_h


def args_init():
    parser = argparse.ArgumentParser("MIG-GT for Node Classification", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)
    
    # MIG-GT 核心参数 (根据原版 config 对齐)
    parser.add_argument("--k_t", type=int, default=3, help="Receptive field (hops) for Text Modality")
    parser.add_argument("--k_v", type=int, default=2, help="Receptive field (hops) for Visual Modality")
    parser.add_argument("--mgdcf_alpha", type=float, default=0.1, help="MGDCF alpha parameter")
    parser.add_argument("--mgdcf_beta", type=float, default=0.9, help="MGDCF beta parameter")
    
    # 根据原版 config，num_samples 通常为 10 或 20
    parser.add_argument("--num_samples", type=int, default=10, help="Number of global samples C for SGT")
    parser.add_argument("--edge_drop", type=float, default=0.0, help="Edge dropout rate for MGDCF")
    
    # TUR Loss 控制
    parser.add_argument("--tur_weight", type=float, default=1.0, help="Weight for Transformer Unsmooth Regularization")
    
    # TUR 采样数对齐原版的 batch_size 8000
    parser.add_argument("--tur_sample_edges", type=int, default=8000, 
                        help="Number of edges to sample for TUR loss (Aligned with original batch_size).")
    
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    return parser

def main():
    parser = args_init()
    args = parser.parse_args()
    
    if args.disable_wandb or wandb is None:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        wandb.init(config=args, reinit=True, entity="tiant-wang")

    device = th.device("cuda:%d" % args.gpu if th.cuda.is_available() and args.gpu != -1 else "cpu")

    graph, labels, train_idx, val_idx, test_idx = load_data(
        args.graph_path, train_ratio=args.train_ratio, val_ratio=args.val_ratio, 
        name=args.data_name, fewshots=args.fewshots
    )

    if args.undirected:
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)

    if args.selfloop:
        graph = graph.remove_self_loop().add_self_loop()

    graph.create_formats_()
    graph = graph.to(device)
    train_idx, val_idx, test_idx = train_idx.to(device), val_idx.to(device), test_idx.to(device)
    labels = labels.to(device)

    text_feature = th.from_numpy(np.load(args.text_feature).astype(np.float32)).to(device)
    visual_feature = th.from_numpy(np.load(args.visual_feature).astype(np.float32)).to(device)
    n_classes = int((labels.max() + 1).item())

    val_results, test_results = [], []

    for run in range(args.n_runs):
        set_seed(args.seed + run)
        
        model = MIGGT_NodeClassifier(args, text_feature.shape[1], visual_feature.shape[1], n_classes).to(device)
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        stopper = EarlyStopping(patience=args.early_stop_patience) if args.early_stop_patience else None

        best_val_score, best_val_result, final_test_result = -1.0, 0.0, 0.0
        total_time = 0

        for epoch in range(1, args.n_epochs + 1):
            tic = time.time()
            model.train()
            optimizer.zero_grad()
            
            # 1. 全图前向传播
            logits, final_h, z_memory_h = model(graph, text_feature, visual_feature)
            
            # 2. 主任务 Loss (仅限 train_idx)
            loss = cross_entropy(logits[train_idx], labels[train_idx], label_smoothing=args.label_smoothing)

            # 3. TUR 损失 (严格还原原版基于边的计算)
            tur_loss = th.tensor(0.0, device=device)
            if args.tur_weight > 0 and args.num_samples > 0:
                src, dst = graph.edges()
                mask = (src != dst)
                src, dst = src[mask], dst[mask]
                # 采样防止 OOM
                if args.tur_sample_edges > 0 and src.size(0) > args.tur_sample_edges:
                    perm = th.randperm(src.size(0), device=device)[:args.tur_sample_edges]
                    src, dst = src[perm], dst[perm]
                
                pos_h = final_h[src]
                pos_z_mem = z_memory_h[dst]
                
                # 完全还原原版公式：unsqueeze(1) @ permute(0,2,1)
                unsmooth_logits = (pos_h.unsqueeze(1) @ pos_z_mem.permute(0, 2, 1)).squeeze(1)
                tur_loss = F.cross_entropy(unsmooth_logits, th.zeros(src.size(0), dtype=th.long, device=device))
                
                loss = loss + args.tur_weight * tur_loss

            loss.backward()
            optimizer.step()

            if epoch % args.eval_steps == 0:
                model.eval()
                with th.no_grad():
                    logits_e, _, _ = model(graph, text_feature, visual_feature)
                    
                val_loss = cross_entropy(logits_e[val_idx], labels[val_idx])
                test_loss = cross_entropy(logits_e[test_idx], labels[test_idx])

                val_pred = th.argmax(logits_e[val_idx], dim=1)
                test_pred = th.argmax(logits_e[test_idx], dim=1)

                val_score = get_metric(val_pred, labels[val_idx], args.metric)
                test_score = get_metric(test_pred, labels[test_idx], args.metric)
                val_score = float(val_score)
                test_score = float(test_score)

                if (wandb is not None) and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
                    wandb.log({
                        "Train_loss": _as_scalar_float(loss),
                        "Val_loss": _as_scalar_float(val_loss),
                        "TUR_loss": _as_scalar_float(tur_loss),
                        f"Val_{args.metric}": _as_scalar_float(val_score),
                        f"Test_{args.metric}": _as_scalar_float(test_score)
                    })

                total_time += time.time() - tic

                if val_score > best_val_score:
                    best_val_score = float(val_score)
                    best_val_result = float(val_score)
                    final_test_result = float(test_score)

                if stopper and stopper.step(val_score):
                    break

        print(f"Run: {run+1}/{args.n_runs} | Best Val {args.metric}: {best_val_result:.4f} | Final Test: {final_test_result:.4f}")
        val_results.append(best_val_result)
        test_results.append(final_test_result)

    test_mean = float(np.mean(test_results))
    test_std = float(np.std(test_results))
    print(f"Average test {args.metric}: {test_mean * 100.0:.3f} ± {test_std * 100.0:.3f}%")

    if getattr(args, "result_csv", None) or getattr(args, "result_csv_all", None):
        row = build_result_row(args=args, method="MIG_GT", full_metric=test_mean, degrade_text=None, degrade_visual=None, extra={"full_std": test_std})
        key_fields = ["dataset", "method", "n_layers", "single_modality", "inductive", "fewshots", "metric"]
        if getattr(args, "result_csv", None):
            update_best_result_csv(args.result_csv, row, key_fields=key_fields, score_field="full")
        if getattr(args, "result_csv_all", None):
            append_result_csv(args.result_csv_all, row)

if __name__ == "__main__":
    main()