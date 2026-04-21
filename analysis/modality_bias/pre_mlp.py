import argparse
import copy
import os
import sys
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(_ROOT)

from GNN.GraphData import load_data, set_seed
from GNN.Utils.LossFunction import cross_entropy
from GNN.Utils.NodeClassification import initialize_optimizer_and_scheduler
from GNN.Utils.model_config import add_common_args

class ModalityEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.proj.reset_parameters()

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.proj(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float, num_layers: int = 2) -> nn.Module:
    if num_layers < 1:
        raise ValueError("num_layers >= 1")
        
    if num_layers == 1:
        return nn.Linear(int(in_dim), int(out_dim))
    
    layers = []
    layers.append(nn.Linear(int(in_dim), int(hidden_dim)))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(float(dropout)))
    
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(int(hidden_dim), int(hidden_dim)))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(float(dropout)))
        
    layers.append(nn.Linear(int(hidden_dim), int(out_dim)))
    
    return nn.Sequential(*layers)

# --------- 基础模型 ---------

class TextOnlyMLP(nn.Module):
    def __init__(self, in_dim, embed_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.enc = ModalityEncoder(in_dim, embed_dim, dropout)
        self.mlp = _make_mlp(embed_dim, hidden_dim, out_dim, dropout)

    def forward(self, text_feat, vis_feat=None):
        return self.mlp(self.enc(text_feat))

class VisualOnlyMLP(nn.Module):
    def __init__(self, in_dim, embed_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.enc = ModalityEncoder(in_dim, embed_dim, dropout)
        self.mlp = _make_mlp(embed_dim, hidden_dim, out_dim, dropout)

    def forward(self, text_feat=None, vis_feat=None):
        return self.mlp(self.enc(vis_feat))

class MultimodalMLP(nn.Module):
    def __init__(self, text_dim, vis_dim, embed_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.enc_t = ModalityEncoder(text_dim, embed_dim, dropout)
        self.enc_v = ModalityEncoder(vis_dim, embed_dim, dropout)
        self.mlp = _make_mlp(embed_dim * 2, hidden_dim, out_dim, dropout)

    def forward(self, text_feat, vis_feat):
        e_t = self.enc_t(text_feat)
        e_v = self.enc_v(vis_feat)
        return self.mlp(th.cat([e_t, e_v], dim=1))

# --- 辅助函数：计算预测分布的信息熵 ---
def compute_entropy(logits):
    """
    计算 Logits 对应的 Shannon Entropy
    H(p) = - sum(p_i * log(p_i))
    """
    probs = F.softmax(logits, dim=1)
    # 加上 1e-9 防止出现 log(0) 导致 NaN
    entropy = -th.sum(probs * th.log(probs + 1e-9), dim=1, keepdim=True)
    return entropy

# --- 基于预测熵的动态路由 Ensemble ---
class EnsembleMLP(nn.Module):
    def __init__(self, text_dim, vis_dim, embed_dim, hidden_dim, out_dim, dropout, temperature=1.0):
        super().__init__()
        
        self.enc_t = ModalityEncoder(text_dim, embed_dim, dropout)
        self.enc_v = ModalityEncoder(vis_dim, embed_dim, dropout)
        
        self.mlp_t = _make_mlp(embed_dim, hidden_dim, out_dim, dropout)
        self.mlp_v = _make_mlp(embed_dim, hidden_dim, out_dim, dropout)
        self.mlp_m = _make_mlp(embed_dim * 2, hidden_dim, out_dim, dropout)
        
        # 温度系数：控制路由的“锐度”
        # T < 1: 更加赢者通吃 (Winner-takes-all)，只听最自信的那个模态
        # T = 1: 正常的 Softmax 分配
        # T > 1: 更加平滑，接近平均值
        self.temperature = temperature

    def forward(self, text_feat, vis_feat, return_weights=False):
        # 1. 提取基础特征并计算 Logits
        e_t = self.enc_t(text_feat)
        e_v = self.enc_v(vis_feat)
        combined_feat = th.cat([e_t, e_v], dim=1)
        
        logits_t = self.mlp_t(e_t)
        logits_v = self.mlp_v(e_v)
        logits_m = self.mlp_m(combined_feat)
        
        # 2. 计算各路预测的熵 (Entropy)
        # 【关键细节】: 必须使用 detach()！
        # 熵是作为“评价裁判”的客观指标。如果不 detach，网络为了减小 Loss，
        # 会通过反向传播强行让某一路的预测分布变得异常尖锐（人为制造低熵），陷入“极度自信的骗子”陷阱。
        ent_t = compute_entropy(logits_t.detach())
        ent_v = compute_entropy(logits_v.detach())
        ent_m = compute_entropy(logits_m.detach())
        
        # 3. 将熵转化为路由权重
        # 逻辑：熵越小 (置信度越高) -> 负熵越大 -> 经过 Softmax 后的权重越大
        ent_scores = th.cat([-ent_t, -ent_v, -ent_m], dim=1)
        route_weights = F.softmax(ent_scores / self.temperature, dim=1)
        
        w_t = route_weights[:, 0:1]
        w_v = route_weights[:, 1:2]
        w_m = route_weights[:, 2:3]
        
        # 4. 动态加权融合
        logits_final = (w_t * logits_t) + (w_v * logits_v) + (w_m * logits_m)
        
        if return_weights:
            return logits_final, route_weights, (ent_t, ent_v, ent_m)
        return logits_final


def train_and_eval(model, text_feat, vis_feat, labels, train_idx, val_idx, test_idx, args, device):
    model = model.to(device)
    optimizer, _ = initialize_optimizer_and_scheduler(args, model)
    
    best_val_acc = -1
    best_test_preds = None
    
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(text_feat, vis_feat)
        loss = cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        
        model.eval()
        with th.no_grad():
            logits = model(text_feat, vis_feat)
            val_preds = th.argmax(logits[val_idx], dim=1)
            val_acc = (val_preds == labels[val_idx]).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_preds = th.argmax(logits[test_idx], dim=1)
                best_test_preds = test_preds.clone()
                
    return best_test_preds


def main():
    parser = argparse.ArgumentParser("Pre-Experiment: MLP Modality Bias Analysis")
    add_common_args(parser)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--out_file", type=str, default="pre_results.txt", help="Path to append the result table")
    args = parser.parse_args()

    set_seed(args.seed)
    device = th.device("cuda:%d" % args.gpu if th.cuda.is_available() and args.gpu != -1 else "cpu")

    graph, labels, train_idx, val_idx, test_idx = load_data(
        args.graph_path, train_ratio=args.train_ratio, val_ratio=args.val_ratio, 
        name=args.data_name, fewshots=args.fewshots
    )
    
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    
    text_feat = th.from_numpy(np.load(args.text_feature).astype(np.float32)).to(device)
    vis_feat = th.from_numpy(np.load(args.visual_feature).astype(np.float32)).to(device)
    n_classes = int((labels.max() + 1).item())

    text_dim = text_feat.shape[1]
    vis_dim = vis_feat.shape[1]
    
    print(f"Dataset: {args.data_name} | Test Nodes: {len(test_idx)}")
    
    print("Training Text-only MLP...")
    model_t = TextOnlyMLP(text_dim, args.embed_dim, args.n_hidden, n_classes, args.dropout)
    preds_t = train_and_eval(model_t, text_feat, vis_feat, labels, train_idx, val_idx, test_idx, args, device)

    print("Training Visual-only MLP...")
    model_v = VisualOnlyMLP(vis_dim, args.embed_dim, args.n_hidden, n_classes, args.dropout)
    preds_v = train_and_eval(model_v, text_feat, vis_feat, labels, train_idx, val_idx, test_idx, args, device)

    print("Training Multimodal MLP...")
    model_m = MultimodalMLP(text_dim, vis_dim, args.embed_dim, args.n_hidden, n_classes, args.dropout)
    preds_m = train_and_eval(model_m, text_feat, vis_feat, labels, train_idx, val_idx, test_idx, args, device)

    print("Training Ensemble MLP (Average Logits)...")
    model_ens = EnsembleMLP(text_dim, vis_dim, args.embed_dim, args.n_hidden, n_classes, args.dropout)
    preds_ens = train_and_eval(model_ens, text_feat, vis_feat, labels, train_idx, val_idx, test_idx, args, device)

    # 统计布尔矩阵
    test_labels = labels[test_idx]
    correct_t = (preds_t == test_labels).cpu().numpy()
    correct_v = (preds_v == test_labels).cpu().numpy()
    correct_m = (preds_m == test_labels).cpu().numpy()
    correct_ens = (preds_ens == test_labels).cpu().numpy()

    # 整体准确率
    acc_t = correct_t.mean() * 100
    acc_v = correct_v.mean() * 100
    acc_m = correct_m.mean() * 100
    acc_ens = correct_ens.mean() * 100
    
    # ---------------- 绘制表格并保存到文件 ----------------
    total_test = len(test_idx)
    
    table_lines = []
    table_lines.append(f"| {'Text':<5} | {'Vis':<5} | {'Multi':<5} | {'Ensemble':<8} | {'Count':<7} | {'Percent':<7} |")
    table_lines.append(f"|-------|-------|-------|----------|---------|---------|")
    
    # 使用 0~15 二进制遍历 2^4 = 16 种组合
    for i in range(15, -1, -1):
        # 取每一位作为布尔值: 8=Text, 4=Vis, 2=Multi, 1=Ensemble
        b_t = bool(i & 8)
        b_v = bool(i & 4)
        b_m = bool(i & 2)
        b_e = bool(i & 1)
        
        # 匹配对应状态的节点
        mask = (correct_t == b_t) & (correct_v == b_v) & (correct_m == b_m) & (correct_ens == b_e)
        count = mask.sum()
        pct = (count / total_test) * 100
        
        sym_t = "O" if b_t else "X"
        sym_v = "O" if b_v else "X"
        sym_m = "O" if b_m else "X"
        sym_e = "O" if b_e else "X"
        
        table_lines.append(f"|   {sym_t}   |   {sym_v}   |   {sym_m}   |    {sym_e}     | {count:7d} | {pct:6.2f}% |")

    result_str = "\n".join(table_lines)
    
    # 打印到控制台
    print("\n" + "="*60)
    print(" " * 20 + "OVERALL ACCURACY")
    print("="*60)
    print(f"Text-Only MLP       : {acc_t:.2f}%")
    print(f"Visual-Only MLP     : {acc_v:.2f}%")
    print(f"Multimodal MLP      : {acc_m:.2f}%")
    print(f"Ensemble MLP        : {acc_ens:.2f}%")
    print("\n" + "="*60)
    print(" " * 15 + "DETAILED OVERLAP STATISTICS (16 Sets)")
    print("="*60)
    print(result_str)

    # 写入聚合文件
    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "a", encoding="utf-8") as f:
            f.write(f"\n[{args.data_name}] Feature: {os.path.basename(args.text_feature).split('_')[1]} | Hidden: {args.n_hidden} | LR: {args.lr}\n")
            f.write(f"------------------------------------------------------------\n")
            f.write(f"Accuracies -> Text: {acc_t:.2f}% | Vis: {acc_v:.2f}% | Multi: {acc_m:.2f}% | Ens: {acc_ens:.2f}%\n")
            f.write(f"------------------------------------------------------------\n")
            f.write(result_str + "\n")
            f.write(f"============================================================\n")

if __name__ == "__main__":
    main()