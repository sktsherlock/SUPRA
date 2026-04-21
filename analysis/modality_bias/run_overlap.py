import os
import sys
import copy
import argparse
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_ROOT)

from GNN.GraphData import load_data, set_seed
from GNN.Utils.NodeClassification import mag_classification, classification
from GNN.Baselines.Early_GNN import SimpleMAGGNN, ModalityEncoder, _build_gnn_backbone, _make_observe_graph_inductive

class SimpleUnimodalGNN(nn.Module):
    def __init__(self, encoder, gnn):
        super().__init__()
        self.encoder = encoder
        self.gnn = gnn

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, graph, feat):
        h = self.encoder(feat)
        return self.gnn(graph, h)

def get_args(dataset, features_path_dict):
    args = SimpleNamespace()
    args.model_name = "SAGE"
    args.n_hidden = 256
    args.n_layers = 3 
    args.dropout = 0.5
    args.lr = 0.001
    args.wd = 5e-4
    args.label_smoothing = 0.1
    args.early_stop_patience = 50
    args.aggregator = "mean"
    args.n_epochs = 500
    args.warmup_epochs = 10
    args.eval_steps = 1
    args.metric = "accuracy"
    args.average = "macro"
    args.gpu = 0
    args.seed = 42
    args.n_runs = 1
    
    args.data_name = dataset
    args.graph_path = features_path_dict[dataset]['graph']
    args.text_feature = features_path_dict[dataset]['text']
    args.visual_feature = features_path_dict[dataset]['visual']
    
    args.undirected = True
    args.selfloop = True
    args.inductive = False
    
    if dataset == "Reddit-M":
        args.train_ratio = 0.6
        args.val_ratio = 0.2
    else:
        args.train_ratio = 0.6
        args.val_ratio = 0.2

    args.mm_proj_dim = 256 
    args.mmcl_weight = 0.0
    args.modality_dropout = 0.0
    args.save_path = None
    args.log_every = 50
    args.min_lr = 1e-5
    
    return args

def run_unimodal(args, modality, device, graph, labels, train_idx, val_idx, test_idx, feature):
    print(f"--- Running Unimodal: {modality} ---")
    set_seed(args.seed)
    
    in_dim = feature.shape[1]
    proj_dim = args.mm_proj_dim
    encoder = ModalityEncoder(in_dim, proj_dim, args.dropout).to(device)
    gnn = _build_gnn_backbone(args, proj_dim, int(labels.max().item()) + 1, device)
    
    model = SimpleUnimodalGNN(encoder, gnn).to(device)
    
    observe_graph = copy.deepcopy(graph)
    if args.inductive:
         observe_graph = _make_observe_graph_inductive(graph, val_idx, test_idx)
    
    best_val, test_res, pred = classification(
        args, graph, observe_graph, model, feature, labels, train_idx, val_idx, test_idx, 
        n_running=1, return_pred=True
    )
    
    test_pred = pred[test_idx].argmax(dim=1)
    test_labels = labels[test_idx]
    correct_mask = (test_pred == test_labels).cpu().numpy()
    
    return correct_mask, test_res

def run_multimodal(args, fusion, device, graph, labels, train_idx, val_idx, test_idx, text_feat, vis_feat):
    print(f"--- Running Multimodal: {fusion} ---")
    set_seed(args.seed)
    args.early_fuse = fusion
    
    proj_dim = args.mm_proj_dim
    text_encoder = ModalityEncoder(text_feat.shape[1], proj_dim, args.dropout).to(device)
    visual_encoder = ModalityEncoder(vis_feat.shape[1], proj_dim, args.dropout).to(device)
    
    gnn_in_dim = proj_dim if fusion == 'sum' else 2 * proj_dim
    gnn = _build_gnn_backbone(args, gnn_in_dim, int(labels.max().item()) + 1, device)
    
    model = SimpleMAGGNN(text_encoder, visual_encoder, gnn, early_fuse=fusion).to(device)
    
    observe_graph = copy.deepcopy(graph)
    if args.inductive:
         observe_graph = _make_observe_graph_inductive(graph, val_idx, test_idx)
         
    best_val, test_res, extra = mag_classification(
        args, graph, observe_graph, model, text_feat, vis_feat, labels, train_idx, val_idx, test_idx, 
        n_running=1, return_extra=True
    )
    
    if 'best_state_dict' in extra:
        model.load_state_dict(extra['best_state_dict'])
    else:
        print("Warning: 'best_state_dict' not found in extra return, using final model state.")

    model.eval()
    with th.no_grad():
        pred = model(graph, text_feat, vis_feat)
        
    test_pred = pred[test_idx].argmax(dim=1)
    test_labels = labels[test_idx]
    correct_mask = (test_pred == test_labels).cpu().numpy()
    
    return correct_mask, test_res

def main():
    os.environ["WANDB_DISABLED"] = "true"
    
    # 请根据实际情况修改路径
    DATA_ROOT = "/hyperai/input/input0/MAGB_Dataset" 
    paths = {
        "Grocery": {
            "graph": f"{DATA_ROOT}/Grocery/GroceryGraph.pt",
            "text": f"{DATA_ROOT}/Grocery/TextFeature/Grocery_roberta_base_256_mean.npy",
            "visual": f"{DATA_ROOT}/Grocery/ImageFeature/Grocery_openai_clip-vit-large-patch14.npy"
        },
        "Reddit-M": {
            "graph": f"{DATA_ROOT}/Reddit-M/RedditMGraph.pt",
            "text": f"{DATA_ROOT}/Reddit-M/TextFeature/RedditM_roberta_base_100_mean.npy",
            "visual": f"{DATA_ROOT}/Reddit-M/ImageFeature/RedditM_openai_clip-vit-large-patch14.npy"
        }
    }
    
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    
    for dataset in ["Grocery", "Reddit-M"]:
        print(f"\n================ Processing Dataset: {dataset} ================")
        args = get_args(dataset, paths)
        
        # Load Data
        graph, labels, train_idx, val_idx, test_idx = load_data(
            args.graph_path, args.train_ratio, args.val_ratio, name=args.data_name
        )
        
        if args.undirected:
            srcs, dsts = graph.all_edges()
            graph.add_edges(dsts, srcs)
        if args.selfloop:
            graph = graph.remove_self_loop().add_self_loop()
            
        graph = graph.to(device)
        labels = labels.to(device)
        train_idx = train_idx.to(device)
        val_idx = val_idx.to(device)
        test_idx = test_idx.to(device)
        text_feat = th.from_numpy(np.load(args.text_feature).astype(np.float32)).to(device)
        vis_feat = th.from_numpy(np.load(args.visual_feature).astype(np.float32)).to(device)
        
        # 1. 单模态
        text_mask, text_acc = run_unimodal(args, "Text", device, graph, labels, train_idx, val_idx, test_idx, text_feat)
        vis_mask, vis_acc = run_unimodal(args, "Visual", device, graph, labels, train_idx, val_idx, test_idx, vis_feat)
        
        # 2. 多模态 (Concat vs Sum)
        masks = {}
        for fusion in ["concat", "sum"]:
            mm_mask, mm_acc = run_multimodal(args, fusion, device, graph, labels, train_idx, val_idx, test_idx, text_feat, vis_feat)
            masks[fusion] = mm_mask
        
        # 3. 结果验证
        diff_count = np.sum(masks['concat'] != masks['sum'])
        print(f"\n[DEBUG] {dataset}: Number of different predictions between Concat and Sum: {diff_count} / {len(test_idx)}")
        
        # 4. 计算统计数据
        stats_common = [
            np.mean(text_mask),                 # Text Correct
            np.mean(vis_mask),                  # Visual Correct
            np.mean(text_mask & vis_mask),      # Intersection (都对)
            np.mean(text_mask | vis_mask)       # Union Unimodal (至少一个对)
        ]
        
        gain_concat = np.mean(text_mask | vis_mask | masks['concat'])
        gain_sum = np.mean(text_mask | vis_mask | masks['sum'])
        
        print(f"Union (Single Models): {stats_common[3]:.4f}")
        print(f"Union (+Concat):       {gain_concat:.4f} (Gain: {gain_concat - stats_common[3]:.4f})")
        print(f"Union (+Sum):          {gain_sum:.4f} (Gain: {gain_sum - stats_common[3]:.4f})")

        # 5. 画图
        labels_chart = ["Text Only", "Visual Only", "Both Correct", "Unimodal Union", "Multi(Concat)", "Multi(Sum)"]
        values = stats_common + [gain_concat, gain_sum]
        colors = ['#CCCCCC', '#CCCCCC', '#AAAAAA', '#888888', '#1f77b4', '#ff7f0e'] # 灰色表示基准，彩色表示多模态
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels_chart, values, color=colors)
        plt.title(f"{dataset} - Multimodal Gain Analysis")
        plt.ylabel("Coverage / Accuracy")
        plt.ylim(min(values) - 0.05, max(values) + 0.02) # 自动缩放Y轴以突显差异
        
        # 在柱子上标数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
            
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f"overlap_{dataset}.png")
        print(f"Plot saved to overlap_{dataset}.png")

if __name__ == "__main__":
    main()