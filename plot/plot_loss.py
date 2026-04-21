# 绘制训练损失曲线，比较文本和视觉模态的收敛速度

import os
import sys
import argparse
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import dgl.nn as dglnn

# 论文级别配置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['figure.dpi'] = 300

# 引入项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GNN.GraphData import load_data, set_seed

# ================= 配置区域 (根据你的 Bash 脚本填入) =================
# 这里我们使用 "clip_roberta" 组的配置，因为它最常用
DATA_ROOT = "/hyperai/input/input0/MAGB_Dataset"

DATASET_CONFIG = {
    "Movies": {
        "graph": "Movies/MoviesGraph.pt",
        "text": "Movies/TextFeature/Movies_roberta_base_512_mean.npy",
        "image": "Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy",
        "split": {"train": 0.6, "val": 0.2}
    },
    "Grocery": {
        "graph": "Grocery/GroceryGraph.pt",
        "text": "Grocery/TextFeature/Grocery_roberta_base_256_mean.npy",
        "image": "Grocery/ImageFeature/Grocery_openai_clip-vit-large-patch14.npy",
        "split": {"train": 0.6, "val": 0.2}
    },
    "Toys": {
        "graph": "Toys/ToysGraph.pt",
        "text": "Toys/TextFeature/Toys_roberta_base_512_mean.npy",
        "image": "Toys/ImageFeature/Toys_openai_clip-vit-large-patch14.npy",
        "split": {"train": 0.6, "val": 0.2}
    },
    "Reddit-M": {
        "graph": "Reddit-M/RedditMGraph.pt",
        "text": "Reddit-M/TextFeature/RedditM_roberta_base_100_mean.npy",
        "image": "Reddit-M/ImageFeature/RedditM_openai_clip-vit-large-patch14.npy",
        "split": {"train": 0.6, "val": 0.2}
    }
    # ,
    # "Reddit-S": {
    #     "graph": "Reddit-S/RedditSGraph.pt",
    #     "text": "Reddit-S/TextFeature/RedditS_roberta_base_100_mean.npy",
    #     "image": "Reddit-S/ImageFeature/RedditS_openai_clip-vit-large-patch14.npy",
    #     "split": {"train": 0.2, "val": 0.2} # Reddit-S 特殊切分
    # }
}

# ================= 简单的 GCN 模型定义 =================
class SimpleGCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers=2, dropout=0.5):
        super(SimpleGCN, self).__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden, allow_zero_in_degree=True))
        # Hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(dglnn.GraphConv(n_hidden, n_hidden, allow_zero_in_degree=True))
        # Output layer
        self.layers.append(dglnn.GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
        return h

# ================= 训练与记录逻辑 =================
def train_and_record(dataset_name, modality, device, epochs=200, patience=50):
    cfg = DATASET_CONFIG[dataset_name]
    
    # 1. Load Data
    print(f"[{dataset_name}] Loading {modality} data...")
    graph_path = os.path.join(DATA_ROOT, cfg["graph"])
    
    # 调用现有的 load_data
    g, labels, train_idx, val_idx, test_idx = load_data(
        graph_path, 
        train_ratio=cfg["split"]["train"], 
        val_ratio=cfg["split"]["val"],
        name=dataset_name
    )
    
    # Load Features
    feat_path = os.path.join(DATA_ROOT, cfg[modality])
    features = th.from_numpy(np.load(feat_path).astype(np.float32))
    
    # Preprocessing
    g = g.remove_self_loop().add_self_loop().to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    
    # Model Setup
    n_classes = int(labels.max().item()) + 1
    in_feats = features.shape[1]
    
    model = SimpleGCN(in_feats, n_hidden=256, n_classes=n_classes, n_layers=3, dropout=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss_fcn = nn.CrossEntropyLoss()

    # Training Loop
    loss_history = []
    best_val_acc = 0
    early_stop_counter = 0
    stop_epoch = epochs

    for epoch in range(epochs):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_idx], labels[train_idx])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())

        # Validation for Early Stopping simulation
        model.eval()
        with th.no_grad():
            val_logits = model(g, features)
            val_pred = val_logits[val_idx].argmax(dim=1)
            val_acc = (val_pred == labels[val_idx]).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= patience:
                stop_epoch = epoch
                # Don't break, keep running to record the full curve for visualization
                # break 
    
    return loss_history, stop_epoch

# ================= 主程序与绘图 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")
    datasets = ["Movies", "Grocery", "Toys", "Reddit-M"]
    
    # Store results
    results = {} 
    
    print("Starting Convergence Analysis...")
    
    for ds in datasets:
        results[ds] = {}
        # Train Text
        print(f"\n--- Processing {ds} (Text) ---")
        loss_text, stop_text = train_and_record(ds, "text", device)
        results[ds]["text_loss"] = loss_text
        results[ds]["text_stop"] = stop_text
        
        # Train Image
        print(f"--- Processing {ds} (Image) ---")
        loss_image, stop_image = train_and_record(ds, "image", device)
        results[ds]["image_loss"] = loss_image
        results[ds]["image_stop"] = stop_image

    # Plotting
    print("\nGenerating Plots...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    
    for i, ds in enumerate(datasets):
        ax = axes[i]
        
        # 获取数据
        text_curve = results[ds]["text_loss"]
        img_curve = results[ds]["image_loss"]
        
        # 为了更好地观察收敛速度，通常看对数坐标，或者前50-100个epoch
        # 这里画全量的线性坐标，但你可以缩放到前100个
        epochs = range(len(text_curve))
        
        ax.plot(epochs, text_curve, label=f'Text (Stop: {results[ds]["text_stop"]})', color='#1f77b4', linewidth=1.5)
        ax.plot(epochs, img_curve, label=f'Visual (Stop: {results[ds]["image_stop"]})', color='#ff7f0e', linewidth=1.5)
        
        ax.set_title(f"{ds}", fontweight='normal')
        ax.set_xlabel("Epoch", fontsize=10)
        if i == 0:
            ax.set_ylabel("Training Loss", fontsize=10)
        
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        
        # 标注 Early Stop 点
        stop_t = results[ds]["text_stop"]
        stop_i = results[ds]["image_stop"]
        if stop_t < len(text_curve):
            ax.axvline(x=stop_t, color='#007aff', linestyle=':', alpha=0.5)
        if stop_i < len(img_curve):
            ax.axvline(x=stop_i, color='#ff4500', linestyle=':', alpha=0.5)

    plt.tight_layout()
    
    # Save as PDF for paper
    output_pdf = "figures/convergence_analysis.pdf"
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
    print(f"\nAnalysis complete! PDF saved to: {output_pdf}")
    
    # Also save PNG for preview
    output_png = "figures/convergence_analysis.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight', format='png', facecolor='white')
    print(f"PNG preview saved to: {output_png}")

if __name__ == "__main__":
    main()