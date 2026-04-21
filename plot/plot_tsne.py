import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import dgl
import os
import argparse
from pathlib import Path
import matplotlib.cm as cm

# ==========================================
# 1. 论文级绘图配置 (复刻参考图风格)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 基础字号
FONT_SIZE_TICKS = 19
FONT_SIZE_LABEL = 20
FONT_SIZE_TITLE = 22

def split_graph(nodes_num, train_ratio, val_ratio, labels, fewshots=None):
    np.random.seed(42)
    indices = np.random.permutation(nodes_num)
    train_size = int(nodes_num * train_ratio)
    val_size = int(nodes_num * val_ratio)
    train_ids = indices[:train_size]
    val_ids = indices[train_size:train_size + val_size]
    test_ids = indices[train_size + val_size:]
    return train_ids, val_ids, test_ids

def load_data(graph_path, train_ratio=0.6, val_ratio=0.2):
    try:
        glist, _ = dgl.load_graphs(graph_path)
        graph = glist[0]
        labels = graph.ndata['label'].numpy()
        train_idx, val_idx, test_idx = split_graph(graph.num_nodes(), train_ratio, val_ratio, labels)
        return graph, labels, train_idx, val_idx, test_idx
    except Exception as e:
        print(f"Failed to load graph from {graph_path}: {e}")
        return None, None, None, None, None

def get_feature_paths(data_root, datasets=["Reddit-M", "Grocery"]):
    paths = {}
    for dataset in datasets:
        prefix = dataset.replace("-", "")
        seq_len = "100" if "Reddit" in dataset else "256"
        key_text = f'{dataset.lower().replace("-", "_")}_text'
        key_visual = f'{dataset.lower().replace("-", "_")}_visual'
        
        paths[key_text] = os.path.join(data_root, dataset, 
                                       f"TextFeature/{prefix}_Llama_3.2_11B_Vision_Instruct_{seq_len}_mean.npy")
        paths[key_visual] = os.path.join(data_root, dataset, 
                                         f"ImageFeature/{prefix}_Llama-3.2-11B-Vision-Instruct_visual.npy")
    return paths

def load_features(path, indices):
    if not os.path.exists(path):
        return None
    try:
        full_feat = np.load(path).astype(np.float32)
        return full_feat[indices]
    except Exception as e:
        return None

def compute_tsne(features, perplexity=30, seed=42):
    print(f"  Computing t-SNE (perplexity={perplexity})...")
    return TSNE(n_components=2, init='pca', learning_rate='auto',
                perplexity=perplexity, random_state=seed, n_jobs=-1).fit_transform(features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/hyperai/input/input0/MAGB_Dataset")
    parser.add_argument("--output_dir", type=str, default="./figures/paper")
    parser.add_argument("--sample_size", type=int, default=2000, help="参考图中点比较密集，可以用2000左右")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    datasets = ["Grocery", "Reddit-M"]
    
    # 1. 加载数据
    dataset_info = {}
    for dataset in datasets:
        prefix = dataset.replace("-", "")
        graph_path = os.path.join(args.data_root, dataset, f"{prefix}Graph.pt")
        print(f"Loading {dataset}...")
        graph, labels, _, _, test_idx = load_data(graph_path)
        
        if graph is None: return
        
        # 采样
        sample_indices = test_idx.copy()
        if len(sample_indices) > args.sample_size:
            np.random.seed(42)
            sample_indices = np.random.choice(sample_indices, args.sample_size, replace=False)
        
        dataset_info[dataset] = {
            'labels': labels,
            'sample_indices': sample_indices,
            'sample_labels': labels[sample_indices]
        }

    # 2. 计算 t-SNE
    feature_paths = get_feature_paths(args.data_root, datasets)
    embeddings = {}
    
    for key, path in feature_paths.items():
        dataset = "Reddit-M" if "reddit" in key else "Grocery"
        sample_indices = dataset_info[dataset]['sample_indices']
        feat = load_features(path, sample_indices)
        if feat is None: continue
        perplexity = 45 if "reddit" in key else 30
        embeddings[key] = compute_tsne(feat, perplexity=perplexity)

    # ==========================================
    # 3. 绘图 (严格复刻参考图风格)
    # ==========================================
    print(f"\n🎨 Plotting...")
    
    # 每个子图正方形，4个子图排列
    fig, axes = plt.subplots(1, 4, figsize=(32, 6), constrained_layout=True)
    
    plot_configs = [
        ('grocery_text', 'Grocery (Text)'),
        ('grocery_visual', 'Grocery (Visual)'),
        ('reddit_m_text', 'Reddit-M (Text)'),
        ('reddit_m_visual', 'Reddit-M (Visual)')
    ]
    
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#0082a6', '#ffffc9', '#ef7368'] 
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256) 

    for idx, (ax, (key, legend_label)) in enumerate(zip(axes, plot_configs)):
        if key not in embeddings: continue
        
        emb = embeddings[key]
        dataset = "Reddit-M" if "reddit" in key else "Grocery"
        labels = dataset_info[dataset]['sample_labels']
        
        scatter = ax.scatter(
            emb[:, 0], emb[:, 1], 
            c=labels, 
            cmap=cmap, 
            s=200,
            marker='*', 
            alpha=0.75,
            linewidths=2.1,
            edgecolors='face',
            label=legend_label
        )
        scatter.set_capstyle('round')
        scatter.set_joinstyle('round')
        
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS, direction='out')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)
            
        # 设置坐标轴范围略微宽松一点，防止点贴边
        # 也可以手动设置 xlim/ylim 来对齐，比如 [-70, 70]
        # ax.set_xlim(-70, 70) 
        # ax.set_ylim(-70, 70)

        # 图例
        legend = ax.legend(loc='upper right', fontsize=FONT_SIZE_LABEL, 
                          handletextpad=0.1, borderpad=0.4, 
                          frameon=True, fancybox=False, edgecolor='black')
        legend.get_frame().set_linewidth(0.5)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label('Classes', rotation=90, labelpad=12, fontsize=FONT_SIZE_LABEL)
        cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS)
        cbar.outline.set_linewidth(0.8)

    # 保存
    save_path = output_path / 'llama_tsne_reference_style.pdf'
    plt.savefig(save_path, dpi=args.dpi, bbox_inches='tight', format='pdf')
    print(f"\n✅ Reference Style PDF saved to: {save_path}")
    
    plt.savefig(output_path / 'llama_tsne_reference_style.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()