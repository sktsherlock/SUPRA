import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import os

# ============================================================================
# 数据配置区域
# ============================================================================

# 扰动比例 (横轴)
perturbation_ratios = [0, 20, 40, 60, 80, 100]  # Full, 20%, 40%, 60%, 80%, 100%

# -------------------- Reddit-M Data --------------------
reddit_m_data = {
    'mllm': {  # Top table
        'GCN': {
            'Visual-only': {'mean': [68.52, 68.14, 66.80, 62.08, 30.75, 0.82], 'std': [0.09, 0.13, 0.19, 0.26, 3.05, 0.25]},
            'EF-GNN':      {'mean': [72.06, 71.89, 71.05, 68.44, 58.21, 40.91], 'std': [0.12, 0.18, 0.53, 1.36, 3.74, 1.70]},
            'EF+CL':       {'mean': [72.26, 72.05, 70.80, 66.10, 53.42, 42.55], 'std': [0.04, 0.17, 0.41, 1.50, 3.00, 0.74]},
            'LF-GNN':      {'mean': [71.34, 71.26, 70.92, 68.89, 59.87, 40.52], 'std': [0.17, 0.10, 0.22, 0.71, 1.84, 0.88]},
            'LF+CL':       {'mean': [71.93, 71.77, 71.26, 69.42, 61.64, 47.18], 'std': [0.08, 0.04, 0.07, 0.63, 1.49, 1.14]},
            'OGM-GNN':     {'mean': [71.39, 71.23, 70.65, 68.57, 60.89, 46.25], 'std': [0.07, 0.15, 0.35, 0.66, 0.96, 0.55]},
            'Ours':        {'mean': [85.24, 85.20, 84.80, 83.75, 78.61, 57.95], 'std': [0.03, 0.04, 0.06, 0.13, 0.16, 1.01]},
        },
        'GraphSAGE': {
            'Visual-only': {'mean': [79.75, 79.05, 76.40, 67.21, 31.16, 1.41], 'std': [0.19, 0.20, 0.32, 1.09, 0.65, 0.09]},
            'EF-GNN':      {'mean': [85.36, 84.75, 83.12, 78.60, 68.11, 51.75], 'std': [0.12, 0.21, 0.59, 1.41, 1.55, 0.82]},
            'EF+CL':       {'mean': [85.52, 84.77, 81.99, 75.45, 66.24, 59.72], 'std': [0.11, 0.03, 0.16, 0.34, 0.42, 0.48]},
            'LF-GNN':      {'mean': [85.12, 84.81, 83.98, 80.84, 68.89, 42.02], 'std': [0.13, 0.12, 0.11, 0.07, 0.24, 0.44]},
            'LF+CL':       {'mean': [85.20, 84.99, 84.05, 80.94, 73.05, 55.55], 'std': [0.12, 0.09, 0.19, 0.24, 0.52, 0.92]},
            'OGM-GNN':     {'mean': [85.02, 84.86, 83.92, 81.28, 71.68, 47.09], 'std': [0.08, 0.07, 0.19, 0.24, 0.46, 0.66]},
            'Ours':        {'mean': [86.06, 85.74, 84.93, 82.48, 74.01, 56.50], 'std': [0.06, 0.17, 0.25, 0.41, 0.80, 0.70]},
        },
        'GAT': {
            'Visual-only': {'mean': [68.27, 68.08, 67.12, 62.41, 37.58, 1.27], 'std': [0.16, 0.10, 0.22, 0.81, 3.61, 0.20]},
            'EF-GNN':      {'mean': [71.96, 71.93, 71.34, 69.00, 59.44, 38.25], 'std': [0.12, 0.26, 0.53, 0.93, 2.08, 1.37]},
            'EF+CL':       {'mean': [72.09, 71.98, 71.06, 67.13, 55.34, 41.47], 'std': [0.10, 0.24, 0.49, 1.23, 1.81, 0.45]},
            'LF-GNN':      {'mean': [71.80, 71.59, 71.10, 69.18, 60.29, 39.03], 'std': [0.10, 0.11, 0.13, 0.16, 0.21, 0.21]},
            'LF+CL':       {'mean': [71.80, 71.73, 71.45, 70.07, 64.87, 51.68], 'std': [0.15, 0.22, 0.27, 0.37, 0.75, 0.76]},
            'OGM-GNN':     {'mean': [71.62, 71.43, 70.76, 68.37, 59.04, 43.96], 'std': [0.14, 0.10, 0.11, 0.33, 0.17, 0.27]},
            'Ours':        {'mean': [85.28, 85.23, 84.88, 83.69, 78.79, 57.62], 'std': [0.18, 0.20, 0.18, 0.29, 0.62, 1.01]},
        }
    },
    'clip': {  # Bottom table
        'GCN': {
            'Visual-only': {'mean': [68.68, 68.30, 66.95, 61.45, 31.03, 1.49], 'std': [0.09, 0.11, 0.29, 1.15, 3.04, 0.09]},
            'EF-GNN':      {'mean': [71.47, 71.17, 70.37, 66.95, 51.31, 23.95], 'std': [0.04, 0.26, 0.28, 0.73, 2.25, 0.27]},
            'EF+CL':       {'mean': [71.55, 71.28, 69.77, 63.18, 45.56, 28.11], 'std': [0.03, 0.04, 0.24, 0.12, 1.27, 1.09]},
            'LF-GNN':      {'mean': [70.89, 70.69, 70.24, 67.50, 53.86, 25.03], 'std': [0.04, 0.21, 0.11, 0.09, 0.55, 0.48]},
            'LF+CL':       {'mean': [71.45, 71.26, 70.93, 68.63, 59.59, 38.71], 'std': [0.05, 0.13, 0.02, 0.08, 0.27, 0.42]},
            'OGM-GNN':     {'mean': [70.63, 70.46, 69.67, 66.18, 52.09, 30.18], 'std': [0.16, 0.16, 0.12, 0.25, 0.31, 0.49]},
            'Ours':        {'mean': [83.91, 83.29, 80.72, 73.31, 55.96, 32.28], 'std': [0.22, 0.29, 0.40, 0.27, 0.66, 0.99]},
        },
        'GraphSAGE': {
            'Visual-only': {'mean': [79.42, 78.14, 74.36, 61.97, 23.24, 1.57], 'std': [0.04, 0.04, 0.12, 0.32, 1.03, 0.05]},
            'EF-GNN':      {'mean': [84.43, 83.70, 81.04, 73.27, 54.23, 27.96], 'std': [0.09, 0.23, 0.36, 0.55, 0.70, 0.59]},
            'EF+CL':       {'mean': [84.53, 83.93, 80.83, 72.56, 56.56, 37.55], 'std': [0.23, 0.22, 0.34, 1.23, 0.61, 0.57]},
            'LF-GNN':      {'mean': [84.24, 84.00, 82.62, 77.13, 56.19, 24.84], 'std': [0.07, 0.20, 0.03, 0.25, 0.49, 0.51]},
            'LF+CL':       {'mean': [84.32, 83.93, 82.50, 77.24, 61.07, 36.34], 'std': [0.23, 0.20, 0.30, 0.18, 0.35, 0.25]},
            'OGM-GNN':     {'mean': [84.11, 83.85, 82.49, 77.38, 58.05, 29.02], 'std': [0.16, 0.16, 0.22, 0.50, 0.36, 0.56]},
            'Ours':        {'mean': [84.42, 84.02, 82.17, 76.17, 57.66, 31.22], 'std': [0.11, 0.11, 0.16, 0.27, 0.47, 0.68]},
        },
        'GAT': {
            'Visual-only': {'mean': [68.67, 68.20, 66.87, 61.67, 33.21, 1.69], 'std': [0.13, 0.07, 0.29, 0.30, 0.23, 0.10]},
            'EF-GNN':      {'mean': [71.26, 71.03, 70.26, 67.46, 53.33, 23.61], 'std': [0.34, 0.34, 0.22, 0.43, 0.67, 0.96]},
            'EF+CL':       {'mean': [71.23, 71.11, 70.02, 66.25, 51.47, 28.61], 'std': [0.11, 0.07, 0.09, 0.23, 1.10, 0.38]},
            'LF-GNN':      {'mean': [71.19, 71.03, 70.33, 67.43, 53.64, 28.07], 'std': [0.23, 0.26, 0.40, 0.84, 1.60, 0.75]},
            'LF+CL':       {'mean': [71.67, 71.64, 71.15, 69.15, 60.15, 41.16], 'std': [0.10, 0.06, 0.09, 0.16, 0.11, 0.28]},
            'OGM-GNN':     {'mean': [70.78, 70.39, 69.32, 65.00, 51.18, 33.17], 'std': [0.11, 0.20, 0.11, 0.04, 0.35, 0.61]},
            'Ours':        {'mean': [83.98, 83.28, 80.98, 74.94, 58.19, 31.34], 'std': [0.17, 0.10, 0.38, 0.57, 0.37, 0.55]},
        }
    }
}

# -------------------- Grocery Data --------------------
grocery_data = {
    'mllm': {  # Top table
        'GCN': {
            'text-only':    {'mean': [73.30, 73.17, 71.84, 66.61, 42.11, 3.63], 'std': [0.42, 0.34, 0.23, 0.70, 1.78, 0.23]},
            'EF-GNN':      {'mean': [73.18, 72.36, 69.34, 62.89, 49.31, 32.42], 'std': [0.41, 0.11, 0.99, 2.88, 5.63, 4.21]},
            'EF+CL':       {'mean': [72.89, 70.67, 63.99, 46.20, 26.06, 20.02], 'std': [0.03, 0.44, 0.72, 0.76, 4.47, 4.98]},
            'LF-GNN':      {'mean': [72.42, 72.09, 71.44, 69.29, 64.66, 51.74], 'std': [0.82, 0.87, 0.72, 1.31, 2.17, 2.72]},
            'LF+CL':       {'mean': [72.02, 71.80, 71.29, 69.95, 67.35, 61.19], 'std': [0.29, 0.17, 0.06, 0.48, 1.00, 0.60]},
            'OGM-GNN':     {'mean': [71.88, 71.43, 70.51, 69.09, 64.71, 56.58], 'std': [0.88, 0.66, 0.41, 1.25, 2.29, 5.50]},
            'Ours':        {'mean': [81.34, 81.15, 80.09, 74.58, 58.72, 33.66], 'std': [0.29, 0.55, 0.14, 0.77, 1.65, 6.10]},
        },
        'GraphSAGE': {
            'text-only':   {'mean': [79.31, 79.17, 77.27, 65.91, 30.85, 4.28], 'std': [0.64, 0.65, 1.12, 3.46, 3.61, 0.25]},
            'EF-GNN':      {'mean': [79.86, 79.08, 73.85, 57.64, 40.73, 26.48], 'std': [0.24, 0.19, 1.86, 5.68, 6.19, 6.17]},
            'EF+CL':       {'mean': [80.09, 79.43, 74.47, 53.42, 33.71, 23.09], 'std': [0.32, 0.30, 1.43, 3.70, 2.77, 0.56]},
            'LF-GNN':      {'mean': [79.46, 79.18, 78.05, 75.58, 70.13, 57.72], 'std': [0.15, 0.21, 0.04, 0.42, 1.07, 1.54]},
            'LF+CL':       {'mean': [79.04, 78.78, 78.32, 76.19, 73.90, 70.09], 'std': [0.22, 0.22, 0.22, 0.18, 0.79, 0.28]},
            'OGM-GNN':     {'mean': [77.82, 77.32, 76.13, 73.74, 69.94, 62.81], 'std': [0.20, 0.39, 0.45, 0.67, 0.78, 1.54]},
            'Ours':        {'mean': [81.10, 81.01, 80.67, 76.79, 61.06, 28.28], 'std': [0.42, 0.24, 0.32, 0.65, 0.55, 0.87]},
        },
        'GAT': {
            'text-only':   {'mean': [72.80, 72.68, 70.21, 54.10, 20.67, 2.16], 'std': [0.67, 0.17, 0.81, 6.20, 9.27, 0.68]},
            'EF-GNN':      {'mean': [72.86, 72.33, 69.58, 59.78, 42.43, 21.04], 'std': [0.70, 0.28, 2.34, 7.41, 10.46, 6.01]},
            'EF+CL':       {'mean': [73.42, 72.42, 67.49, 49.88, 30.50, 19.20], 'std': [0.66, 0.75, 1.17, 3.83, 1.44, 1.89]},
            'LF-GNN':      {'mean': [73.16, 72.55, 71.49, 69.86, 67.08, 62.70], 'std': [0.46, 0.47, 0.31, 0.57, 0.21, 0.28]},
            'LF+CL':       {'mean': [71.31, 70.88, 70.48, 69.05, 67.04, 64.88], 'std': [0.91, 0.69, 1.06, 1.01, 0.21, 0.19]},
            'OGM-GNN':     {'mean': [71.86, 71.52, 70.19, 67.79, 63.83, 58.39], 'std': [1.04, 0.87, 0.31, 0.73, 1.80, 2.77]},
            'Ours':        {'mean': [81.28, 81.22, 80.03, 76.53, 59.61, 23.29], 'std': [0.50, 0.61, 0.28, 1.10, 2.55, 5.83]},
        }
    },
    'clip': {  # Bottom table
        'GCN': {
            'text-only':   {'mean': [69.30, 68.98, 66.72, 52.43, 23.85, 3.87], 'std': [0.09, 0.36, 0.87, 1.47, 2.98, 0.20]},
            'EF-GNN':      {'mean': [71.30, 69.84, 67.18, 64.77, 62.10, 57.83], 'std': [0.38, 0.50, 0.66, 0.96, 0.84, 1.32]},
            'EF+CL':       {'mean': [71.72, 69.80, 66.68, 63.22, 60.13, 56.42], 'std': [0.56, 0.27, 0.73, 1.68, 0.81, 1.16]},
            'LF-GNN':      {'mean': [70.96, 70.81, 70.41, 68.79, 63.06, 53.90], 'std': [0.78, 0.49, 0.13, 1.01, 0.70, 0.71]},
            'LF+CL':       {'mean': [70.58, 70.38, 69.39, 68.74, 66.26, 62.14], 'std': [0.27, 0.38, 0.09, 0.18, 0.61, 0.66]},
            'OGM-GNN':     {'mean': [71.14, 71.12, 70.27, 68.39, 64.59, 57.52], 'std': [0.21, 0.34, 0.20, 0.06, 0.57, 0.93]},
            'Ours':        {'mean': [73.69, 72.93, 71.65, 68.29, 62.33, 52.43], 'std': [1.05, 0.78, 1.17, 0.71, 0.85, 1.31]},
        },
        'GraphSAGE': {
            'text-only':   {'mean': [73.60, 72.54, 65.21, 45.08, 18.06, 4.95], 'std': [0.85, 0.22, 0.76, 1.48, 1.97, 0.32]},
            'EF-GNN':      {'mean': [75.94, 74.39, 71.91, 67.97, 64.58, 58.29], 'std': [0.16, 0.67, 0.96, 0.56, 1.01, 0.99]},
            'EF+CL':       {'mean': [76.16, 74.10, 70.64, 66.46, 62.27, 56.67], 'std': [0.16, 0.40, 0.34, 0.92, 1.53, 1.11]},
            'LF-GNN':      {'mean': [75.47, 74.67, 73.70, 69.93, 65.77, 59.46], 'std': [0.07, 0.37, 0.65, 0.72, 1.42, 1.23]},
            'LF+CL':       {'mean': [76.03, 75.35, 73.76, 71.11, 67.96, 63.76], 'std': [0.43, 0.38, 0.40, 0.43, 0.64, 0.48]},
            'OGM-GNN':     {'mean': [75.69, 75.03, 74.04, 69.87, 63.85, 55.85], 'std': [0.28, 0.21, 0.84, 0.14, 1.52, 2.18]},
            'Ours':        {'mean': [76.43, 75.17, 73.36, 67.60, 60.28, 49.33], 'std': [0.26, 0.28, 0.69, 1.03, 1.90, 0.66]},
        },
        'GAT': {
            'text-only':   {'mean': [67.90, 66.49, 61.11, 45.34, 17.22, 3.34], 'std': [0.98, 1.35, 3.18, 4.61, 4.37, 0.28]},
            'EF-GNN':      {'mean': [70.45, 69.08, 67.78, 64.50, 61.34, 55.00], 'std': [0.49, 0.44, 0.50, 0.60, 0.75, 1.72]},
            'EF+CL':       {'mean': [71.53, 70.55, 68.12, 64.38, 58.45, 50.50], 'std': [0.06, 0.18, 0.23, 0.87, 1.63, 1.93]},
            'LF-GNN':      {'mean': [69.70, 69.47, 68.82, 67.02, 64.19, 58.44], 'std': [0.47, 0.67, 0.62, 0.94, 1.60, 2.59]},
            'LF+CL':       {'mean': [70.05, 69.36, 69.69, 68.00, 66.84, 64.73], 'std': [0.61, 0.36, 0.62, 0.25, 0.44, 0.46]},
            'OGM-GNN':     {'mean': [69.54, 69.38, 67.40, 64.54, 59.24, 52.09], 'std': [0.43, 0.50, 0.91, 0.18, 0.72, 2.13]},
            'Ours':        {'mean': [73.64, 72.54, 70.61, 67.68, 63.29, 57.09], 'std': [0.15, 0.17, 0.15, 0.25, 0.10, 0.80]},
        }
    }
}

gnn_models = ['GCN', 'GraphSAGE', 'GAT']

# ============================================================================
# 绘图配置与美化
# ============================================================================

sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks")

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1.0,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'legend.frameon': False,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# 定义样式映射
method_styles = {
    # 'Visual-only':  {'c': '#7f7f7f', 'm': 'X', 'l': '--', 'lw': 1.5, 'ms': 6,  'z': 1}, 
    # 'text-only':    {'c': '#7f7f7f', 'm': 'X', 'l': '--', 'lw': 1.5, 'ms': 6,  'z': 1}, # New mapping for Grocery
    'EF-GNN':       {'c': '#1f77b4', 'm': 'o', 'l': '-',  'lw': 1.5, 'ms': 6,  'z': 2}, 
    'EF+CL':        {'c': '#aec7e8', 'm': 's', 'l': '-',  'lw': 1.5, 'ms': 6,  'z': 2}, 
    'LF-GNN':       {'c': '#2ca02c', 'm': '^', 'l': '-',  'lw': 1.5, 'ms': 6,  'z': 2}, 
    'LF+CL':        {'c': '#98df8a', 'm': 'v', 'l': '-',  'lw': 1.5, 'ms': 6,  'z': 2}, 
    'OGM-GNN':      {'c': '#9467bd', 'm': 'D', 'l': '-',  'lw': 1.5, 'ms': 5,  'z': 2}, 
    'Ours':         {'c': '#d62728', 'm': '*', 'l': '-',  'lw': 2.5, 'ms': 9,  'z': 4}, 
}

Y_LIMIT_CONFIG = {
    'Grocery': {
        'mllm': 8,
        'clip': 48
    },
    'Reddit-M': {
        'mllm': 28,
        'clip': 18
    }
}

Y_LIMIT_CONFIG2 = {
    'Grocery': {
        'mllm': 92,
        'clip': 82
    },
    'Reddit-M': {
        'mllm': 92,
        'clip': 92
    }
}

# ============================================================================
# 绘图函数
# ============================================================================

def plot_combined_robustness(data_dict, dataset_name, output_dir='figures/robustness'):
    """
    绘制 2x3 组合对比图 (Top: MLLM, Bottom: CLIP)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5)) 
    plt.subplots_adjust(wspace=0.15, hspace=0.35, top=0.88, bottom=0.1)
    
    legend_handles = []
    legend_labels = []
    
    # 提取 MLLM 和 CLIP 数据
    rows_config = [
        (data_dict.get('mllm', {}), 'MLLM'),
        (data_dict.get('clip', {}), 'CLIP+RoBERTa')
    ]
    
    for row_idx, (feat_data, feature_name) in enumerate(rows_config):
        feat_key = 'mllm' if row_idx == 0 else 'clip'
        y_bottom = Y_LIMIT_CONFIG.get(dataset_name, {}).get(feat_key, 0)
        y_top = Y_LIMIT_CONFIG2.get(dataset_name, {}).get(feat_key, 100)
        for col_idx, gnn_name in enumerate(gnn_models):
            ax = axes[row_idx, col_idx]
            
            # 如果某个模型数据缺失，跳过
            if gnn_name not in feat_data:
                ax.set_visible(False)
                continue
                
            gnn_data = feat_data[gnn_name]
            
            # 排序
            available_methods = [m for m in method_styles.keys() if m in gnn_data]
            excluded_methods = ['Visual-only', 'text-only']
            available_methods = [m for m in available_methods if m not in excluded_methods]
            sorted_methods = sorted(available_methods, key=lambda m: method_styles[m]['z'])

            for method_name in sorted_methods:
                stats = gnn_data[method_name]
                mean_vals = stats['mean']
                std_vals = stats['std']
                
                valid_indices = [i for i, v in enumerate(mean_vals) if v is not None]
                if not valid_indices:
                    continue
                
                x_vals = [perturbation_ratios[i] for i in valid_indices]
                y_mean = np.array([mean_vals[i] for i in valid_indices])
                y_std = np.array([std_vals[i] for i in valid_indices])
                
                style = method_styles[method_name]
                
                line, = ax.plot(x_vals, y_mean, 
                               color=style['c'], marker=style['m'], linestyle=style['l'],
                               linewidth=style['lw'], markersize=style['ms'],
                               markeredgecolor='white', markeredgewidth=0.5,
                               zorder=style['z'], label=method_name)
                
                ax.fill_between(x_vals, y_mean - y_std, y_mean + y_std, 
                               color=style['c'], alpha=0.15, linewidth=0, zorder=style['z']-10)
                
                # 收集图例 (只收集一次，尽量选一个覆盖全的子图)
                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(method_name)

            # 坐标轴设置
            ax.set_xticks(perturbation_ratios)
            ax.set_xticklabels(['Full', '20%', '40%', '60%', '80%', '100%'], fontsize=18)
            ax.grid(True)
            ax.set_ylim([y_bottom, y_top]) 
            
            # 第一行显示标题
            if row_idx == 0:
                ax.set_title(f'{gnn_name}', fontsize=20, fontweight='bold', pad=15)
            
            # 第一列显示特征名称
            if col_idx == 0:
                ax.set_ylabel(f'{feature_name}\nAccuracy (%)', fontsize=20, fontweight='bold')
            
            # 只有最后一行显示 x label
            if row_idx == 1:
                ax.set_xlabel('Visual Perturbation (%)', fontsize=20, fontweight='bold')
            
            sns.despine(ax=ax)

    # 统一图例
    fig.legend(legend_handles, legend_labels, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 1.05),  # 稍微向上移动以容纳两行
               ncol=3, 
               fontsize=18,
               frameon=False,
               columnspacing=1.0)
    
    filename = f'robustness_{dataset_name.lower().replace("-", "_")}.pdf'
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f'✓ Saved: {save_path}')
    plt.close()


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Robustness Analysis: Reddit-M & Grocery")
    print("=" * 70)
    
    # 绘制 Reddit-M
    print(f"\nPlotting Reddit-M...")
    plot_combined_robustness(reddit_m_data, 'Reddit-M')
    
    # 绘制 Grocery
    print(f"\nPlotting Grocery...")
    plot_combined_robustness(grocery_data, 'Grocery')
    
    print("\nAll plots generated successfully.")
