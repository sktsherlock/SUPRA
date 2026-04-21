# 文本和图像特征的 NTK 余弦相似度 & 谱范数/最大特征值

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

DATA_ROOT = "/hyperai/input/input0/MAGB_Dataset"
DATASETS = ["Movies", "Grocery", "Toys", "Reddit-M", "Reddit-S"]

# Feature paths (Group: clip_roberta)
# Based on TEXT_FEATURE_BY_DS_GROUP["...|clip_roberta"]
FEATURE_PATHS = {
    "Movies": {
        "text": "TextFeature/Movies_roberta_base_512_mean.npy",
        "image": "ImageFeature/Movies_openai_clip-vit-large-patch14.npy"
    },
    "Grocery": {
        "text": "TextFeature/Grocery_roberta_base_256_mean.npy",
        "image": "ImageFeature/Grocery_openai_clip-vit-large-patch14.npy"
    },
    "Toys": {
        "text": "TextFeature/Toys_roberta_base_512_mean.npy",
        "image": "ImageFeature/Toys_openai_clip-vit-large-patch14.npy"
    },
    "Reddit-M": {
        "text": "TextFeature/RedditM_roberta_base_100_mean.npy",
        "image": "ImageFeature/RedditM_openai_clip-vit-large-patch14.npy"
    },
    "Reddit-S": {
        "text": "TextFeature/RedditS_roberta_base_100_mean.npy",
        "image": "ImageFeature/RedditS_openai_clip-vit-large-patch14.npy"
    }
}

OUTPUT_DIR = "figures/ntk_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_feature(dataset, modality):
    """Load feature matrix from .npy file"""
    rel_path = FEATURE_PATHS[dataset][modality]
    full_path = os.path.join(DATA_ROOT, dataset, rel_path)
    
    print(f"Loading {dataset} {modality} from {full_path}...")
    
    if not os.path.exists(full_path):
        print(f"Error: File not found: {full_path}")
        return None
        
    try:
        data = np.load(full_path)
        return torch.from_numpy(data).float()
    except Exception as e:
        print(f"Error loading {full_path}: {e}")
        return None

def analyze_kernels(X1, X2, label1="Text", label2="Image", normalize=False):
    """
    Compute Cosine Similarity of NTK matrices and Max Eigenvalues.
    Ref: <K1, K2>_F / (||K1||_F ||K2||_F)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X1 = X1.to(device)
    X2 = X2.to(device)
    
    if X1.shape[0] != X2.shape[0]:
        min_len = min(X1.shape[0], X2.shape[0])
        print(f"Warning: Shape mismatch {X1.shape} vs {X2.shape}. Truncating to {min_len}.")
        X1 = X1[:min_len]
        X2 = X2[:min_len]

    # Compute raw norms for reporting
    raw_norm1 = torch.norm(X1, dim=1).mean().item()
    raw_norm2 = torch.norm(X2, dim=1).mean().item()
    print(f"  [Raw] Avg Norm {label1}: {raw_norm1:.4f}")
    print(f"  [Raw] Avg Norm {label2}: {raw_norm2:.4f}")

    # Optional: L2 Normalize features
    if normalize:
        print("  [Info] Applying L2 normalization to features before Kernel computation...")
        X1 = torch.nn.functional.normalize(X1, p=2, dim=1)
        X2 = torch.nn.functional.normalize(X2, p=2, dim=1)

    with torch.no_grad():
        # 1. Compute Gram matrices inner products efficiently using the trace property
        # <K1, K2>_F = || X1.T @ X2 ||_F^2
        print("  Computing Cross-Covariance...")
        cov_12 = torch.mm(X1.T, X2)
        numerator = torch.norm(cov_12) ** 2
        
        # ||K1||_F = || X1.T @ X1 ||_F
        print(f"  Computing {label1} Self-Covariance...")
        cov_11 = torch.mm(X1.T, X1)
        norm1 = torch.norm(cov_11)
        
        # ||K2||_F = || X2.T @ X2 ||_F
        print(f"  Computing {label2} Self-Covariance...")
        cov_22 = torch.mm(X2.T, X2)
        norm2 = torch.norm(cov_22)
        
        similarity = (numerator / (norm1 * norm2)).item()
        
        # 2. Compute Spectral Norms (Lambda_max)
        # Lambda_max(K) = Lambda_max(X X^T) = Lambda_max(X^T X)
        print("  Computing Eigenvalues...")
        
        # Eigvals of cov_11
        l1 = torch.linalg.eigvalsh(cov_11)[-1].item()
        
        # Eigvals of cov_22
        l2 = torch.linalg.eigvalsh(cov_22)[-1].item()
        
    return {
        "similarity": similarity,
        f"lambda_max_{label1}": l1,
        f"lambda_max_{label2}": l2,
        "avg_norm_Text": raw_norm1,
        "avg_norm_Image": raw_norm2,
        "lambda_ratio": l1 / l2 if l2 > 0 else 0
    }

def main():
    results_raw = []
    results_norm = []
    
    for dataset in DATASETS:
        print(f"\nProcessing {dataset}...")
        feat_text = load_feature(dataset, "text")
        feat_img = load_feature(dataset, "image")
        
        if feat_text is None or feat_img is None:
            print(f"Skipping {dataset} due to missing files.")
            continue
            
        # Analysis 1: Raw Dynamics
        print("--- Mode: Raw Features ---")
        res_raw = analyze_kernels(feat_text, feat_img, normalize=False)
        res_raw["dataset"] = dataset
        results_raw.append(res_raw)
        
        # Analysis 2: Normalized Geometry
        print("--- Mode: Normalized Features ---")
        res_norm = analyze_kernels(feat_text, feat_img, normalize=True)
        res_norm["dataset"] = dataset
        results_norm.append(res_norm)
        
        print(f"  => [Raw] Lambda Ratio: {res_raw['lambda_ratio']:.4f}")
        print(f"  => [Norm] Lambda Ratio: {res_norm['lambda_ratio']:.4f}")

    if not results_raw:
        print("No results to plot.")
        return

    # ================= Plotting =================
    sns.set_style("whitegrid")
    
    # --- Plot 1: Similarity Comparison (Raw vs Norm) ---
    plt.figure(figsize=(12, 6))
    
    datasets = [r["dataset"] for r in results_raw]
    sim_raw = [r["similarity"] for r in results_raw]
    sim_norm = [r["similarity"] for r in results_norm]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    plt.bar(x - width/2, sim_raw, width, label='Raw Similarity', color='purple', alpha=0.9)
    plt.bar(x + width/2, sim_norm, width, label='Normalized Similarity', color='mediumorchid', alpha=0.9)
    
    plt.title("NTK Cosine Similarity: Raw vs Normalized Features", fontsize=14)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(x, datasets)
    plt.legend()
    
    # Add values
    for i, v in enumerate(sim_raw):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate(sim_norm):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ntk_similarity_comparison.png"), dpi=300)
    print("Saved similarity comparison plot.")
    
    # --- Plot 2: Raw Eigenvalues ---
    plt.figure(figsize=(10, 6))
    l_text = [r["lambda_max_Text"] for r in results_raw]
    l_img = [r["lambda_max_Image"] for r in results_raw]
    
    plt.bar(x - width/2, l_text, width, label='Text $\lambda_{max}$', color='skyblue')
    plt.bar(x + width/2, l_img, width, label='Image $\lambda_{max}$', color='salmon')
    
    plt.ylabel("Maximum Eigenvalue $\lambda_{max}$ (Log Scale)")
    plt.title("Spectral Bias (Raw Features): Dominant Eigenvalues")
    plt.xticks(x, datasets)
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ntk_eigenvalues_raw.png"), dpi=300)
    print("Saved raw eigenvalues plot.")

    # --- Plot 3: Normalized Eigenvalues ---
    plt.figure(figsize=(10, 6))
    l_text_n = [r["lambda_max_Text"] for r in results_norm]
    l_img_n = [r["lambda_max_Image"] for r in results_norm]
    
    plt.bar(x - width/2, l_text_n, width, label='Text $\lambda_{max}$', color='dodgerblue')
    plt.bar(x + width/2, l_img_n, width, label='Image $\lambda_{max}$', color='orangered')
    
    # Add text for ratios
    for i in range(len(datasets)):
        ratio = l_text_n[i] / l_img_n[i] if l_img_n[i] > 0 else 0
        plt.text(i, max(l_text_n[i], l_img_n[i]) * 1.1, f'Ratio:{ratio:.1f}x', ha='center', fontsize=9, fontweight='bold')

    plt.ylabel("Maximum Eigenvalue $\lambda_{max}$ (Log Scale)")
    plt.title("Spectral Bias (Normalized Features): Intrinsic Complexity")
    plt.xticks(x, datasets)
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ntk_eigenvalues_normalized.png"), dpi=300)
    print("Saved normalized eigenvalues plot.")
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
