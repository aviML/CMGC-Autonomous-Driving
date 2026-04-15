import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc

# Set academic plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def extract_r_matrix(s, modality):
    """Bulletproof extraction of physical signatures with correct keys."""
    if modality == 'lidar':
        depths = np.array(s.get('lidar_depth_means', []), dtype=np.float32)
        if len(depths) == 0: return None
        
        stds = np.array(s.get('lidar_depth_stds', np.zeros_like(depths)), dtype=np.float32)
        
        # CORRECT KEY: 'lidar_log_counts'
        log_counts_raw = s.get('lidar_log_counts', None)
        if log_counts_raw is None:
            log_counts = np.zeros_like(depths, dtype=np.float32)
        else:
            log_counts = np.array(log_counts_raw, dtype=np.float32)
                 
        return np.column_stack((depths, stds, log_counts))
        
    elif modality == 'radar':
        depths = s.get('radar_depths', [])
        if len(depths) == 0: return None
        
        rcs = s.get('radar_rcs', np.zeros_like(depths))
        
        if 'radar_velocities' in s:
            vel = np.linalg.norm(s['radar_velocities'], axis=1)
        else:
            vel = np.zeros_like(depths)
            
        return np.column_stack((depths, rcs, vel))
    return None


def get_rho_distributions(base_dir, max_train_files=250, modality='lidar'):
    """Extracts rho distributions using full held-out scoring and correct CCA metrics."""
    base_path = Path(base_dir)
    print(f"\n--- Processing {modality.upper()} from {base_path.name} ---")
    
    all_clear_files = sorted(list((base_path / "clear").glob("*.pkl")))
    if not all_clear_files:
        print(f"  [WARNING] No clear files found.")
        return {}
    
    # Split scenes to avoid in-sample overfitting (1 chunk = 1 scene in our pipeline)
    train_files, test_clear_files = train_test_split(all_clear_files, test_size=0.3, random_state=42)
    
    # Cap training files to 250 (covers >50% of the ~421 train scenes) for a highly representative manifold
    train_files_subset = train_files[:max_train_files]
    print(f"Fitting Manifold on {len(train_files_subset)} clear training chunks...")
        
    train_V, train_R = [], []
    rng = np.random.default_rng(42)
    
    # 1. FIT MANIFOLD
    for f_path in train_files_subset:
        with open(f_path, 'rb') as f:
            data = pickle.load(f)
        frames = list(data.values())[0] if isinstance(data, dict) else data
        
        for s in frames:
            V = np.array(s.get('lidar_patch_features', s.get('radar_patch_features', [])), dtype=np.float32)
            R = extract_r_matrix(s, modality)
            if len(V) > 0 and R is not None:
                # 10% patch subsampling controls RAM, allowing us to use a large chunk count
                idx = rng.choice(len(V), max(1, int(len(V)*0.10)), replace=False)
                train_V.append(V[idx])
                train_R.append(R[idx])

    V_train, R_train = np.vstack(train_V), np.vstack(train_R)
    
    scaler = StandardScaler().fit(R_train)
    pca = PCA(n_components=32).fit(V_train)
    cca = CCA(n_components=1).fit(pca.transform(V_train), scaler.transform(R_train))
    
    del train_V, train_R, V_train, R_train; gc.collect()

    # 2. SCORE FUNCTION (No caps on files, scores the entire distribution)
    def score_condition(cond_folder, specific_files=None):
        if specific_files:
            files = specific_files
        else:
            cond_path = base_path / cond_folder
            if not cond_path.exists(): return []
            files = sorted(list(cond_path.glob("*.pkl"))) # Evaluate all files
            
        if not files: return []
        
        rhos = []
        for f_path in tqdm(files, desc=f"Scoring {cond_folder}", leave=False):
            with open(f_path, 'rb') as f:
                data = pickle.load(f)
            frames = list(data.values())[0] if isinstance(data, dict) else data
            
            for s in frames:
                V = np.array(s.get('lidar_patch_features', s.get('radar_patch_features', [])), dtype=np.float32)
                R = extract_r_matrix(s, modality)
                if len(V) < 2 or R is None: continue
                    
                V_c, R_c = cca.transform(pca.transform(V), scaler.transform(R))
                delta = V_c[:, 0] - R_c[:, 0]
                rho = float(np.mean(np.clip(1.0 - 0.5 * delta**2, 0.0, 1.0)))
                rhos.append(rho)
        return rhos

    return {
        'Clear': score_condition("clear", specific_files=test_clear_files), # Score strictly held-out data
        'Night': score_condition("night"),
        'Rain': score_condition("rain"),
        'Rain+Night': score_condition("rain_night_mixed")
    }

def plot_distributions():
    # Update these paths if your root directory is different
    lidar_dir = "/home/avnish/radar_camera_PR/outputs/cmgc_lidar_dinov2/"
    radar_dir = "/home/avnish/radar_camera_PR/outputs/cmgc_radar_dinov2/"
    
    lidar_dist = get_rho_distributions(lidar_dir, max_train_files=250, modality='lidar')
   
    if 'Rain+Night' in lidar_dist and lidar_dist['Rain+Night']:
        rn_rhos = np.array(lidar_dist['Rain+Night'])
        print(f"\n[DIAGNOSTIC] Rain+Night LiDAR Bimodal Check:")
        print(f"  Total Rain+Night frames scored: {len(rn_rhos)}")
        print(f"  Frames below 0.70 (The collapse): {(rn_rhos < 0.70).sum()}")
        print(f"  Frames above 0.70 (The survivors): {(rn_rhos >= 0.70).sum()}")
        print(f"------------------------------------------\n")
    # ----------------------------------------
    radar_dist = get_rho_distributions(radar_dir, max_train_files=250, modality='radar')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {'Clear': '#2ca02c', 'Night': '#ff7f0e', 'Rain': '#1f77b4', 'Rain+Night': '#d62728'}
    
    # Left Plot: LiDAR (The True Signal)
    if lidar_dist:
        for cond, rhos in lidar_dist.items():
            if rhos:
                sns.kdeplot(rhos, ax=axes[0], label=cond, color=colors[cond], fill=True, alpha=0.3, linewidth=2)
        
        axes[0].set_title("A: Dense LiDAR Grounding (Signal Detected)", fontsize=16, fontweight='bold')
        axes[0].set_xlabel(r"Canonical Correlation $|\rho|$", fontsize=14)
        axes[0].set_ylabel("Density", fontsize=14)
        axes[0].legend(loc='upper left', fontsize=12)
        axes[0].set_xlim(0.4, 1.0)

    # Right Plot: Radar (The Confound)
    if radar_dist:
        for cond, rhos in radar_dist.items():
            if rhos:
                sns.kdeplot(rhos, ax=axes[1], label=cond, color=colors[cond], fill=True, alpha=0.3, linewidth=2)
                
        axes[1].set_title("B: Sparse Radar Grounding (Signal Confounded)", fontsize=16, fontweight='bold')
        axes[1].set_xlabel(r"Canonical Correlation $|\rho|$", fontsize=14)
        axes[1].set_ylabel("Density", fontsize=14)
        axes[1].legend(loc='upper left', fontsize=12)
        axes[1].set_xlim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig("rho_distributions.png", dpi=300, bbox_inches='tight')
    print("\n[SUCCESS] Plot saved to rho_distributions.png")

if __name__ == "__main__":
    plot_distributions()
