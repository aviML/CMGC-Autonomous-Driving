import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc
import json
from datetime import datetime
import scipy.stats as stats

def compute_significance_and_cohens(clear_rhos, adv_rhos):
    """Computes statistical significance using Welch's t-test directly on raw values (valid via CLT)."""
    if not clear_rhos or not adv_rhos: return 1.0, 0.0
    
    n1, n2 = len(clear_rhos), len(adv_rhos)
    if n1 < 30 or n2 < 30:
        print(f"  [WARN] Small sample size (clear={n1}, adv={n2}): Welch's t-test p-value may be unreliable.")
        
    t_stat, p_val = stats.ttest_ind(clear_rhos, adv_rhos, equal_var=False)
    
    var1, var2 = np.var(clear_rhos, ddof=1), np.var(adv_rhos, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (np.mean(clear_rhos) - np.mean(adv_rhos)) / (pooled_std + 1e-8)
    
    return float(p_val), float(cohens_d)

def evaluate_streaming_radar(base_dir="/home/avnish/radar_camera_PR/outputs/cmgc_radar_dinov2/"):
    base_path = Path(base_dir)
    
    clear_dir = base_path / "clear"
    night_dir = base_path / "night"
    rain_dir = base_path / "rain"
    extreme_dir = base_path / "rain_night_mixed"
    
    clear_files = sorted(list(clear_dir.glob("*.pkl")))
    if not clear_files:
        print(f"[ERROR] No baseline data found in {clear_dir}")
        return

    rng_sample = np.random.default_rng(seed=42)
    rng_shuffle = np.random.default_rng(seed=123)

    # =========================================================================
    # PASS 1: Scene Discovery & Strict Splitting
    # =========================================================================
    print("PASS 1: Scanning Clear Baseline for Sequence-Level Train/Test Split...")
    baseline_scenes = set()
    
    for f_path in tqdm(clear_files, desc="Scanning Clear Chunks"):
        with open(f_path, 'rb') as f:
            chunk_data = pickle.load(f)
            
            if isinstance(chunk_data, dict):
                if 'clear' not in chunk_data:
                    print(f"  [WARN] 'clear' key missing in {f_path.name}, skipping.")
                    del chunk_data
                    gc.collect()
                    continue
                frames = chunk_data['clear']
            else:
                frames = chunk_data
                
            for s in frames:
                baseline_scenes.add(s['scene_token'])
            del chunk_data
            gc.collect()

    baseline_scenes = sorted(list(baseline_scenes))
    train_scenes, test_scenes = train_test_split(baseline_scenes, test_size=0.3, random_state=42)
    train_set, test_set = set(train_scenes), set(test_scenes)
    
    print(f"  -> Discovered {len(baseline_scenes)} Clear Scenes.")
    print(f"  -> Training Manifold on {len(train_scenes)} Scenes, Testing on {len(test_scenes)} Scenes.")

    # =========================================================================
    # PASS 2: Stream & Fit the Manifold
    # =========================================================================
    print("\nPASS 2: Streaming Training Data & Fitting PCA/CCA...")
    train_V, train_R = [], []
    
    for f_path in tqdm(clear_files, desc="Extracting Train Patches"):
        with open(f_path, 'rb') as f:
            chunk_data = pickle.load(f)
            
        if isinstance(chunk_data, dict):
            if 'clear' not in chunk_data: 
                del chunk_data
                gc.collect()
                continue
            frames = chunk_data['clear']
        else:
            frames = chunk_data
        
        for s in frames:
            if s['scene_token'] in train_set:
                V = np.array(s.get('radar_patch_features', []), dtype=np.float32)
                
                # LOUD WARNING GUARDS FOR RADAR
                depths_raw = s.get('radar_depths', None)
                if depths_raw is None:
                    print(f"  [WARN] 'radar_depths' missing for token {s.get('sample_token', '?')}")
                    continue
                depths = np.array(depths_raw, dtype=np.float32)
                
                rcs_raw = s.get('radar_rcs', None)
                if rcs_raw is None:
                    print(f"  [WARN] 'radar_rcs' missing for token {s.get('sample_token', '?')}")
                    continue
                rcs = np.array(rcs_raw, dtype=np.float32)
                
                vel_raw = s.get('radar_velocities', None)
                if vel_raw is None:
                    print(f"  [WARN] 'radar_velocities' missing for token {s.get('sample_token', '?')}")
                    continue
                velocities = np.array(vel_raw, dtype=np.float32)
                
                if len(V) == 0 or len(depths) == 0: continue
                
                vel_mag = np.linalg.norm(velocities, axis=1) if len(velocities) > 0 else np.zeros_like(depths)
                
                R = np.column_stack((depths, rcs, vel_mag))
                n_patches = len(V)
                
                if n_patches > 0:
                    sample_size = max(1, int(n_patches * 0.10)) 
                    idx = rng_sample.choice(n_patches, sample_size, replace=False)
                    train_V.append(V[idx])
                    train_R.append(R[idx])
                
        del chunk_data
        gc.collect()

    if not train_V:
        print("\n[ERROR] No valid training patches extracted. Check clear_dir and scene split.")
        return

    V_train = np.vstack(train_V)
    R_train = np.vstack(train_R)
    print(f"  -> Memory Safe: Fitting Manifold on {len(V_train)} spatial patches.")

    # MANDATORY FOR RADAR
    scaler_R = StandardScaler()
    R_train_scaled = scaler_R.fit_transform(R_train)

    pca = PCA(n_components=32)
    V_train_pca = pca.fit_transform(V_train)
    
    cca = CCA(n_components=1)
    cca.fit(V_train_pca, R_train_scaled)
    
    del train_V, train_R, V_train, R_train, V_train_pca, R_train_scaled
    gc.collect()

    # =========================================================================
    # PASS 3: Stream & Score
    # =========================================================================
    print("\nPASS 3: Streaming Evaluation...")
    
    def score_directory(directory, condition_key, is_test_baseline=False):
        if not directory.exists(): return [], [], []
        files = sorted(list(directory.glob("*.pkl")))
        if not files: return [], [], []
        
        rhos, physics_l2, rhos_shuff = [], [], []
        
        for f_path in tqdm(files, desc=f"Scoring {condition_key.upper()}"):
            with open(f_path, 'rb') as f:
                chunk_data = pickle.load(f)
                
            if isinstance(chunk_data, dict):
                if condition_key not in chunk_data:
                    print(f"\n  [WARN] Key '{condition_key}' not in chunk {f_path.name}, skipping to prevent leakage.")
                    del chunk_data
                    gc.collect()
                    continue
                frames = chunk_data[condition_key]
            else:
                frames = chunk_data
            
            for s in frames:
                if is_test_baseline and s['scene_token'] not in test_set: continue
                    
                V = np.array(s.get('radar_patch_features', []), dtype=np.float32)
                
                # LOUD WARNING GUARDS FOR RADAR
                depths_raw = s.get('radar_depths', None)
                if depths_raw is None:
                    print(f"  [WARN] 'radar_depths' missing for token {s.get('sample_token', '?')}")
                    continue
                depths = np.array(depths_raw, dtype=np.float32)
                
                if len(V) < 2 or len(depths) == 0: continue
                
                rcs_raw = s.get('radar_rcs', None)
                if rcs_raw is None:
                    print(f"  [WARN] 'radar_rcs' missing for token {s.get('sample_token', '?')}")
                    continue
                rcs = np.array(rcs_raw, dtype=np.float32)
                
                vel_raw = s.get('radar_velocities', None)
                if vel_raw is None:
                    print(f"  [WARN] 'radar_velocities' missing for token {s.get('sample_token', '?')}")
                    continue
                velocities = np.array(vel_raw, dtype=np.float32)
                
                vel_mag = np.linalg.norm(velocities, axis=1) if len(velocities) > 0 else np.zeros_like(depths)
                
                R = np.column_stack((depths, rcs, vel_mag))
                
                R_scaled = scaler_R.transform(R)
                V_pca = pca.transform(V)
                
                V_c, R_c = cca.transform(V_pca, R_scaled)
                
                delta = V_c[:, 0] - R_c[:, 0]
                rho = float(np.mean(np.clip(1.0 - 0.5 * delta**2, 0.0, 1.0)))
                rhos.append(rho)
                
                frame_phys_l2 = float(np.mean(np.linalg.norm(R_scaled, axis=1)))
                physics_l2.append(frame_phys_l2)
                
                R_shuff_scaled = R_scaled.copy()
                rng_shuffle.shuffle(R_shuff_scaled) 
                
                V_c_shuff, R_c_shuff = cca.transform(V_pca, R_shuff_scaled)
                delta_shuff = V_c_shuff[:, 0] - R_c_shuff[:, 0]
                rho_shuff = float(np.mean(np.clip(1.0 - 0.5 * delta_shuff**2, 0.0, 1.0)))
                rhos_shuff.append(rho_shuff)
                
            del chunk_data
            gc.collect()
            
        return rhos, physics_l2, rhos_shuff

    results = {}
    results['test_clear'] = score_directory(clear_dir, 'clear', is_test_baseline=True)
    results['night'] = score_directory(night_dir, 'night')
    results['rain'] = score_directory(rain_dir, 'rain')
    results['rain_night_mixed'] = score_directory(extreme_dir, 'rain_night_mixed')

    # =========================================================================
    # FINAL STATISTICS & JSON EXPORT
    # =========================================================================
    print("\n" + "="*75)
    print("FINAL RESULTS: SPARSE RADAR BASELINE")
    print("="*75)
    
    clear_rhos, clear_phys_l2, clear_shuff = results['test_clear']
    if not clear_rhos:
        print("[ERROR] No Clear baseline frames scored. Exiting.")
        return
        
    print(f"Test-Clear |Rho|      : {np.mean(clear_rhos):.4f} ± {np.std(clear_rhos):.4f} (n={len(clear_rhos)})")
    print(f"Test-Clear Physics L2 : {np.mean(clear_phys_l2):.4f} ± {np.std(clear_phys_l2):.4f}")
    print(f"Test-Clear Shuffled   : {np.mean(clear_shuff):.4f} ± {np.std(clear_shuff):.4f} (Control)")
    
    json_export = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "test_clear_mean": float(np.mean(clear_rhos)), 
            "test_clear_std": float(np.std(clear_rhos)),
            "test_clear_phys_l2_mean": float(np.mean(clear_phys_l2)),
            "test_clear_n": len(clear_rhos),
            "shuffle_control_mean": float(np.mean(clear_shuff)),
            "shuffle_control_std": float(np.std(clear_shuff))
        },
        "conditions": {}
    }
    
    for condition in ['night', 'rain', 'rain_night_mixed']:
        adv_rhos, adv_phys_l2, adv_shuff = results.get(condition, ([], [],[]))
        if not adv_rhos: continue
            
        y_true = [0] * len(clear_rhos) + [1] * len(adv_rhos)
        y_scores_rho = [-x for x in (clear_rhos + adv_rhos)]
        y_scores_phys = clear_phys_l2 + adv_phys_l2
        
        auroc_rho = roc_auc_score(y_true, y_scores_rho)
        auroc_phys = roc_auc_score(y_true, y_scores_phys)
        p_val, cohens_d = compute_significance_and_cohens(clear_rhos, adv_rhos)
        
        print("-" * 75)
        print(f"Condition: {condition.upper()} (n={len(adv_rhos)})")
        print(f"  |Rho| Mean        : {np.mean(adv_rhos):.4f} ± {np.std(adv_rhos):.4f}")
        print(f"  Physics L2 Mean   : {np.mean(adv_phys_l2):.4f} ± {np.std(adv_phys_l2):.4f}")
        print(f"  Vision CMGC AUROC : {auroc_rho:.4f}")
        print(f"  Physics AUROC     : {auroc_phys:.4f}")
        print(f"  Welch's t-test    : p={p_val:.2e}")
        print(f"  Cohen's d         : {cohens_d:.3f}")
        
        json_export["conditions"][condition] = {
            "n_frames": len(adv_rhos),
            "vision_cmgc_auroc": float(auroc_rho),
            "physics_auroc": float(auroc_phys),
            "physics_l2_mean": float(np.mean(adv_phys_l2)),
            "welch_p_value": p_val,
            "cohens_d": cohens_d
        }

    output_file = base_path / "final_radar_baseline_results.json"
    with open(output_file, 'w') as f:
        json.dump(json_export, f, indent=4)
    print(f"\n[SUCCESS] Results successfully saved to: {output_file}")

if __name__ == "__main__":
    evaluate_streaming_radar()
