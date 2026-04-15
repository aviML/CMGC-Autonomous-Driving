import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import gc
import json
from datetime import datetime
import scipy.stats as stats

def compute_significance_and_cohens(clear_rhos, adv_rhos):
    """Computes significance using Welch's t-test and Cohen's d for effect size."""
    if not clear_rhos or not adv_rhos: return 1.0, 0.0
    
    n1, n2 = len(clear_rhos), len(adv_rhos)
    if n1 < 30 or n2 < 30:
        print(f"  [WARN] Small sample size (n_clear={n1}, n_adv={n2}).")
        
    t_stat, p_val = stats.ttest_ind(clear_rhos, adv_rhos, equal_var=False)
    
    var1, var2 = np.var(clear_rhos, ddof=1), np.var(adv_rhos, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (np.mean(clear_rhos) - np.mean(adv_rhos)) / (pooled_std + 1e-8)
    
    return float(p_val), float(cohens_d)

def evaluate_radiate_radar(base_dir="/home/avnish/radar_camera_PR/outputs/radiate_features/"):
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"[ERROR] Directory missing: {base_path}")
        return

    all_files = list(base_path.glob("*.pkl"))
    
    baseline_files = [f for f in all_files if "city" in f.name or "rural" in f.name]
    
    ood_files = {
        "night": [f for f in all_files if "night" in f.name],
        "fog": [f for f in all_files if "fog" in f.name or "tiny" in f.name],
        "snow": [f for f in all_files if "snow" in f.name]
    }

    if not baseline_files:
        print(f"[ERROR] No baseline (city/rural) files found in {base_dir}")
        return

    rng_sample = np.random.default_rng(seed=42)
    rng_shuffle = np.random.default_rng(seed=123)

    print(f"PASS 1: RADIATE Topology Check")
    print(f"  -> Using 100% of Baseline ({len(baseline_files)} chunks) for Nominal Fit due to small scene count.")

    # =========================================================================
    # PASS 2: Stream & Fit the Manifold (RAM-Safe)
    # =========================================================================
    print("\nPASS 2: Streaming Training Data & Fitting PCA/CCA...")
    train_V, train_R = [], []
    
    for f_path in tqdm(baseline_files, desc="Extracting Nominal Patches"):
        with open(f_path, 'rb') as f:
            chunk_data = pickle.load(f)
            
        if 'clear' not in chunk_data:
            print(f"  [WARN] 'clear' key missing in {f_path.name}, skipping.")
            del chunk_data
            gc.collect()
            continue
            
        frames = chunk_data['clear']
        
        for s in frames:
            V = np.array(s.get('radar_patch_features', []), dtype=np.float32)
            
            depths_raw = s.get('radar_depths', None)
            if depths_raw is None: continue
            depths = np.array(depths_raw, dtype=np.float32)
            
            rcs_raw = s.get('radar_rcs', None)
            if rcs_raw is None: continue
            rcs = np.array(rcs_raw, dtype=np.float32)
            
            if len(V) == 0 or len(depths) == 0: continue
            
            # Navtech Physics: strictly [Depth, RCS]
            R = np.column_stack((depths, rcs))
            n_patches = len(V)
            
            if n_patches > 0:
                sample_size = max(1, int(n_patches * 0.10))
                idx = rng_sample.choice(n_patches, sample_size, replace=False)
                train_V.append(V[idx])
                train_R.append(R[idx])
                
        del chunk_data
        gc.collect()

    if not train_V:
        print("\n[ERROR] No valid training patches extracted.")
        return

    V_train = np.vstack(train_V)
    R_train = np.vstack(train_R)
    print(f"  -> Memory Safe: Fitting Manifold on {len(V_train)} spatial patches.")

    scaler_R = StandardScaler()
    R_train_scaled = scaler_R.fit_transform(R_train)
    
    # ANTI-OVERFITTING GUARD: Reduced to 16 components for small dataset
    pca = PCA(n_components=16)
    V_train_pca = pca.fit_transform(V_train)
    
    cca = CCA(n_components=1)
    cca.fit(V_train_pca, R_train_scaled)
    
    del train_V, train_R, V_train, R_train, V_train_pca, R_train_scaled
    gc.collect()

    # =========================================================================
    # PASS 3: Stream & Score Evaluator
    # =========================================================================
    print("\nPASS 3: Streaming Evaluation...")
    
    def score_radiate_set(file_list, condition_label, expected_key):
        if not file_list: return [], [], []
        rhos, physics_l2, rhos_shuff = [], [], []
        
        for f_path in tqdm(file_list, desc=f"Scoring {condition_label.upper()}"):
            with open(f_path, 'rb') as f:
                chunk_data = pickle.load(f)
                
            if expected_key not in chunk_data:
                del chunk_data
                gc.collect()
                continue
                
            frames = chunk_data[expected_key]
            
            for s in frames:
                V = np.array(s.get('radar_patch_features', []), dtype=np.float32)
                
                depths_raw = s.get('radar_depths', None)
                if depths_raw is None: continue
                depths = np.array(depths_raw, dtype=np.float32)
                
                rcs_raw = s.get('radar_rcs', None)
                if rcs_raw is None: continue
                rcs = np.array(rcs_raw, dtype=np.float32)
                
                if len(V) < 2 or len(depths) == 0: continue
                
                # Navtech Physics: strictly [Depth, RCS]
                R = np.column_stack((depths, rcs))
                
                R_scaled = scaler_R.transform(R)
                V_pca = pca.transform(V)
                V_c, R_c = cca.transform(V_pca, R_scaled)
                
                delta = V_c[:, 0] - R_c[:, 0]
                rho = float(np.mean(np.clip(1.0 - 0.5 * delta**2, 0.0, 1.0)))
                rhos.append(rho)
                
                
                # FIX: Navtech physics anomaly is directional attenuation (lower RCS)
                # We negate mean RCS so higher score = more anomalous
                frame_phys_score = float(-np.mean(rcs))
                physics_l2.append(frame_phys_score)
                
                R_shuff_scaled = R_scaled.copy()
                rng_shuffle.shuffle(R_shuff_scaled)
                V_c_s, R_c_s = cca.transform(V_pca, R_shuff_scaled)
                delta_shuff = V_c_s[:, 0] - R_c_s[:, 0]
                rhos_shuff.append(float(np.mean(np.clip(1.0 - 0.5 * delta_shuff**2, 0.0, 1.0))))
                
            del chunk_data
            gc.collect()
            
        return rhos, physics_l2, rhos_shuff

    # Run evaluations
    results = {}
    results['test_clear'] = score_radiate_set(baseline_files, 'clear_baseline', 'clear')
    results['night'] = score_radiate_set(ood_files['night'], 'night', 'night')
    results['fog'] = score_radiate_set(ood_files['fog'], 'fog', 'fog')
    results['snow'] = score_radiate_set(ood_files['snow'], 'snow', 'snow')

    # =========================================================================
    # FINAL STATISTICS & JSON EXPORT
    # =========================================================================
    print("\n" + "="*75)
    print("FINAL RESULTS: RADIATE NAVTECH RADAR")
    print("="*75)
    
    clear_rhos, clear_phys_l2, clear_shuff = results['test_clear']
    if not clear_rhos:
        print("[ERROR] No Clear baseline frames scored. Exiting.")
        return
        
    print(f"Test-Clear |Rho|      : {np.mean(clear_rhos):.4f} ± {np.std(clear_rhos):.4f} (n={len(clear_rhos)})")
    print(f"Test-Clear Physics score (neg rcs) : {np.mean(clear_phys_l2):.4f} ± {np.std(clear_phys_l2):.4f}")
    print(f"Test-Clear Shuffled   : {np.mean(clear_shuff):.4f} ± {np.std(clear_shuff):.4f} (Control)")
    
    json_export = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "test_clear_mean": float(np.mean(clear_rhos)), 
            "test_clear_std": float(np.std(clear_rhos)),
            "test_clear_phys_score_mean": float(np.mean(clear_phys_l2)),
            "test_clear_n": len(clear_rhos),
            "shuffle_control_mean": float(np.mean(clear_shuff)),
            "shuffle_control_std": float(np.std(clear_shuff))
        },
        "conditions": {}
    }
    
    for condition in ['night', 'fog', 'snow']:
        adv_rhos, adv_phys_l2, adv_shuff = results.get(condition, ([], [], []))
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
        print(f"  Physics score (neg rcs)   : {np.mean(adv_phys_l2):.4f} ± {np.std(adv_phys_l2):.4f}")
        print(f"  Vision CMGC AUROC : {auroc_rho:.4f}")
        print(f"  Physics AUROC     : {auroc_phys:.4f}")
        print(f"  Welch's t-test    : p={p_val:.2e}")
        print(f"  Cohen's d         : {cohens_d:.3f}")
        
        json_export["conditions"][condition] = {
            "n_frames": len(adv_rhos),
            "vision_cmgc_auroc": float(auroc_rho),
            "physics_auroc": float(auroc_phys),
            "cohens_d": cohens_d,
            "welch_p_value": p_val,
            "shuffle_control_mean": float(np.mean(adv_shuff)),
            "shuffle_control_std": float(np.std(adv_shuff))
        }

    output_file = base_path / "final_radiate_baseline_results.json"
    with open(output_file, 'w') as f:
        json.dump(json_export, f, indent=4)
    print(f"\n[SUCCESS] Results saved to: {output_file}")

if __name__ == "__main__":
    evaluate_radiate_radar()
