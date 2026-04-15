"""
Traffic Density Confound Plot
Shows correlation between radar return count (traffic density proxy)
and the physics baseline anomaly score for Night scenes on nuScenes.

Uses the same StandardScaler-based anomaly score as the eval script
so the r value matches the paper's reported r = -0.179, p < 1e-25.
"""

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)


def fit_scaler_on_clear(
    clear_dir="/home/avnish/radar_camera_PR/outputs/cmgc_radar_dinov2/clear/",
    max_files=40
):
    """
    Fit a StandardScaler on clear radar physics features [depth, RCS, vel_mag]
    — identical to what the eval script does — so the anomaly score is normalised.
    """
    clear_path = Path(clear_dir)
    clear_files = sorted(list(clear_path.glob("*.pkl")))[:max_files]

    if not clear_files:
        raise RuntimeError(f"No clear radar chunks found in {clear_dir}")

    train_R = []
    for f_path in tqdm(clear_files, desc="Fitting scaler on clear"):
        with open(f_path, 'rb') as f:
            try:
                chunk_data = pickle.load(f)
            except EOFError:
                continue

        if isinstance(chunk_data, dict):
            if 'clear' not in chunk_data:
                continue
            frames = chunk_data['clear']
        else:
            frames = chunk_data

        for s in frames:
            depths = np.array(s.get('radar_depths', []), dtype=np.float32)
            if len(depths) == 0:
                continue
            rcs = np.array(s.get('radar_rcs', np.zeros_like(depths)), dtype=np.float32)

            if 'radar_velocities' in s:
                vel = np.linalg.norm(
                    np.array(s['radar_velocities'], dtype=np.float32), axis=1
                )
            else:
                vel = np.zeros_like(depths)

            R = np.column_stack((depths, rcs, vel))
            train_R.append(R)

    if not train_R:
        raise RuntimeError("No valid clear frames for scaler fitting.")

    R_train = np.vstack(train_R)
    scaler  = StandardScaler().fit(R_train)
    print(f"  Scaler fitted on {len(train_R)} clear frames.")
    return scaler


def plot_confound(
    night_dir="/home/avnish/radar_camera_PR/outputs/cmgc_radar_dinov2/night/",
    clear_dir="/home/avnish/radar_camera_PR/outputs/cmgc_radar_dinov2/clear/"
):
    # ── Fit scaler on clear data (same as eval script) ────────────────────
    print("Step 1: Fitting StandardScaler on clear radar data...")
    scaler = fit_scaler_on_clear(clear_dir, max_files=40)

    # ── Extract night frames ──────────────────────────────────────────────
    night_path  = Path(night_dir)
    chunk_files = sorted(list(night_path.glob("*.pkl")))

    if not chunk_files:
        print(f"[ERROR] No radar chunks found in {night_dir}")
        return

    n_points_list  = []
    anomaly_scores = []

    print("\nStep 2: Extracting traffic density and anomaly scores from Night scenes...")
    for f_path in tqdm(chunk_files, desc="Processing night chunks"):
        with open(f_path, 'rb') as f:
            try:
                chunk_data = pickle.load(f)
            except EOFError:
                continue

        # Strict key guard
        if isinstance(chunk_data, dict):
            if 'night' not in chunk_data:
                continue
            frames = chunk_data['night']
        else:
            frames = chunk_data

        for s in frames:
            # Traffic density proxy — total radar returns in this frame
            n_pts = s.get('n_radar_points', 0)
            if n_pts == 0:
                continue

            depths = np.array(s.get('radar_depths', []), dtype=np.float32)
            if len(depths) == 0:
                continue

            rcs = np.array(s.get('radar_rcs', np.zeros_like(depths)), dtype=np.float32)
            if 'radar_velocities' in s:
                vel = np.linalg.norm(
                    np.array(s['radar_velocities'], dtype=np.float32), axis=1
                )
            else:
                vel = np.zeros_like(depths)

            R = np.column_stack((depths, rcs, vel))

            # Physics anomaly score — mean L2 norm in scaled space
            # Identical formula to eval script:
            # frame_phys_l2 = float(np.mean(np.linalg.norm(R_scaled, axis=1)))
            R_scaled   = scaler.transform(R)
            phys_score = float(np.mean(np.linalg.norm(R_scaled, axis=1)))

            n_points_list.append(n_pts)
            anomaly_scores.append(phys_score)

    if not n_points_list:
        print("[ERROR] No valid data points extracted. Check night directory and keys.")
        return

    x = np.array(n_points_list)
    y = np.array(anomaly_scores)

    # ── Pearson correlation ───────────────────────────────────────────────
    r_val, p_val = pearsonr(x, y)
    print(f"\n  Pearson r = {r_val:.3f}  (p = {p_val:.2e})")
    print(f"  n = {len(x)} night frames")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(x, y, alpha=0.4, c='#1f77b4', edgecolors='none', s=30, zorder=2)

    # Line of best fit
    m, b    = np.polyfit(x, y, 1)
    x_range = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_range, m * x_range + b,
            color='#d62728', linewidth=2, linestyle='--',
            label=fr'Linear trend  $r = {r_val:.3f}$,  $p < 10^{{{int(np.floor(np.log10(p_val)))}}}$',
            zorder=3)

    ax.set_title(
        "The Evaluation Confound: Radar Anomaly Score vs. Traffic Density\n"
        "nuScenes Night Frames — Continental 77 GHz Radar",
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel("Physical traffic density (total radar returns per frame)",
                  fontsize=12)
    ax.set_ylabel("Physics baseline anomaly score\n(mean L2 in standardised feature space)",
                  fontsize=12)

    ax.legend(fontsize=12, loc='upper right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.grid(False)

    plt.tight_layout()
    output_img = "traffic_density_confound.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Plot saved to {output_img}")


if __name__ == "__main__":
    plot_confound()
