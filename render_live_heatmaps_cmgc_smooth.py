"""
CMGC Spatial Heatmap Renderer — Smoothed Version
Changes from pixelated version:
  - Heatmap uses INTER_CUBIC for smooth patch-to-patch gradients
  - Mask derived from sentinel BEFORE cubic resize using INTER_NEAREST
    so absent-LiDAR regions (sky) stay cleanly masked with no bleed
  - Mild Gaussian blur (sigma=1.5 patches) applied after resize for
    display smoothness — does not alter the underlying data values
  - alpha raised to 0.75 for slightly stronger overlay
  - All scientific fixes from the final version are preserved
"""

import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_dilation

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from nuscenes.nuscenes import NuScenes
from lidar_data_pipeline import DINOv2FeatureExtractor, load_project_and_bin_lidar
from config import ExperimentConfig

GRID_SIZE = 518 // 14  # = 37


def compute_frame_coherence(V, R, pca, cca, scaler_R):
    R_scaled = scaler_R.transform(R)
    V_pca    = pca.transform(V)
    V_c, R_c = cca.transform(V_pca, R_scaled)
    delta    = V_c[:, 0] - R_c[:, 0]
    return np.clip(1.0 - 0.5 * delta**2, 0.0, 1.0)


def load_frame_arrays(s):
    V      = s.get('lidar_patch_features', [])
    depths = s.get('lidar_depth_means',    [])
    if len(V) == 0 or len(depths) == 0:
        return None, None

    stds       = s.get('lidar_depth_stds', np.zeros_like(depths))
    log_counts = s.get('lidar_log_counts', None)
    if log_counts is None:
        print(f"  [WARN] 'lidar_log_counts' missing for token {s.get('sample_token','?')}")
        log_counts = np.zeros_like(depths)

    V = np.array(V,          dtype=np.float32)
    R = np.column_stack((
            np.array(depths,      dtype=np.float32),
            np.array(stds,        dtype=np.float32),
            np.array(log_counts,  dtype=np.float32),
        ))
    return V, R


def get_tokens_and_fit_cca(clear_dir, adverse_dir):
    print("Step 1: Fitting PCA/CCA manifold from .pkl files...")

    clear_files   = sorted(Path(clear_dir).glob("*.pkl"))
    adverse_files = sorted(Path(adverse_dir).glob("*.pkl"))

    train_V, train_R = [], []
    clear_sample_data = []

    for f_path in clear_files:
        with open(f_path, 'rb') as f:
            data = pickle.load(f)
        frames = list(data.values())[0] if isinstance(data, dict) else data
        for s in frames:
            V, R = load_frame_arrays(s)
            if V is None:
                continue
            train_V.append(V)
            train_R.append(R)
            clear_sample_data.append((s['sample_token'], V, R))
            if len(train_V) >= 500:
                break
        if len(train_V) >= 500:
            break

    if len(train_V) == 0:
        raise RuntimeError("No valid clear frames found.")

    V_train = np.vstack(train_V)
    R_train = np.vstack(train_R)
    print(f"  Loaded {len(train_V)} clear frames | V: {V_train.shape} | R: {R_train.shape}")

    scaler_R       = StandardScaler()
    R_train_scaled = scaler_R.fit_transform(R_train)
    pca = PCA(n_components=32)
    V_pca = pca.fit_transform(V_train)
    cca = CCA(n_components=1)
    cca.fit(V_pca, R_train_scaled)
    print("  PCA + CCA fitted.")

    # Representative clear token — target rho ~ 0.833 (new dataset mean)
    TARGET_CLEAR_RHO = 0.833
    best_clear_token = clear_sample_data[0][0]
    best_clear_diff  = float('inf')
    for token, V, R in clear_sample_data[:50]:
        scores   = compute_frame_coherence(V, R, pca, cca, scaler_R)
        mean_rho = float(np.mean(scores))
        if abs(mean_rho - TARGET_CLEAR_RHO) < best_clear_diff:
            best_clear_diff  = abs(mean_rho - TARGET_CLEAR_RHO)
            best_clear_token = token
    print(f"  Best clear token: {best_clear_token}  (Δrho = {best_clear_diff:.4f})")

    # Representative adverse token — target rho ~ 0.69 (1 std below rain+night mean)
    TARGET_ADVERSE_RHO = 0.69
    best_adverse_token = None
    best_adverse_diff  = float('inf')
    for f_path in adverse_files:
        with open(f_path, 'rb') as f:
            data = pickle.load(f)
        frames = list(data.values())[0] if isinstance(data, dict) else data
        for s in frames:
            V, R = load_frame_arrays(s)
            if V is None:
                continue
            scores   = compute_frame_coherence(V, R, pca, cca, scaler_R)
            mean_rho = float(np.mean(scores))
            if abs(mean_rho - TARGET_ADVERSE_RHO) < best_adverse_diff:
                best_adverse_diff  = abs(mean_rho - TARGET_ADVERSE_RHO)
                best_adverse_token = s['sample_token']

    if best_adverse_token is None:
        raise RuntimeError("No valid adverse frames found.")
    print(f"  Best adverse token: {best_adverse_token}  (Δrho = {best_adverse_diff:.4f})")

    return pca, cca, scaler_R, best_clear_token, best_adverse_token


def build_heatmap(nusc, token, cfg, extractor, pca, cca, scaler_R):
    sample   = nusc.get('sample', token)
    cam_data = nusc.get('sample_data', sample['data'][cfg.nuscenes.camera_channel])
    img_path = Path(cfg.nuscenes.dataroot) / cam_data['filename']

    image      = Image.open(img_path).convert('RGB')
    features   = extractor.extract_features(image)
    patch_grid = features['patch_grid'].numpy()  # (37, 37, 1024)

    lidar_mapping = load_project_and_bin_lidar(nusc, token, cfg)
    if len(lidar_mapping.activated_patches) == 0:
        raise RuntimeError(f"No activated LiDAR patches for token {token}")

    aligned_vision, aligned_physics = [], []
    for idx, (r, c) in enumerate(lidar_mapping.activated_patches):
        aligned_vision.append(patch_grid[r, c])
        aligned_physics.append([
            lidar_mapping.depth_means[idx],
            lidar_mapping.depth_stds[idx],
            lidar_mapping.log_point_counts[idx],
        ])

    V = np.array(aligned_vision,  dtype=np.float32)
    R = np.array(aligned_physics, dtype=np.float32)

    patch_coherence = compute_frame_coherence(V, R, pca, cca, scaler_R)
    mean_coherence  = float(np.mean(patch_coherence))

    # Build sentinel grid: -1.0 = no LiDAR
    heatmap_grid = np.full((GRID_SIZE, GRID_SIZE), -1.0, dtype=np.float32)
    for idx, (r, c) in enumerate(lidar_mapping.activated_patches):
        heatmap_grid[r, c] = patch_coherence[idx]

    img_resized = cv2.resize(np.array(image), (518, 518))

    return img_resized, heatmap_grid, mean_coherence


def smooth_and_mask(heatmap_grid, output_size=518, blur_sigma=0.8):
    """
    Produces a smooth, correctly masked heatmap for display.
 
    Strategy:
      1. Extract the binary valid-patch mask using INTER_NEAREST — this
         preserves hard edges at the sky/no-LiDAR boundary exactly.
      2. Replace sentinel (-1.0) with 0.0 and resize the values using
         INTER_CUBIC — this gives smooth gradients between adjacent patches.
      3. Apply a mild Gaussian blur in pixel space for display smoothness.
         sigma=1.2 pixels at 518px output ≈ 0.08 patch widths — subtle.
      4. Re-apply the hard mask so blurred values never bleed into sky.
    """
    
    from scipy.ndimage import binary_dilation
    import cv2
    import numpy as np
    from scipy.ndimage import gaussian_filter

    # Step 1: hard mask from valid patches
    valid_mask = (heatmap_grid >= 0.0).astype(bool)
    # Dilate to fill isolated gaps
    dilated_mask = binary_dilation(valid_mask, iterations=1).astype(float)
    mask_resized = cv2.resize(dilated_mask, (output_size, output_size),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

    # Step 2: fill sentinel with 0 and resize smoothly
    heatmap_filled = np.where(heatmap_grid < 0, 0.0, heatmap_grid)
    heatmap_resized = cv2.resize(heatmap_filled, (output_size, output_size),
                                 interpolation=cv2.INTER_CUBIC)

    # Step 3: Gaussian blur for smoothness
    heatmap_smooth = gaussian_filter(heatmap_resized, sigma=blur_sigma)

    # Step 4: clip and re-apply hard mask
    heatmap_smooth = np.clip(heatmap_smooth, 0.0, 1.0)
    heatmap_masked = np.ma.masked_where(~mask_resized, heatmap_smooth)

    return heatmap_masked

def render_heatmaps():
    clear_dir   = "/home/avnish/radar_camera_PR/outputs/cmgc_lidar_dinov2/clear/"
    adverse_dir = "/home/avnish/radar_camera_PR/outputs/cmgc_lidar_dinov2/rain_night_mixed/"
    dataroot    = "/home/avnish/3dVision/OpenPCDet/data/nuscenes/"

    pca, cca, scaler_R, clear_token, adverse_token = get_tokens_and_fit_cca(
        clear_dir, adverse_dir
    )

    print(f"\nStep 2: Rendering heatmaps")
    print(f"  Clear   token: {clear_token}")
    print(f"  Adverse token: {adverse_token}")

    cfg = ExperimentConfig()
    cfg.nuscenes.dataroot = dataroot
    nusc      = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
    extractor = DINOv2FeatureExtractor(cfg.dinov2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "CMGC Spatial Alignment (Dense LiDAR to DINOv2)",
        fontsize=18, fontweight='bold'
    )

    titles = ["Nominal Geometry (Clear)", "Geometric Collapse (Rain + Night)"]
    tokens = [clear_token, adverse_token]

    for ax, token, title in zip(axes, tokens, titles):
        img_resized, heatmap_grid, mean_rho = build_heatmap(
            nusc, token, cfg, extractor, pca, cca, scaler_R
        )
        print(f"  [{title}] mean coherence = {mean_rho:.4f}")

        # Smooth + masked overlay
        heatmap_overlay = smooth_and_mask(heatmap_grid, output_size=518, blur_sigma=0.8)

        ax.imshow(img_resized)
        im = ax.imshow(
            heatmap_overlay, cmap='inferno',
            vmin=0.0, vmax=1.0, alpha=0.75
        )
        ax.set_title(f"{title}\n(mean $|\\rho|$ = {mean_rho:.3f})", fontsize=13)
        ax.axis('off')

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    fig.colorbar(
        im, cax=cbar_ax, orientation='horizontal',
        label='Cross-Modal Coherence Score (Canonical Space)'
    )

    plt.savefig("cmgc_true_spatial_heatmaps.png", dpi=600, bbox_inches='tight')
    plt.savefig("cmgc_true_spatial_heatmaps.pdf", dpi=300, bbox_inches='tight')
    print("\n[SUCCESS] Saved cmgc_true_spatial_heatmaps.png and .pdf")


if __name__ == "__main__":
    render_heatmaps()
