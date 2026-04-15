"""
CMGC Spatial Heatmap Renderer — Extended Version with Batch Comparison
New function: render_condition_comparison()
  - Selects representative tokens for ALL conditions (clear, night, rain, rain+night)
  - Renders a 2×2 grid comparison figure
  - Optionally renders individual per-condition PNGs
  - All scientific fixes from the smoothed version are preserved
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
from scipy.ndimage import gaussian_filter, binary_dilation

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from nuscenes.nuscenes import NuScenes
from lidar_data_pipeline import DINOv2FeatureExtractor, load_project_and_bin_lidar
from config import ExperimentConfig

GRID_SIZE = 518 // 14  # = 37


# ============================================================================
# CORE FUNCTIONS (unchanged from smoothed version)
# ============================================================================

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


def fit_manifold(clear_dir, max_frames=500):
    """Fit PCA + CCA + StandardScaler on clear data. Returns frozen pipeline."""
    print("Fitting PCA/CCA manifold from clear data...")
    clear_files = sorted(Path(clear_dir).glob("*.pkl"))

    train_V, train_R = [], []
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
            if len(train_V) >= max_frames:
                break
        if len(train_V) >= max_frames:
            break

    if len(train_V) == 0:
        raise RuntimeError("No valid clear frames found.")

    V_train = np.vstack(train_V)
    R_train = np.vstack(train_R)
    print(f"  Fitted on {len(train_V)} frames | V: {V_train.shape} | R: {R_train.shape}")

    scaler_R       = StandardScaler()
    R_train_scaled = scaler_R.fit_transform(R_train)
    pca            = PCA(n_components=32)
    V_pca          = pca.fit_transform(V_train)
    cca            = CCA(n_components=1)
    cca.fit(V_pca, R_train_scaled)
    print("  PCA + CCA fitted.")

    return pca, cca, scaler_R


def find_representative_token(pkl_dir, condition_key, target_rho,
                              pca, cca, scaler_R, max_files=None):
    path  = Path(pkl_dir)
    files = sorted(path.glob("*.pkl"))
    if max_files:
        files = files[:max_files]

    best_token = None
    best_diff  = float('inf')

    # COMPOUND KEY FIX: LiDAR pipeline saves rain+night as 'night+rain'
    # Try both the requested key and the compound variant
    key_variants = [condition_key, 'night+rain', 'rain+night']

    for f_path in files:
        with open(f_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            # Try each key variant until one works
            frames = None
            for key in key_variants:
                if key in data:
                    frames = data[key]
                    break
            if frames is None:
                continue
        else:
            frames = data

        for s in frames:
            V, R = load_frame_arrays(s)
            if V is None:
                continue
            scores   = compute_frame_coherence(V, R, pca, cca, scaler_R)
            mean_rho = float(np.mean(scores))
            if abs(mean_rho - target_rho) < best_diff:
                best_diff  = abs(mean_rho - target_rho)
                best_token = s['sample_token']

    if best_token is None:
        raise RuntimeError(
            f"No valid frames found in {pkl_dir} for key '{condition_key}' "
            f"(also tried: {key_variants[1:]}). "
            f"Check pkl keys with: pickle.load(open(file,'rb')).keys()"
        )
    print(f"  [{condition_key}] token: {best_token}  (|Δrho| = {best_diff:.4f})")
    return best_token


def build_heatmap(nusc, token, cfg, extractor, pca, cca, scaler_R):
    """Extract DINOv2 + LiDAR for a token, return img, heatmap grid, mean rho."""
    sample   = nusc.get('sample', token)
    cam_data = nusc.get('sample_data', sample['data'][cfg.nuscenes.camera_channel])
    img_path = Path(cfg.nuscenes.dataroot) / cam_data['filename']

    image      = Image.open(img_path).convert('RGB')
    features   = extractor.extract_features(image)
    patch_grid = features['patch_grid'].numpy()

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

    heatmap_grid = np.full((GRID_SIZE, GRID_SIZE), -1.0, dtype=np.float32)
    for idx, (r, c) in enumerate(lidar_mapping.activated_patches):
        heatmap_grid[r, c] = patch_coherence[idx]

    img_resized = cv2.resize(np.array(image), (518, 518))
    return img_resized, heatmap_grid, mean_coherence


def smooth_and_mask(heatmap_grid, output_size=518, blur_sigma=0.8):
    """Smooth + mask heatmap for display. See smoothed version docstring."""
    valid_mask    = (heatmap_grid >= 0.0).astype(bool)
    dilated_mask  = binary_dilation(valid_mask, iterations=1).astype(float)
    mask_resized  = cv2.resize(dilated_mask, (output_size, output_size),
                               interpolation=cv2.INTER_NEAREST).astype(bool)

    heatmap_filled  = np.where(heatmap_grid < 0, 0.0, heatmap_grid)
    heatmap_resized = cv2.resize(heatmap_filled, (output_size, output_size),
                                 interpolation=cv2.INTER_CUBIC)
    heatmap_smooth  = gaussian_filter(heatmap_resized, sigma=blur_sigma)
    heatmap_smooth  = np.clip(heatmap_smooth, 0.0, 1.0)
    return np.ma.masked_where(~mask_resized, heatmap_smooth)


def _render_single_panel(ax, img_resized, heatmap_grid, mean_rho,
                         title, blur_sigma=0.8):
    """Render one heatmap panel onto a given matplotlib axis."""
    overlay = smooth_and_mask(heatmap_grid, output_size=518, blur_sigma=blur_sigma)
    ax.imshow(img_resized)
    im = ax.imshow(overlay, cmap='inferno', vmin=0.0, vmax=1.0, alpha=0.75)
    ax.set_title(f"{title}\n(mean $|\\rho|$ = {mean_rho:.3f})", fontsize=13)
    ax.axis('off')
    return im


# ============================================================================
# ORIGINAL FUNCTION — clear vs rain+night (unchanged)
# ============================================================================

def render_heatmaps():
    clear_dir   = "/home/avnish/radar_camera_PR/outputs/cmgc_lidar_dinov2/clear/"
    adverse_dir = "/home/avnish/radar_camera_PR/outputs/cmgc_lidar_dinov2/rain_night_mixed/"
    dataroot    = "/home/avnish/3dVision/OpenPCDet/data/nuscenes/"

    pca, cca, scaler_R = fit_manifold(clear_dir)

    print("\nSelecting representative tokens...")
    clear_token   = find_representative_token(
        clear_dir, 'clear', 0.833, pca, cca, scaler_R)
    # Add this temporarily before the find_representative_token call for rain_night
    import pickle
    f = sorted(Path(base + "rain_night_mixed/").glob("*.pkl"))[0]
    d = pickle.load(open(f, 'rb'))
    print(f"rain_night_mixed chunk keys: {list(d.keys())}")
    adverse_token = find_representative_token(
        adverse_dir, 'rain_night_mixed', 0.69, pca, cca, scaler_R)

    cfg           = ExperimentConfig()
    cfg.nuscenes.dataroot = dataroot
    nusc          = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
    extractor     = DINOv2FeatureExtractor(cfg.dinov2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("CMGC Spatial Alignment (Dense LiDAR to DINOv2)",
                 fontsize=18, fontweight='bold')

    for ax, token, title in zip(
        axes,
        [clear_token, adverse_token],
        ["Nominal Geometry (Clear)", "Geometric Collapse (Rain + Night)"]
    ):
        img, hmap, rho = build_heatmap(nusc, token, cfg, extractor, pca, cca, scaler_R)
        print(f"  [{title}] mean coherence = {rho:.4f}")
        im = _render_single_panel(ax, img, hmap, rho, title)

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                 label='Cross-Modal Coherence Score (Canonical Space)')

    plt.savefig("cmgc_true_spatial_heatmaps.png", dpi=600, bbox_inches='tight')
    plt.savefig("cmgc_true_spatial_heatmaps.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Saved cmgc_true_spatial_heatmaps.png and .pdf")


# ============================================================================
# NEW FUNCTION — full 4-condition comparison for supplementary
# ============================================================================

def render_condition_comparison(
    output_dir="./supplementary_heatmaps/",
    save_individual=True,
    blur_sigma=0.8,
):
    """
    Renders cross-condition coherence comparison heatmaps for supplementary material.

    Produces:
      1. A 2×2 grid figure comparing Clear / Night / Rain / Rain+Night
         → saved as  cmgc_4condition_comparison.png  and  .pdf
      2. (if save_individual=True) Four individual per-condition PNGs:
         → cmgc_heatmap_clear.png
         → cmgc_heatmap_night.png
         → cmgc_heatmap_rain.png
         → cmgc_heatmap_rain_night.png

    Token selection targets:
      Clear       → mean rho closest to 0.833 (dataset mean)
      Night       → mean rho closest to 0.774 (dataset mean)
      Rain        → mean rho closest to 0.824 (dataset mean)
      Rain+Night  → mean rho closest to 0.690 (1 std below mean, shows collapse)
    """

    # ── Paths ──────────────────────────────────────────────────────────────
    base       = "/home/avnish/radar_camera_PR/outputs/cmgc_lidar_dinov2/"
    dataroot   = "/home/avnish/3dVision/OpenPCDet/data/nuscenes/"
    out_path   = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    condition_dirs = {
        "clear":           (base + "clear/",           "clear",           0.833),
        "night":           (base + "night/",           "night",           0.774),
        "rain":            (base + "rain/",            "rain",            0.824),
        "rain_night":      (base + "rain_night_mixed/","rain_night_mixed", 0.690),
    }

    titles = {
        "clear":      "Clear\n(Nominal Geometry)",
        "night":      "Night\n(Luminance Collapse)",
        "rain":       "Rain\n(Photometric Perturbation)",
        "rain_night": "Rain + Night\n(Compound Degradation)",
    }

    # ── Fit manifold once on clear data ────────────────────────────────────
    pca, cca, scaler_R = fit_manifold(base + "clear/")

    # ── Setup nuScenes + extractor ──────────────────────────────────────────
    cfg               = ExperimentConfig()
    cfg.nuscenes.dataroot = dataroot
    nusc              = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
    extractor         = DINOv2FeatureExtractor(cfg.dinov2)

    # ── Find representative token for each condition ────────────────────────
    print("\nSelecting representative tokens for all conditions...")
    tokens = {}
    for cond, (cdir, ckey, target) in condition_dirs.items():
        tokens[cond] = find_representative_token(
            cdir, ckey, target, pca, cca, scaler_R
        )

    # ── Build all heatmaps ─────────────────────────────────────────────────
    print("\nBuilding heatmaps...")
    results = {}
    for cond, token in tokens.items():
        print(f"  Rendering [{cond}]...")
        img, hmap, rho = build_heatmap(
            nusc, token, cfg, extractor, pca, cca, scaler_R
        )
        results[cond] = (img, hmap, rho)
        print(f"    mean |ρ| = {rho:.4f}")

    # ── 1. Save individual per-condition PNGs ──────────────────────────────
    if save_individual:
        print("\nSaving individual condition heatmaps...")
        for cond, (img, hmap, rho) in results.items():
            fig, ax = plt.subplots(1, 1, figsize=(9, 7))
            fig.suptitle("CMGC Coherence — DINOv2 ViT-L/14 + Dense LiDAR",
                         fontsize=14, fontweight='bold')
            im = _render_single_panel(
                ax, img, hmap, rho, titles[cond], blur_sigma
            )
            cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.03])
            fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                         label='Cross-Modal Coherence Score $|\\rho|$')
            fname = out_path / f"cmgc_heatmap_{cond}.png"
            plt.savefig(fname, dpi=400, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {fname}")

    # ── 2. Save 2×2 grid comparison figure ────────────────────────────────
    print("\nRendering 2×2 condition comparison grid...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        "CMGC Cross-Modal Coherence — All Weather Conditions\n"
        "DINOv2 ViT-L/14 · Dense LiDAR · nuScenes",
        fontsize=16, fontweight='bold', y=0.98
    )

    order = ["clear", "night", "rain", "rain_night"]
    for ax, cond in zip(axes.flat, order):
        img, hmap, rho = results[cond]
        im = _render_single_panel(ax, img, hmap, rho, titles[cond], blur_sigma)

    # Shared colorbar at bottom
    fig.subplots_adjust(bottom=0.08, hspace=0.15, wspace=0.05)
    cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                 label='Cross-Modal Coherence Score $|\\rho|$  '
                       '(warm = high coherence · dark = collapse)')

    grid_png = out_path / "cmgc_4condition_comparison.png"
    grid_pdf = out_path / "cmgc_4condition_comparison.pdf"
    plt.savefig(grid_png, dpi=400, bbox_inches='tight')
    plt.savefig(grid_pdf, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SUCCESS] 2×2 grid saved:")
    print(f"  {grid_png}")
    print(f"  {grid_pdf}")
    if save_individual:
        print(f"  + 4 individual PNGs in {out_path}/")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="CMGC heatmap renderer — main figure and supplementary"
    )
    parser.add_argument(
        "--mode",
        choices=["main", "supplementary", "both"],
        default="both",
        help="main = clear vs rain+night only; "
             "supplementary = 4-condition grid + individuals; "
             "both = run both"
    )
    parser.add_argument(
        "--output_dir",
        default="./supplementary_heatmaps/",
        help="Output directory for supplementary figures"
    )
    parser.add_argument(
        "--no_individual",
        action="store_true",
        help="Skip saving individual per-condition PNGs"
    )
    args = parser.parse_args()

    if args.mode in ("main", "both"):
        render_heatmaps()

    if args.mode in ("supplementary", "both"):
        render_condition_comparison(
            output_dir=args.output_dir,
            save_individual=not args.no_individual,
        )
