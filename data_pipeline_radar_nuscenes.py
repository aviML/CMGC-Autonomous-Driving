"""
CMGC Radar-DINOv2 Data Pipeline (v3 — Production)

Changes from v1/v2:
    - Saves image_mean_feature (D,) and image_cov_stats instead of full patch_grid
      → ~5KB/sample vs ~5.6MB/sample (1000x reduction)
    - Computes vanilla L2 scores at extraction time (proven Δ=0.78 over spatial confound)
    - Saves per-radar-point band-relative features for spatial confound control
    - Stores sufficient statistics for both vanilla L2 and MCD approaches
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle
import json
import gc
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from pyquaternion import Quaternion

from config import NuScenesConfig, DINOv2Config, ExperimentConfig


# =============================================================================
# 1. RADAR DATA LOADING & PROJECTION (unchanged from v1)
# =============================================================================

@dataclass
class RadarProjection:
    """Radar points projected into camera image coordinates."""
    pixel_coords: np.ndarray   # (2, N)
    depths: np.ndarray         # (N,)
    rcs: np.ndarray            # (N,)
    velocities: np.ndarray     # (N, 2)
    dynprop: np.ndarray        # (N,)
    pdh0: np.ndarray           # (N,)
    n_points: int
    points_cam_3d: np.ndarray  # (3, N)


def load_and_project_radar(
    nusc: NuScenes,
    sample_token: str,
    cfg: NuScenesConfig,
) -> RadarProjection:
    """
    Load radar with multi-sweep accumulation, project onto front camera.
    """
    sample = nusc.get('sample', sample_token)
    
    cam_data = nusc.get('sample_data', sample['data'][cfg.camera_channel])
    cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    cam_intrinsic = np.array(cam_cs['camera_intrinsic'])
    
    # Aggregate radar sweeps in camera reference frame
    all_pc, all_times = RadarPointCloud.from_file_multisweep(
        nusc,
        sample_rec=sample,
        chan=cfg.radar_channel,
        ref_chan=cfg.camera_channel,
        nsweeps=cfg.radar_nsweeps,
        min_distance=cfg.radar_min_distance,
    )
    
    points = all_pc.points  # (18, N)
    
    if points.shape[1] == 0:
        return _empty_radar_projection()
    
    # Quality filtering
    mask = (
        np.isin(points[14, :].astype(int), cfg.radar_invalid_states) &
        np.isin(points[3, :].astype(int), cfg.radar_dynprop_states) &
        np.isin(points[11, :].astype(int), cfg.radar_ambig_states) &
        (points[5, :] >= cfg.radar_min_rcs)
    )
    points = points[:, mask]
    
    if points.shape[1] == 0:
        return _empty_radar_projection()
    
    # Extract attributes
    rcs = points[5, :].copy()
    vx_comp = points[8, :].copy()
    vy_comp = points[9, :].copy()
    dynprop = points[3, :].copy()
    pdh0 = points[15, :].copy()
    points_3d = points[:3, :]
    
    # Filter behind camera
    front = points_3d[2, :] > 0
    points_3d = points_3d[:, front]
    rcs, vx_comp, vy_comp = rcs[front], vx_comp[front], vy_comp[front]
    dynprop, pdh0 = dynprop[front], pdh0[front]
    
    if points_3d.shape[1] == 0:
        return _empty_radar_projection()
    
    depths = points_3d[2, :].copy()
    pixel_coords = view_points(points_3d, cam_intrinsic, normalize=True)[:2, :]
    
    # Filter within image bounds (1600x900)
    in_bounds = (
        (pixel_coords[0, :] >= 0) & (pixel_coords[0, :] < 1600) &
        (pixel_coords[1, :] >= 0) & (pixel_coords[1, :] < 900)
    )
    
    return RadarProjection(
        pixel_coords=pixel_coords[:, in_bounds],
        depths=depths[in_bounds],
        rcs=rcs[in_bounds],
        velocities=np.stack([vx_comp[in_bounds], vy_comp[in_bounds]], axis=1),
        dynprop=dynprop[in_bounds],
        pdh0=pdh0[in_bounds],
        n_points=int(in_bounds.sum()),
        points_cam_3d=points_3d[:, in_bounds],
    )


def _empty_radar_projection():
    return RadarProjection(
        pixel_coords=np.zeros((2, 0)), depths=np.zeros(0),
        rcs=np.zeros(0), velocities=np.zeros((0, 2)),
        dynprop=np.zeros(0), pdh0=np.zeros(0),
        n_points=0, points_cam_3d=np.zeros((3, 0)),
    )


# =============================================================================
# 2. DINOv2 FEATURE EXTRACTION (unchanged from v1)
# =============================================================================

class DINOv2FeatureExtractor:
    """Extract patch-level and CLS features from DINOv2 ViT-L/14."""
    
    def __init__(self, cfg: DINOv2Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', cfg.model_name)
        except Exception:
            print(f"Could not load {cfg.model_name}, trying fallback...")
            self.model = torch.hub.load('facebookresearch/dinov2', cfg.model_name_fallback)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.image_size = cfg.image_size
        self.patch_size = cfg.patch_size
        self.grid_size = cfg.image_size // cfg.patch_size
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        x = torch.from_numpy(np.array(image)).float().permute(2, 0, 1).unsqueeze(0)
        x = x / 255.0
        x = x.to(self.device)
        x = (x - self.mean) / self.std
        return x
    
    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        x = self.preprocess(image)
        output = self.model.forward_features(x)
        
        cls_token = output['x_norm_clstoken']
        patch_tokens = output['x_norm_patchtokens']
        
        N = patch_tokens.shape[1]
        H_grid = W_grid = int(N ** 0.5)
        assert H_grid * W_grid == N, f"Non-square patch grid: {N}"
        
        patch_grid = patch_tokens.squeeze(0).reshape(H_grid, W_grid, -1)
        
        return {
            'cls_token': cls_token.squeeze(0).cpu(),
            'patch_tokens': patch_tokens.squeeze(0).cpu(),
            'patch_grid': patch_grid.cpu(),
            'grid_size': (H_grid, W_grid),
        }


# =============================================================================
# 3. RADAR-TO-PATCH MAPPING
# =============================================================================

def map_radar_to_patches(
    radar_proj: RadarProjection,
    img_size: Tuple[int, int],   # (W, H) = (1600, 900)
    grid_size: Tuple[int, int],  # (H_grid, W_grid) = (37, 37)
    patch_radius: int = 1,
) -> List[Dict]:
    """Map each radar point to its corresponding DINOv2 patch neighborhood."""
    if radar_proj.n_points == 0:
        return []
    
    img_w, img_h = img_size
    H_grid, W_grid = grid_size
    
    mappings = []
    for i in range(radar_proj.n_points):
        px, py = radar_proj.pixel_coords[:, i]
        
        px_resized = px * 518.0 / img_w
        py_resized = py * 518.0 / img_h
        
        patch_col = max(0, min(int(px_resized // 14), W_grid - 1))
        patch_row = max(0, min(int(py_resized // 14), H_grid - 1))
        
        patch_indices = []
        for dr in range(-patch_radius, patch_radius + 1):
            for dc in range(-patch_radius, patch_radius + 1):
                r, c = patch_row + dr, patch_col + dc
                if 0 <= r < H_grid and 0 <= c < W_grid:
                    patch_indices.append((r, c))
        
        mappings.append({
            'center_patch': (patch_row, patch_col),
            'neighborhood_patches': patch_indices,
            'pixel_coord': (px, py),
            'depth': float(radar_proj.depths[i]),
            'rcs': float(radar_proj.rcs[i]),
            'velocity': radar_proj.velocities[i].tolist(),
            'dynprop': int(radar_proj.dynprop[i]),
            'pdh0': float(radar_proj.pdh0[i]),
        })
    
    return mappings


# =============================================================================
# 4. SAMPLE RESULT (Lightweight — no patch_grid)
# =============================================================================

@dataclass
class SampleResult:
    """
    Processed result for a single nuScenes sample.
    
    Stores sufficient statistics for coherence scoring without the full patch grid:
    - image_mean_feature: for vanilla L2 scoring (proven Δ=0.78)
    - radar_patch_features: DINOv2 features at radar projection locations
    - radar_l2_scores: pre-computed vanilla L2 distances (for fast evaluation)
    - image_patch_l2_stats: mean/std of L2 distances across ALL patches (reference)
    """
    sample_token: str
    scene_token: str
    scene_description: str
    timestamp: int
    
    # DINOv2 global features
    cls_token: np.ndarray              # (D,) CLS token
    image_mean_feature: np.ndarray     # (D,) mean of all 37x37 patch features
    
    # Image-level patch statistics (for reference distribution)
    image_patch_l2_mean: float         # mean L2 dist of all patches from image_mean
    image_patch_l2_std: float          # std of L2 dists
    image_patch_l2_median: float       # median L2 dist
    image_patch_l2_p95: float          # 95th percentile
    
    # Radar info
    n_radar_points: int
    radar_mappings: List[Dict]
    
    # Per-radar-point DINOv2 features and pre-computed scores
    radar_patch_features: np.ndarray   # (N_radar, D)
    radar_l2_scores: np.ndarray        # (N_radar,) L2 distance from image_mean
    
    # Radar physical attributes
    radar_rcs: np.ndarray              # (N_radar,)
    radar_depths: np.ndarray           # (N_radar,)
    radar_velocities: np.ndarray       # (N_radar, 2)
    
    # Spatial band info for confound control
    radar_row_mean: float              # mean patch row of radar points
    radar_row_std: float               # spread of radar in vertical axis
    
    # Band-relative L2 stats: L2 distances of all patches in the radar's row band
    band_l2_mean: float
    band_l2_std: float


# =============================================================================
# 5. PROCESS SINGLE SAMPLE
# =============================================================================

def process_sample(
    nusc: NuScenes,
    sample_token: str,
    feature_extractor: DINOv2FeatureExtractor,
    cfg: ExperimentConfig,
) -> Optional[SampleResult]:


    """
    Full pipeline for a single sample.
    Computes and caches all necessary statistics without storing patch_grid.
    """
    sample = nusc.get('sample', sample_token)
    scene = nusc.get('scene', sample['scene_token'])
    
    # 1. Radar projection
    try:
        radar_proj = load_and_project_radar(nusc, sample_token, cfg.nuscenes)
    except FileNotFoundError:
        print(f"    [Warning] Missing radar file for {sample_token}. Skipping.")
        return None
        
    if radar_proj.n_points == 0:
        return None
    
    # 2. DINOv2 features
    try:
        cam_data = nusc.get('sample_data', sample['data'][cfg.nuscenes.camera_channel])
        img_path = Path(cfg.nuscenes.dataroot) / cam_data['filename']
        image = Image.open(img_path).convert('RGB')
        features = feature_extractor.extract_features(image)
    except FileNotFoundError:
        print(f"    [Warning] Missing image file for {sample_token}. Skipping.")
        return None
    
    # 3. Map radar to patches
    H, W = features['grid_size']
    mappings = map_radar_to_patches(
        radar_proj, img_size=(1600, 900), grid_size=(H, W),
        patch_radius=cfg.coherence.radar_patch_radius,
    )
    
    # 4. Extract features and compute statistics
    patch_grid = features['patch_grid'].numpy()  # (H, W, D)
    D = patch_grid.shape[-1]
    all_patches = patch_grid.reshape(-1, D)       # (H*W, D)
    
    # Global image statistics
    image_mean = all_patches.mean(axis=0)          # (D,)
    all_l2 = np.linalg.norm(all_patches - image_mean, axis=1)  # (H*W,)
    
    # Radar patch features
    radar_features = np.zeros((len(mappings), D))
    radar_rows = []
    for i, m in enumerate(mappings):
        feats = [patch_grid[r, c] for (r, c) in m['neighborhood_patches']]
        radar_features[i] = np.mean(feats, axis=0)
        radar_rows.append(m['center_patch'][0])
    
    radar_rows = np.array(radar_rows, dtype=float)
    
    # Pre-compute vanilla L2 scores for radar points
    radar_l2 = np.linalg.norm(radar_features - image_mean, axis=1)
    
    # Band-relative statistics (for spatial confound control)
    row_min = max(0, int(radar_rows.min()) - 2)
    row_max = min(H - 1, int(radar_rows.max()) + 2)
    band_patches = patch_grid[row_min:row_max+1, :, :].reshape(-1, D)
    band_l2 = np.linalg.norm(band_patches - image_mean, axis=1)
    
    return SampleResult(
        sample_token=sample_token,
        scene_token=sample['scene_token'],
        scene_description=scene['description'],
        timestamp=sample['timestamp'],
        cls_token=features['cls_token'].numpy(),
        image_mean_feature=image_mean,
        image_patch_l2_mean=float(all_l2.mean()),
        image_patch_l2_std=float(all_l2.std()),
        image_patch_l2_median=float(np.median(all_l2)),
        image_patch_l2_p95=float(np.percentile(all_l2, 95)),
        n_radar_points=radar_proj.n_points,
        radar_mappings=mappings,
        radar_patch_features=radar_features,
        radar_l2_scores=radar_l2,
        radar_rcs=radar_proj.rcs,
        radar_depths=radar_proj.depths,
        radar_velocities=radar_proj.velocities,
        radar_row_mean=float(radar_rows.mean()),
        radar_row_std=float(radar_rows.std()) if len(radar_rows) > 1 else 0.0,
        band_l2_mean=float(band_l2.mean()),
        band_l2_std=float(band_l2.std()),
    )



# =============================================================================
# 6. SCENE CONDITION CLASSIFICATION (STRICT ISOLATION)
# =============================================================================

def classify_scene_conditions(description: str) -> str:
    """
    Classify nuScenes scene into strictly mutually exclusive conditions.
    """
    desc_lower = description.lower()
    conditions = []
    
    if 'rain' in desc_lower: conditions.append('rain')
    if 'night' in desc_lower or 'dark' in desc_lower: conditions.append('night')
    if 'fog' in desc_lower: conditions.append('fog')
    
    # Strict mutually exclusive routing
    if 'rain' in conditions and 'night' in conditions:
        return 'rain_night_mixed'
    elif 'rain' in conditions:
        return 'rain'
    elif 'night' in conditions:
        return 'night'
    elif 'fog' in conditions:
        return 'fog'
    else:
        return 'clear'

# =============================================================================
# 7. BATCH PROCESSING (V2 STREAMING ARCHITECTURE)
# =============================================================================

def process_split(
    nusc: NuScenes,
    feature_extractor: DINOv2FeatureExtractor,
    cfg: ExperimentConfig,
    output_dir: Path,
    target_condition: str = 'clear',
    scene_tokens: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
):
    import gc
    import torch
    import pickle
    
    scenes = nusc.scene if scene_tokens is None else [nusc.get('scene', t) for t in scene_tokens]
    total_scenes = len(scenes)
    sample_count = 0
    
    chunk_idx = 0 
    results_chunk = {}
    
    for scene_idx, scene in enumerate(scenes):
        condition = classify_scene_conditions(scene['description'])
        
        # STRICT FILTER: Only process the condition requested via CLI
        if condition != target_condition:
            continue
            
        if condition not in results_chunk:
            results_chunk[condition] = []
            
        sample_token = scene['first_sample_token']
        scene_sample_count = 0
        
        print(f"\n[{scene_idx+1}/{total_scenes}] Starting Scene: {scene['description']} ({condition})")
        
        while sample_token:
            result = process_sample(nusc, sample_token, feature_extractor, cfg)
            if result is not None:
                # Convert to lightweight dict immediately to save memory
                lightweight_result = {
                    'sample_token': result.sample_token,
                    'scene_token': result.scene_token,
                    'scene_description': result.scene_description,
                    'timestamp': result.timestamp,
                    'cls_token': result.cls_token,
                    'image_mean_feature': result.image_mean_feature,
                    'image_patch_l2_mean': result.image_patch_l2_mean,
                    'image_patch_l2_std': result.image_patch_l2_std,
                    'image_patch_l2_median': result.image_patch_l2_median,
                    'image_patch_l2_p95': result.image_patch_l2_p95,
                    'n_radar_points': result.n_radar_points,
                    'radar_patch_features': result.radar_patch_features,
                    'radar_l2_scores': result.radar_l2_scores,
                    'radar_rcs': result.radar_rcs,
                    'radar_depths': result.radar_depths,
                    'radar_velocities': result.radar_velocities,
                    'radar_row_mean': result.radar_row_mean,
                    'radar_row_std': result.radar_row_std,
                    'band_l2_mean': result.band_l2_mean,
                    'band_l2_std': result.band_l2_std,
                }
                results_chunk[condition].append(lightweight_result)
            
            sample = nusc.get('sample', sample_token)
            sample_token = sample['next'] if sample['next'] != '' else None
            
            sample_count += 1
            scene_sample_count += 1
            
            # GPU Sweeper
            if sample_count % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            if max_samples and sample_count >= max_samples:
                break
        
        # CHUNKING LOGIC: Save to disk and flush RAM at the end of every valid scene
        if len(results_chunk.get(condition, [])) > 0:
            chunk_path = output_dir / f"radar_{condition}_chunk_{chunk_idx:03d}.pkl"
            print(f">>> SAVING CHUNK {chunk_idx} to {chunk_path} (Reclaiming System RAM)... condition: {condition}")
            with open(chunk_path, 'wb') as f:
                pickle.dump(results_chunk, f)
            
            results_chunk = {condition: []}
            chunk_idx += 1
            gc.collect()

        if max_samples and sample_count >= max_samples:
            break

# =============================================================================
# 8. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="CMGC Radar-DINOv2 Pipeline v3 (V2 Architecture)")
    parser.add_argument("--dataroot", type=str, required=True, help="/data/sets/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--condition", type=str, required=True, help="'clear', 'night', 'rain', 'rain_night_mixed'")
    parser.add_argument("--output_dir", type=str, default="./outputs/cmgc_radar_dinov2")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    cfg = ExperimentConfig()
    cfg.nuscenes.dataroot = args.dataroot
    cfg.nuscenes.version = args.version
    cfg.dinov2.device = args.device
    
    # Create the specific subfolder for the target condition
    target_dir_path = Path(args.output_dir) / args.condition
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading nuScenes {args.version}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    
    print("Loading DINOv2...")
    extractor = DINOv2FeatureExtractor(cfg.dinov2)
    
    print(f"Executing Dense Radar extraction for condition: [{args.condition.upper()}]")
    process_split(nusc, extractor, cfg, target_dir_path, target_condition=args.condition, max_samples=args.max_samples)
    
    print(f"\nExtraction complete. Chunks saved to {target_dir_path}")
