"""
CMGC LiDAR-DINOv2 Data Pipeline (Production)

Architectural Guarantees:
    - Dense Patch Binning: Maps dense Velodyne returns into corresponding DINOv2 patches.
    - Strict SE(3) Alignment: Chains Lidar -> Ego -> Ego -> Camera transforms to handle rolling ego-motion.
    - Geometric Scaling: Scales native 1600x900 LiDAR projections to the resized 518x518 DINOv2 grid.
    - RAM Optimization: Discards full DINOv2 patch grid, saving only activated patches and global stats.
    - Output Independence: Saves to chunks >= 200 to prevent collisions with Radar data.
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
from PIL import Image

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from pyquaternion import Quaternion

from config import NuScenesConfig, DINOv2Config, ExperimentConfig


# =============================================================================
# 1. DINOv2 FEATURE EXTRACTOR (Identical to your v3 script)
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
# 2. LIDAR PROJECTION AND DENSE BINNING
# =============================================================================

@dataclass
class LidarPatchMapping:
    """Aggregated LiDAR physics for DINOv2 patches."""
    activated_patches: List[Tuple[int, int]] # (row, col)
    depth_means: np.ndarray                  # (N_activated,)
    intensity_means: np.ndarray              # (N_activated,)
    depth_stds: np.ndarray                   # (N_activated,)
    log_point_counts: np.ndarray             # (N_activated,)
    total_projected_points: int


def load_project_and_bin_lidar(
    nusc: NuScenes,
    sample_token: str,
    cfg: ExperimentConfig
) -> LidarPatchMapping:
    """
    Load LiDAR, transform to camera frame, project, and bin into DINOv2 patches.
    """
    sample = nusc.get('sample', sample_token)
    
    cam_data = nusc.get('sample_data', sample['data'][cfg.nuscenes.camera_channel])
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    
    pcl_path = Path(nusc.dataroot) / lidar_data['filename']
    if not pcl_path.exists():
        raise FileNotFoundError(f"Missing LiDAR file: {pcl_path}")
        
    pc = LidarPointCloud.from_file(str(pcl_path))
    
    # SE(3) Alignments (Lidar -> Lidar Ego -> Cam Ego -> Cam)
    cs_record_lidar = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record_lidar['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record_lidar['translation']))
    
    poserecord_lidar = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    pc.rotate(Quaternion(poserecord_lidar['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord_lidar['translation']))
    
    poserecord_cam = nusc.get('ego_pose', cam_data['ego_pose_token'])
    pc.translate(-np.array(poserecord_cam['translation']))
    pc.rotate(Quaternion(poserecord_cam['rotation']).rotation_matrix.T)
    
    cs_record_cam = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record_cam['translation']))
    pc.rotate(Quaternion(cs_record_cam['rotation']).rotation_matrix.T)
    
    # Filter points behind camera
    depths = pc.points[2, :]
    mask_front = depths > 0.1
    pc.points = pc.points[:, mask_front]
    depths = depths[mask_front]
    intensities = pc.points[3, :] 
    
    if len(depths) == 0:
        return _empty_lidar_mapping()
    
    # Project to 2D
    camera_intrinsic = np.array(cs_record_cam['camera_intrinsic'])
    points_2d = view_points(pc.points[:3, :], camera_intrinsic, normalize=True)
    
    orig_w, orig_h = cam_data['width'], cam_data['height']
    u, v = points_2d[0, :], points_2d[1, :]
    
    # Filter to image bounds
    mask_img = (u >= 0) & (u < orig_w) & (v >= 0) & (v < orig_h)
    u, v = u[mask_img], v[mask_img]
    depths, intensities = depths[mask_img], intensities[mask_img]
    total_pts = len(u)
    
    if total_pts == 0:
        return _empty_lidar_mapping()
    
    # --- CRITICAL GEOMETRIC SCALING ---
    # Scale native (u,v) to the resized DINOv2 image resolution
    scale_x = cfg.dinov2.image_size / orig_w
    scale_y = cfg.dinov2.image_size / orig_h
    u_scaled = u * scale_x
    v_scaled = v * scale_y
    
    # Bin into DINOv2 patches
    patch_size = cfg.dinov2.patch_size
    H_grid = W_grid = cfg.dinov2.image_size // patch_size
    
    patch_cols = np.clip((u_scaled // patch_size).astype(int), 0, W_grid - 1)
    patch_rows = np.clip((v_scaled // patch_size).astype(int), 0, H_grid - 1)
    
    # Aggregate physics per patch
    patch_dict = {}
    for i in range(total_pts):
        r, c = patch_rows[i], patch_cols[i]
        idx = (r, c)
        if idx not in patch_dict:
            patch_dict[idx] = {'z': [], 'i': []}
        patch_dict[idx]['z'].append(depths[i])
        patch_dict[idx]['i'].append(intensities[i])
        
    activated_patches = []
    z_means, i_means, z_stds, log_counts = [], [], [], []
    
    for (r, c), vals in patch_dict.items():
        z_arr = np.array(vals['z'])
        i_arr = np.array(vals['i'])
        
        activated_patches.append((r, c))
        z_means.append(np.mean(z_arr))
        i_means.append(np.mean(i_arr))
        z_stds.append(np.std(z_arr))
        log_counts.append(np.log(len(z_arr) + 1))
        
    return LidarPatchMapping(
        activated_patches=activated_patches,
        depth_means=np.array(z_means, dtype=np.float32),
        intensity_means=np.array(i_means, dtype=np.float32),
        depth_stds=np.array(z_stds, dtype=np.float32),
        log_point_counts=np.array(log_counts, dtype=np.float32),
        total_projected_points=total_pts
    )

def _empty_lidar_mapping():
    return LidarPatchMapping([], np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), 0)


# =============================================================================
# 3. SAMPLE RESULT CLASS (Lightweight)
# =============================================================================

@dataclass
class SampleResultLidar:
    """Lightweight representation of DINOv2 + LiDAR features for CCA evaluation."""
    sample_token: str
    scene_token: str
    scene_description: str
    timestamp: int
    
    cls_token: np.ndarray              # (D,)
    image_mean_feature: np.ndarray     # (D,)
    
    image_patch_l2_mean: float         
    image_patch_l2_std: float          
    
    # Aligned Features
    lidar_patch_features: np.ndarray   # (N_activated, D) DINOv2 features at LiDAR patches
    lidar_l2_scores: np.ndarray        # (N_activated,) pre-computed L2 distances
    
    # Physics Matrix
    lidar_depth_means: np.ndarray      # (N_activated,)
    lidar_intensity_means: np.ndarray  # (N_activated,)
    lidar_depth_stds: np.ndarray       # (N_activated,)
    lidar_log_counts: np.ndarray       # (N_activated,)
    
    total_projected_points: int


# =============================================================================
# 4. PROCESS SINGLE SAMPLE
# =============================================================================

def process_sample(
    nusc: NuScenes,
    sample_token: str,
    feature_extractor: DINOv2FeatureExtractor,
    cfg: ExperimentConfig,
) -> Optional[SampleResultLidar]:

    sample = nusc.get('sample', sample_token)
    scene = nusc.get('scene', sample['scene_token'])
    
    # 1. LiDAR extraction & binning
    try:
        lidar_mapping = load_project_and_bin_lidar(nusc, sample_token, cfg)
    except FileNotFoundError:
        print(f"    [Warning] Missing LiDAR file. Skipping {sample_token}.")
        return None
        
    if lidar_mapping.total_projected_points < 100 or len(lidar_mapping.activated_patches) == 0:
        return None
        
    # 2. Vision extraction
    try:
        cam_data = nusc.get('sample_data', sample['data'][cfg.nuscenes.camera_channel])
        img_path = Path(cfg.nuscenes.dataroot) / cam_data['filename']
        image = Image.open(img_path).convert('RGB')
        features = feature_extractor.extract_features(image)
    except FileNotFoundError:
        print(f"    [Warning] Missing camera file. Skipping {sample_token}.")
        return None
        
    patch_grid = features['patch_grid'].numpy()  # (H, W, D)
    D = patch_grid.shape[-1]
    
    # 3. Global Stats (Memory Efficient)
    all_patches = patch_grid.reshape(-1, D)
    image_mean = all_patches.mean(axis=0)
    all_l2 = np.linalg.norm(all_patches - image_mean, axis=1)
    
    # 4. Extract aligned DINOv2 features for activated LiDAR patches
    aligned_vision = []
    for (r, c) in lidar_mapping.activated_patches:
        aligned_vision.append(patch_grid[r, c])
        
    aligned_vision = np.array(aligned_vision)
    lidar_l2_scores = np.linalg.norm(aligned_vision - image_mean, axis=1)
    
    return SampleResultLidar(
        sample_token=sample_token,
        scene_token=sample['scene_token'],
        scene_description=scene['description'],
        timestamp=sample['timestamp'],
        cls_token=features['cls_token'].numpy(),
        image_mean_feature=image_mean,
        image_patch_l2_mean=float(all_l2.mean()),
        image_patch_l2_std=float(all_l2.std()),
        lidar_patch_features=aligned_vision,
        lidar_l2_scores=lidar_l2_scores,
        lidar_depth_means=lidar_mapping.depth_means,
        lidar_intensity_means=lidar_mapping.intensity_means,
        lidar_depth_stds=lidar_mapping.depth_stds,
        lidar_log_counts=lidar_mapping.log_point_counts,
        total_projected_points=lidar_mapping.total_projected_points
    )

# =============================================================================
# 5. CONDITION UTILS & PIPELINE EXECUTION
# =============================================================================

def classify_scene_conditions(description: str) -> str:
    desc_lower = description.lower()
    conditions = []
    if 'rain' in desc_lower: conditions.append('rain')
    if 'night' in desc_lower or 'dark' in desc_lower: conditions.append('night')
    if 'fog' in desc_lower: conditions.append('fog')
    if not conditions: conditions.append('clear')
    return '+'.join(sorted(conditions))


def process_split(
    nusc: NuScenes,
    feature_extractor: DINOv2FeatureExtractor,
    cfg: ExperimentConfig,
    output_dir: Path,
    target_condition: str = 'clear',
    max_samples: Optional[int] = None,
):
    import gc
    import pickle
    
    scenes = nusc.scene
    total_scenes = len(scenes)
    sample_count = 0
    
    # Use chunk indices >= 200 for LiDAR to never overwrite Radar chunks (001-199)
    chunk_idx = 200 
    results_chunk = {}
    
    for scene_idx, scene in enumerate(scenes):
        condition = classify_scene_conditions(scene['description'])
        
        if target_condition not in condition:
            continue
            
        if condition not in results_chunk:
            results_chunk[condition] = []
            
        print(f"\n[{scene_idx+1}/{total_scenes}] Starting LiDAR Scene: {scene['description']} ({condition})")
        sample_token = scene['first_sample_token']
        
        while sample_token:
            result = process_sample(nusc, sample_token, feature_extractor, cfg)
            
            if result is not None:
                # Store as dict to guarantee safe pickling and loading
                results_chunk[condition].append({
                    'sample_token': result.sample_token,
                    'scene_token': result.scene_token,
                    'scene_description': result.scene_description,
                    'timestamp': result.timestamp,
                    'cls_token': result.cls_token.astype(np.float16),
                    'image_mean_feature': result.image_mean_feature.astype(np.float16),
                    'image_patch_l2_mean': result.image_patch_l2_mean,
                    'image_patch_l2_std': result.image_patch_l2_std,
                    'lidar_patch_features': result.lidar_patch_features.astype(np.float32),
                    'lidar_l2_scores': result.lidar_l2_scores.astype(np.float32),
                    'lidar_depth_means': result.lidar_depth_means.astype(np.float32),
                    'lidar_intensity_means': result.lidar_intensity_means.astype(np.float32),
                    'lidar_depth_stds': result.lidar_depth_stds.astype(np.float32),
                    'lidar_log_counts': result.lidar_log_counts.astype(np.float32),
                    'total_projected_points': result.total_projected_points
                })
                
                
                

            
            sample = nusc.get('sample', sample_token)
            sample_token = sample['next'] if sample['next'] != '' else None
            sample_count += 1
            
            # GPU/RAM Sweeper
            if sample_count % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            if max_samples and sample_count >= max_samples:
                break
                
        # Chunking: Save after every valid scene
        if len(results_chunk.get(condition, [])) > 0:
            chunk_path = output_dir / f"lidar_results_chunk_{chunk_idx:03d}.pkl"
            print(f">>> SAVING CHUNK {chunk_idx} to {chunk_path}...")
            with open(chunk_path, 'wb') as f:
                pickle.dump(results_chunk, f)
            
            results_chunk = {condition: []}
            chunk_idx += 1
            gc.collect()

        if max_samples and sample_count >= max_samples:
            break

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CMGC LiDAR-DINOv2 Pipeline")
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--condition", type=str, required=True, help="e.g., 'clear', 'night', 'rain'")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    cfg = ExperimentConfig()
    cfg.nuscenes.dataroot = args.dataroot
    cfg.nuscenes.version = args.version
    cfg.dinov2.device = args.device
    
    out_dir_path = Path(args.output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading nuScenes {args.version}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    
    print("Initializing DINOv2 Feature Extractor...")
    extractor = DINOv2FeatureExtractor(cfg.dinov2)
    
    print(f"Executing Dense LiDAR extraction for condition: [{args.condition.upper()}]")
    process_split(nusc, extractor, cfg, out_dir_path, target_condition=args.condition, max_samples=args.max_samples)
    
    print("\nExtraction complete.")
