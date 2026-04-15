"""
CMGC LiDAR-CLIP Data Pipeline (Production-Sanitized)

Optimizations:
    - Zero-Reference Tensor Deletion: Prevents GPU memory fragmentation.
    - Per-Sample Cache Clearing: Keeps VRAM footprint constant.
    - FP16 Precision: Reduces VRAM usage by 50%.
    - Scene-Level Persistence: Protects system RAM from dictionary bloat.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle
import gc
from tqdm import tqdm

import torch
from PIL import Image

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


# from config import NuScenesConfig, ExperimentConfig

# =============================================================================
# 1. SANITIZED CLIP FEATURE EXTRACTOR
# =============================================================================

from transformers import CLIPVisionModel, CLIPImageProcessor

class CLIPFeatureExtractor:
    """Extract patch-level features with strict memory sanitization."""
    
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP-ViT-B/16 on {self.device} (FP16)...")
        
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch16",
            use_safetensors=True,
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()
        
    @torch.no_grad()
    def extract_features(self, image: Image.Image):
        # Maintain geometric alignment with 224x224 squish
        image_squished = image.resize((224, 224), Image.BICUBIC)
        
        inputs = self.processor(
            images=image_squished, 
            return_tensors="pt", 
            do_resize=False, 
            do_center_crop=False
        ).to(self.device).to(torch.float16)
        
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract last hidden state and move to CPU immediately to free VRAM
        hidden_states = outputs.hidden_states[-1].squeeze(0).cpu().to(torch.float32)
        
        cls_token = hidden_states[0].numpy()
        patch_tokens = hidden_states[1:].numpy()
        
        # Reshape to 14x14 grid
        H_grid = W_grid = int(patch_tokens.shape[0] ** 0.5)
        patch_grid = patch_tokens.reshape(H_grid, W_grid, -1)
        
        # --- CRITICAL SANITIZATION STEP ---
        del outputs, inputs, hidden_states
        torch.cuda.empty_cache() # Flush fragments
        
        return {
            'cls_token': cls_token,
            'patch_grid': patch_grid,
            'grid_size': (H_grid, W_grid)
        }

# =============================================================================
# 2. LIDAR PROJECTION (Logic remains identical for consistency)
# =============================================================================

@dataclass
class LidarPatchMapping:
    activated_patches: List[Tuple[int, int]]
    depth_means: np.ndarray
    intensity_means: np.ndarray
    depth_stds: np.ndarray
    log_point_counts: np.ndarray
    total_projected_points: int

def load_project_and_bin_lidar(nusc: NuScenes, sample_token: str, cam_channel: str) -> LidarPatchMapping:
    sample = nusc.get('sample', sample_token)
    cam_data = nusc.get('sample_data', sample['data'][cam_channel])
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    
    pc = LidarPointCloud.from_file(str(Path(nusc.dataroot) / lidar_data['filename']))
    
    # SE(3) Transform Chain
    for record_type, token_key in [('calibrated_sensor', 'calibrated_sensor_token'), ('ego_pose', 'ego_pose_token')]:
        ref = nusc.get(record_type, lidar_data[token_key])
        pc.rotate(Quaternion(ref['rotation']).rotation_matrix)
        pc.translate(np.array(ref['translation']))
    
    for record_type, token_key in [('ego_pose', 'ego_pose_token'), ('calibrated_sensor', 'calibrated_sensor_token')]:
        ref = nusc.get(record_type, cam_data[token_key])
        pc.translate(-np.array(ref['translation']))
        pc.rotate(Quaternion(ref['rotation']).rotation_matrix.T)
        
    depths = pc.points[2, :]
    mask = depths > 0.1
    pc.points, depths = pc.points[:, mask], depths[mask]
    
    if len(depths) == 0: return _empty_lidar_mapping()
    
    intrinsic = np.array(nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])['camera_intrinsic'])
    pts_2d = view_points(pc.points[:3, :], intrinsic, normalize=True)
    
    u, v = pts_2d[0, :], pts_2d[1, :]
    mask_img = (u >= 0) & (u < cam_data['width']) & (v >= 0) & (v < cam_data['height'])
    u, v, depths = u[mask_img], v[mask_img], depths[mask_img]
    
    if len(u) == 0: return _empty_lidar_mapping()
    
    # Geometric Scaling to 224x224 (CLIP Grid)
    u_s, v_s = u * (224 / cam_data['width']), v * (224 / cam_data['height'])
    p_cols, p_rows = np.clip(u_s // 16, 0, 13).astype(int), np.clip(v_s // 16, 0, 13).astype(int)
    
    patch_dict = {}
    for i in range(len(u)):
        idx = (p_rows[i], p_cols[i])
        if idx not in patch_dict: patch_dict[idx] = []
        patch_dict[idx].append(depths[i])
        
    act, z_m, z_s, l_c = [], [], [], []
    for (r, c), zs in patch_dict.items():
        act.append((r, c))
        z_m.append(np.mean(zs))
        z_s.append(np.std(zs))
        l_c.append(np.log(len(zs) + 1))
        
    return LidarPatchMapping(act, np.array(z_m), np.zeros_like(z_m), np.array(z_s), np.array(l_c), len(u))

def _empty_lidar_mapping():
    return LidarPatchMapping([], np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), 0)

# =============================================================================
# 3. PRODUCTION EXECUTION LOOP
# =============================================================================

def process_full_clear(nusc: NuScenes, extractor: CLIPFeatureExtractor, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter only for 'clear' scenes
    clear_scenes = [s for s in nusc.scene if 'rain' not in s['description'].lower() and 
                    'night' not in s['description'].lower() and 'dark' not in s['description'].lower()]
    
    print(f"Starting extraction for {len(clear_scenes)} Clear scenes...")
    
    for idx, scene in enumerate(tqdm(clear_scenes)):
        scene_results = []
        sample_token = scene['first_sample_token']
        
        while sample_token:
            # 1. LiDAR Physics
            lidar = load_project_and_bin_lidar(nusc, sample_token, 'CAM_FRONT')
            
            if lidar.total_projected_points >= 100:
                # 2. Vision Features
                sample = nusc.get('sample', sample_token)
                cam_path = Path(nusc.dataroot) / nusc.get('sample_data', sample['data']['CAM_FRONT'])['filename']
                features = extractor.extract_features(Image.open(cam_path).convert('RGB'))
                
                # 3. Align & Store
                aligned_v = np.array([features['patch_grid'][r, c] for (r, c) in lidar.activated_patches])
                
                scene_results.append({
                    'sample_token': sample_token,
                    'scene_token': scene['token'],
                    'lidar_patch_features': aligned_v.astype(np.float32),
                    'lidar_depth_means': lidar.depth_means.astype(np.float32),
                    'lidar_depth_stds': lidar.depth_stds.astype(np.float32),
                    'lidar_log_counts': lidar.log_point_counts.astype(np.float32)
                })
            
            sample_token = nusc.get('sample', sample_token)['next']
            if not sample_token: break

        # Save Scene Chunk and clear RAM
        if scene_results:
            with open(output_dir / f"scene_{idx:03d}.pkl", 'wb') as f:
                pickle.dump({'clear': scene_results}, f)
        
        del scene_results
        gc.collect()

if __name__ == "__main__":
    # Update these paths for your workstation
    DATA_ROOT = "/home/avnish/3dVision/OpenPCDet/data/nuscenes"
    OUT_DIR = "/home/avnish/radar_camera_PR/outputs/cmgc_lidar_clip/clear_full"
    
    nusc = NuScenes(version='v1.0-trainval', dataroot=DATA_ROOT, verbose=True)
    extractor = CLIPFeatureExtractor()
    process_full_clear(nusc, extractor, Path(OUT_DIR))
