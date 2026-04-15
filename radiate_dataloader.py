"""
RADIATE to CMGC Pipeline (Production Grade)

Features:
- Safe Resumption: Automatically detects existing chunks and fast-forwards.
- Fault Tolerance: Gracefully catches missing/corrupted files.
- Memory Management: Active garbage collection and CUDA cache clearing.
- Geometric Projection: Mathematically maps BEV radar to perspective vision.
"""

import os
import cv2
import numpy as np
import torch
import pickle
import argparse
import gc
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Re-use your DINOv2 extractor
from config import DINOv2Config, ExperimentConfig
from data_pipeline import DINOv2FeatureExtractor

class RadiateExtractor:
    def __init__(self, radiate_root: str, sequence: str):
        self.seq_dir = Path(radiate_root) / sequence
        self.radar_dir = self.seq_dir / "Navtech_Cartesian"
        self.camera_dir = self.seq_dir / "zed_left"
        self.sequence_name = sequence
        
        self.cart_res = 0.175
        self.center = 576
        
        # ZED Camera Intrinsics
        self.K = np.array([
            [678.0, 0.0, 640.0],
            [0.0, 678.0, 360.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Extrinsics: Radar to Camera mapping based on Sanity Check Geometry
        self.R_cam_radar = np.array([
            [ 1.0,  0.0,  0.0],  # Cam X (Right) = Radar X (Right)
            [ 0.0,  0.0, -1.0],  # Cam Y (Down)  = -Radar Z (Up)
            [ 0.0,  1.0,  0.0]   # Cam Z (Fwd)   = Radar Y (Fwd)
        ])
        self.T_cam_radar = np.array([0.0, 1.0, 0.5])

    def get_synchronized_frames(self):
        radar_files = sorted(os.listdir(self.radar_dir))
        cam_files = sorted(os.listdir(self.camera_dir))
        
        cam_ts = np.array([float(f.replace('.png', '')) for f in cam_files])
        
        for r_file in radar_files:
            r_ts = float(r_file.replace('.png', ''))
            idx = np.argmin(np.abs(cam_ts - r_ts))
            
            if abs(cam_ts[idx] - r_ts) < 0.05:
                yield (
                    self.radar_dir / r_file,
                    self.camera_dir / cam_files[idx],
                    r_ts
                )

    def extract_radar_points(self, radar_img: np.ndarray, intensity_thresh: int = 80):
        mask = np.zeros_like(radar_img)
        pt_left = (int(self.center + 500 * np.cos(np.deg2rad(270 - 55))), 
                   int(self.center + 500 * np.sin(np.deg2rad(270 - 55))))
        pt_right = (int(self.center + 500 * np.cos(np.deg2rad(270 + 55))), 
                    int(self.center + 500 * np.sin(np.deg2rad(270 + 55))))
        
        triangle = np.array([[self.center, self.center], pt_left, pt_right])
        cv2.fillPoly(mask, [triangle], 255)
        
        masked_radar = cv2.bitwise_and(radar_img, mask)
        y_coords, x_coords = np.where(masked_radar > intensity_thresh)
        intensities = masked_radar[y_coords, x_coords]
        
        if len(x_coords) == 0:
            return None, None
            
        # Convert to physical 3D space
        X_rad = (x_coords - self.center) * self.cart_res
        Y_rad = (self.center - y_coords) * self.cart_res
        Z_rad = np.zeros_like(X_rad)
        pts_3d_radar = np.vstack((X_rad, Y_rad, Z_rad)).T
        
        depths = np.linalg.norm(pts_3d_radar[:, :2], axis=1)
        valid = (depths > 2.0) & (depths < 70.0)
        
        pts_3d_radar = pts_3d_radar[valid]
        intensities = intensities[valid]
        depths = depths[valid]
        
        if len(pts_3d_radar) == 0:
            return None, None
            
        # Project to Camera Pixel Space
        pts_3d_cam = (self.R_cam_radar @ pts_3d_radar.T).T + self.T_cam_radar
        pts_2d_hom = (self.K @ pts_3d_cam.T).T
        u = pts_2d_hom[:, 0] / (pts_2d_hom[:, 2] + 1e-6)
        v = pts_2d_hom[:, 1] / (pts_2d_hom[:, 2] + 1e-6)
        
        in_view = (u >= 0) & (u < 1280) & (v >= 0) & (v < 720) & (pts_3d_cam[:, 2] > 0)
        uv_coords = np.vstack((u[in_view], v[in_view])).T
        
        return uv_coords, {
            'rcs': intensities[in_view],
            'depths': depths[in_view],
            'velocities': np.zeros((in_view.sum(), 2)) 
        }

def process_radiate_sequence(radiate_root: str, sequence: str, feature_extractor, out_dir: Path):
    extractor = RadiateExtractor(radiate_root, sequence)
    condition = "clear" if "city" in sequence or "rural" in sequence else "fog" if "fog" in sequence else "snow" if "snow" in sequence else "night"
    
    # --- RESUMPTION LOGIC ---
    chunk_size = 200
    existing_chunks = list(out_dir.glob(f"radiate_{sequence}_chunk_*.pkl"))
    
    if existing_chunks:
        last_chunk_num = max([int(p.stem.split('_')[-1]) for p in existing_chunks])
        chunk_idx = last_chunk_num + 1
        start_frame_idx = chunk_idx * chunk_size
        print(f"\n[INFO] Found {len(existing_chunks)} existing chunks for {sequence}.")
        print(f"[INFO] Resuming extraction from frame {start_frame_idx} (Chunk {chunk_idx})...")
    else:
        chunk_idx = 0
        start_frame_idx = 0
        print(f"\n[INFO] Starting fresh extraction for {sequence} ({condition})...")
    
    frames = list(extractor.get_synchronized_frames())
    
    # Fast-forward the loop if resuming
    frames_to_process = frames[start_frame_idx:]
    if not frames_to_process:
        print(f"[INFO] {sequence} is already fully processed. Skipping.")
        return

    results = {condition: []}
    sample_count = 0
    
    for radar_path, cam_path, ts in tqdm(frames_to_process, desc=f"Processing {sequence}"):
        
        # --- ERROR HANDLING ---
        try:
            radar_img = cv2.imread(str(radar_path), cv2.IMREAD_GRAYSCALE)
            if radar_img is None:
                raise FileNotFoundError("cv2.imread returned None")
                
            cam_img = Image.open(cam_path).convert('RGB')
        except Exception as e:
            print(f"\n[WARNING] Failed to load files for timestamp {ts}: {e}. Skipping.")
            continue
            
        # Extract Radar Points
        uv_coords, phys_data = extractor.extract_radar_points(radar_img)
        if uv_coords is None or len(uv_coords) < 5:
            continue
            
        # Extract DINOv2
        try:
            with torch.no_grad():
                features = feature_extractor.extract_features(cam_img)
        except Exception as e:
            print(f"\n[WARNING] DINOv2 extraction failed for timestamp {ts}: {e}. Skipping.")
            continue
            
        H, W = features['grid_size']
        patch_tokens = features['patch_tokens'].squeeze(0)
        
        patch_x = np.clip((uv_coords[:, 0] / 1280 * W).astype(int), 0, W - 1)
        patch_y = np.clip((uv_coords[:, 1] / 720 * H).astype(int), 0, H - 1)
        patch_indices = patch_y * W + patch_x
        
        radar_patch_features = patch_tokens[patch_indices].cpu().numpy()
        
        sample_res = {
            'sample_token': f"{sequence}_{ts}",
            'n_radar_points': len(uv_coords),
            'radar_patch_features': radar_patch_features,
            'radar_rcs': phys_data['rcs'].astype(np.float32), 
            'radar_depths': phys_data['depths'].astype(np.float32),
            'radar_velocities': phys_data['velocities'].astype(np.float32),
            'cls_token': features['cls_token'].squeeze().cpu().numpy(),
        }
        
        results[condition].append(sample_res)
        sample_count += 1
        
        # MEMORY MANAGEMENT: Clear cache every 50 valid frames
        if sample_count % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Chunking
        if sample_count % chunk_size == 0:
            out_file = out_dir / f"radiate_{sequence}_chunk_{chunk_idx:03d}.pkl"
            with open(out_file, 'wb') as f:
                pickle.dump(results, f)
            results = {condition: []}
            chunk_idx += 1
            
    # Save final partial chunk
    if len(results[condition]) > 0:
        out_file = out_dir / f"radiate_{sequence}_chunk_{chunk_idx:03d}.pkl"
        with open(out_file, 'wb') as f:
            pickle.dump(results, f)
            
    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiate_root", type=str, default="/home/avnish/3dVision/OpenPCDet/data/radiate")
    parser.add_argument("--output_dir", type=str, default="./outputs/radiate_features")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = ExperimentConfig()
    feature_extractor = DINOv2FeatureExtractor(cfg.dinov2)
    
    sequences_to_process = [
        # --- The Clear Baseline (Train & Control) ---
        "city_1_0", "city_1_1", "city_1_3", 
        "city_2_0", "city_3_0", "city_3_1",
        "rural_1_1", "rural_1_3",
        
        # --- The Atmospheric Failure (Experiment 1) ---
        "fog_6_0", "fog_8_0", "fog_8_1", "fog_8_2",
        "tiny_foggy",
        "snow_1_0",
        
        # --- The Illumination Failure (Experiment 2) ---
        "night_1_0", "night_1_2", 
        "night_1_3", "night_1_4", "night_1_5"
    ]
    
    for seq in sequences_to_process:
        process_radiate_sequence(args.radiate_root, seq, feature_extractor, out_dir)
        
    print("\n[SUCCESS] Extraction complete! Ready for CCA evaluation.")
