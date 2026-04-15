"""
Configuration for CMGC Radar-DINOv2 Cross-Modal Coherence Pipeline
Target: Pattern Recognition Special Issue — Foundation Models for Anomaly Detection
Deadline: March 30, 2026
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class NuScenesConfig:
    """nuScenes dataset configuration."""
    dataroot: str = "/data/sets/nuscenes"
    version: str = "v1.0-trainval"
    
    camera_channel: str = "CAM_FRONT"
    radar_channel: str = "RADAR_FRONT"
    
    # Radar sweep accumulation
    radar_nsweeps: int = 3
    radar_min_distance: float = 1.0
    
    # Radar quality filters
    radar_dynprop_states: List[int] = field(default_factory=lambda: list(range(7)))
    radar_invalid_states: List[int] = field(default_factory=lambda: [0])
    radar_ambig_states: List[int] = field(default_factory=lambda: list(range(5)))
    radar_min_rcs: float = -5.0


@dataclass
class DINOv2Config:
    """DINOv2 feature extraction configuration."""
    model_name: str = "dinov2_vitl14_reg"
    model_name_fallback: str = "dinov2_vitl14"
    patch_size: int = 14
    image_size: int = 518  # 518/14 = 37 patches per side
    batch_size: int = 8
    device: str = "cuda"
    feature_dim: int = 1024  # ViT-L


@dataclass
class CoherenceConfig:
    """Cross-modal coherence scoring configuration."""
    regularization_eps: float = 1e-5
    
    # MCD parameters
    mcd_support_fraction: float = 0.5
    pca_dim: int = 64
    
    # Spatial matching
    radar_patch_radius: int = 1  # 3x3 neighborhood
    
    # Anomaly threshold
    anomaly_percentile: float = 95.0


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    nuscenes: NuScenesConfig = field(default_factory=NuScenesConfig)
    dinov2: DINOv2Config = field(default_factory=DINOv2Config)
    coherence: CoherenceConfig = field(default_factory=CoherenceConfig)
    
    output_dir: str = "./outputs/cmgc_radar_dinov2"
    
    # Adverse = physically degrading conditions only
    # 'cloudy' excluded: overcast doesn't meaningfully degrade camera or radar
    # 'rain': camera blur/noise + radar backscatter
    # 'night': camera low-light degradation, radar unaffected
    # 'fog': camera visibility loss, radar partially affected
    adverse_keywords: List[str] = field(default_factory=lambda: [
        "rain", "night", "fog",
    ])
