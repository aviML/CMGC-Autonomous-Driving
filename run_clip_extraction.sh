#!/bin/bash

# Configuration
DATAROOT="/home/avnish/3dVision/OpenPCDet/data/nuscenes"
OUTPUT_BASE="/home/avnish/radar_camera_PR/outputs/cmgc_lidar_clip"
VERSION="v1.0-trainval"

# Ensure output directories exist
mkdir -p "${OUTPUT_BASE}/clear"
mkdir -p "${OUTPUT_BASE}/night"
mkdir -p "${OUTPUT_BASE}/rain"

echo "=== STARTING FULL CLIP EXTRACTION PIPELINE ==="

# 1. CLEAR Split (The primary architectural baseline)
echo "[1/3] Extracting CLEAR scenes..."
python clip_data_pipeline.py \
    --dataroot $DATAROOT \
    --version $VERSION \
    --condition "clear" \
    --output_dir "${OUTPUT_BASE}/clear"

# 2. NIGHT Split 
echo "[2/3] Extracting NIGHT scenes..."
python clip_data_pipeline.py \
    --dataroot $DATAROOT \
    --version $VERSION \
    --condition "night" \
    --output_dir "${OUTPUT_BASE}/night"

# 3. RAIN Split
echo "[3/3] Extracting RAIN scenes..."
python clip_data_pipeline.py \
    --dataroot $DATAROOT \
    --version $VERSION \
    --condition "rain" \
    --output_dir "${OUTPUT_BASE}/rain"

echo "=== ALL EXTRACTIONS COMPLETE ==="
