#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "======================================================================="
echo "PHASE 1: Evaluating nuScenes Dense LiDAR Baseline"
echo "======================================================================="
python coherence_eval_lidar_streaming.py

echo -e "\n======================================================================="
echo "PHASE 2: Evaluating nuScenes Sparse Radar Baseline"
echo "======================================================================="
python coherence_eval_radar_streaming.py

echo -e "\n======================================================================="
echo "PHASE 3: Evaluating RADIATE Sparse Radar Baseline"
echo "======================================================================="
python coherence_eval_radiate_streaming.py

echo -e "\n======================================================================="
echo "[SUCCESS] All Cross-Modal Geometric Collapse (CMGC) evaluations complete!"
echo "======================================================================="
