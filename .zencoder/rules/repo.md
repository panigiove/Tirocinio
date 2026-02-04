---
description: Repository Information Overview
alwaysApply: true
---

# Tirocinio - Player Tracking & Pose Estimation Information

## Summary
This repository contains a Python-based video processing pipeline designed for multi-view player tracking and pose estimation, specifically tailored for sports analysis. It leverages **YOLOv11** for object detection and pose estimation, integrated with **SAHI** (Slicing Aided Hyper Inference) to improve detection of small objects in high-resolution frames. The tracking is handled by a custom Hungarian-based tracker that fuses IoU, appearance, and pose information.

## Structure
- **Root**: Contains the core Python scripts for video processing, tracking logic, and utility functions.
- **Tracking/**: Contains project materials, presentation slides, and a sub-directory `material4project/` which holds calibration data and rectified video resources.
- **tracking_12.v2i.yolov11/**: (Ignored by git) Likely contains a dataset in YOLOv11 format for training or evaluation.

## Language & Runtime
**Language**: Python  
**Environment**: Requires CUDA-enabled environment for efficient YOLO and Pose inference.  
**Package Manager**: None explicitly identified (no `requirements.txt` or `poetry.lock`), but dependencies are listed below.

## Dependencies
**Main Dependencies**:
- `ultralytics`: Provides the YOLOv11 model implementation and pose estimation.
- `sahi`: Slicing Aided Hyper Inference for detecting small objects.
- `opencv-python` (`cv2`): For video capture, frame manipulation, and visualization.
- `numpy`: For numerical operations and coordinate transformations.
- `scipy`: Specifically `linear_sum_assignment` for the Hungarian matching algorithm.
- `torch`: Backend for Ultralytics models.

## Main Files & Entry Points
- **`process_video.py`**: The primary single-view processing script implementing the YOLO + SAHI + Tracker pipeline.
- **`process_video_two_views.py`**: An extension of the pipeline supporting dual-camera views with cross-view synchronization and 3D triangulation.
- **`iou_tracker.py`**: Implementation of the `IOUTracker` class, which combines multiple cues (IoU, Appearance, Pose, World distance) for stable tracking.
- **`appearence_utils.py`**: Utilities for calculating appearance feature vectors using HSV histograms and Lab space color features.
- **`parameter_tuning.py`**: A systematic script for testing different combinations of confidence thresholds, IoU weights, and appearance descriptors.
- **`yolo11_pose.py`**: Specific logic for YOLOv11 pose estimation and model training experiments.

## Build & Installation
Since the project is script-based, installation typically involves setting up a Python environment and installing the required libraries:
```bash
pip install ultralytics sahi opencv-python numpy scipy torch
```

## Testing & Validation
The project does not use a formal testing framework like `pytest`. Instead, it relies on:
- **`parameter_tuning.py`**: Performs sweeps over hyperparameters to evaluate tracking stability.
- **`tune_sweeper.py`**: Implements parallelized hyperparameter sweeps using multiprocessing.
- **Manual Verification**: Processing sample clips (e.g., `test_clip1.mp4`) and inspecting the generated output videos (`test_*_v1.mp4`) for tracking continuity and accuracy.
