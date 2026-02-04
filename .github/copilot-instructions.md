# AI Coding Agent Instructions for Tirocinio Project

## Project Overview
This is a computer vision project for multi-view object tracking in team sports videos (primarily soccer/basketball). The system detects players using YOLO, tracks them across frames using a multi-cue IOU tracker, and supports cross-view matching with 3D triangulation.

## Architecture & Data Flow

### Core Pipeline Components
- **Detection**: YOLOv11 models (`yolo11x.pt`) with SAHI sliced inference for large images
- **Tracking**: Hungarian algorithm-based matcher combining IoU, appearance, and pose cues
- **Multi-view**: Cross-camera tracking with 3D world coordinate projection
- **Output**: Annotated videos + CSV event logs (`track_events.csv`)

### Key Files & Responsibilities
- `process_video.py`: Single-view pipeline with ROI masking, detection, pose estimation, and tracking
- `process_video_two_views.py`: Multi-view processing with cross-view matching and triangulation
- `iou_tracker.py`: Core tracking logic with Hungarian matching and event logging
- `appearence_utils.py`: Team-specific appearance descriptors (HSV/LAB histograms from jersey areas)
- `Tracking/camera_data*/`: Camera calibration files for 3D projection

### Data Flow
1. Video frames → ROI trapezoid masking → SAHI+YOLO detection
2. Detections → Appearance extraction + Pose estimation → Tracker update
3. Tracker → Hungarian matching → Event logging → Frame annotation

## Critical Developer Workflows

### Running the Pipeline
```python
# Single view
from process_video import yolo_sahi_pose_tracking
yolo_sahi_pose_tracking('video_path.mp4')

# Multi-view (requires camera calibration)
python process_video_two_views.py  # See file for parameter setup
```

### Hyperparameter Tuning
```python
python tune_sweeper.py  # Grid search over detection/tracking parameters
```
Results saved to `tuning_results.json` and `best_run_results.json`

### Model Setup
- YOLO weights are pre-downloaded (`.pt` files in root)
- Install dependencies: `pip install ultralytics sahi opencv-python numpy scipy`

## Project-Specific Patterns & Conventions

### ROI & Masking
- Trapezoidal field masks defined by corner coordinates
- Applied before detection to focus on playable area
- See `TRAPEZOID_TOP_LEFT`, etc. constants in processing scripts

### Appearance Extraction
- Team jersey colors from upper body (60% of bbox height)
- HSV histogram (16H×8S bins) + LAB mean color per horizontal strip
- L2-normalized descriptor for cosine similarity matching
- Example: `compute_team_appearence(frame, bbox)`

### Pose Estimation Strategy
- Multi-attempt fallback: no padding → 25% padding with lower thresholds
- Keypoints normalized to bbox coordinates for pose matching
- Example attempts configuration:
```python
pose_attempts = [
    {'pad': 0.0, 'conf': None, 'iou': None},
    {'pad': 0.25, 'conf': 0.03, 'iou': 0.005}
]
```

### Tracking Events & Logging
- CSV format: `frame,track_id,event` (events: added, lost, temporary_missed, refound)
- Global ID management for cross-view consistency
- Statistics tracking: active tracks, refound rate, etc.

### Camera Calibration Integration
- Intrinsic/extrinsic parameters loaded from `Tracking/camera_data*/calib/`
- Homography computation for ground plane projection
- Triangulation for cross-view 3D positioning

## Integration Points & Dependencies

### External Libraries
- **ultralytics YOLO**: Detection and pose estimation models
- **SAHI**: Sliced inference for high-resolution images
- **OpenCV**: Image processing, camera calibration, projection
- **scipy**: Hungarian algorithm for optimal matching

### File Formats
- **Videos**: MP4 input/output with OpenCV VideoWriter
- **Annotations**: YOLO format (.txt) for initial track seeding
- **Calibration**: JSON/camera parameter files
- **Models**: PyTorch .pt files (YOLOv11 variants)

## Common Development Tasks

### Adding New Tracking Cues
1. Extend detection dict in `process_video.py` with new feature
2. Add weight parameter to `IOUTracker.__init__`
3. Implement similarity function in `iou_tracker.py::_cost_matrix`
4. Update Hungarian cost computation

### Modifying ROI Masks
- Edit trapezoid coordinates in processing script constants
- Test mask visualization: `cv.imshow('mask', mask)`

### Cross-View Setup
1. Calibrate cameras → save parameters to `Tracking/camera_data*/`
2. Initialize separate trackers with shared `GlobalIDManager`
3. Call `assign_cross_view_tracks()` after per-view updates

### Performance Optimization
- Adjust SAHI slice parameters for speed vs accuracy trade-off
- Tune pose estimation attempts (reduce fallbacks for speed)
- Modify `max_missed_frames` for track persistence vs false positives

## Debugging & Validation

### Common Issues
- **Empty detections**: Check SAHI confidence thresholds, ROI mask coverage
- **Broken tracks**: Adjust `match_threshold`, `iou_weight`, `appearance_weight`
- **Pose failures**: Verify YOLO pose model loading, check bbox padding
- **Cross-view mismatches**: Validate camera calibration, check triangulation

### Validation Scripts
- `diagnose_alignment.py`: Camera calibration verification
- `Python_test.py`: Basic functionality tests
- `list_and_clean.py`: Dataset management utilities

### Output Analysis
- Track event CSVs for temporal consistency
- Video overlays for visual validation
- Statistics in tracker for performance metrics

## Code Style & Structure

### Naming Conventions
- Functions: `snake_case` (e.g., `compute_team_appearence`)
- Variables: `snake_case` with descriptive names
- Constants: `UPPER_CASE` for mask coordinates, thresholds

### Error Handling
- Graceful fallbacks (pose estimation attempts)
- Bounds checking for bbox coordinates
- File existence checks for annotations

### Performance Considerations
- CUDA device specification for GPU acceleration
- Frame-by-frame processing with FPS monitoring
- Memory-efficient cropped processing for pose estimation