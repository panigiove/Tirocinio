"""
Script to rectify YOLO format annotations to match rectified videos.

The annotations are in the original (distorted) image coordinate system,
and need to be transformed to match the rectified video coordinate system.

CONFIGURATION:
Modify the variables below to configure the script behavior.
"""

import cv2
import numpy as np
import json
import os
import re
import glob

# ================= CONFIGURATION =================
# Single file mode: Set annotation_file to process one file
# Batch mode: Set annotation_dir to process all .txt files in a directory
ANNOTATION_FILE = None  # e.g., 'tracking_12.v4i.yolov11/train/labels/out4_frame_0001_png.rf.13ad8a164866d2d0e8151ff7a71f7908.txt'
ANNOTATION_DIR = 'tracking_12.v4i.yolov11/train/labels/'

# Image dimensions (original/distorted image size)
IMG_WIDTH = 3840
IMG_HEIGHT = 2160

# Output settings (for single file mode)
OUTPUT_PATH = None  # If None, overwrites input file

# Batch mode settings
OUTPUT_SUFFIX = "_rectified"  # Suffix to add to output filenames
OVERWRITE = False  # If True, overwrites original files in batch mode

# Manual overrides (optional - leave as None for auto-detection)
CAM_INDEX = None  # e.g., "4" for cam_4
CALIB_PATH = None  # Path to calibration JSON file
# =================================================


def load_calibration(calib_path):
    """Load camera calibration data."""
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist


def find_calibration_for_cam(cam_index):
    """
    Find calibration file for a camera index.
    Tries multiple paths similar to rectified_videos.py
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MATERIAL_DIR = os.path.join(BASE_DIR, "Tracking", "material4project", "3D Tracking Material")
    
    candidates = [
        # Current layout used by rectified_videos.py
        os.path.join(
            MATERIAL_DIR,
            "camera_data_with_Rvecs_2ndversion",
            f"cam_{cam_index}",
            "calib",
            "camera_calib.json",
        ),
        # Older dataset fallbacks
        os.path.join(MATERIAL_DIR, "camera_data", f"cam_{cam_index}", "calib", "camera_calib.json"),
        os.path.join(MATERIAL_DIR, "camera_data", f"cam_{cam_index}", "camera_calib.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # fallback: recursive search
    pattern = os.path.join(MATERIAL_DIR, "**", f"cam_{cam_index}", "calib", "camera_calib.json")
    found = glob.glob(pattern, recursive=True)
    if found:
        return found[0]

    pattern2 = os.path.join(MATERIAL_DIR, "**", f"cam_{cam_index}", "*camera_calib.json")
    found2 = glob.glob(pattern2, recursive=True)
    if found2:
        return found2[0]

    return None


def yolo_to_xyxy(center_x, center_y, width, height, img_width, img_height):
    """
    Convert YOLO format (normalized center, width, height) to pixel coordinates (x1, y1, x2, y2).
    """
    x1 = (center_x - width / 2) * img_width
    y1 = (center_y - height / 2) * img_height
    x2 = (center_x + width / 2) * img_width
    y2 = (center_y + height / 2) * img_height
    return x1, y1, x2, y2


def xyxy_to_yolo(x1, y1, x2, y2, img_width, img_height):
    """
    Convert pixel coordinates (x1, y1, x2, y2) to YOLO format (normalized center, width, height).
    """
    center_x = ((x1 + x2) / 2) / img_width
    center_y = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Clamp to [0, 1]
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return center_x, center_y, width, height


def rectify_bbox(bbox_xyxy, mtx, dist, newcameramtx):
    """
    Rectify a bounding box using exactly the same logic as the video rectification.
    Matches rectified_videos.py: cv2.undistort(frame, mtx, dist, None, newcameramtx)
    """
    x1, y1, x2, y2 = bbox_xyxy
    
    # Create a grid of points to sample the distorted bounding box
    n_samples = 15
    x_coords = np.linspace(x1, x2, n_samples)
    y_coords = np.linspace(y1, y2, n_samples)
    
    edge_points = []
    # Grid of points including edges and interior
    for x in x_coords:
        for y in y_coords:
            edge_points.append([x, y])
    
    all_points = np.array([edge_points], dtype=np.float32)
    
    # Use undistortPoints with P=newcameramtx to match cv2.undistort(..., newcameramtx)
    rectified_points = cv2.undistortPoints(all_points, mtx, dist, R=None, P=newcameramtx)
    rectified_points = rectified_points.reshape(-1, 2)
    
    # Find the bounding box that contains all rectified points
    min_x = float(np.min(rectified_points[:, 0]))
    max_x = float(np.max(rectified_points[:, 0]))
    min_y = float(np.min(rectified_points[:, 1]))
    max_y = float(np.max(rectified_points[:, 1]))
    
    return min_x, min_y, max_x, max_y


def rectify_yolo_annotation_file(annotation_path, calib_path, original_img_width, original_img_height, output_path=None):
    """
    Rectify all annotations in a YOLO format annotation file.
    """
    # Load calibration
    mtx, dist = load_calibration(calib_path)
    
    # Check if we need to scale the camera matrix (matches video resolution)
    # Calibration is typically for 3840x2160.
    calib_res = (3840, 2160)
    if (original_img_width, original_img_height) != calib_res:
        sw = original_img_width / calib_res[0]
        sh = original_img_height / calib_res[1]
        mtx[0, 0] *= sw
        mtx[0, 2] *= sw
        mtx[1, 1] *= sh
        mtx[1, 2] *= sh
        print(f"  Note: Scaled camera matrix for {original_img_width}x{original_img_height}")

    # Match rectified_videos.py: newCameraMatrix with alpha=0
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (original_img_width, original_img_height), 0, (original_img_width, original_img_height)
    )

    # Read annotations
    rectified_annotations = []
    
    with open(annotation_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(' ')
            class_id = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert YOLO to pixel coordinates (original image)
            x1, y1, x2, y2 = yolo_to_xyxy(center_x, center_y, width, height, 
                                          original_img_width, original_img_height)
            
            # Rectify the bounding box
            rect_x1, rect_y1, rect_x2, rect_y2 = rectify_bbox(
                (x1, y1, x2, y2), mtx, dist, newcameramtx
            )
            
            # Convert back to YOLO format
            # Note: rectified image has same dimensions as original
            rect_center_x, rect_center_y, rect_width, rect_height = xyxy_to_yolo(
                rect_x1, rect_y1, rect_x2, rect_y2, original_img_width, original_img_height
            )
            
            # Write rectified annotation
            rectified_annotations.append(
                f"{class_id} {rect_center_x:.10f} {rect_center_y:.10f} {rect_width:.10f} {rect_height:.10f}\n"
            )
    
    # Write output
    if output_path is None:
        output_path = annotation_path
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(rectified_annotations)
    
    return len(rectified_annotations)


def extract_cam_index_from_filename(filename):
    """Extract camera index from filename (e.g., out4_frame_0001 -> 4)."""
    match = re.search(r'out(\d+)', filename)
    if match:
        return match.group(1)
    return None


def batch_rectify_annotations(annotation_dir, img_width=3840, img_height=2160, output_suffix="_rectified", overwrite=False):
    """
    Batch process all YOLO annotation files in a directory.
    
    Args:
        annotation_dir: Directory containing annotation files
        img_width: Original image width
        img_height: Original image height
        output_suffix: Suffix to add to output filenames (if overwrite=False)
        overwrite: If True, overwrites original files. If False, creates new files with suffix.
    
    Returns:
        Dictionary mapping input files to output files and processing status
    """
    results = {}
    annotation_files = glob.glob(os.path.join(annotation_dir, "*.txt"))
    
    for anno_file in annotation_files:
        if output_suffix in anno_file:
            continue
        try:
            cam_index = extract_cam_index_from_filename(os.path.basename(anno_file))
            if cam_index is None:
                print(f"Warning: Could not extract camera index from {anno_file}, skipping")
                results[anno_file] = {"status": "skipped", "reason": "no_cam_index"}
                continue
            
            calib_path = find_calibration_for_cam(cam_index)
            if calib_path is None:
                print(f"Warning: Could not find calibration for cam_{cam_index}, skipping {anno_file}")
                results[anno_file] = {"status": "skipped", "reason": "no_calibration"}
                continue
            
            if overwrite:
                output_path = anno_file
            else:
                base, ext = os.path.splitext(anno_file)
                output_path = base + output_suffix + ext
            
            count = rectify_yolo_annotation_file(
                anno_file, calib_path, img_width, img_height, output_path
            )
            results[anno_file] = {"status": "success", "output": output_path, "count": count}
            print(f"DONE: {os.path.basename(anno_file)} -> {os.path.basename(output_path)} ({count} annotations)")
        except Exception as e:
            results[anno_file] = {"status": "error", "error": str(e)}
            print(f"ERROR processing {anno_file}: {e}")
    
    return results


def main():
    """Main function - processes annotations based on configuration variables."""
    
    # Batch mode
    if ANNOTATION_DIR is not None:
        if not os.path.isdir(ANNOTATION_DIR):
            print(f"Error: {ANNOTATION_DIR} is not a directory")
            return
        
        print(f"Batch processing annotations in: {ANNOTATION_DIR}")
        print(f"Image dimensions: {IMG_WIDTH}x{IMG_HEIGHT}")
        if OVERWRITE:
            print("Mode: Overwriting original files")
        else:
            print(f"Mode: Creating new files with suffix '{OUTPUT_SUFFIX}'")
        print()
        
        results = batch_rectify_annotations(
            ANNOTATION_DIR,
            IMG_WIDTH,
            IMG_HEIGHT,
            OUTPUT_SUFFIX,
            OVERWRITE
        )
        
        print()
        print("Summary:")
        success = sum(1 for r in results.values() if r.get("status") == "success")
        total = len(results)
        print(f"  Success: {success}/{total}")
        if success < total:
            skipped = sum(1 for r in results.values() if r.get("status") == "skipped")
            errors = sum(1 for r in results.values() if r.get("status") == "error")
            if skipped > 0:
                print(f"  Skipped: {skipped}")
            if errors > 0:
                print(f"  Errors: {errors}")
        return
    
    # Single file mode
    if ANNOTATION_FILE is None:
        print("Error: Please set either ANNOTATION_FILE or ANNOTATION_DIR in the configuration section")
        print("\nExample for single file:")
        print("  ANNOTATION_FILE = 'path/to/annotation.txt'")
        print("\nExample for batch processing:")
        print("  ANNOTATION_DIR = 'path/to/labels/'")
        return
    
    if not os.path.exists(ANNOTATION_FILE):
        print(f"Error: Annotation file not found: {ANNOTATION_FILE}")
        return
    
    # Determine camera index
    cam_index = CAM_INDEX
    if cam_index is None:
        cam_index = extract_cam_index_from_filename(os.path.basename(ANNOTATION_FILE))
        if cam_index is None:
            print("Error: Could not extract camera index from filename. Please set CAM_INDEX in configuration")
            return
    
    # Find calibration file
    calib_path = CALIB_PATH
    if calib_path is None:
        calib_path = find_calibration_for_cam(cam_index)
        if calib_path is None:
            print(f"Error: Could not find calibration file for cam_{cam_index}")
            return
    
    print(f"Processing: {ANNOTATION_FILE}")
    print(f"Camera: cam_{cam_index}")
    print(f"Calibration: {calib_path}")
    print(f"Image dimensions: {IMG_WIDTH}x{IMG_HEIGHT}")
    
    # Rectify annotations
    count = rectify_yolo_annotation_file(
        ANNOTATION_FILE,
        calib_path,
        IMG_WIDTH,
        IMG_HEIGHT,
        OUTPUT_PATH
    )
    
    output_path = OUTPUT_PATH if OUTPUT_PATH else ANNOTATION_FILE
    print(f"Rectified {count} annotations -> {output_path}")


if __name__ == "__main__":
    main()
