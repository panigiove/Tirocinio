import json
import os
from pathlib import Path

import cv2
import numpy as np

# ===== CONFIG =====
INPUT_DIR = Path(r"tracking_12.v4i.yolov11/train/images")
OUTPUT_DIR = Path(r"tracking_12.v4i.yolov11/train/images_rectified")

CALIB_PATHS = {
    "out13": Path(
        r"Tracking/material4project/3D Tracking Material/camera_data/cam_13/calib/camera_calib.json"
    ),
    "out4": Path(
        r"Tracking/material4project/3D Tracking Material/camera_data/cam_4/calib/camera_calib.json"
    ),
}

# File glob for images
IMAGE_GLOB = "*.jpg"


def load_calibration(calib_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with calib_path.open("r", encoding="utf-8") as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist

def rectified_output_path(in_path: Path, out_dir: Path) -> Path:
    return out_dir / in_path.name


def _pick_cam_key(name: str) -> str | None:
    if name.startswith("out13"):
        return "out13"
    if name.startswith("out4"):
        return "out4"
    return None


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input dir not found: {INPUT_DIR}")
    for key, path in CALIB_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found for {key}: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(INPUT_DIR.glob(IMAGE_GLOB))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {INPUT_DIR} with glob {IMAGE_GLOB}")

    # Build maps per camera key and image size

    for idx, img_path in enumerate(image_paths, start=1):
        cam_key = _pick_cam_key(img_path.name)
        if cam_key is None:
            print(f"Skip (unknown camera prefix): {img_path.name}")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skip unreadable: {img_path}")
            continue

        height, width = img.shape[:2]
        
        mtx, dist = load_calibration(CALIB_PATHS[cam_key])
        
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        pts = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
        pts = pts.reshape(-1, 1, 2)

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 0.25, (width, height))
        rectified = cv2.undistort(img, mtx, dist, None, newcameramtx)
        out_path = rectified_output_path(img_path, OUTPUT_DIR)
        cv2.imwrite(str(out_path), rectified)

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(image_paths)}")

    print(f"Done. Rectified images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
