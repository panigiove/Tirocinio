import json
from pathlib import Path

import cv2
import numpy as np


def resolve_calibration_path(cam_index: str) -> Path:
    """Resolve calibration path used by rectified_videos.py."""
    base = Path(r"Tracking/material4project/3D Tracking Material/camera_data_with_Rvecs_2ndversion")
    return base / f"cam_{cam_index}" / "calib" / "camera_calib.json"


def resolve_img_points_path(cam_index: str) -> Path:
    """Resolve img_points path from the same camera-data root."""
    base = Path(r"Tracking/material4project/3D Tracking Material/camera_data_with_Rvecs_2ndversion")
    return base / f"cam_{cam_index}" / "calib" / "img_points.json"


# Paths
RECTIFIED_VIDEO_PATH = Path(
    r"Tracking/material4project/Rectified videos/tracking_12/out13.mp4"
)
CAM_INDEX = "13"
CALIB_PATH = resolve_calibration_path(CAM_INDEX)
IMG_POINTS_PATH = resolve_img_points_path(CAM_INDEX)

# If you only want a single frame overlay, set FRAME_INDEX to a non-negative index.
# Set to None to process the entire video.
FRAME_INDEX = None

# Optional outputs
OUTPUT_IMAGE_PATH = Path("cam13_img_corners_rectified_overlay.png")
OUTPUT_POINTS_PATH = Path("cam13_img_corners_rectified.json")

# Display frames live
SHOW_WINDOW = True


def load_calibration(calib_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with calib_path.open("r", encoding="utf-8") as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist


def load_img_corners(img_points_path: Path) -> np.ndarray:
    with img_points_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    img_corners = data.get("img_corners", [])
    if not img_corners:
        raise ValueError("img_corners is empty in img_points.json")
    return np.array(img_corners, dtype=np.float32).reshape(-1, 2)


def remap_points_with_map(
    img_corners: np.ndarray,
    mtx: np.ndarray,
    dist: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    # Use the same remap mapping as rectified_videos.py
    rectified = []
    for (u, v) in img_corners:
        u_i, v_i = int(round(u)), int(round(v))
        if u_i < 0 or u_i >= width or v_i < 0 or v_i >= height:
            rectified.append([np.nan, np.nan])
            continue

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (u_i, v_i), 1, 255, -1)
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 0, (width, height))

        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        pts = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
        pts = pts.reshape(-1, 1, 2)
        
        remapped = cv2.undistort(mask, mtx, dist, None, newcameramtx)
        ys, xs = np.where(remapped > 0)
        if xs.size == 0:
            rectified.append([np.nan, np.nan])
        else:
            rectified.append([float(xs.mean()), float(ys.mean())])

    return np.array(rectified, dtype=np.float32)


def main() -> None:
    if not RECTIFIED_VIDEO_PATH.exists():
        raise FileNotFoundError(f"Rectified video not found: {RECTIFIED_VIDEO_PATH}")
    if not CALIB_PATH.exists():
        raise FileNotFoundError(f"Calibration file not found: {CALIB_PATH}")
    if not IMG_POINTS_PATH.exists():
        raise FileNotFoundError(f"img_points.json not found: {IMG_POINTS_PATH}")

    mtx, dist = load_calibration(CALIB_PATH)
    img_corners = load_img_corners(IMG_POINTS_PATH)

    cap = cv2.VideoCapture(str(RECTIFIED_VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {RECTIFIED_VIDEO_PATH}")

    if FRAME_INDEX is not None and FRAME_INDEX >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rectified_corners = remap_points_with_map(img_corners, mtx, dist, width, height)
    if OUTPUT_POINTS_PATH:
        OUTPUT_POINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rectified_img_corners_indexed": [
                {"index": idx, "point": [float(u), float(v)]}
                for idx, (u, v) in enumerate(rectified_corners)
            ],
        }
        with OUTPUT_POINTS_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved rectified points to {OUTPUT_POINTS_PATH}")

    saved_image = False
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # Draw rectified corners
        for idx, (u, v) in enumerate(rectified_corners):
            if np.isnan(u) or np.isnan(v):
                continue
            u_i, v_i = int(round(u)), int(round(v))
            cv2.circle(frame, (u_i, v_i), 6, (0, 255, 255), -1)
            cv2.putText(
                frame,
                str(idx),
                (u_i + 8, v_i - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if OUTPUT_IMAGE_PATH and not saved_image:
            OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(OUTPUT_IMAGE_PATH), frame)
            print(f"Saved overlay image to {OUTPUT_IMAGE_PATH}")
            saved_image = True

        if SHOW_WINDOW:
            cv2.imshow("cam13 img_corners (rectified)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        frame_idx += 1
        if FRAME_INDEX is not None and frame_idx > 0:
            break

    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
