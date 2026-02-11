import json
from pathlib import Path

import cv2
import numpy as np
import ctypes

# ===== CONFIG =====
H_PATH = Path(r"homography_cam4_to_cam13.json")

SRC_IMAGE_PATH = Path(
    r"tracking_12.v4i.yolov11/train/images_rectified/out4_frame_0002_png.rf.13ad8a164866d2d0e8151ff7a71f7908.jpg"
)
DST_IMAGE_PATH = Path(
    r"tracking_12.v4i.yolov11/train/images_rectified/out13_frame_0002_png.rf.68fc7ed57b749ab73105139ae9ec4f7e.jpg"
)

POINT_RADIUS = 8
POINT_THICKNESS = -1  # filled
SRC_NATIVE_COLOR = (0, 255, 255)      # yellow
DST_NATIVE_COLOR = (255, 255, 0)      # cyan
PROJECTED_COLOR = (0, 0, 255)         # red
TEXT_COLOR = (255, 255, 255)
TEXT_SCALE = 0.8
TEXT_THICKNESS = 2
TEXT_OFFSET = (10, -10)


def _screen_size() -> tuple[int, int]:
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def _fit_to_screen(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return img
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _load_image(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {img_path}")
    return img


def _label_path_from_image(img_path: Path) -> Path:
    # Rectified images use labels/<image_stem>_rectified.txt
    if img_path.parent.name == "images_rectified":
        return img_path.parent.parent / "labels" / f"{img_path.stem}_rectified.txt"
    if img_path.parent.name == "images":
        return img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    return img_path.with_suffix(".txt")


def _load_bottom_centers(label_path: Path, img_w: int, img_h: int) -> tuple[np.ndarray, list[str]]:
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    points: list[list[float]] = []
    ids: list[str] = []

    with label_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            ann_id = parts[0]
            x_c, y_c, _, h = map(float, parts[1:5])

            x_px = x_c * img_w
            y_bottom_px = (y_c + h / 2.0) * img_h

            points.append([x_px, y_bottom_px])
            ids.append(ann_id)

    if not points:
        return np.empty((0, 2), dtype=np.float32), []

    return np.array(points, dtype=np.float32), ids


def _draw_points(
    img: np.ndarray,
    pts: np.ndarray,
    ids: list[str],
    color: tuple[int, int, int],
    suffix: str,
) -> None:
    h, w = img.shape[:2]
    for (x, y), ann_id in zip(pts, ids):
        if not (0 <= x < w and 0 <= y < h):
            continue

        xi = int(round(x))
        yi = int(round(y))
        cv2.circle(img, (xi, yi), POINT_RADIUS, color, POINT_THICKNESS)
        cv2.putText(
            img,
            f"{ann_id}{suffix}",
            (xi + TEXT_OFFSET[0], yi + TEXT_OFFSET[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            TEXT_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )


def main() -> None:
    if not H_PATH.exists():
        raise FileNotFoundError(f"Homography file not found: {H_PATH}")
    if not SRC_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Source image not found: {SRC_IMAGE_PATH}")
    if not DST_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Destination image not found: {DST_IMAGE_PATH}")

    with H_PATH.open("r", encoding="utf-8") as f:
        H = np.array(json.load(f)["H"], dtype=np.float64)
    H_inv = np.linalg.inv(H)

    src_img = _load_image(SRC_IMAGE_PATH)
    dst_img = _load_image(DST_IMAGE_PATH)

    src_h, src_w = src_img.shape[:2]
    dst_h, dst_w = dst_img.shape[:2]

    src_label_path = _label_path_from_image(SRC_IMAGE_PATH)
    dst_label_path = _label_path_from_image(DST_IMAGE_PATH)

    src_pts, src_ids = _load_bottom_centers(src_label_path, src_w, src_h)
    dst_pts, dst_ids = _load_bottom_centers(dst_label_path, dst_w, dst_h)

    src_to_dst = src_pts
    if len(src_pts):
        src_to_dst = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)

    dst_to_src = dst_pts
    if len(dst_pts):
        dst_to_src = cv2.perspectiveTransform(dst_pts.reshape(-1, 1, 2), H_inv).reshape(-1, 2)

    # Native bottom-center points in each image.
    _draw_points(src_img, src_pts, src_ids, SRC_NATIVE_COLOR, "")
    _draw_points(dst_img, dst_pts, dst_ids, DST_NATIVE_COLOR, "")

    # Cross-view projected bottom-centers from the other image (marked with ').
    _draw_points(dst_img, src_to_dst, src_ids, PROJECTED_COLOR, "'")
    _draw_points(src_img, dst_to_src, dst_ids, PROJECTED_COLOR, "'")

    screen_w, screen_h = _screen_size()
    max_w = int(screen_w * 0.9)
    max_h = int(screen_h * 0.9)

    src_disp = _fit_to_screen(src_img, max_w, max_h)
    dst_disp = _fit_to_screen(dst_img, max_w, max_h)

    cv2.namedWindow("cam4 rectified - annotations + projections", cv2.WINDOW_NORMAL)
    cv2.namedWindow("cam13 rectified - annotations + projections", cv2.WINDOW_NORMAL)
    cv2.imshow("cam4 rectified - annotations + projections", src_disp)
    cv2.imshow("cam13 rectified - annotations + projections", dst_disp)

    print(f"Loaded src labels: {src_label_path}")
    print(f"Loaded dst labels: {dst_label_path}")
    print("Legend: yellow/cyan = native bottom centers, red + apostrophe = projected bottom centers from the other view.")
    print("Press any key in an image window to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()