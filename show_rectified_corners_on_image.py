import json
from pathlib import Path

import cv2
import numpy as np

# Inputs
RECTIFIED_POINTS_JSON = Path("cam4_img_corners_rectified.json")
IMAGE_PATH = Path(
    r"tracking_12.v4i.yolov11/train/images_rectified/out4_frame_0002_png.rf.13ad8a164866d2d0e8151ff7a71f7908.jpg"
)

SHOW_WINDOW = True
WINDOW_NAME = "rectified img_corners"
WINDOW_MAX_WIDTH = 1400
WINDOW_MAX_HEIGHT = 810
WINDOW_OFFSET_X = 200
WINDOW_OFFSET_Y = 100


def load_rectified_points(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    points = data.get("rectified_img_corners_indexed", [])
    if not points:
        raise ValueError("rectified_img_corners_indexed is missing or empty")
    return points


def draw_points(img: np.ndarray, points: list[dict]) -> np.ndarray:
    for item in points:
        idx = item.get("index")
        pt = item.get("point")
        if pt is None or len(pt) != 2:
            continue
        u, v = pt
        if np.isnan(u) or np.isnan(v):
            continue
        u_i, v_i = int(round(u)), int(round(v))
        cv2.circle(img, (u_i, v_i), 6, (0, 255, 255), -1)
        cv2.putText(
            img,
            str(idx),
            (u_i + 8, v_i - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return img


def main() -> None:
    if not RECTIFIED_POINTS_JSON.exists():
        raise FileNotFoundError(f"Points JSON not found: {RECTIFIED_POINTS_JSON}")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    points = load_rectified_points(RECTIFIED_POINTS_JSON)
    img = cv2.imread(str(IMAGE_PATH))
    if img is None:
        raise RuntimeError(f"Failed to read image: {IMAGE_PATH}")

    out = draw_points(img, points)
    if SHOW_WINDOW:
        h, w = out.shape[:2]
        scale = min(WINDOW_MAX_WIDTH / w, WINDOW_MAX_HEIGHT / h, 1.0)
        disp = out if scale >= 1.0 else cv2.resize(
            out, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
        cv2.imshow(WINDOW_NAME, disp)
        cv2.moveWindow(WINDOW_NAME, WINDOW_OFFSET_X, WINDOW_OFFSET_Y)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
