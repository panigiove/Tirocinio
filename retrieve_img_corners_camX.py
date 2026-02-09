import json
from pathlib import Path

import cv2
import numpy as np

VIDEO_PATH = Path(r"Tracking/material4project/video/tracking_12/out13.mp4")
JSON_PATH = Path(
    r"Tracking/material4project/3D Tracking Material/camera_data_with_Rvecs/camera_data/cam_13/calib/img_points.json"
)

# If you only want a single frame overlay, set FRAME_INDEX to a non-negative index.
# Set to None to process the entire video.
FRAME_INDEX = None

# Optional outputs
OUTPUT_IMAGE_PATH = Path("cam13_img_corners_overlay.png")
OUTPUT_VIDEO_PATH = Path("cam13_img_corners_overlay.mp4")

# Display frames live
SHOW_WINDOW = True


def main() -> None:
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"img_points.json not found: {JSON_PATH}")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    with JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    img_corners = data.get("img_corners", [])
    if not img_corners:
        raise ValueError("img_corners is empty in img_points.json")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    if FRAME_INDEX is not None and FRAME_INDEX >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if OUTPUT_VIDEO_PATH:
        OUTPUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (width, height))

    saved_image = False
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # Draw corners
        for idx, (u, v) in enumerate(img_corners):
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

        if writer is not None:
            writer.write(frame)

        if SHOW_WINDOW:
            cv2.imshow("cam13 img_corners", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        frame_idx += 1
        if FRAME_INDEX is not None and frame_idx > 0:
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved overlay video to {OUTPUT_VIDEO_PATH}")
    if SHOW_WINDOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
