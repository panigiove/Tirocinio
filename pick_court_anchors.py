import argparse
import json
import os
import sys

import cv2 as cv


def parse_args():
    p = argparse.ArgumentParser(description="Pick court anchor points and save JSON.")
    p.add_argument("--video", required=True, help="Path to video file")
    p.add_argument("--frame", type=int, default=0, help="Frame index to use")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--length", type=float, default=40.0, help="Court length in meters")
    p.add_argument("--width", type=float, default=20.0, help="Court width in meters")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.video):
        print(f"Video not found: {args.video}")
        return 1

    cap = cv.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Failed to open: {args.video}")
        return 1

    if args.frame > 0:
        cap.set(cv.CAP_PROP_POS_FRAMES, args.frame)

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("Failed to read frame")
        return 1

    points = []
    order = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]

    def redraw(img):
        vis = img.copy()
        for idx, (x, y) in enumerate(points):
            cv.circle(vis, (x, y), 5, (0, 0, 255), -1)
            cv.putText(vis, f"{idx+1}:{order[idx]}", (x + 8, y - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return vis

    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))

    window = "Pick 4 court corners"
    cv.namedWindow(window, cv.WINDOW_NORMAL)
    cv.setMouseCallback(window, on_mouse)

    print("Click points in this exact order:")
    for i, name in enumerate(order, 1):
        print(f"  {i}. {name}")
    print("Keys: [u] undo, [r] reset, [s] save, [q] quit")

    while True:
        vis = redraw(frame)
        cv.imshow(window, vis)
        key = cv.waitKey(10) & 0xFF
        if key == ord('u') and points:
            points.pop()
        elif key == ord('r'):
            points.clear()
        elif key == ord('s'):
            if len(points) < 4:
                print("Need 4 points before saving.")
                continue
            pixel_points = [[int(x), int(y)] for x, y in points]
            L = float(args.length)
            W = float(args.width)
            world_points = [
                [0.0, 0.0],    # top-left
                [L, 0.0],      # top-right
                [L, W],        # bottom-right
                [0.0, W],      # bottom-left
            ]
            payload = {
                "pixel_points": pixel_points,
                "world_points": world_points,
                "meta": {
                    "length": L,
                    "width": W,
                    "order": order
                }
            }
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved: {args.out}")
            break
        elif key == ord('q'):
            break

    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
