import argparse
import os
from typing import List, Tuple, Optional

import cv2 as cv
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Extract court lines and visualize reconstruction.")
    p.add_argument("--video1", required=True, help="Path to view1 video")
    p.add_argument("--video2", required=True, help="Path to view2 video")
    p.add_argument("--frames1", default="15,350", help="Comma-separated frame ids for view1")
    p.add_argument("--frames2", default="236", help="Comma-separated frame ids for view2")
    p.add_argument("--size", default="1440x810", help="Resize frames to this size (WxH)")
    p.add_argument("--use_line_mask", action="store_true", help="Use white-line mask before edges")
    p.add_argument("--canny1", type=int, default=50, help="Canny low threshold")
    p.add_argument("--canny2", type=int, default=150, help="Canny high threshold")
    p.add_argument("--hough_thresh", type=int, default=80, help="HoughLinesP threshold")
    p.add_argument("--min_line_length", type=int, default=100, help="HoughLinesP min line length")
    p.add_argument("--max_line_gap", type=int, default=20, help="HoughLinesP max line gap")
    return p.parse_args()


def _parse_frames(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_size(s: str) -> Tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def _court_line_mask(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (0, 0, 160), (180, 60, 255))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=1)
    return mask


def _median_frame(frames):
    if not frames:
        return None
    stack = np.stack(frames, axis=0).astype(np.float32)
    med = np.median(stack, axis=0)
    return med.astype(np.uint8)


def _read_frames(video_path: str, frame_ids: List[int], size: Tuple[int, int]):
    if not os.path.exists(video_path):
        return []
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frames = []
    for fid in frame_ids:
        cap.set(cv.CAP_PROP_POS_FRAMES, int(fid))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if size is not None:
            frame = cv.resize(frame, size)
        frames.append(frame)
    cap.release()
    return frames


def _detect_boundary_lines(frame, use_line_mask, canny1, canny2, hough_thresh, min_line_length, max_line_gap):
    img = frame.copy()
    if use_line_mask:
        mask = _court_line_mask(img)
        img = cv.bitwise_and(img, img, mask=mask)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, canny1, canny2, apertureSize=3)
    lines = cv.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=hough_thresh,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    if lines is None:
        return None, None

    horizontals = []
    verticals = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle < 20 or angle > 160:
            y = (y1 + y2) / 2.0
            horizontals.append((y, (x1, y1, x2, y2)))
        elif 70 < angle < 110:
            x = (x1 + x2) / 2.0
            verticals.append((x, (x1, y1, x2, y2)))

    if len(horizontals) < 2 or len(verticals) < 2:
        return lines, None

    top = min(horizontals, key=lambda t: t[0])[1]
    bottom = max(horizontals, key=lambda t: t[0])[1]
    left = min(verticals, key=lambda t: t[0])[1]
    right = max(verticals, key=lambda t: t[0])[1]

    def line_from_points(x1, y1, x2, y2):
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        return a, b, c

    def intersect(l1, l2):
        a1, b1, c1 = l1
        a2, b2, c2 = l2
        d = a1 * b2 - a2 * b1
        if abs(d) < 1e-6:
            return None
        x = (b1 * c2 - b2 * c1) / d
        y = (c1 * a2 - c2 * a1) / d
        return [x, y]

    L_top = line_from_points(*top)
    L_bottom = line_from_points(*bottom)
    L_left = line_from_points(*left)
    L_right = line_from_points(*right)

    tl = intersect(L_top, L_left)
    tr = intersect(L_top, L_right)
    br = intersect(L_bottom, L_right)
    bl = intersect(L_bottom, L_left)
    if tl is None or tr is None or br is None or bl is None:
        return lines, None

    return lines, [tl, tr, br, bl]


def _draw_reconstruction(frame, lines, corners):
    vis = frame.copy()
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
    if corners is not None:
        pts = np.array(corners, dtype=np.int32)
        cv.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        for i, (x, y) in enumerate(pts):
            cv.circle(vis, (x, y), 5, (0, 0, 255), -1)
            cv.putText(vis, f"{i+1}", (x + 6, y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis


def main():
    args = parse_args()
    size = _parse_size(args.size)
    frames1 = _parse_frames(args.frames1)
    frames2 = _parse_frames(args.frames2)

    f1s = _read_frames(args.video1, frames1, size)
    f2s = _read_frames(args.video2, frames2, size)
    img1 = _median_frame(f1s)
    img2 = _median_frame(f2s)

    if img1 is None or img2 is None:
        print("Could not read frames for one or both views.")
        return 1

    lines1, corners1 = _detect_boundary_lines(
        img1, args.use_line_mask, args.canny1, args.canny2,
        args.hough_thresh, args.min_line_length, args.max_line_gap
    )
    lines2, corners2 = _detect_boundary_lines(
        img2, args.use_line_mask, args.canny1, args.canny2,
        args.hough_thresh, args.min_line_length, args.max_line_gap
    )

    vis1 = _draw_reconstruction(img1, lines1, corners1)
    vis2 = _draw_reconstruction(img2, lines2, corners2)

    cv.imshow("View1 Court Lines", vis1)
    cv.imshow("View2 Court Lines", vis2)
    print("Press any key to exit.")
    cv.waitKey(0)
    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
