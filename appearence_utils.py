# appearance_utils.py
# Appearance and pose utilities tuned for team-sport scenarios

import cv2 as cv
import numpy as np


def compute_team_appearance(frame, bbox, h_bins=32, s_bins=12, upper_fraction=0.6):
    """
    Extract an appearance descriptor robust to team sports:
    - HSV histogram (H,S) from upper body (jersey area)
    - Mean LAB color
    - L2-normalized descriptor
    """
    if bbox is None:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w - 1, x2); y2 = min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    height = y2 - y1
    top_h = max(1, int(height * upper_fraction))
    crop = frame[y1:y1 + top_h, x1:x2]
    if crop.size == 0:
        return None

    # HSV histogram (jersey color)
    hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    cv.normalize(hist, hist)
    hist = hist.flatten().astype(np.float32)

    # LAB mean color (illumination-robust)
    lab = cv.cvtColor(crop, cv.COLOR_BGR2LAB)
    mean_lab = np.array(cv.mean(lab)[:3], dtype=np.float32) / 255.0

    desc = np.concatenate([hist, mean_lab])
    norm = np.linalg.norm(desc)
    if norm > 0:
        desc /= norm
    return desc


def keypoints_to_pose_vec(keypoints, bbox):
    """
    Convert absolute keypoints to a normalized pose vector relative to bbox.
    Used for pose-based matching inside the tracker.
    """
    if keypoints is None or bbox is None:
        return None

    kpts = np.array(keypoints, dtype=np.float32)
    if kpts.size == 0:
        return None

    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    norm = np.zeros_like(kpts)
    norm[:, 0] = (kpts[:, 0] - x1) / w
    norm[:, 1] = (kpts[:, 1] - y1) / h

    return norm.flatten()

