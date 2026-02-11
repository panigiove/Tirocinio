import cv2 as cv
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
import csv
from scipy.optimize import linear_sum_assignment

import json
import os
import re

from iou_tracker import IOUTracker, GlobalIDManager
from appearence_utils import compute_team_appearence, keypoints_to_pose_vec

# ================= ROI / MASK =================
# Mask for View 1
TRAPEZ_TOP_LEFT_1 = (200, 340)
TRAPEZ_TOP_RIGHT_1 = (1240, 340)
TRAPEZ_BOTTOM_LEFT_1 = (0, 665)
TRAPEZ_BOTTOM_RIGHT_1 = (1440, 635)
CURVE_HEIGHT_1 = 70

# Mask for View 2
TRAPEZ_TOP_LEFT_2 = (0, 0)
TRAPEZ_TOP_RIGHT_2 = (1440, 0)
TRAPEZ_BOTTOM_LEFT_2 = (0, 810)
TRAPEZ_BOTTOM_RIGHT_2 = (1440, 810)
CURVE_HEIGHT_2 = 0

# ================= UTILITIES =================

def round_to_multiple(value, multiple):
    return multiple * ((value + multiple - 1) // multiple)


def create_trapezoid_mask(frame_shape, tl, tr, bl, br, curve_height=0):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts = [np.array([tl, tr, br, bl], dtype=np.int32)]
    if curve_height > 0:
        curve = [tl, tr, br]
        for i in range(1, 20):
            t = i / 20
            x = br[0] + (bl[0] - br[0]) * t
            y = br[1] + (bl[1] - br[1]) * t + curve_height * (4 * t * (1 - t))
            curve.append((int(x), int(y)))
        curve.append(bl)
        pts = [np.array(curve, dtype=np.int32)]
    cv.fillPoly(mask, pts, 255)
    return mask, pts[0]


def apply_trapezoid_mask(frame, mask):
    return cv.bitwise_and(frame, frame, mask=mask)


def pad_bbox(bbox, pad_ratio, frame_shape):
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pw, ph = int(w * pad_ratio), int(h * pad_ratio)
    return (
        max(0, x1 - pw),
        max(0, y1 - ph),
        min(frame_shape[1] - 1, x2 + pw),
        min(frame_shape[0] - 1, y2 + ph)
    )


def process_detection_pose(frame, bbox, pose_model, pose_attempts, pose_conf_threshold, pose_iou_threshold):
    """
    Optimized single detection pose estimation with early exit.
    """
    if pose_model is None:
        return None, None
    
    pose_kpts, pose_vec = None, None
    
    # Try first attempt
    pb = pad_bbox(bbox, pose_attempts[0]['pad'], frame.shape)
    px1, py1, px2, py2 = pb
    crop = frame[py1:py2, px1:px2]
    if crop.size > 0:
        res = pose_model.predict(
            crop, 
            conf=pose_attempts[0].get('conf') or pose_conf_threshold,
            iou=pose_attempts[0].get('iou') or pose_iou_threshold,
            imgsz=round_to_multiple(max(crop.shape[:2]), 32),
            device='cuda',
            verbose=False
        )
        if res and len(res[0].keypoints.xy) > 0:
            k = res[0].keypoints.xy.cpu().numpy()[0]
            pose_kpts = k + np.array([px1, py1])
            pose_vec = keypoints_to_pose_vec(pose_kpts, bbox)
            return pose_kpts, pose_vec
    
    # Fallback to second attempt
    if len(pose_attempts) > 1:
        pb = pad_bbox(bbox, pose_attempts[1]['pad'], frame.shape)
        px1, py1, px2, py2 = pb
        crop = frame[py1:py2, px1:px2]
        if crop.size > 0:
            res = pose_model.predict(
                crop,
                conf=pose_attempts[1].get('conf') or pose_conf_threshold,
                iou=pose_attempts[1].get('iou') or pose_iou_threshold,
                imgsz=round_to_multiple(max(crop.shape[:2]), 32),
                device='cuda',
                verbose=False
            )
            if res and len(res[0].keypoints.xy) > 0:
                k = res[0].keypoints.xy.cpu().numpy()[0]
                pose_kpts = k + np.array([px1, py1])
                pose_vec = keypoints_to_pose_vec(pose_kpts, bbox)
    
    return pose_kpts, pose_vec


def draw_text_with_bg(img, text, pos, scale=0.6):
    font = cv.FONT_HERSHEY_SIMPLEX
    (tw, th), b = cv.getTextSize(text, font, scale, 2)
    x, y = pos
    cv.rectangle(img, (x, y - th - 10), (x + tw + 20, y + b + 10), (0, 0, 0), -1)
    cv.putText(img, text, (x + 10, y), font, scale, (255, 255, 255), 2)


def parse_yolo_annotations(anno_file, img_width, img_height, frame, pose_model, pose_attempts, 
                          pose_conf_threshold, pose_iou_threshold, tracker=None):
    """Parse YOLO format annotations from file."""
    detections = []
    if not os.path.exists(anno_file):
        return detections
    
    try:
        with open(anno_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                center_x, center_y, width, height = map(float, parts[1:])
                
                # Convert normalized YOLO format to pixel coordinates
                x1 = int((center_x - width / 2) * img_width)
                y1 = int((center_y - height / 2) * img_height)
                x2 = int((center_x + width / 2) * img_width)
                y2 = int((center_y + height / 2) * img_height)
                bbox = (x1, y1, x2, y2)

                appearance = compute_team_appearence(frame, bbox)
                pose_kpts, pose_vec = process_detection_pose(
                    frame, bbox, pose_model, pose_attempts, 
                    pose_conf_threshold, pose_iou_threshold
                )
                world_pos = tracker.project_to_world(bbox) if tracker else None

                detections.append({
                    'bbox': bbox,
                    'appearance': appearance,
                    'keypoints': pose_kpts,
                    'pose_vec': pose_vec,
                    'world_pos': world_pos,
                    'score': 1.0,
                    'global_id': class_id,
                    'gid_source': 'annotation'
                })
    except Exception as e:
        print(f"Error parsing annotations from {anno_file}: {e}")
    
    return detections


def load_annotations_from_dir(annotations_dir, video_prefix):
    """Load all annotation files from a directory."""
    annotations = {}
    if not annotations_dir or not os.path.exists(annotations_dir):
        return annotations
    
    for file in os.listdir(annotations_dir):
        if file.startswith(video_prefix + '_') and file.endswith('_rectified.txt'):
            match = re.search(r'frame_(\d+)', file)
            if match:
                frame_id = int(match.group(1))
                anno_path = os.path.join(annotations_dir, file)
                if frame_id not in annotations:
                    annotations[frame_id] = []
                annotations[frame_id].append(anno_path)
    
    return annotations


def _min_annotated_frame(*annotations_sets):
    frames = []
    for ann in annotations_sets:
        if ann:
            frames.extend(list(ann.keys()))
    return min(frames) if frames else None


def draw_tracks(img, tracks, max_tracks, connections, tracker_self):
    """Draw tracks, keypoints and cross-view projections on a frame."""
    valid_tracks = [t for t in tracks if t['missed'] <= tracker_self.max_missed]
    
    # Filter by active tracks only (missed == 0 and not deduced)
    valid_tracks_filtered = []
    for t in valid_tracks:
        if t['missed'] == 0 and not t.get('was_deduced'):
            valid_tracks_filtered.append(t)
    
    valid_tracks_filtered.sort(key=lambda x: x['global_id'])
    
    if max_tracks is not None:
        valid_tracks_filtered = valid_tracks_filtered[:max_tracks]
    
    # Draw tracks
    for t in valid_tracks_filtered:
        x1, y1, x2, y2 = map(int, t['bbox'])
        gid = t['global_id']
        col = (gid * 37 % 255, gid * 17 % 255, gid * 97 % 255)
        
        thickness = 1 if (t.get('was_deduced') or t['missed'] > 0) else 2
        cv.rectangle(img, (x1, y1), (x2, y2), col, thickness)
        
        # Label
        cx = (x1 + x2) // 2
        cy = y1
        label = f'GID:{gid}' + (' (D)' if t.get('was_deduced') else '')
        cv.putText(img, label, (cx, cy - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        
        # Keypoints
        if t.get('keypoints') is not None and t['missed'] == 0:
            kpts = t['keypoints'].astype(int)
            for a, b in connections:
                p1, p2 = kpts[a], kpts[b]
                if p1[0] > 0 and p2[0] > 0:
                    cv.line(img, tuple(p1), tuple(p2), col, thickness)
            for p in kpts:
                if p[0] > 0:
                    cv.circle(img, tuple(p), 3, col, -1)
                    cv.circle(img, tuple(p), 3, (255, 255, 255), 1)

    return img

def _augment_event_with_world_pos(event, tracker):
    """Attach world position (x,y,z) to an event if track has it."""
    if tracker is None:
        return event
    tid = event.get('track_id')
    if tid is None:
        return event
    track = next((t for t in tracker.tracks if t.get('track_id') == tid), None)
    wp = track.get('world_pos') if track else None
    if wp is not None and len(wp) >= 3:
        event['world_x'] = float(wp[0])
        event['world_y'] = float(wp[1])
        event['world_z'] = float(wp[2])
        event['world_pos_json'] = json.dumps([float(wp[0]), float(wp[1]), float(wp[2])])
        event['has_world_pos'] = 1
    else:
        event['world_x'] = ''
        event['world_y'] = ''
        event['world_z'] = ''
        event['world_pos_json'] = ''
        event['has_world_pos'] = 0
    return event

# ================= MAIN =================

def _bbox_area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_intersection(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _suppress_nested_detections(detections, contain_thresh=0.9, area_ratio_thresh=0.6):
    if len(detections) < 2:
        return detections
    keep = [True] * len(detections)
    areas = [_bbox_area(d['bbox']) for d in detections]
    for i in range(len(detections)):
        if not keep[i]:
            continue
        for j in range(len(detections)):
            if i == j or not keep[j]:
                continue
            if areas[i] <= 0 or areas[j] <= 0:
                continue
            # Consider j nested in i
            if areas[j] >= areas[i]:
                continue
            inter = _bbox_intersection(detections[i]['bbox'], detections[j]['bbox'])
            contain = inter / areas[j] if areas[j] > 0 else 0.0
            area_ratio = areas[j] / areas[i]
            if contain >= contain_thresh and area_ratio <= area_ratio_thresh:
                # Prefer higher score; keep annotations
                if detections[j].get('gid_source') == 'annotation':
                    continue
                if detections[j].get('score', 0.0) <= detections[i].get('score', 0.0):
                    keep[j] = False
    return [d for d, k in zip(detections, keep) if k]

def load_calibration(calib_path):
    if not calib_path or not os.path.exists(calib_path):
        return None
    with open(calib_path, 'r') as f:
        return json.load(f)


def load_indexed_points(points_path):
    if not points_path or not os.path.exists(points_path):
        return {}
    with open(points_path, 'r') as f:
        data = json.load(f)
    out = {}
    for item in data.get('rectified_img_corners_indexed', []):
        idx = item.get('index')
        pt = item.get('point')
        if idx is None or pt is None or len(pt) != 2:
            continue
        out[int(idx)] = np.array([float(pt[0]), float(pt[1])], dtype=np.float32)
    return out


def load_court_lines(lines_path, points_by_idx):
    if not lines_path or not os.path.exists(lines_path):
        return []
    with open(lines_path, 'r') as f:
        data = json.load(f)
    lines = []
    for item in data.get('rectified_court_lines', []):
        line_id = item.get('line_id')
        si = item.get('start_point_index')
        ei = item.get('end_point_index')
        if line_id is None or si is None or ei is None:
            continue
        p1 = points_by_idx.get(int(si))
        p2 = points_by_idx.get(int(ei))
        if p1 is None or p2 is None:
            continue
        lines.append({
            'line_id': int(line_id),
            'p1': p1.copy(),
            'p2': p2.copy()
        })
    return lines


def compute_common_ref_points(points_by_idx1, points_by_idx2):
    """
    Compute reference points using only corners common to both views.
    Returns (ref1, ref2) in each view's pixel space (unscaled).
    """
    if not points_by_idx1 or not points_by_idx2:
        return None, None
    common_idx = sorted(set(points_by_idx1.keys()) & set(points_by_idx2.keys()))
    if not common_idx:
        return None, None
    pts1 = np.array([points_by_idx1[i] for i in common_idx], dtype=np.float32)
    pts2 = np.array([points_by_idx2[i] for i in common_idx], dtype=np.float32)
    if pts1.size == 0 or pts2.size == 0:
        return None, None
    return pts1.mean(axis=0), pts2.mean(axis=0)


def load_line_correspondences(path, view1_key, view2_key):
    if not path or not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        data = json.load(f)
    pairs = []
    for item in data.get('lines_corrispondency', []):
        if view1_key in item and view2_key in item:
            pairs.append((int(item[view1_key]), int(item[view2_key])))
    return pairs


def scale_lines(lines, sx, sy):
    if sx is None or sy is None:
        return lines
    out = []
    for l in lines:
        p1 = l['p1']
        p2 = l['p2']
        out.append({
            'line_id': l['line_id'],
            'p1': np.array([p1[0] * sx, p1[1] * sy], dtype=np.float32),
            'p2': np.array([p2[0] * sx, p2[1] * sy], dtype=np.float32),
        })
    return out


def draw_court_lines(img, lines, color=(0, 255, 0), thickness=2, label=True):
    if img is None or not lines:
        return img
    for l in lines:
        p1 = l['p1']
        p2 = l['p2']
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))
        cv.line(img, (x1, y1), (x2, y2), color, thickness)
        if label:
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            cv.putText(img, str(l['line_id']), (mx + 4, my - 4),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def _signed_point_line_distance(pt, line, ref_pt=None):
    """
    Signed distance from point to line. If ref_pt is provided, the sign is
    oriented so that ref_pt lies on the positive side for consistency.
    """
    p1 = line['p1']
    p2 = line['p2']
    v = p2 - p1
    denom = np.linalg.norm(v)
    if denom <= 1e-6:
        return 0.0
    n = np.array([-v[1], v[0]], dtype=np.float32)  # perpendicular
    d = float(np.dot((pt - p1), n) / denom)
    if ref_pt is not None:
        ref_side = float(np.dot((ref_pt - p1), n))
        if ref_side < 0:
            d = -d
    return d


def _line_signature(pt, lines, ref_pt=None):
    if pt is None or not lines:
        return None
    dists = np.array([_signed_point_line_distance(pt, l, ref_pt=ref_pt) for l in lines],
                     dtype=np.float32)
    norm = np.linalg.norm(dists)
    if norm <= 1e-6:
        return None
    return dists / norm


def _smooth_penalty(value, threshold, margin, min_scale=0.4):
    if value is None or threshold is None or margin is None or margin <= 0:
        return 1.0
    if value <= threshold:
        return 1.0
    if value >= threshold + margin:
        return min_scale
    t = (value - threshold) / margin
    return 1.0 - t * (1.0 - min_scale)

def detect_view(frame, frame_id, annotations, size, mask, det_model,
                sahi_conf_threshold, sahi_iou_threshold, slice_h, slice_w,
                slice_overlap, pose_model, pose_attempts, pose_conf_threshold,
                pose_iou_threshold, tracker,
                suppress_nested=True, nested_contain_thresh=0.9, nested_area_ratio=0.6):
    detections = []
    if frame_id in annotations:
        for anno_file in annotations[frame_id]:
            detections.extend(parse_yolo_annotations(
                anno_file, size[0], size[1], frame, pose_model, pose_attempts,
                pose_conf_threshold, pose_iou_threshold, tracker=tracker
            ))
    else:
        roi = apply_trapezoid_mask(frame, mask)
        mask_2d = np.any(roi > 0, axis=2)
        if np.any(mask_2d):
            coords = np.where(mask_2d)
            min_y, max_y = coords[0].min(), coords[0].max()
            min_x, max_x = coords[1].min(), coords[1].max()
            cropped = roi[min_y:max_y + 1, min_x:max_x + 1]

            preds = get_sliced_prediction(
                cropped, det_model, slice_height=slice_h, slice_width=slice_w,
                overlap_height_ratio=slice_overlap, overlap_width_ratio=slice_overlap,
                postprocess_match_metric='IOU', postprocess_match_threshold=sahi_iou_threshold,
                verbose=0
            )

            if preds and preds.object_prediction_list:
                for p in preds.object_prediction_list:
                    if p.category.id == 0 and p.score.value >= sahi_conf_threshold:
                        bx1, by1, bx2, by2 = map(int, p.bbox.to_xyxy())
                        bbox = (bx1 + min_x, by1 + min_y, bx2 + min_x, by2 + min_y)
                        appearance = compute_team_appearence(frame, bbox)
                        world_pos = tracker.project_to_world(bbox)
                        detections.append({
                            'bbox': bbox,
                            'appearance': appearance,
                            'world_pos': world_pos,
                            'score': p.score.value
                        })

    if suppress_nested:
        detections = _suppress_nested_detections(
            detections,
            contain_thresh=nested_contain_thresh,
            area_ratio_thresh=nested_area_ratio
        )

    for d in detections:
        if d.get('keypoints') is None:
            kpts, vec = process_detection_pose(
                frame, d['bbox'], pose_model, pose_attempts,
                pose_conf_threshold, pose_iou_threshold
            )
            d['keypoints'] = kpts
            d['pose_vec'] = vec
        if d.get('keypoints') is not None:
            kpts = d['keypoints']
            candidates = []
            for idx in (15, 16, 11, 12):
                if idx < len(kpts):
                    x, y = kpts[idx]
                    if x > 0 and y > 0:
                        candidates.append((x, y))
            if candidates:
                px, py = max(candidates, key=lambda p: p[1])
                d['world_pos'] = tracker.project_pixel_to_world(px, py)

    return detections


def merge_track_global_id(tracker, track, target_gid, frame_id):
    if track['global_id'] == target_gid:
        return True
    existing = tracker.global_id_to_track.get(target_gid)
    if existing is not None and existing is not track:
        return False
    tracker.merge_global_ids(track['global_id'], target_gid, frame_id)
    return True


def _bbox_bottom_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)


def _reproj_error(tracker, world_pos, bbox):
    if world_pos is None:
        return None
    px = tracker.project_to_pixel(world_pos)
    if px is None:
        return None
    return float(np.linalg.norm(px - _bbox_bottom_center(bbox)))


def _pose_anchor_pixel(track):
    kpts = track.get('keypoints')
    if kpts is None:
        return None
    candidates = []
    for idx in (15, 16, 11, 12, 13, 14):
        if idx < len(kpts):
            x, y = kpts[idx]
            if x > 0 and y > 0:
                candidates.append((x, y))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p[1])


def _track_anchor_pixel(track):
    p = _pose_anchor_pixel(track)
    if p is not None:
        return np.array(p, dtype=np.float32)
    bbox = track.get('bbox')
    if bbox is None:
        return None
    return _bbox_bottom_center(bbox)


def _triangulate_world_pos(tracker1, tracker2, p1, p2, max_reproj_error=50.0):
    if not hasattr(tracker1, 'P') or not hasattr(tracker2, 'P'):
        return None
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)

    if tracker1.is_rectified:
        p1_norm = cv.undistortPoints(p1.reshape(1, 1, 2), tracker1.mtx, None, P=np.eye(3))
        p2_norm = cv.undistortPoints(p2.reshape(1, 1, 2), tracker2.mtx, None, P=np.eye(3))
    else:
        p1_norm = cv.undistortPoints(p1.reshape(1, 1, 2), tracker1.mtx, tracker1.dist, P=np.eye(3))
        p2_norm = cv.undistortPoints(p2.reshape(1, 1, 2), tracker2.mtx, tracker2.dist, P=np.eye(3))

    p4d = cv.triangulatePoints(tracker1.P, tracker2.P,
                               p1_norm.reshape(2, 1),
                               p2_norm.reshape(2, 1))
    p3d = (p4d[:3] / p4d[3]).flatten()

    # Validate reprojection
    pts_3d = np.array([p3d], dtype=np.float32)
    img_pts1, _ = cv.projectPoints(pts_3d, tracker1.rvec, tracker1.tvec, tracker1.mtx, tracker1.dist)
    img_pts2, _ = cv.projectPoints(pts_3d, tracker2.rvec, tracker2.tvec, tracker2.mtx, tracker2.dist)
    err1 = np.linalg.norm(img_pts1[0][0] - p1)
    err2 = np.linalg.norm(img_pts2[0][0] - p2)
    if err1 > max_reproj_error or err2 > max_reproj_error:
        return None
    return p3d


class TrackletStore:
    def __init__(self):
        self.stats = {}

    def update(self, view_id, tracks):
        for t in tracks:
            if not t.get('updated'):
                continue
            key = (view_id, t['track_id'])
            s = self.stats.get(key)
            if s is None:
                s = {
                    'count': 0,
                    'app_sum': None,
                    'app_count': 0,
                    'pose_sum': None,
                    'pose_count': 0,
                    'world_sum': None,
                    'world_count': 0,
                    'anchor_gid': None
                }
                self.stats[key] = s
            s['count'] += 1

            if t.get('gid_source') == 'annotation' and s['anchor_gid'] is None:
                s['anchor_gid'] = t.get('global_id')

            app = t.get('appearance')
            if app is not None:
                if s['app_sum'] is None:
                    s['app_sum'] = app.astype(np.float32).copy()
                else:
                    s['app_sum'] += app
                s['app_count'] += 1

            pose = t.get('pose_vec')
            if pose is not None:
                if s['pose_sum'] is None:
                    s['pose_sum'] = pose.astype(np.float32).copy()
                else:
                    s['pose_sum'] += pose
                s['pose_count'] += 1

            wp = t.get('world_pos')
            if wp is not None and len(wp) >= 2:
                wp_arr = np.array([wp[0], wp[1], 0.0], dtype=np.float32)
                if s['world_sum'] is None:
                    s['world_sum'] = wp_arr.copy()
                else:
                    s['world_sum'] += wp_arr
                s['world_count'] += 1

    def get_features(self, view_id, track_id, min_len=5):
        s = self.stats.get((view_id, track_id))
        if not s:
            return None
        if s['anchor_gid'] is None and s['count'] < min_len:
            return None

        app = None
        if s['app_count'] > 0:
            app = s['app_sum'] / float(s['app_count'])
            norm = np.linalg.norm(app)
            if norm > 0:
                app = app / norm

        pose = None
        if s['pose_count'] > 0:
            pose = s['pose_sum'] / float(s['pose_count'])

        wp = None
        if s['world_count'] > 0:
            wp = s['world_sum'] / float(s['world_count'])

        return {
            'appearance': app,
            'pose_vec': pose,
            'world_pos': wp,
            'anchor_gid': s['anchor_gid']
        }

    def get_anchor_gid(self, view_id, track_id):
        s = self.stats.get((view_id, track_id))
        if not s:
            return None
        return s.get('anchor_gid')


def associate_tracks_to_view1(tracker1, tracker_other, frame_id,
                              match_threshold, app_weight=0.4, world_weight=0.6,
                              max_world_dist=None, world_dist_tolerance=0.0,
                              reproj_max_px=120.0, reproj_soft_margin=40.0,
                              line_sig_lines_anchor=None, line_sig_lines_other=None,
                              line_sig_ref_anchor=None, line_sig_ref_other=None,
                              line_sig_thresh=0.15,
                              line_sig_debug_frames=None,
                              prev_assoc=None, sticky_bonus=0.08,
                              conflict_log=None, conflict_view_id=None,
                              view_id_other=None,
                              tracklet_min_frames=5,
                              tracklet_store=None,
                              use_pose_triangulation=True,
                              pose_triangulation_max_reproj=50.0,
                              always_overwrite_gid=False,
                              anchor_view_id=1,):
    active1 = [t for t in tracker1.tracks if t['updated']]
    active_other = [t for t in tracker_other.tracks if t['updated']]

    if not active1 or not active_other:
        return

    pose_points1 = []
    pose_points2 = []
    if use_pose_triangulation:
        for t in active1:
            pose_points1.append(_pose_anchor_pixel(t))
        for t in active_other:
            pose_points2.append(_pose_anchor_pixel(t))

    w_norm = app_weight + world_weight
    if w_norm == 0:
        return
    app_weight /= w_norm
    world_weight /= w_norm

    line_sigs_1 = None
    line_sigs_2 = None
    if line_sig_lines_anchor and line_sig_lines_other:
        line_sigs_1 = []
        for t in active1:
            p1_px = _track_anchor_pixel(t)
            line_sigs_1.append(_line_signature(p1_px, line_sig_lines_anchor, ref_pt=line_sig_ref_anchor))
        line_sigs_2 = []
        for t in active_other:
            p2_px = _track_anchor_pixel(t)
            line_sigs_2.append(_line_signature(p2_px, line_sig_lines_other, ref_pt=line_sig_ref_other))

    cost = np.zeros((len(active1), len(active_other)), dtype=np.float32)
    for i, t1 in enumerate(active1):
        for j, t2 in enumerate(active_other):
            anchor_gid1 = tracklet_store.get_anchor_gid(anchor_view_id, t1['track_id']) if tracklet_store else None
            anchor_gid2 = tracklet_store.get_anchor_gid(view_id_other, t2['track_id']) if tracklet_store else None
            t1_annot = t1.get('gid_source') == 'annotation' or anchor_gid1 is not None
            t2_annot = t2.get('gid_source') == 'annotation' or anchor_gid2 is not None
            gid1 = anchor_gid1 if anchor_gid1 is not None else t1.get('global_id')
            gid2 = anchor_gid2 if anchor_gid2 is not None else t2.get('global_id')
            if t1_annot and t2_annot and gid1 != gid2:
                if conflict_log is not None:
                    conflict_log.append({
                        'frame': frame_id,
                        'view': conflict_view_id,
                        'view1_track_id': t1['track_id'],
                        'view1_global_id': gid1,
                        'other_track_id': t2['track_id'],
                        'other_global_id': gid2,
                        'reason': 'annotated_gid_mismatch'
                    })
                # Hard-force annotations: anchor view wins even if mismatched
                cost[i, j] = 0.0
                continue
            elif t1_annot and t2_annot and gid1 == gid2:
                cost[i, j] = 0.0
                continue
            app1 = t1.get('appearance')
            app2 = t2.get('appearance')
            pose1 = t1.get('pose_vec')
            pose2 = t2.get('pose_vec')
            wp1 = t1.get('world_pos')
            wp2 = t2.get('world_pos')

            if tracklet_store is not None:
                f1 = tracklet_store.get_features(anchor_view_id, t1['track_id'], tracklet_min_frames)
                if f1 is not None:
                    if f1.get('appearance') is not None:
                        app1 = f1['appearance']
                    if f1.get('pose_vec') is not None:
                        pose1 = f1['pose_vec']
                    if f1.get('world_pos') is not None:
                        wp1 = f1['world_pos']
                f2 = tracklet_store.get_features(view_id_other, t2['track_id'], tracklet_min_frames)
                if f2 is not None:
                    if f2.get('appearance') is not None:
                        app2 = f2['appearance']
                    if f2.get('pose_vec') is not None:
                        pose2 = f2['pose_vec']
                    if f2.get('world_pos') is not None:
                        wp2 = f2['world_pos']

            app_sim = tracker1._appearance_sim(app1, app2)

            if use_pose_triangulation:
                p1 = pose_points1[i]
                p2 = pose_points2[j]
                if p1 is not None and p2 is not None:
                    tri = _triangulate_world_pos(
                        tracker1, tracker_other, p1, p2,
                        max_reproj_error=pose_triangulation_max_reproj
                    )
                    if tri is not None:
                        wp1 = tri
                        wp2 = tri

            world_sim = tracker1._world_pos_sim(wp1, wp2)
            
            sim = app_weight * app_sim + world_weight * world_sim

            if line_sigs_1 is not None and line_sigs_2 is not None:
                sig1 = line_sigs_1[i]
                sig2 = line_sigs_2[j]
                if sig1 is not None and sig2 is not None:
                    sig_dist = float(np.linalg.norm(sig1 - sig2))
                    if line_sig_debug_frames is not None and frame_id in line_sig_debug_frames:
                        print(
                            f"[line_sig] frame={frame_id} view_other={view_id_other} "
                            f"t1={t1.get('track_id')} t2={t2.get('track_id')} "
                            f"gid1={t1.get('global_id')} gid2={t2.get('global_id')} "
                            f"dist={sig_dist:.4f} thresh={line_sig_thresh}"
                        )
                    if sig_dist > line_sig_thresh:
                        cost[i, j] = 1.0 - sim + 1.0
                        continue

            if wp1 is not None and wp2 is not None:
                dist = np.linalg.norm(np.array(wp1[:2]) - np.array(wp2[:2]))
                if max_world_dist is not None and dist > (max_world_dist + world_dist_tolerance):
                    sim *= _smooth_penalty(
                        dist, max_world_dist + world_dist_tolerance,
                        max(1.0, world_dist_tolerance), min_scale=0.4
                    )

                if reproj_max_px is not None:
                    err_1 = _reproj_error(tracker1, wp2, t1['bbox'])
                    err_2 = _reproj_error(tracker_other, wp1, t2['bbox'])
                    worst_err = max(e for e in (err_1, err_2) if e is not None) if (err_1 or err_2) else None
                    if worst_err is not None:
                        if worst_err > (reproj_max_px - reproj_soft_margin):
                            sim *= _smooth_penalty(
                                worst_err, reproj_max_px - reproj_soft_margin,
                                reproj_soft_margin, min_scale=0.4
                            )

            if prev_assoc is not None:
                prev_gid = prev_assoc.get(t2['track_id'])
                if prev_gid is not None and prev_gid == t1['global_id']:
                    sim = min(1.0, sim + sticky_bonus)
                if t2_annot and gid1 == gid2:
                    sim = min(1.0, sim + 0.2)

            cost[i, j] = 1.0 - sim

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_other_ids = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= (1.0 - match_threshold):
            t1 = active1[r]
            t2 = active_other[c]
            sim = 1.0 - cost[r, c]
            # Decide target GID
            anchor_gid1 = tracklet_store.get_anchor_gid(anchor_view_id, t1['track_id']) if tracklet_store else None
            anchor_gid2 = tracklet_store.get_anchor_gid(view_id_other, t2['track_id']) if tracklet_store else None
            t1_annot = t1.get('gid_source') == 'annotation' or anchor_gid1 is not None
            t2_annot = t2.get('gid_source') == 'annotation' or anchor_gid2 is not None
            if t2_annot and not t1_annot:
                target_gid = anchor_gid2 if anchor_gid2 is not None else t2['global_id']
            elif t1_annot and not t2_annot:
                target_gid = anchor_gid1 if anchor_gid1 is not None else t1['global_id']
            elif t1_annot and t2_annot:
                target_gid = anchor_gid1 if anchor_gid1 is not None else t1['global_id']
            else:
                if always_overwrite_gid:
                    s1 = t1.get('start_frame', frame_id)
                    s2 = t2.get('start_frame', frame_id)
                    if s2 < s1:
                        target_gid = t2['global_id']
                    else:
                        target_gid = t1['global_id']
                else:
                    target_gid = t1['global_id']

            merge_track_global_id(tracker1, t1, target_gid, frame_id)
            merge_track_global_id(tracker_other, t2, target_gid, frame_id)
            if t2_annot and not t1_annot:
                t1['gid_source'] = 'annotation'
            matched_other_ids.add(t2['track_id'])
            if prev_assoc is not None:
                prev_assoc[t2['track_id']] = t2['global_id'] if t2.get('gid_source') == 'annotation' else t1['global_id']

    if prev_assoc is not None:
        active_other_ids = {t['track_id'] for t in active_other}
        for tid in list(prev_assoc.keys()):
            if tid not in active_other_ids:
                del prev_assoc[tid]


def yolo_sahi_pose_tracking(
    source1,
    source2,
    calib_path1=None,
    calib_path2=None,
    annotations_dir1=None,
    annotations_dir2=None,
    output_path1='output_view1.mp4',
    output_path2='output_view2.mp4',
    output_csv_path1='track_events_view1.csv',
    output_csv_path2='track_events_view2.csv',
    start_frame=0,
    auto_start_from_annotations=True,
    enable_pose=True,
    size=(1440, 810),
    # detection View 1
    sahi_conf_threshold1=0.45,
    sahi_iou_threshold1=0.40,
    slice_h1=480,
    slice_w1=480,
    slice_overlap1=0.3,
    # pose View 1
    pose_conf_threshold1=0.09,
    pose_iou_threshold1=0.02,
    pose_attempts1=(
        {'pad': 0.0, 'conf': None, 'iou': None},
        {'pad': 0.25, 'conf': 0.03, 'iou': 0.005},
    ),
    # tracker View 1
    match_threshold1=0.45,
    iou_weight1=0.50,
    appearance_weight1=0.35,
    pose_weight1=0.0,
    ema_alpha1=0.5,
    max_velocity1=800.0,
    suppress_nested_tracks1=True,
    nested_track_contain_thresh1=0.9,
    nested_track_area_ratio1=0.6,
    nested_track_app_sim1=0.7,
    cross_view_match_threshold1=0.15,
    world_dist_weight1=0.5,
    world_dist_tolerance1=150.0,
    reproj_max_px1=120.0,
    reproj_soft_margin1=40.0,
    max_missed_frames1=80,
    # detection View 2
    sahi_conf_threshold2=0.7,
    sahi_iou_threshold2=0.55,
    slice_h2=640,
    slice_w2=640,
    slice_overlap2=0.05,
    # pose View 2
    pose_conf_threshold2=0.15,
    pose_iou_threshold2=0.02,
    pose_attempts2=(
        {'pad': 0.0, 'conf': None, 'iou': None},
        {'pad': 0.25, 'conf': 0.01, 'iou': 0.005},
    ),
    # tracker View 2
    match_threshold2=0.35,
    iou_weight2=0.45,
    appearance_weight2=0.35,
    pose_weight2=0.0,
    ema_alpha2=0.5,
    max_velocity2=800.0,
    suppress_nested_tracks2=True,
    nested_track_contain_thresh2=0.9,
    nested_track_area_ratio2=0.6,
    nested_track_app_sim2=0.7,
    cross_view_match_threshold2=0.15,
    world_dist_weight2=0.5,
    world_dist_tolerance2=150.0,
    reproj_max_px2=120.0,
    reproj_soft_margin2=40.0,
    max_missed_frames2=300,
    # tracklet-based reid
    tracklet_min_frames=5,
    use_pose_triangulation=True,
    pose_triangulation_max_reproj=50.0,
    suppress_nested_detections=True,
    nested_contain_thresh=0.9,
    nested_area_ratio=0.6,
    always_overwrite_gid=True,
    # line-based alignment + debug
    court_lines_path1=None,
    court_lines_path2=None,
    court_corners_path1=None,
    court_corners_path2=None,
    court_line_correspondence_path=None,
    court_line_view1_key='line_id_cam13',
    court_line_view2_key='line_id_cam4',
    draw_court_lines_debug=True,
    line_sig_debug_frames=(12,),
    # logging
    conflict_log_path='annotation_conflicts.csv'
):
    """
    Multi-view tracking with single-view tracking + cross-view association.

    Flow:
    - Detect per-view and estimate pose
    - Track per-view independently
    - Associate tracks across views using appearance + 3D position
    """
    print("Starting FIXED yolo_sahi_pose_tracking")
    
    # Skeleton connections
    connections = [
        (0,1),(0,2),(1,3),(2,4),
        (5,6),(5,7),(6,8),(7,9),(8,10),
        (5,11),(6,12),(11,12),
        (11,13),(12,14),(13,15),(14,16)
    ]

    print("Loading models...")
    det_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path='yolo11x.pt',
        confidence_threshold=min(sahi_conf_threshold1, sahi_conf_threshold2),
        image_size=round_to_multiple(640, 32),
        device='cuda:0'
    )
    pose_model = YOLO('yolo11x-pose.pt') if enable_pose else None
    print("Models loaded successfully")

    # Load calibrations
    calib1 = load_calibration(calib_path1)
    calib2 = load_calibration(calib_path2)

    video_prefix1 = os.path.basename(source1).split('.')[0] if source1 else None
    video_prefix2 = os.path.basename(source2).split('.')[0] if source2 else None

    shared_id_manager = GlobalIDManager()

    # FIX: Create trackers with unified max_missed_frames and is_rectified=True
    tracker1 = IOUTracker(
        match_threshold=match_threshold1,
        max_missed_frames=max_missed_frames1,  # Unified
        iou_weight=iou_weight1,
        appearance_weight=appearance_weight1,
        pose_weight=pose_weight1,
        ema_alpha=ema_alpha1,
        max_velocity=max_velocity1,
        camera_params=calib1,
        global_id_manager=shared_id_manager,
        cross_view_match_threshold=cross_view_match_threshold1,
        world_dist_weight=world_dist_weight1,
        world_dist_tolerance=world_dist_tolerance1,
        is_rectified=True,  # FIX: Videos are rectified
        suppress_nested_tracks=suppress_nested_tracks1,
        nested_contain_thresh=nested_track_contain_thresh1,
        nested_area_ratio=nested_track_area_ratio1,
        nested_app_sim_thresh=nested_track_app_sim1
    )
    
    tracker2 = IOUTracker(
        match_threshold=match_threshold2,
        max_missed_frames=max_missed_frames2,  # Unified
        iou_weight=iou_weight2,
        appearance_weight=appearance_weight2,
        pose_weight=pose_weight2,
        ema_alpha=ema_alpha2,
        max_velocity=max_velocity2,
        camera_params=calib2,
        global_id_manager=shared_id_manager,
        cross_view_match_threshold=cross_view_match_threshold2,
        world_dist_weight=world_dist_weight2,
        world_dist_tolerance=world_dist_tolerance2,
        is_rectified=True,  # FIX: Videos are rectified
        suppress_nested_tracks=suppress_nested_tracks2,
        nested_contain_thresh=nested_track_contain_thresh2,
        nested_area_ratio=nested_track_area_ratio2,
        nested_app_sim_thresh=nested_track_app_sim2
    )

    tracklet_store = None

    # Open videos
    cap1 = cv.VideoCapture(source1)
    cap2 = cv.VideoCapture(source2)

    fps1 = int(cap1.get(cv.CAP_PROP_FPS)) or 25
    fps2 = int(cap2.get(cv.CAP_PROP_FPS)) or 25

    # Create masks
    mask1, _ = create_trapezoid_mask((size[1], size[0]),
                                    TRAPEZ_TOP_LEFT_1, TRAPEZ_TOP_RIGHT_1, 
                                    TRAPEZ_BOTTOM_LEFT_1, TRAPEZ_BOTTOM_RIGHT_1, 
                                    CURVE_HEIGHT_1)
    mask2, _ = create_trapezoid_mask((size[1], size[0]),
                                    TRAPEZ_TOP_LEFT_2, TRAPEZ_TOP_RIGHT_2, 
                                    TRAPEZ_BOTTOM_LEFT_2, TRAPEZ_BOTTOM_RIGHT_2, 
                                    CURVE_HEIGHT_2)
    # Create video writers
    writer1 = cv.VideoWriter(output_path1, cv.VideoWriter_fourcc(*'mp4v'), fps1, size)
    writer2 = cv.VideoWriter(output_path2, cv.VideoWriter_fourcc(*'mp4v'), fps2, size)

    # Create CSV writers
    csv_file1 = open(output_csv_path1, 'w', newline='')
    csv_writer1 = csv.DictWriter(csv_file1, fieldnames=['frame', 'track_id', 'event', 'view', 'global_id', 'matched_with_other_view_track_id', 'new_global_id', 'old_global_id', 'world_x', 'world_y', 'world_z', 'world_pos_json', 'has_world_pos'])
    csv_writer1.writeheader()

    csv_file2 = open(output_csv_path2, 'w', newline='')
    csv_writer2 = csv.DictWriter(csv_file2, fieldnames=['frame', 'track_id', 'event', 'view', 'global_id', 'matched_with_other_view_track_id', 'new_global_id', 'old_global_id', 'world_x', 'world_y', 'world_z', 'world_pos_json', 'has_world_pos'])
    csv_writer2.writeheader()


    # Load annotations
    annotations1 = load_annotations_from_dir(annotations_dir1, video_prefix1)
    annotations2 = load_annotations_from_dir(annotations_dir2, video_prefix2)
    if auto_start_from_annotations:
        first_annot_frame = _min_annotated_frame(annotations1, annotations2)
        if first_annot_frame is not None and start_frame < first_annot_frame:
            start_frame = first_annot_frame

    frame_id = start_frame
    fps_hist = deque(maxlen=30)
    prev_assoc_v2 = {}
    prev_assoc_v1 = {}
    conflict_log = []
    conflict_writer = None
    conflict_file = None
    if conflict_log_path:
        conflict_file = open(conflict_log_path, 'w', newline='')
        conflict_writer = csv.DictWriter(conflict_file, fieldnames=[
            'frame', 'view', 'view1_track_id', 'view1_global_id',
            'other_track_id', 'other_global_id', 'reason'
        ])
        conflict_writer.writeheader()

    # Skip to start_frame
    for _ in range(start_frame):
        cap1.read()
        cap2.read()

    # ===== PROCESS INITIAL FRAME =====
    ok1, frame1 = cap1.read()
    ok2, frame2 = cap2.read()
    
    if not (ok1 and ok2):
        print("Error: Could not read initial frames")
        return

    orig_h1, orig_w1 = frame1.shape[:2]
    orig_h2, orig_w2 = frame2.shape[:2]
    frame1 = cv.resize(frame1, size)
    frame2 = cv.resize(frame2, size)
    print(f"Processing initial frame {frame_id}")

    line_debug1 = []
    line_debug2 = []
    common_lines_1 = []
    common_lines_2 = []
    line_sig_ref_1 = None
    line_sig_ref_2 = None
    if court_lines_path1 and court_lines_path2 and court_corners_path1 and court_corners_path2 and court_line_correspondence_path:
        sx1 = size[0] / float(orig_w1) if orig_w1 > 0 else 1.0
        sy1 = size[1] / float(orig_h1) if orig_h1 > 0 else 1.0
        sx2 = size[0] / float(orig_w2) if orig_w2 > 0 else 1.0
        sy2 = size[1] / float(orig_h2) if orig_h2 > 0 else 1.0

        pts1 = load_indexed_points(court_corners_path1)
        pts2 = load_indexed_points(court_corners_path2)
        lines1 = load_court_lines(court_lines_path1, pts1)
        lines2 = load_court_lines(court_lines_path2, pts2)
        line_debug1 = scale_lines(lines1, sx1, sy1)
        line_debug2 = scale_lines(lines2, sx2, sy2)
        ref1, ref2 = compute_common_ref_points(pts1, pts2)
        if ref1 is not None:
            line_sig_ref_1 = np.array([ref1[0] * sx1, ref1[1] * sy1], dtype=np.float32)
        if ref2 is not None:
            line_sig_ref_2 = np.array([ref2[0] * sx2, ref2[1] * sy2], dtype=np.float32)

        corr = load_line_correspondences(
            court_line_correspondence_path, court_line_view1_key, court_line_view2_key
        )
        lines1_by_id = {l['line_id']: l for l in line_debug1}
        lines2_by_id = {l['line_id']: l for l in line_debug2}
        for lid1, lid2 in corr:
            l1 = lines1_by_id.get(lid1)
            l2 = lines2_by_id.get(lid2)
            if l1 is not None and l2 is not None:
                common_lines_1.append(l1)
                common_lines_2.append(l2)
    initial_detections1 = detect_view(
        frame1, frame_id, annotations1, size, mask1, det_model,
        sahi_conf_threshold1, sahi_iou_threshold1, slice_h1, slice_w1,
        slice_overlap1, pose_model, pose_attempts1, pose_conf_threshold1,
        pose_iou_threshold1, tracker1,
        suppress_nested=suppress_nested_detections,
        nested_contain_thresh=nested_contain_thresh,
        nested_area_ratio=nested_area_ratio
    )

    initial_detections2 = detect_view(
        frame2, frame_id, annotations2, size, mask2, det_model,
        sahi_conf_threshold2, sahi_iou_threshold2, slice_h2, slice_w2,
        slice_overlap2, pose_model, pose_attempts2, pose_conf_threshold2,
        pose_iou_threshold2, tracker2,
        suppress_nested=suppress_nested_detections,
        nested_contain_thresh=nested_contain_thresh,
        nested_area_ratio=nested_area_ratio
    )

    # Update trackers with initial detections
    tracks1, _, _ = tracker1.update(
        initial_detections1, frame_id, finalize=True,
        resolve_conflicts=(frame_id not in annotations1)
    )
    tracks2, _, _ = tracker2.update(
        initial_detections2, frame_id, finalize=True,
        resolve_conflicts=(frame_id not in annotations2)
    )
    # tracklet aggregation disabled

    # Associate tracks across views (View 1 as anchor)
    associate_tracks_to_view1(
        tracker1, tracker2, frame_id,
        tracker1.cross_view_match_threshold,
        app_weight=appearance_weight1, world_weight=world_dist_weight1,
        max_world_dist=2000.0, world_dist_tolerance=world_dist_tolerance1,
        reproj_max_px=reproj_max_px1, reproj_soft_margin=reproj_soft_margin1,
        prev_assoc=prev_assoc_v2, sticky_bonus=0.10,
        conflict_log=conflict_log, conflict_view_id=2,
        view_id_other=2,
        tracklet_min_frames=tracklet_min_frames,
        tracklet_store=tracklet_store,
        use_pose_triangulation=use_pose_triangulation,
        pose_triangulation_max_reproj=pose_triangulation_max_reproj,
        always_overwrite_gid=always_overwrite_gid,
        line_sig_lines_anchor=common_lines_1,
        line_sig_lines_other=common_lines_2,
        line_sig_ref_anchor=line_sig_ref_1,
        line_sig_ref_other=line_sig_ref_2,
        line_sig_thresh=0.15,
        line_sig_debug_frames=line_sig_debug_frames,
        anchor_view_id=1,
    )
    associate_tracks_to_view1(
        tracker2, tracker1, frame_id,
        tracker2.cross_view_match_threshold,
        app_weight=appearance_weight2, world_weight=world_dist_weight2,
        max_world_dist=2000.0, world_dist_tolerance=world_dist_tolerance2,
        reproj_max_px=reproj_max_px2, reproj_soft_margin=reproj_soft_margin2,
        prev_assoc=prev_assoc_v1, sticky_bonus=0.10,
        conflict_log=conflict_log, conflict_view_id=1,
        view_id_other=1,
        tracklet_min_frames=tracklet_min_frames,
        tracklet_store=tracklet_store,
        use_pose_triangulation=use_pose_triangulation,
        pose_triangulation_max_reproj=pose_triangulation_max_reproj,
        always_overwrite_gid=always_overwrite_gid,
        line_sig_lines_anchor=common_lines_2,
        line_sig_lines_other=common_lines_1,
        line_sig_ref_anchor=line_sig_ref_2,
        line_sig_ref_other=line_sig_ref_1,
        line_sig_thresh=0.15,
        line_sig_debug_frames=line_sig_debug_frames,
        anchor_view_id=2
    )
    if conflict_writer and conflict_log:
        for row in conflict_log:
            conflict_writer.writerow(row)
        conflict_log.clear()

    # Log initial events
    for event in tracker1.get_frame_events():
        event['view'] = 1
        _augment_event_with_world_pos(event, tracker1)
        csv_writer1.writerow(event)
    for event in tracker2.get_frame_events():
        event['view'] = 2
        _augment_event_with_world_pos(event, tracker2)
        csv_writer2.writerow(event)

    # Save first frame
    roi1 = apply_trapezoid_mask(frame1, mask1)
    roi2 = apply_trapezoid_mask(frame2, mask2)
    if draw_court_lines_debug:
        roi1 = draw_court_lines(roi1, line_debug1, color=(0, 255, 0), thickness=2, label=True)
        roi2 = draw_court_lines(roi2, line_debug2, color=(0, 255, 0), thickness=2, label=True)
    out1 = draw_tracks(roi1.copy(), tracks1, None, connections, tracker1)
    out2 = draw_tracks(roi2.copy(), tracks2, None, connections, tracker2)
    writer1.write(out1)
    writer2.write(out2)
    

    frame_id += 1

    # ===== MAIN PROCESSING LOOP =====
    print("Starting main processing loop...")
    
    while cap1.isOpened() and cap2.isOpened():
        ok1, frame1 = cap1.read()
        ok2, frame2 = cap2.read()
        
        if not (ok1 and ok2):
            break

        frame1 = cv.resize(frame1, size)
        frame2 = cv.resize(frame2, size)
        start_time = time.time()

        # ===== STEP 1: DETECTION + POSE =====
        detections1 = detect_view(
            frame1, frame_id, annotations1, size, mask1, det_model,
            sahi_conf_threshold1, sahi_iou_threshold1, slice_h1, slice_w1,
            slice_overlap1, pose_model, pose_attempts1, pose_conf_threshold1,
            pose_iou_threshold1, tracker1,
            suppress_nested=suppress_nested_detections,
            nested_contain_thresh=nested_contain_thresh,
            nested_area_ratio=nested_area_ratio
        )
        detections2 = detect_view(
            frame2, frame_id, annotations2, size, mask2, det_model,
            sahi_conf_threshold2, sahi_iou_threshold2, slice_h2, slice_w2,
            slice_overlap2, pose_model, pose_attempts2, pose_conf_threshold2,
            pose_iou_threshold2, tracker2,
            suppress_nested=suppress_nested_detections,
            nested_contain_thresh=nested_contain_thresh,
            nested_area_ratio=nested_area_ratio
        )
        # ===== STEP 4: UPDATE TRACKERS (finalize=True) =====
        tracks1, _, _ = tracker1.update(
            detections1, frame_id, finalize=True,
            resolve_conflicts=(frame_id not in annotations1)
        )
        tracks2, _, _ = tracker2.update(
            detections2, frame_id, finalize=True,
            resolve_conflicts=(frame_id not in annotations2)
        )
        # tracklet aggregation disabled

        # ===== STEP 5: CROSS-VIEW ASSOCIATION (VIEW 1 AS ANCHOR) =====
        associate_tracks_to_view1(
            tracker1, tracker2, frame_id,
            tracker1.cross_view_match_threshold,
            app_weight=appearance_weight1, world_weight=world_dist_weight1,
            max_world_dist=2000.0, world_dist_tolerance=world_dist_tolerance1,
            reproj_max_px=reproj_max_px1, reproj_soft_margin=reproj_soft_margin1,
            prev_assoc=prev_assoc_v2, sticky_bonus=0.10,
            conflict_log=conflict_log, conflict_view_id=2,
            view_id_other=2,
            tracklet_min_frames=tracklet_min_frames,
            tracklet_store=tracklet_store,
            use_pose_triangulation=use_pose_triangulation,
            pose_triangulation_max_reproj=pose_triangulation_max_reproj,
            always_overwrite_gid=always_overwrite_gid,
            line_sig_lines_anchor=common_lines_1,
            line_sig_lines_other=common_lines_2,
            line_sig_ref_anchor=line_sig_ref_1,
            line_sig_ref_other=line_sig_ref_2,
            line_sig_thresh=0.15,
            line_sig_debug_frames=line_sig_debug_frames,
            anchor_view_id=1
        )
        associate_tracks_to_view1(
            tracker2, tracker1, frame_id,
            tracker2.cross_view_match_threshold,
            app_weight=appearance_weight2, world_weight=world_dist_weight2,
            max_world_dist=2000.0, world_dist_tolerance=world_dist_tolerance2,
            reproj_max_px=reproj_max_px2, reproj_soft_margin=reproj_soft_margin2,
            prev_assoc=prev_assoc_v1, sticky_bonus=0.10,
            conflict_log=conflict_log, conflict_view_id=1,
            view_id_other=1,
            tracklet_min_frames=tracklet_min_frames,
            tracklet_store=tracklet_store,
            use_pose_triangulation=use_pose_triangulation,
            pose_triangulation_max_reproj=pose_triangulation_max_reproj,
            always_overwrite_gid=always_overwrite_gid,
            line_sig_lines_anchor=common_lines_2,
            line_sig_lines_other=common_lines_1,
            line_sig_ref_anchor=line_sig_ref_2,
            line_sig_ref_other=line_sig_ref_1,
            line_sig_thresh=0.15,
            line_sig_debug_frames=line_sig_debug_frames,
            anchor_view_id=2
        )
        if conflict_writer and conflict_log:
            for row in conflict_log:
                conflict_writer.writerow(row)
            conflict_log.clear()

        # ===== LOG EVENTS =====
        for event in tracker1.get_frame_events():
            if 'view' not in event:
                event['view'] = 1
            _augment_event_with_world_pos(event, tracker1)
            csv_writer1.writerow(event)
        
        for event in tracker2.get_frame_events():
            if 'view' not in event:
                event['view'] = 2
            _augment_event_with_world_pos(event, tracker2)
            csv_writer2.writerow(event)

        # Refresh tracks after all operations
        tracks1 = tracker1.tracks
        tracks2 = tracker2.tracks

        # ===== DRAWING =====
        roi1 = apply_trapezoid_mask(frame1, mask1)
        roi2 = apply_trapezoid_mask(frame2, mask2)
        if draw_court_lines_debug:
            roi1 = draw_court_lines(roi1, line_debug1, color=(0, 255, 0), thickness=2, label=True)
            roi2 = draw_court_lines(roi2, line_debug2, color=(0, 255, 0), thickness=2, label=True)
        out1 = draw_tracks(roi1.copy(), tracks1, None, connections, tracker1)
        out2 = draw_tracks(roi2.copy(), tracks2, None, connections, tracker2)

        dt = time.time() - start_time
        fps_hist.append(1 / dt if dt > 0 else 0)
        fps_avg = sum(fps_hist) / len(fps_hist) if fps_hist else 0
        
        draw_text_with_bg(out1, f'FPS {int(fps_avg)}', (30, 50))
        draw_text_with_bg(out1, f'Frame {frame_id}', (30, 100))
        draw_text_with_bg(out2, f'FPS {int(fps_avg)}', (30, 50))
        draw_text_with_bg(out2, f'Frame {frame_id}', (30, 100))

        writer1.write(out1)
        writer2.write(out2)

        cv.imshow('Tracking View 1', out1)
        cv.imshow('Tracking View 2', out2)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    # Cleanup
    cap1.release()
    cap2.release()
    
    writer1.release()
    writer2.release()
    
    cv.destroyAllWindows()

    csv_file1.close()
    csv_file2.close()
    if conflict_file:
        conflict_file.close()

    print("Processing complete!")


if __name__ == '__main__':
    try:
        yolo_sahi_pose_tracking(
            source1='Tracking/material4project/Rectified videos/tracking_12/out13.mp4',
            source2='Tracking/material4project/Rectified videos/tracking_12/out4.mp4',
            calib_path1='Tracking/material4project/3D Tracking Material/camera_data_with_Rvecs/camera_data/cam_13/calib/camera_calib.json',
            calib_path2='Tracking/material4project/3D Tracking Material/camera_data_with_Rvecs/camera_data/cam_4/calib/camera_calib.json',
            annotations_dir1='tracking_12.v4i.yolov11/train/labels/',
            annotations_dir2='tracking_12.v4i.yolov11/train/labels/',
            output_path1='output_view1_improved.mp4',
            output_path2='output_view2_improved.mp4',
            start_frame=1,
            enable_pose=True,
            court_lines_path1='court_lines_cam13.json',
            court_lines_path2='court_lines_cam4.json',
            court_corners_path1='cam13_img_corners_rectified.json',
            court_corners_path2='cam4_img_corners_rectified.json',
            court_line_correspondence_path='court_line_corrispondency.json',
            court_line_view1_key='line_id_cam13',
            court_line_view2_key='line_id_cam4',
            draw_court_lines_debug=True,
        )
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
