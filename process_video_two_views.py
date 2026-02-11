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

# ================= UTILITIES =================

def round_to_multiple(value, multiple):
    return multiple * ((value + multiple - 1) // multiple)


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


def resize_for_display(img, max_width=1280, max_height=720):
    if img is None:
        return img
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return img
    scale = min(max_width / float(w), max_height / float(h), 1.0)
    if scale >= 1.0:
        return img
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return cv.resize(img, (out_w, out_h), interpolation=cv.INTER_AREA)


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

def _augment_event_with_view_and_projected_pos(event, tracker, homography_to_other):
    """Attach own-view bottom center and projected point in the other view."""
    event['view_x'] = ''
    event['view_y'] = ''
    event['view_pos_json'] = ''
    event['has_view_pos'] = 0
    event['projected_other_x'] = ''
    event['projected_other_y'] = ''
    event['projected_other_pos_json'] = ''
    event['has_projected_other_pos'] = 0

    if tracker is None:
        return event

    tid = event.get('track_id')
    if tid is None:
        return event

    track = next((t for t in tracker.tracks if t.get('track_id') == tid), None)
    bbox = track.get('bbox') if track else None
    if bbox is None:
        return event

    view_pt = _bbox_bottom_center(bbox)
    if np.all(np.isfinite(view_pt)):
        vx, vy = float(view_pt[0]), float(view_pt[1])
        event['view_x'] = vx
        event['view_y'] = vy
        event['view_pos_json'] = json.dumps([vx, vy])
        event['has_view_pos'] = 1

        if homography_to_other is not None:
            proj_pt = project_point_with_homography(view_pt, homography_to_other)
            if np.all(np.isfinite(proj_pt)):
                px, py = float(proj_pt[0]), float(proj_pt[1])
                event['projected_other_x'] = px
                event['projected_other_y'] = py
                event['projected_other_pos_json'] = json.dumps([px, py])
                event['has_projected_other_pos'] = 1
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


def load_homographies_for_views(homography_path):
    if not homography_path or not os.path.exists(homography_path):
        raise FileNotFoundError(f"Homography file not found: {homography_path}")
    with open(homography_path, 'r') as f:
        data = json.load(f)

    if 'H' not in data or 'H_inv' not in data:
        raise ValueError(
            f"Homography file {homography_path} must contain both 'H' and 'H_inv'."
        )

    # In this project file:
    # - H maps cam4 -> cam13
    # - H_inv maps cam13 -> cam4
    h_view2_to_view1 = np.array(data['H'], dtype=np.float32)
    h_view1_to_view2 = np.array(data['H_inv'], dtype=np.float32)
    return h_view1_to_view2, h_view2_to_view1


def project_point_with_homography(pt_xy, homography):
    src = np.array([[pt_xy]], dtype=np.float32)
    dst = cv.perspectiveTransform(src, homography)
    return dst[0, 0]


def scale_homography_for_resized_views(
    homography,
    src_orig_size,
    dst_orig_size,
    src_proc_size,
    dst_proc_size,
):
    """
    Convert a homography defined on original resolutions to processing resolutions.

    homography maps src_orig pixels -> dst_orig pixels.
    Returned matrix maps src_proc pixels -> dst_proc pixels.
    """
    src_ow, src_oh = src_orig_size
    dst_ow, dst_oh = dst_orig_size
    src_pw, src_ph = src_proc_size
    dst_pw, dst_ph = dst_proc_size

    sx_src = float(src_pw) / float(src_ow)
    sy_src = float(src_ph) / float(src_oh)
    sx_dst = float(dst_pw) / float(dst_ow)
    sy_dst = float(dst_ph) / float(dst_oh)

    s_src = np.array([[sx_src, 0.0, 0.0], [0.0, sy_src, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    s_dst = np.array([[sx_dst, 0.0, 0.0], [0.0, sy_dst, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    h_scaled = s_dst @ homography @ np.linalg.inv(s_src)
    return h_scaled.astype(np.float32)


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

def _is_point_above_line_segment(pt, line, ref_pt):
    p1 = line['p1']
    p2 = line['p2']
    x = float(pt[0])

    min_x = min(float(p1[0]), float(p2[0]))
    max_x = max(float(p1[0]), float(p2[0]))
    if x < (min_x - 1.0) or x > (max_x + 1.0):
        return False

    v = p2 - p1
    denom = np.linalg.norm(v)
    if denom <= 1e-6:
        return False

    n = np.array([-v[1], v[0]], dtype=np.float32)
    d = float(np.dot((pt - p1), n) / denom)
    ref_side = float(np.dot((ref_pt - p1), n))
    if ref_side < 0:
        d = -d

    return d < 0.0


def _filter_detections_above_lines(detections, lines, ref_pt):
    if not lines:
        return detections

    if ref_pt is None:
        xs = []
        ys = []
        for l in lines:
            xs.extend([float(l['p1'][0]), float(l['p2'][0])])
            ys.extend([float(l['p1'][1]), float(l['p2'][1])])
        ref_pt = np.array([np.mean(xs), max(ys) + 100.0], dtype=np.float32)

    valid = []
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        bottom_center = np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)
        invalid = False
        for line in lines:
            if _is_point_above_line_segment(bottom_center, line, ref_pt):
                invalid = True
                break
        if not invalid:
            valid.append(d)
    return valid


def detect_view(frame, frame_id, annotations, det_model,
                sahi_conf_threshold, sahi_iou_threshold, slice_h, slice_w,
                slice_overlap, pose_model, pose_attempts, pose_conf_threshold,
                pose_iou_threshold, tracker,
                suppress_nested=True, nested_contain_thresh=0.9, nested_area_ratio=0.6,
                invalid_if_above_lines=None, invalid_ref_pt=None):
    detections = []
    frame_h, frame_w = frame.shape[:2]
    if frame_id in annotations:
        for anno_file in annotations[frame_id]:
            detections.extend(parse_yolo_annotations(
                anno_file, frame_w, frame_h, frame, pose_model, pose_attempts,
                pose_conf_threshold, pose_iou_threshold, tracker=tracker
            ))
    else:
        preds = get_sliced_prediction(
            frame, det_model, slice_height=slice_h, slice_width=slice_w,
            overlap_height_ratio=slice_overlap, overlap_width_ratio=slice_overlap,
            postprocess_match_metric='IOU', postprocess_match_threshold=sahi_iou_threshold,
            verbose=0
        )

        if preds and preds.object_prediction_list:
            for p in preds.object_prediction_list:
                if p.category.id == 0 and p.score.value >= sahi_conf_threshold:
                    bx1, by1, bx2, by2 = map(int, p.bbox.to_xyxy())
                    bbox = (bx1, by1, bx2, by2)
                    appearance = compute_team_appearence(frame, bbox)
                    world_pos = tracker.project_to_world(bbox)
                    detections.append({
                        'bbox': bbox,
                        'appearance': appearance,
                        'world_pos': world_pos,
                        'score': p.score.value
                    })

    if invalid_if_above_lines:
        detections = _filter_detections_above_lines(
            detections, invalid_if_above_lines, invalid_ref_pt
        )

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
        # Force bbox-bottom-center anchoring for world coordinates.
        d['world_pos'] = tracker.project_to_world(d['bbox']) if tracker else None

    return detections


def merge_track_global_id(tracker, track, target_gid, frame_id):
    if track['global_id'] == target_gid:
        return True
    if hasattr(tracker, 'reassign_track_global_id'):
        return bool(
            tracker.reassign_track_global_id(
                track, target_gid, frame_id, allow_target_conflict=False
            )
        )
    tracker.merge_global_ids(track['global_id'], target_gid, frame_id)
    return track.get('global_id') == target_gid


def _bbox_bottom_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)


def associate_tracks_to_view1(
    tracker1, tracker_other, frame_id,
    homography_anchor_to_other,
    max_projected_dist_px=120.0,
):
    active1 = [t for t in tracker1.tracks if t.get('updated') and t.get('bbox') is not None]
    active_other = [t for t in tracker_other.tracks if t.get('updated') and t.get('bbox') is not None]
    if not active1 or not active_other:
        return

    projected = []
    for t1 in active1:
        p1 = _bbox_bottom_center(t1['bbox'])
        p_other = project_point_with_homography(p1, homography_anchor_to_other)
        projected.append(p_other)

    other_points = [_bbox_bottom_center(t2['bbox']) for t2 in active_other]

    large = 1e9
    cost = np.full((len(active1), len(active_other)), large, dtype=np.float32)
    for i, p_proj in enumerate(projected):
        if not np.all(np.isfinite(p_proj)):
            continue
        for j, p_other in enumerate(other_points):
            cost[i, j] = float(np.linalg.norm(p_proj - p_other))

    row_ind, col_ind = linear_sum_assignment(cost)

    for r, c in zip(row_ind, col_ind):
        dist = float(cost[r, c])
        if dist > max_projected_dist_px:
            continue

        t1 = active1[r]
        t2 = active_other[c]

        gid1 = t1['global_id']
        target_gid = gid1
        merged1 = merge_track_global_id(tracker1, t1, target_gid, frame_id)
        merged2 = merge_track_global_id(tracker_other, t2, target_gid, frame_id)
        if merged1 and merged2 and t1.get('gid_source') == 'annotation':
            t2['gid_source'] = 'annotation'


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
    size=(1440, 810),  # Processing resolution (display uses same max size).
    # detection View 1
    sahi_conf_threshold1=0.40,
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
    max_velocity1=500.0,
    suppress_nested_tracks1=True,
    nested_track_contain_thresh1=0.9,
    nested_track_area_ratio1=0.6,
    nested_track_app_sim1=0.7,
    cross_view_max_dist_px1=120.0,
    max_missed_frames1=80,
    # detection View 2
    sahi_conf_threshold2=0.75,
    sahi_iou_threshold2=0.50,
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
    max_velocity2=500.0,
    suppress_nested_tracks2=True,
    nested_track_contain_thresh2=0.9,
    nested_track_area_ratio2=0.68,
    nested_track_app_sim2=0.7,
    cross_view_max_dist_px2=120.0,
    max_missed_frames2=80,
    homography_path='homography_cam4_to_cam13.json',
    suppress_nested_detections=True,
    nested_contain_thresh=0.9,
    nested_area_ratio=0.68,
    # line-based filtering + debug
    invalid_view1_line_ids=(12, 13, 14),
    court_lines_path1=None,
    court_corners_path1=None,
):
    """
    Multi-view tracking with single-view tracking + cross-view association.

    Flow:
    - Detect per-view and estimate pose
    - Track per-view independently
    - Associate tracks across views using homography-projected bottom centers
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
        is_rectified=True,  # FIX: Videos are rectified
        suppress_nested_tracks=suppress_nested_tracks2,
        nested_contain_thresh=nested_track_contain_thresh2,
        nested_area_ratio=nested_track_area_ratio2,
        nested_app_sim_thresh=nested_track_app_sim2
    )

    h_view1_to_view2_raw, h_view2_to_view1_raw = load_homographies_for_views(homography_path)
    process_size = (int(size[0]), int(size[1]))
    if process_size[0] <= 0 or process_size[1] <= 0:
        raise ValueError(f"Invalid processing size: {process_size}")

    # Open videos
    cap1 = cv.VideoCapture(source1)
    cap2 = cv.VideoCapture(source2)

    fps1 = int(cap1.get(cv.CAP_PROP_FPS)) or 25
    fps2 = int(cap2.get(cv.CAP_PROP_FPS)) or 25

    # Create CSV writers
    csv_file1 = open(output_csv_path1, 'w', newline='')
    csv_writer1 = csv.DictWriter(csv_file1, fieldnames=['frame', 'track_id', 'event', 'view', 'global_id', 'matched_with_other_view_track_id', 'new_global_id', 'old_global_id', 'view_x', 'view_y', 'view_pos_json', 'has_view_pos', 'projected_other_x', 'projected_other_y', 'projected_other_pos_json', 'has_projected_other_pos'])
    csv_writer1.writeheader()

    csv_file2 = open(output_csv_path2, 'w', newline='')
    csv_writer2 = csv.DictWriter(csv_file2, fieldnames=['frame', 'track_id', 'event', 'view', 'global_id', 'matched_with_other_view_track_id', 'new_global_id', 'old_global_id', 'view_x', 'view_y', 'view_pos_json', 'has_view_pos', 'projected_other_x', 'projected_other_y', 'projected_other_pos_json', 'has_projected_other_pos'])
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

    # Skip to start_frame
    for _ in range(start_frame):
        cap1.read()
        cap2.read()

    # ===== PROCESS INITIAL FRAME =====
    ok1, frame1 = cap1.read()
    ok2, frame2 = cap2.read()
    
    if not (ok1 and ok2):
        print("Error: Could not read initial frames")
        cap1.release()
        cap2.release()
        csv_file1.close()
        csv_file2.close()
        return

    orig_h1, orig_w1 = frame1.shape[:2]
    orig_h2, orig_w2 = frame2.shape[:2]
    frame1 = cv.resize(frame1, process_size)
    frame2 = cv.resize(frame2, process_size)

    # Scale homographies from original camera resolution to processing resolution.
    h_view1_to_view2 = scale_homography_for_resized_views(
        h_view1_to_view2_raw,
        src_orig_size=(orig_w1, orig_h1),
        dst_orig_size=(orig_w2, orig_h2),
        src_proc_size=process_size,
        dst_proc_size=process_size,
    )
    h_view2_to_view1 = scale_homography_for_resized_views(
        h_view2_to_view1_raw,
        src_orig_size=(orig_w2, orig_h2),
        dst_orig_size=(orig_w1, orig_h1),
        src_proc_size=process_size,
        dst_proc_size=process_size,
    )

    # Writers use the resized processing resolution.
    writer1 = cv.VideoWriter(output_path1, cv.VideoWriter_fourcc(*'mp4v'), fps1, process_size)
    writer2 = cv.VideoWriter(output_path2, cv.VideoWriter_fourcc(*'mp4v'), fps2, process_size)

    print(f"Processing initial frame {frame_id}")

    line_debug1 = []
    invalid_view1_lines = []
    invalid_view1_ref_pt = np.array([process_size[0] * 0.5, process_size[1] - 1.0], dtype=np.float32)
    sx1 = process_size[0] / float(orig_w1) if orig_w1 > 0 else 1.0
    sy1 = process_size[1] / float(orig_h1) if orig_h1 > 0 else 1.0

    if court_lines_path1 and court_corners_path1:
        pts1 = load_indexed_points(court_corners_path1)
        lines1 = load_court_lines(court_lines_path1, pts1)
        line_debug1 = scale_lines(lines1, sx1, sy1)
        invalid_ids = {int(v) for v in invalid_view1_line_ids}
        invalid_view1_lines = [l for l in line_debug1 if l['line_id'] in invalid_ids]

    initial_detections1 = detect_view(
        frame1, frame_id, annotations1, det_model,
        sahi_conf_threshold1, sahi_iou_threshold1, slice_h1, slice_w1,
        slice_overlap1, pose_model, pose_attempts1, pose_conf_threshold1,
        pose_iou_threshold1, tracker1,
        suppress_nested=suppress_nested_detections,
        nested_contain_thresh=nested_contain_thresh,
        nested_area_ratio=nested_area_ratio,
        invalid_if_above_lines=invalid_view1_lines,
        invalid_ref_pt=invalid_view1_ref_pt
    )

    initial_detections2 = detect_view(
        frame2, frame_id, annotations2, det_model,
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
    # Associate tracks across views (one-way: View 1 -> View 2)
    associate_tracks_to_view1(
        tracker1, tracker2, frame_id,
        homography_anchor_to_other=h_view1_to_view2,
        max_projected_dist_px=cross_view_max_dist_px1
    )

    # Log initial events
    for event in tracker1.get_frame_events():
        event['view'] = 1
        _augment_event_with_view_and_projected_pos(event, tracker1, h_view1_to_view2)
        csv_writer1.writerow(event)
    for event in tracker2.get_frame_events():
        event['view'] = 2
        _augment_event_with_view_and_projected_pos(event, tracker2, h_view2_to_view1)
        csv_writer2.writerow(event)

    # Save first frame
    roi1 = frame1
    roi2 = frame2
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

        frame1 = cv.resize(frame1, process_size)
        frame2 = cv.resize(frame2, process_size)
        start_time = time.time()

        # ===== STEP 1: DETECTION + POSE =====
        detections1 = detect_view(
            frame1, frame_id, annotations1, det_model,
            sahi_conf_threshold1, sahi_iou_threshold1, slice_h1, slice_w1,
            slice_overlap1, pose_model, pose_attempts1, pose_conf_threshold1,
            pose_iou_threshold1, tracker1,
            suppress_nested=suppress_nested_detections,
            nested_contain_thresh=nested_contain_thresh,
            nested_area_ratio=nested_area_ratio,
            invalid_if_above_lines=invalid_view1_lines,
            invalid_ref_pt=invalid_view1_ref_pt
        )
        detections2 = detect_view(
            frame2, frame_id, annotations2, det_model,
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
        # ===== STEP 5: CROSS-VIEW ASSOCIATION (ONE-WAY: VIEW 1 -> VIEW 2) =====
        associate_tracks_to_view1(
            tracker1, tracker2, frame_id,
            homography_anchor_to_other=h_view1_to_view2,
            max_projected_dist_px=cross_view_max_dist_px1
        )

        # ===== LOG EVENTS =====
        for event in tracker1.get_frame_events():
            if 'view' not in event:
                event['view'] = 1
            _augment_event_with_view_and_projected_pos(event, tracker1, h_view1_to_view2)
            csv_writer1.writerow(event)
        
        for event in tracker2.get_frame_events():
            if 'view' not in event:
                event['view'] = 2
            _augment_event_with_view_and_projected_pos(event, tracker2, h_view2_to_view1)
            csv_writer2.writerow(event)

        # Refresh tracks after all operations
        tracks1 = tracker1.tracks
        tracks2 = tracker2.tracks

        # ===== DRAWING =====
        roi1 = frame1
        roi2 = frame2
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

        disp1 = resize_for_display(out1, max_width=size[0], max_height=size[1])
        disp2 = resize_for_display(out2, max_width=size[0], max_height=size[1])
        cv.imshow('Tracking View 1', disp1)
        cv.imshow('Tracking View 2', disp2)
        
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

    print("Processing complete!")


if __name__ == '__main__':
    try:
        yolo_sahi_pose_tracking(
            source1='Tracking/material4project/Rectified videos/tracking_12/out13.mp4',
            source2='Tracking/material4project/Rectified videos/tracking_12/out4.mp4',
            calib_path1='Tracking/material4project/3D Tracking Material/camera_data_with_Rvecs_2ndversion/camera_data/cam_13/calib/camera_calib.json',
            calib_path2='Tracking/material4project/3D Tracking Material/camera_data_with_Rvecs_2ndversion/camera_data/cam_4/calib/camera_calib.json',
            annotations_dir1='tracking_12.v4i.yolov11/train/labels/',
            annotations_dir2='tracking_12.v4i.yolov11/train/labels/',
            output_path1='output_view1_improved.mp4',
            output_path2='output_view2_improved.mp4',
            start_frame=1,
            enable_pose=True,
            court_lines_path1='court_lines_cam13.json',
            court_corners_path1='cam13_img_corners_rectified.json',
        )
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
