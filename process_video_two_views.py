import cv2 as cv
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
import csv

import json
import os
import re
from scipy.optimize import linear_sum_assignment

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
    Run pose estimation with early exit and return how many attempts were executed.
    """
    if pose_model is None:
        return None, None, 0

    attempts_cfg = tuple(pose_attempts) if pose_attempts else ({'pad': 0.0, 'conf': None, 'iou': None},)
    attempts_used = 0

    for attempt in attempts_cfg:
        attempts_used += 1
        pb = pad_bbox(bbox, attempt.get('pad', 0.0), frame.shape)
        px1, py1, px2, py2 = pb
        crop = frame[py1:py2, px1:px2]
        if crop.size <= 0:
            continue

        res = pose_model.predict(
            crop,
            conf=attempt.get('conf') or pose_conf_threshold,
            iou=attempt.get('iou') or pose_iou_threshold,
            imgsz=round_to_multiple(max(crop.shape[:2]), 32),
            device='cuda',
            verbose=False
        )
        if res and len(res[0].keypoints.xy) > 0:
            k = res[0].keypoints.xy.cpu().numpy()[0]
            pose_kpts = k + np.array([px1, py1])
            pose_vec = keypoints_to_pose_vec(pose_kpts, bbox)
            return pose_kpts, pose_vec, attempts_used

    return None, None, attempts_used


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
                pose_kpts, pose_vec, pose_attempts_used = process_detection_pose(
                    frame, bbox, pose_model, pose_attempts, 
                    pose_conf_threshold, pose_iou_threshold
                )
                world_pos = tracker.project_to_world(bbox) if tracker else None

                detections.append({
                    'bbox': bbox,
                    'appearance': appearance,
                    'keypoints': pose_kpts,
                    'pose_vec': pose_vec,
                    'pose_attempts_used': int(pose_attempts_used),
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


def _count_valid_yolo_boxes(anno_files):
    total = 0
    for anno_path in anno_files or []:
        if not anno_path or not os.path.exists(anno_path):
            continue
        try:
            with open(anno_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        total += 1
        except Exception:
            continue
    return total


def infer_slot_count_from_first_annotation_frame(annotations):
    if not annotations:
        return None, None
    first_frame = min(annotations.keys())
    count = _count_valid_yolo_boxes(annotations.get(first_frame, []))
    if count <= 0:
        return first_frame, None
    return first_frame, int(count)


def first_annotation_frame(annotations):
    if not annotations:
        return None
    return min(annotations.keys())


def keep_annotation_gids_only_on_initial_frame(detections, frame_id, initial_annotation_frame):
    """
    Keep annotation-provided global IDs only on the first annotated frame.
    For subsequent annotated frames, keep boxes but drop annotation GID forcing.
    """
    if initial_annotation_frame is None or frame_id == initial_annotation_frame:
        return detections
    for d in detections:
        if d.get('gid_source') == 'annotation':
            d['gid_source'] = None
            d['global_id'] = None
    return detections


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

def draw_cross_view_associations(img, associations, view_id):
    """Draw projected/native point pairs for cross-view matches."""
    if img is None or not associations:
        return img

    for assoc in associations:
        gid = assoc.get('global_id')
        if gid is None:
            continue
        col = (gid * 37 % 255, gid * 17 % 255, gid * 97 % 255)

        if view_id == 1:
            native = assoc.get('native_view1')
            projected = assoc.get('projected_view1')
        else:
            native = assoc.get('native_view2')
            projected = assoc.get('projected_view2')

        native_valid = native is not None and np.all(np.isfinite(native))
        proj_valid = projected is not None and np.all(np.isfinite(projected))

        if native_valid:
            nx, ny = int(round(float(native[0]))), int(round(float(native[1])))
            cv.circle(img, (nx, ny), 4, col, -1)
            cv.circle(img, (nx, ny), 5, (255, 255, 255), 1)
        if proj_valid:
            px, py = int(round(float(projected[0]))), int(round(float(projected[1])))
            cv.drawMarker(
                img, (px, py), col,
                markerType=cv.MARKER_TILTED_CROSS,
                markerSize=10,
                thickness=2
            )
            cv.putText(
                img, f'P{gid}', (px + 6, py - 6),
                cv.FONT_HERSHEY_SIMPLEX, 0.4, col, 1
            )
        if native_valid and proj_valid:
            cv.line(img, (nx, ny), (px, py), col, 1)

    return img


def _build_association_lookup(associations):
    """Build per-track lookups for event logging in each view."""
    by_view1_track = {}
    by_view2_track = {}
    for assoc in associations or []:
        tid1 = assoc.get('track_id_view1')
        tid2 = assoc.get('track_id_view2')
        dist = assoc.get('distance_px')
        if tid1 is not None:
            by_view1_track[tid1] = {
                'other_track_id': tid2,
                'projected_other_pt': assoc.get('projected_view2'),
                'native_other_pt': assoc.get('native_view2'),
                'distance_px': dist,
            }
        if tid2 is not None:
            by_view2_track[tid2] = {
                'other_track_id': tid1,
                'projected_other_pt': assoc.get('projected_view1'),
                'native_other_pt': assoc.get('native_view1'),
                'distance_px': dist,
            }
    return by_view1_track, by_view2_track


def _compute_pose_attempt_metrics(detections):
    attempts = []
    for d in detections or []:
        v = d.get('pose_attempts_used', 0)
        try:
            v = int(v)
        except (TypeError, ValueError):
            v = 0
        attempts.append(max(0, v))

    max_attempts = max(attempts) if attempts else 0
    return {
        'pose_ran_second_attempt': int(any(v > 1 for v in attempts)),
        'pose_ran_more_than_second_attempt': int(any(v > 2 for v in attempts)),
        'pose_max_attempts_used': int(max_attempts),
    }


def _augment_event_with_pose_attempts(event, pose_attempt_metrics):
    event['pose_ran_second_attempt'] = int(pose_attempt_metrics.get('pose_ran_second_attempt', 0))
    event['pose_ran_more_than_second_attempt'] = int(
        pose_attempt_metrics.get('pose_ran_more_than_second_attempt', 0)
    )
    event['pose_max_attempts_used'] = int(pose_attempt_metrics.get('pose_max_attempts_used', 0))
    return event


def _augment_event_with_view_and_projected_pos(event, tracker, homography_to_other, association_info_by_track=None):
    """Attach own-view bottom center and projected point in the other view."""
    event['view_x'] = ''
    event['view_y'] = ''
    event['view_pos_json'] = ''
    event['has_view_pos'] = 0
    event['projected_other_x'] = ''
    event['projected_other_y'] = ''
    event['projected_other_pos_json'] = ''
    event['has_projected_other_pos'] = 0
    event['matched_with_other_view_track_id'] = ''
    event['assoc_projected_other_x'] = ''
    event['assoc_projected_other_y'] = ''
    event['assoc_projected_other_pos_json'] = ''
    event['has_assoc_projected_other_pos'] = 0
    event['assoc_native_other_x'] = ''
    event['assoc_native_other_y'] = ''
    event['assoc_native_other_pos_json'] = ''
    event['has_assoc_native_other_pos'] = 0
    event['assoc_match_dist_px'] = ''

    if tracker is None:
        return event

    tid = event.get('track_id')
    if tid is None:
        return event

    assoc = (association_info_by_track or {}).get(tid)
    if assoc is not None:
        if assoc.get('other_track_id') is not None:
            event['matched_with_other_view_track_id'] = int(assoc['other_track_id'])
        proj_other = assoc.get('projected_other_pt')
        if proj_other is not None and np.all(np.isfinite(proj_other)):
            px, py = float(proj_other[0]), float(proj_other[1])
            event['assoc_projected_other_x'] = px
            event['assoc_projected_other_y'] = py
            event['assoc_projected_other_pos_json'] = json.dumps([px, py])
            event['has_assoc_projected_other_pos'] = 1
        native_other = assoc.get('native_other_pt')
        if native_other is not None and np.all(np.isfinite(native_other)):
            nx, ny = float(native_other[0]), float(native_other[1])
            event['assoc_native_other_x'] = nx
            event['assoc_native_other_y'] = ny
            event['assoc_native_other_pos_json'] = json.dumps([nx, ny])
            event['has_assoc_native_other_pos'] = 1
        if assoc.get('distance_px') is not None:
            event['assoc_match_dist_px'] = float(assoc['distance_px'])

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
        bbox = d['bbox']
        bottom_center = np.array([(bbox[0] + bbox[2]) / 2.0, bbox[3]], dtype=np.float32)
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
        if d.get('pose_attempts_used') is None:
            kpts, vec, attempts_used = process_detection_pose(
                frame, d['bbox'], pose_model, pose_attempts,
                pose_conf_threshold, pose_iou_threshold
            )
            d['keypoints'] = kpts
            d['pose_vec'] = vec
            d['pose_attempts_used'] = int(attempts_used)
        elif d.get('pose_attempts_used') is not None:
            try:
                d['pose_attempts_used'] = int(d['pose_attempts_used'])
            except (TypeError, ValueError):
                d['pose_attempts_used'] = 0
        # Force bbox-bottom-center anchoring for world coordinates.
        d['world_pos'] = tracker.project_to_world(d['bbox']) if tracker else None

    return detections


def merge_track_global_id(tracker, track, target_gid, frame_id, allow_target_conflict=False):
    if track['global_id'] == target_gid:
        return True
    if hasattr(tracker, 'reassign_track_global_id'):
        return bool(
            tracker.reassign_track_global_id(
                track, target_gid, frame_id, allow_target_conflict=allow_target_conflict
            )
        )
    tracker.merge_global_ids(track['global_id'], target_gid, frame_id)
    return track.get('global_id') == target_gid


def _bbox_bottom_center(bbox):
    return np.array([(bbox[0] + bbox[2]) / 2.0, bbox[3]], dtype=np.float32)


def associate_tracks_to_view1(
    tracker1, tracker_other, frame_id,
    homography_anchor_to_other,
    homography_other_to_anchor=None,
    max_projected_dist_px=120.0,
    sticky_gid_by_other_track=None,
):
    # Consider every non-lost track so cross-view GID assignment can run every frame.
    max_missed_1 = int(getattr(tracker1, 'max_missed', 0))
    max_missed_other = int(getattr(tracker_other, 'max_missed', 0))
    active1 = [
        t for t in tracker1.tracks
        if t.get('bbox') is not None and t.get('missed', 0) <= max_missed_1
    ]
    active_other = [
        t for t in tracker_other.tracks
        if t.get('bbox') is not None and t.get('missed', 0) <= max_missed_other
    ]
    if not active1 or not active_other:
        return []

    active1 = sorted(active1, key=lambda t: t.get('track_id', 10**9))
    active_other = sorted(active_other, key=lambda t: t.get('track_id', 10**9))

    projected_to_other = []
    native_view1 = []
    for t1 in active1:
        native_pt = _bbox_bottom_center(t1['bbox'])
        native_view1.append(native_pt)
        projected_to_other.append(project_point_with_homography(native_pt, homography_anchor_to_other))
    native_view2 = [_bbox_bottom_center(t2['bbox']) for t2 in active_other]

    n1 = len(active1)
    n2 = len(active_other)
    very_large = float(max_projected_dist_px) + 1e6
    cost = np.full((n1, n2), very_large, dtype=np.float32)

    for i, p_proj in enumerate(projected_to_other):
        if p_proj is None or not np.all(np.isfinite(p_proj)):
            continue
        gid1 = active1[i].get('global_id')
        for j, p_other in enumerate(native_view2):
            if p_other is None or not np.all(np.isfinite(p_other)):
                continue
            other_tid = active_other[j].get('track_id')
            if sticky_gid_by_other_track is not None and other_tid is not None:
                locked_gid = sticky_gid_by_other_track.get(other_tid)
                if locked_gid is not None and locked_gid != gid1:
                    continue
            dist = float(np.linalg.norm(p_proj - p_other))
            if dist <= max_projected_dist_px:
                cost[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(cost)
    associations = []
    for r, c in zip(row_ind, col_ind):
        dist = float(cost[r, c])
        if dist > max_projected_dist_px:
            continue

        t1 = active1[r]
        t2 = active_other[c]
        gid1 = t1['global_id']
        other_tid = t2.get('track_id')

        target_gid = gid1
        merged1 = merge_track_global_id(
            tracker1, t1, target_gid, frame_id, allow_target_conflict=True
        )
        merged2 = merge_track_global_id(
            tracker_other, t2, target_gid, frame_id, allow_target_conflict=True
        )
        if not (merged1 and merged2):
            continue

        if sticky_gid_by_other_track is not None and other_tid is not None:
            sticky_gid_by_other_track[other_tid] = target_gid
        if t1.get('gid_source') == 'annotation':
            t2['gid_source'] = 'annotation'

        p2 = projected_to_other[r]
        n1_pt = native_view1[r]
        n2_pt = native_view2[c]
        p1 = None
        if homography_other_to_anchor is not None and n2_pt is not None and np.all(np.isfinite(n2_pt)):
            p1 = project_point_with_homography(n2_pt, homography_other_to_anchor)

        associations.append({
            'track_id_view1': t1.get('track_id'),
            'track_id_view2': t2.get('track_id'),
            'global_id': target_gid,
            'native_view1': n1_pt,
            'projected_view1': p1,
            'native_view2': n2_pt,
            'projected_view2': p2,
            'distance_px': dist,
        })
    return associations


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
    enable_pose=True,
    size=(1440, 810),  # Processing resolution (display uses same max size).
    # detection View 1
    sahi_conf_threshold1=0.43,
    sahi_iou_threshold1=0.37,
    slice_h1=480,
    slice_w1=480,
    slice_overlap1=0.3,
    # pose View 1
    pose_conf_threshold1=0.03,
    pose_iou_threshold1=0.005,
    # tracker View 1
    match_threshold1=0.45,
    iou_weight1=0.55,
    appearance_weight1=0.35,
    pose_weight1=0.25,
    ema_alpha1=0.5,
    max_velocity1=200.0,
    nested_track_contain_thresh1=0.9,
    nested_track_area_ratio1=0.6,
    nested_track_app_sim1=0.7,
    cross_view_max_dist_px1=120.0,
    max_missed_frames1=80,
    # detection View 2
    sahi_conf_threshold2=0.55,
    sahi_iou_threshold2=0.35,
    slice_h2=640,
    slice_w2=640,
    slice_overlap2=0.15,
    # pose View 2
    pose_conf_threshold2=0.10,
    pose_iou_threshold2=0.02,
    pose_attempts2=(
        {'pad': 0.0, 'conf': None, 'iou': None},
        {'pad': 0.25, 'conf': 0.01, 'iou': 0.005},
    ),
    # tracker View 2
    match_threshold2=0.45,
    iou_weight2=0.55,
    appearance_weight2=0.35,
    pose_weight2=0.05,
    ema_alpha2=0.5,
    max_velocity2=300.0,
    nested_track_contain_thresh2=0.9,
    nested_track_area_ratio2=0.6,
    nested_track_app_sim2=0.7,
    max_missed_frames2=80,
    homography_path='homography_cam4_to_cam13.json',
    nested_contain_thresh=0.9,
    nested_area_ratio=0.68,
    gid_pool_size=None,
    gid_reuse_cooldown_frames=0,
    sticky_cross_view_gid_lock=False,
    fixed_slots_mode=False,
    fixed_slots_count=None,
    annotation_gid_first_frame_only=True,
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

    # Load annotations early so fixed-slot initialization can infer roster size.
    annotations1 = load_annotations_from_dir(annotations_dir1, video_prefix1)
    annotations2 = load_annotations_from_dir(annotations_dir2, video_prefix2)
    initial_annotation_frame1 = first_annotation_frame(annotations1)
    initial_annotation_frame2 = first_annotation_frame(annotations2)
    if annotation_gid_first_frame_only:
        print(
            f"Annotation GID init-only mode: view1 first frame={initial_annotation_frame1}, "
            f"view2 first frame={initial_annotation_frame2}"
        )

    if fixed_slots_mode:
        slot_count = fixed_slots_count
        if slot_count is None:
            first_frame1, inferred1 = infer_slot_count_from_first_annotation_frame(annotations1)
            if inferred1 is not None:
                slot_count = inferred1
                print(f"Fixed slots inferred from view1 frame {first_frame1}: {slot_count}")
            else:
                first_frame2, inferred2 = infer_slot_count_from_first_annotation_frame(annotations2)
                if inferred2 is not None:
                    slot_count = inferred2
                    print(f"Fixed slots inferred from view2 frame {first_frame2}: {slot_count}")
        if slot_count is None or int(slot_count) <= 0:
            raise ValueError(
                "fixed_slots_mode=True requires fixed_slots_count or valid first-frame annotations."
            )
        fixed_slots_count = int(slot_count)
        print(f"Fixed slots mode enabled: {fixed_slots_count} slots per view")
    else:
        fixed_slots_count = None

    if gid_pool_size is None:
        # Unlimited GID mode: do not use fixed-pool allocation/cooldown.
        shared_id_manager = GlobalIDManager()
    else:
        shared_id_manager = GlobalIDManager(
            pool_size=gid_pool_size,
            cooldown_frames=max(0, int(gid_reuse_cooldown_frames)),
        )
    sticky_gid_view2 = {} if sticky_cross_view_gid_lock else None

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
        max_tracks=fixed_slots_count,
        is_rectified=True,  # FIX: Videos are rectified
        prune_tracks=not fixed_slots_mode,
        fixed_slots_mode=fixed_slots_mode,
        # Nested suppression is intentionally done only at detection stage (pre-tracker update).
        suppress_nested_tracks=False,
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
        max_tracks=fixed_slots_count,
        is_rectified=True,  # FIX: Videos are rectified
        prune_tracks=not fixed_slots_mode,
        fixed_slots_mode=fixed_slots_mode,
        # Nested suppression is intentionally done only at detection stage (pre-tracker update).
        suppress_nested_tracks=False,
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
    csv_writer1 = csv.DictWriter(
        csv_file1,
        fieldnames=[
            'frame', 'track_id', 'event', 'view', 'global_id',
            'matched_with_other_view_track_id', 'new_global_id', 'old_global_id',
            'swapped_with_track_id',
            'detections_in_frame',
            'pose_ran_second_attempt', 'pose_ran_more_than_second_attempt', 'pose_max_attempts_used',
            'view_x', 'view_y', 'view_pos_json', 'has_view_pos',
            'projected_other_x', 'projected_other_y', 'projected_other_pos_json', 'has_projected_other_pos',
            'assoc_projected_other_x', 'assoc_projected_other_y', 'assoc_projected_other_pos_json', 'has_assoc_projected_other_pos',
            'assoc_native_other_x', 'assoc_native_other_y', 'assoc_native_other_pos_json', 'has_assoc_native_other_pos',
            'assoc_match_dist_px',
        ]
    )
    csv_writer1.writeheader()

    csv_file2 = open(output_csv_path2, 'w', newline='')
    csv_writer2 = csv.DictWriter(
        csv_file2,
        fieldnames=[
            'frame', 'track_id', 'event', 'view', 'global_id',
            'matched_with_other_view_track_id', 'new_global_id', 'old_global_id',
            'swapped_with_track_id',
            'detections_in_frame',
            'pose_ran_second_attempt', 'pose_ran_more_than_second_attempt', 'pose_max_attempts_used',
            'view_x', 'view_y', 'view_pos_json', 'has_view_pos',
            'projected_other_x', 'projected_other_y', 'projected_other_pos_json', 'has_projected_other_pos',
            'assoc_projected_other_x', 'assoc_projected_other_y', 'assoc_projected_other_pos_json', 'has_assoc_projected_other_pos',
            'assoc_native_other_x', 'assoc_native_other_y', 'assoc_native_other_pos_json', 'has_assoc_native_other_pos',
            'assoc_match_dist_px',
        ]
    )
    csv_writer2.writeheader()


    frame_id = 0
    fps_hist = deque(maxlen=30)

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

    # View 1: force a single pose attempt using the former fallback settings.
    pose_attempts_view1_runtime = (
        {'pad': 0.25, 'conf': pose_conf_threshold1, 'iou': pose_iou_threshold1},
    )

    initial_detections1 = detect_view(
        frame1, frame_id, annotations1, det_model,
        sahi_conf_threshold1, sahi_iou_threshold1, slice_h1, slice_w1,
        slice_overlap1, pose_model, pose_attempts_view1_runtime, pose_conf_threshold1,
        pose_iou_threshold1, tracker1,
        suppress_nested=True,
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
        suppress_nested=True,
        nested_contain_thresh=nested_contain_thresh,
        nested_area_ratio=nested_area_ratio
    )
    if annotation_gid_first_frame_only:
        initial_detections1 = keep_annotation_gids_only_on_initial_frame(
            initial_detections1, frame_id, initial_annotation_frame1
        )
        initial_detections2 = keep_annotation_gids_only_on_initial_frame(
            initial_detections2, frame_id, initial_annotation_frame2
        )

    # Update trackers with initial detections
    tracks1 = tracker1.update(
        initial_detections1, frame_id, finalize=True,
        resolve_conflicts=((not fixed_slots_mode) and (frame_id not in annotations1))
    )[0]
    tracks2 = tracker2.update(
        initial_detections2, frame_id, finalize=True,
        resolve_conflicts=((not fixed_slots_mode) and (frame_id not in annotations2))
    )[0]
    if sticky_gid_view2 is not None:
        active_view2_tids = {t.get('track_id') for t in tracker2.tracks}
        stale_keys = [tid for tid in sticky_gid_view2.keys() if tid not in active_view2_tids]
        for tid in stale_keys:
            del sticky_gid_view2[tid]
    # Associate tracks across views (one-way: View 1 -> View 2)
    cross_view_associations = associate_tracks_to_view1(
        tracker1, tracker2, frame_id,
        homography_anchor_to_other=h_view1_to_view2,
        homography_other_to_anchor=h_view2_to_view1,
        max_projected_dist_px=cross_view_max_dist_px1,
        sticky_gid_by_other_track=sticky_gid_view2,
    )
    assoc_lookup_view1, assoc_lookup_view2 = _build_association_lookup(cross_view_associations)
    pose_metrics_view1 = _compute_pose_attempt_metrics(initial_detections1)
    pose_metrics_view2 = _compute_pose_attempt_metrics(initial_detections2)

    # Log initial events
    for event in tracker1.get_frame_events():
        event['view'] = 1
        event['detections_in_frame'] = len(initial_detections1)
        _augment_event_with_pose_attempts(event, pose_metrics_view1)
        _augment_event_with_view_and_projected_pos(
            event, tracker1, h_view1_to_view2, association_info_by_track=assoc_lookup_view1
        )
        csv_writer1.writerow(event)
    for event in tracker2.get_frame_events():
        event['view'] = 2
        event['detections_in_frame'] = len(initial_detections2)
        _augment_event_with_pose_attempts(event, pose_metrics_view2)
        _augment_event_with_view_and_projected_pos(
            event, tracker2, h_view2_to_view1, association_info_by_track=assoc_lookup_view2
        )
        csv_writer2.writerow(event)

    # Save first frame
    roi1 = frame1
    roi2 = frame2
    out1 = draw_tracks(roi1.copy(), tracks1, None, connections, tracker1)
    out2 = draw_tracks(roi2.copy(), tracks2, None, connections, tracker2)
    out1 = draw_cross_view_associations(out1, cross_view_associations, view_id=1)
    out2 = draw_cross_view_associations(out2, cross_view_associations, view_id=2)
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
            slice_overlap1, pose_model, pose_attempts_view1_runtime, pose_conf_threshold1,
            pose_iou_threshold1, tracker1,
            suppress_nested=True,
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
            suppress_nested=True,
            nested_contain_thresh=nested_contain_thresh,
            nested_area_ratio=nested_area_ratio
        )
        if annotation_gid_first_frame_only:
            detections1 = keep_annotation_gids_only_on_initial_frame(
                detections1, frame_id, initial_annotation_frame1
            )
            detections2 = keep_annotation_gids_only_on_initial_frame(
                detections2, frame_id, initial_annotation_frame2
            )
        # ===== STEP 4: UPDATE TRACKERS (finalize=True) =====
        tracks1 = tracker1.update(
            detections1, frame_id, finalize=True,
            resolve_conflicts=((not fixed_slots_mode) and (frame_id not in annotations1))
        )[0]
        tracks2 = tracker2.update(
            detections2, frame_id, finalize=True,
            resolve_conflicts=((not fixed_slots_mode) and (frame_id not in annotations2))
        )[0]
        if sticky_gid_view2 is not None:
            active_view2_tids = {t.get('track_id') for t in tracker2.tracks}
            stale_keys = [tid for tid in sticky_gid_view2.keys() if tid not in active_view2_tids]
            for tid in stale_keys:
                del sticky_gid_view2[tid]
        # ===== STEP 5: CROSS-VIEW ASSOCIATION (ONE-WAY: VIEW 1 -> VIEW 2) =====
        cross_view_associations = associate_tracks_to_view1(
            tracker1, tracker2, frame_id,
            homography_anchor_to_other=h_view1_to_view2,
            homography_other_to_anchor=h_view2_to_view1,
            max_projected_dist_px=cross_view_max_dist_px1,
            sticky_gid_by_other_track=sticky_gid_view2,
        )
        assoc_lookup_view1, assoc_lookup_view2 = _build_association_lookup(cross_view_associations)
        pose_metrics_view1 = _compute_pose_attempt_metrics(detections1)
        pose_metrics_view2 = _compute_pose_attempt_metrics(detections2)

        # ===== LOG EVENTS =====
        for event in tracker1.get_frame_events():
            if 'view' not in event:
                event['view'] = 1
            event['detections_in_frame'] = len(detections1)
            _augment_event_with_pose_attempts(event, pose_metrics_view1)
            _augment_event_with_view_and_projected_pos(
                event, tracker1, h_view1_to_view2, association_info_by_track=assoc_lookup_view1
            )
            csv_writer1.writerow(event)
        
        for event in tracker2.get_frame_events():
            if 'view' not in event:
                event['view'] = 2
            event['detections_in_frame'] = len(detections2)
            _augment_event_with_pose_attempts(event, pose_metrics_view2)
            _augment_event_with_view_and_projected_pos(
                event, tracker2, h_view2_to_view1, association_info_by_track=assoc_lookup_view2
            )
            csv_writer2.writerow(event)

        # Refresh tracks after all operations
        tracks1 = tracker1.tracks
        tracks2 = tracker2.tracks

        # ===== DRAWING =====
        roi1 = frame1
        roi2 = frame2
        out1 = draw_tracks(roi1.copy(), tracks1, None, connections, tracker1)
        out2 = draw_tracks(roi2.copy(), tracks2, None, connections, tracker2)
        out1 = draw_cross_view_associations(out1, cross_view_associations, view_id=1)
        out2 = draw_cross_view_associations(out2, cross_view_associations, view_id=2)

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
            output_path1='output_view1.mp4',
            output_path2='output_view2.mp4',
            enable_pose=True,
            fixed_slots_mode=True,
            court_lines_path1='court_lines_cam13.json',
            court_corners_path1='cam13_img_corners_rectified.json',
        )
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
