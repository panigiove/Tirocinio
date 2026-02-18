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


def process_detection_pose(frame, bbox, pose_model, pose_conf_threshold, pose_iou_threshold):
    """
    Run a single pose estimation pass on the detection crop.
    """
    if pose_model is None:
        return None, None, 0

    pb = pad_bbox(bbox, 0.0, frame.shape)
    px1, py1, px2, py2 = pb
    crop = frame[py1:py2, px1:px2]
    if crop.size <= 0:
        return None, None, 1

    res = pose_model.predict(
        crop,
        conf=pose_conf_threshold,
        iou=pose_iou_threshold,
        imgsz=round_to_multiple(max(crop.shape[:2]), 32),
        device='cuda',
        verbose=False
    )
    if res and len(res[0].keypoints.xy) > 0:
        k = res[0].keypoints.xy.cpu().numpy()[0]
        pose_kpts = k + np.array([px1, py1])
        pose_vec = keypoints_to_pose_vec(pose_kpts, bbox)
        return pose_kpts, pose_vec, 1

    return None, None, 1


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


def parse_yolo_annotations(anno_file, img_width, img_height, frame, pose_model,
                          pose_conf_threshold, pose_iou_threshold, world_homography=None):
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
                    frame, bbox, pose_model,
                    pose_conf_threshold, pose_iou_threshold
                )
                world_pos = _bbox_to_reference_plane(bbox, world_homography)

                detections.append({
                    'bbox': bbox,
                    'appearance': appearance,
                    'keypoints': pose_kpts,
                    'pose_vec': pose_vec,
                    'pose_attempts_used': int(pose_attempts_used),
                    'world_pos': world_pos,
                    'score': 1.0,
                    'global_id': class_id,
                })
    except Exception as e:
        print(f"Error parsing annotations from {anno_file}: {e}")
    
    return detections


def load_annotations_from_dir(annotations_dir, video_prefix):
    """Load annotation files from a directory, keeping only the earliest frame."""
    if not annotations_dir or not os.path.exists(annotations_dir):
        return {}

    annotations = {}
    for file in os.listdir(annotations_dir):
        if file.startswith(video_prefix + '_') and file.endswith('_rectified.txt'):
            match = re.search(r'frame_(\d+)', file)
            if match:
                frame_id = int(match.group(1))
                anno_path = os.path.join(annotations_dir, file)
                if frame_id not in annotations:
                    annotations[frame_id] = []
                annotations[frame_id].append(anno_path)

    if not annotations:
        return {}

    first_frame = min(annotations.keys())
    dropped_frames = [fid for fid in sorted(annotations.keys()) if fid != first_frame]
    if dropped_frames:
        print(
            f"Keeping only first annotation frame {first_frame}; "
            f"ignoring later annotation frames {dropped_frames}"
        )
    if first_frame != 0:
        print(
            f"Remapping annotation frame {first_frame} to processing frame 0 "
            f"for {video_prefix}"
        )
    return {0: annotations[first_frame]}


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


def _extract_yolo_gids(anno_files):
    gids = set()
    for anno_path in anno_files or []:
        if not anno_path or not os.path.exists(anno_path):
            continue
        try:
            with open(anno_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        gids.add(int(parts[0]))
                    except (TypeError, ValueError):
                        continue
        except Exception:
            continue
    return gids


def infer_fixed_roster_from_first_annotation_frame(annotations):
    if not annotations:
        return None, None, []
    first_frame = min(annotations.keys())
    anno_files = annotations.get(first_frame, [])
    count = _count_valid_yolo_boxes(anno_files)
    gids = sorted(_extract_yolo_gids(anno_files))
    if count <= 0 or not gids:
        return first_frame, None, []
    return first_frame, int(count), gids


def draw_tracks(img, tracks, connections, tracker_self):
    """Draw tracks, keypoints and cross-view projections on a frame."""
    valid_tracks = [t for t in tracks if t['missed'] == 0]
    
    valid_tracks.sort(key=lambda x: x['global_id'])
    
    # Draw tracks
    for t in valid_tracks:
        x1, y1, x2, y2 = map(int, t['bbox'])
        gid = t['global_id']
        col = (gid * 37 % 255, gid * 17 % 255, gid * 97 % 255)
        
        thickness = 2
        cv.rectangle(img, (x1, y1), (x2, y2), col, thickness)
        
        # Label
        cx = (x1 + x2) // 2
        cy = y1
        label = f'GID:{gid}'
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


def _draw_projected_bottom_centers(
    canvas,
    items,
    homography_to_canvas,
    view_tag,
    marker_type='circle',
):
    """Draw projected bbox bottom centers for one view on a destination canvas."""
    if canvas is None or items is None:
        return 0

    h, w = canvas.shape[:2]
    drawn = 0

    for item in items:
        if not isinstance(item, dict):
            continue
        bbox = item.get('bbox')
        if bbox is None:
            continue
        # Skip stale track slots if this is a track list.
        if int(item.get('missed', 0)) > 0:
            continue

        pt = _bbox_bottom_center(bbox)
        if homography_to_canvas is not None:
            pt = project_point_with_homography(pt, homography_to_canvas)
        if pt is None or not np.all(np.isfinite(pt)):
            continue

        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        gid = item.get('global_id')
        if gid is None:
            col = (0, 200, 255) if view_tag == 'V1' else (0, 255, 0)
        else:
            col = (
                int(gid * 37 % 255),
                int(gid * 17 % 255),
                int(gid * 97 % 255),
            )

        if marker_type == 'cross':
            cv.drawMarker(
                canvas, (x, y), col,
                markerType=cv.MARKER_TILTED_CROSS,
                markerSize=12,
                thickness=2
            )
        else:
            cv.circle(canvas, (x, y), 5, col, -1)
            cv.circle(canvas, (x, y), 6, (255, 255, 255), 1)

        label_id = gid if gid is not None else item.get('track_id')
        if label_id is not None:
            cv.putText(
                canvas, f'{view_tag}:{label_id}', (x + 6, y - 6),
                cv.FONT_HERSHEY_SIMPLEX, 0.45, col, 1
            )
        drawn += 1

    return drawn


def _collect_projected_bottom_centers_by_gid(items, homography_to_canvas):
    """Collect projected bbox bottom centers grouped by global ID."""
    points_by_gid = {}
    if items is None:
        return points_by_gid

    for item in items:
        if not isinstance(item, dict):
            continue
        if int(item.get('missed', 0)) > 0:
            continue

        gid = item.get('global_id')
        bbox = item.get('bbox')
        if gid is None or bbox is None:
            continue

        pt = _bbox_bottom_center(bbox)
        if homography_to_canvas is not None:
            pt = project_point_with_homography(pt, homography_to_canvas)
        if pt is None or not np.all(np.isfinite(pt)):
            continue

        try:
            gid = int(gid)
        except (TypeError, ValueError):
            continue
        points_by_gid.setdefault(gid, []).append(
            np.array([float(pt[0]), float(pt[1])], dtype=np.float32)
        )

    return points_by_gid


def render_court_projection_frame(
    court_template_img,
    view1_items,
    view2_items,
    homography_view1_to_court,
    homography_view2_to_court,
    frame_id=None,
):
    """
    Draw only shared tracks on the court template.

    For each GID visible in both views, draw the midpoint between the two projected
    bottom-center points (one from each view).
    """
    if court_template_img is None:
        return None

    out = court_template_img.copy()
    h, w = out.shape[:2]

    view1_points = _collect_projected_bottom_centers_by_gid(
        view1_items, homography_view1_to_court
    )
    view2_points = _collect_projected_bottom_centers_by_gid(
        view2_items, homography_view2_to_court
    )

    shared_gids = sorted(set(view1_points.keys()) & set(view2_points.keys()))
    midpoint_count = 0

    for gid in shared_gids:
        p1 = np.mean(np.asarray(view1_points[gid], dtype=np.float32), axis=0)
        p2 = np.mean(np.asarray(view2_points[gid], dtype=np.float32), axis=0)
        midpoint = 0.5 * (p1 + p2)
        if midpoint is None or not np.all(np.isfinite(midpoint)):
            continue

        x = int(round(float(midpoint[0])))
        y = int(round(float(midpoint[1])))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        col = (
            int(gid * 37 % 255),
            int(gid * 17 % 255),
            int(gid * 97 % 255),
        )
        cv.circle(out, (x, y), 6, col, -1)
        cv.circle(out, (x, y), 7, (255, 255, 255), 1)
        cv.putText(
            out, f'GID:{gid}', (x + 7, y - 7),
            cv.FONT_HERSHEY_SIMPLEX, 0.45, col, 1
        )
        midpoint_count += 1

    if frame_id is not None:
        draw_text_with_bg(out, f'Frame {int(frame_id)}', (20, 35))
    draw_text_with_bg(out, f'Shared tracks: {midpoint_count}', (20, 70), scale=0.55)
    return out


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


def _augment_event_with_view_and_projected_pos(
    event,
    tracker,
    homography_to_other,
    association_info_by_track=None,
    track_by_id=None,
):
    """Attach own-view bottom center and projected point in the other view."""
    event['view_x'] = ''
    event['view_y'] = ''
    event['view_pos_json'] = ''
    event['projected_other_x'] = ''
    event['projected_other_y'] = ''
    event['projected_other_pos_json'] = ''
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

    track = None
    if track_by_id is not None:
        track = track_by_id.get(tid)
    if track is None:
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

        if homography_to_other is not None:
            proj_pt = project_point_with_homography(view_pt, homography_to_other)
            if np.all(np.isfinite(proj_pt)):
                px, py = float(proj_pt[0]), float(proj_pt[1])
                event['projected_other_x'] = px
                event['projected_other_y'] = py
                event['projected_other_pos_json'] = json.dumps([px, py])
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


def _suppress_nested_detections(
    detections,
    overlap_thresh=0.9,
):
    n = len(detections)
    if n < 2:
        return detections
    keep = [True] * n
    bboxes = [d['bbox'] for d in detections]
    areas = [_bbox_area(b) for b in bboxes]
    for i in range(n):
        if not keep[i]:
            continue
        ai = areas[i]
        if ai <= 0:
            continue
        bi = bboxes[i]
        for j in range(n):
            if i == j or not keep[j]:
                continue
            aj = areas[j]
            if aj <= 0:
                continue
            # Consider j nested in i
            if aj >= ai:
                continue
            inter = _bbox_intersection(bi, bboxes[j])
            overlap = inter / aj
            if overlap >= overlap_thresh:
                keep[j] = False
    return [d for d, k in zip(detections, keep) if k]


def load_homography_matrix(homography_path, matrix_key='H'):
    """Load a single 3x3 homography matrix from JSON."""
    if not homography_path or not os.path.exists(homography_path):
        raise FileNotFoundError(f"Homography file not found: {homography_path}")

    with open(homography_path, 'r') as f:
        data = json.load(f)

    matrix = None
    if isinstance(data, dict):
        if matrix_key in data:
            matrix = data[matrix_key]
        elif 'H' in data:
            matrix = data['H']
        elif 'homography' in data:
            matrix = data['homography']
        elif 'matrix' in data:
            matrix = data['matrix']
    elif isinstance(data, list):
        matrix = data

    if matrix is None:
        raise ValueError(
            f"Homography file {homography_path} does not contain a matrix key "
            f"('{matrix_key}', 'H', 'homography', or 'matrix')."
        )

    h = np.array(matrix, dtype=np.float32)
    if h.shape != (3, 3):
        raise ValueError(f"Invalid homography shape in {homography_path}: {h.shape}")
    return h


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
                slice_overlap, pose_model, pose_conf_threshold,
                pose_iou_threshold, world_homography=None,
                nested_overlap_thresh=0.9,
                invalid_if_above_lines=None, invalid_ref_pt=None):
    detections = []
    frame_h, frame_w = frame.shape[:2]
    using_annotations = frame_id in annotations
    if using_annotations:
        for anno_file in annotations[frame_id]:
            detections.extend(parse_yolo_annotations(
                anno_file, frame_w, frame_h, frame, pose_model,
                pose_conf_threshold, pose_iou_threshold, world_homography=world_homography
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
                    detections.append({
                        'bbox': bbox,
                        'appearance': appearance,
                        'score': p.score.value
                    })

    if (not using_annotations) and invalid_if_above_lines:
        detections = _filter_detections_above_lines(
            detections, invalid_if_above_lines, invalid_ref_pt
        )


    if not using_annotations:
        detections = _suppress_nested_detections(
            detections,
            overlap_thresh=nested_overlap_thresh,
        )


    for d in detections:
        if d.get('pose_attempts_used') is None:
            kpts, vec, attempts_used = process_detection_pose(
                frame, d['bbox'], pose_model,
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
        # Map to a shared reference plane (or keep native bottom-center if no homography).
        if d.get('world_pos') is None:
            d['world_pos'] = _bbox_to_reference_plane(d['bbox'], world_homography)

    return detections


def merge_track_global_id(tracker, track, target_gid, frame_id):
    if track['global_id'] == target_gid:
        return True
    if not hasattr(tracker, 'reassign_track_global_id'):
        return False
    return bool(
        tracker.reassign_track_global_id(
            track, target_gid, frame_id
        )
    )


def _bbox_bottom_center(bbox):
    return np.array([(bbox[0] + bbox[2]) / 2.0, bbox[3]], dtype=np.float32)


def _bbox_to_reference_plane(bbox, homography_to_ref=None):
    pt = _bbox_bottom_center(bbox)
    if homography_to_ref is not None:
        pt = project_point_with_homography(pt, homography_to_ref)
    if pt is None or not np.all(np.isfinite(pt)):
        return None
    return np.array([float(pt[0]), float(pt[1]), 0.0], dtype=np.float32)


def inject_reseed_hints_between_views(
    detections_target,
    tracker_source,
    tracker_target,
    homography_source_to_target,
    max_projected_dist_px=35.0,
    skip_gids_already_active_in_target=True,
):
    """
    Inject GID hints into target detections using projected active tracks from source.
    The hints are consumed only if a detection remains unmatched inside tracker2.update.
    """
    if (
        not detections_target
        or tracker_source is None
        or tracker_target is None
        or homography_source_to_target is None
    ):
        return 0

    det_indices = [
        j for j, d in enumerate(detections_target)
        if d.get('bbox') is not None and d.get('global_id') is None
    ]
    if not det_indices:
        return 0

    active_target_gids = set()
    if skip_gids_already_active_in_target:
        active_target_gids = {
            int(t.get('global_id'))
            for t in tracker_target.tracks
            if t.get('missed', 0) == 0 and t.get('global_id') is not None
        }

    source_points = []
    source_gids = []
    for ts in tracker_source.tracks:
        if ts.get('missed', 0) != 0 or ts.get('bbox') is None:
            continue
        gid = ts.get('global_id')
        if gid is None:
            continue
        gid = int(gid)
        if gid in active_target_gids:
            continue

        p1 = _bbox_bottom_center(ts['bbox'])
        p2 = project_point_with_homography(p1, homography_source_to_target)
        if p2 is None or not np.all(np.isfinite(p2)):
            continue

        source_points.append(p2)
        source_gids.append(gid)

    if not source_points:
        return 0

    n_s = len(source_points)
    n_d = len(det_indices)
    very_large = float(max_projected_dist_px) + 1e6
    cost = np.full((n_s, n_d), very_large, dtype=np.float32)

    for i, proj_pt in enumerate(source_points):
        for j, det_idx in enumerate(det_indices):
            det_pt = _bbox_bottom_center(detections_target[det_idx]['bbox'])
            if det_pt is None or not np.all(np.isfinite(det_pt)):
                continue
            dist = float(np.linalg.norm(proj_pt - det_pt))
            if dist <= max_projected_dist_px:
                cost[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(cost)
    assigned = 0
    for r, c in zip(row_ind, col_ind):
        if float(cost[r, c]) > max_projected_dist_px:
            continue
        det_idx = det_indices[c]
        detections_target[det_idx]['global_id'] = int(source_gids[r])
        assigned += 1

    return int(assigned)


def inject_view2_reseed_hints_from_view1(
    detections_view2,
    tracker_view1,
    tracker_view2,
    homography_view1_to_view2,
    max_projected_dist_px=35.0,
):
    return inject_reseed_hints_between_views(
        detections_target=detections_view2,
        tracker_source=tracker_view1,
        tracker_target=tracker_view2,
        homography_source_to_target=homography_view1_to_view2,
        max_projected_dist_px=max_projected_dist_px,
        skip_gids_already_active_in_target=True,
    )


def inject_view1_reseed_hints_from_view2(
    detections_view1,
    tracker_view2,
    tracker_view1,
    homography_view2_to_view1,
    max_projected_dist_px=35.0,
):
    # Do not skip source GIDs that are already active in view1:
    # this allows correction of in-view ID swaps (same IDs active on wrong people).
    return inject_reseed_hints_between_views(
        detections_target=detections_view1,
        tracker_source=tracker_view2,
        tracker_target=tracker_view1,
        homography_source_to_target=homography_view2_to_view1,
        max_projected_dist_px=max_projected_dist_px,
        skip_gids_already_active_in_target=False,
    )


def associate_tracks_to_view1(
    tracker1, tracker_other, frame_id,
    homography_anchor_to_other,
    homography_other_to_anchor=None,
    max_projected_dist_px=120.0,
):
    # Consider only currently visible tracks to avoid lingering
    # cross-view points after occlusion/exit.
    # GID reassignment is applied only for confident occlusion recovery:
    # exactly one side was refound in this frame.
    active1 = [
        t for t in tracker1.tracks
        if t.get('bbox') is not None and t.get('missed', 0) == 0
    ]
    active_other = [
        t for t in tracker_other.tracks
        if t.get('bbox') is not None and t.get('missed', 0) == 0
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
        for j, p_other in enumerate(native_view2):
            if p_other is None or not np.all(np.isfinite(p_other)):
                continue
            dist = float(np.linalg.norm(p_proj - p_other))
            if dist <= max_projected_dist_px:
                cost[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(cost)
    associations = []

    def _select_target_gid(t1, t2):
        gid1 = t1.get('global_id')
        gid2 = t2.get('global_id')

        if gid1 is None and gid2 is None:
            return None, False
        if gid1 is None:
            return gid2, True
        if gid2 is None:
            return gid1, True
        if gid1 == gid2:
            return gid1, False

        refound1 = bool(t1.get('was_refound_in_this_frame'))
        refound2 = bool(t2.get('was_refound_in_this_frame'))
        # Reassign only for confident occlusion recovery:
        # exactly one side was refound in this frame.
        if refound2 and not refound1:
            return gid1, True
        if refound1 and not refound2:
            return gid2, True

        # Otherwise, keep existing identities and only report the association.
        return None, False

    for r, c in zip(row_ind, col_ind):
        dist = float(cost[r, c])
        if dist > max_projected_dist_px:
            continue

        t1 = active1[r]
        t2 = active_other[c]
        gid1 = t1.get('global_id')
        gid2 = t2.get('global_id')
        target_gid, should_reassign = _select_target_gid(t1, t2)

        gid_reassigned = False
        if should_reassign and target_gid is not None:
            prev_gid1 = gid1
            prev_gid2 = gid2
            merge_track_global_id(
                tracker1, t1, target_gid, frame_id
            )
            merge_track_global_id(
                tracker_other, t2, target_gid, frame_id
            )
            gid1_now = t1.get('global_id')
            gid2_now = t2.get('global_id')
            gid_reassigned = int((prev_gid1 != gid1_now) or (prev_gid2 != gid2_now))

        p2 = projected_to_other[r]
        n1_pt = native_view1[r]
        n2_pt = native_view2[c]
        p1 = None
        if homography_other_to_anchor is not None and n2_pt is not None and np.all(np.isfinite(n2_pt)):
            p1 = project_point_with_homography(n2_pt, homography_other_to_anchor)

        gid1_now = t1.get('global_id')
        gid2_now = t2.get('global_id')
        if gid1_now is None:
            assoc_gid = gid2_now
        elif gid2_now is None:
            assoc_gid = gid1_now
        elif gid1_now == gid2_now:
            assoc_gid = gid1_now
        else:
            assoc_gid = None
        associations.append({
            'track_id_view1': t1.get('track_id'),
            'track_id_view2': t2.get('track_id'),
            'global_id': assoc_gid,
            'native_view1': n1_pt,
            'projected_view1': p1,
            'native_view2': n2_pt,
            'projected_view2': p2,
            'distance_px': dist,
            'gid_reassigned': int(gid_reassigned),
        })
    return associations


def yolo_sahi_pose_tracking(
    source1,
    source2,
    annotations_dir1=None,
    annotations_dir2=None,
    output_path1='output_view1.mp4',
    output_path2='output_view2.mp4',
    output_csv_path1='track_events_view1.csv',
    output_csv_path2='track_events_view2.csv',
    size=(1440, 810),  # Processing resolution (display uses same max size).
    # detection View 1
    sahi_conf_threshold1=0.50,
    sahi_iou_threshold1=0.37,
    slice_h1=320,
    slice_w1=320,
    slice_overlap1=0.3,
    # pose View 1
    pose_conf_threshold1=0.03,
    pose_iou_threshold1=0.005,
    # tracker View 1
    match_threshold1=0.55,
    iou_weight1=0.60,
    appearance_weight1=0.35,
    pose_weight1=0.25,
    ema_alpha1=0.5,
    max_velocity1=45.0,
    cross_view_max_dist_px1=35.0,
    # detection View 2
    sahi_conf_threshold2=0.45,
    sahi_iou_threshold2=0.40,
    slice_h2=480,
    slice_w2=480,
    slice_overlap2=0.25,
    # pose View 2
    pose_conf_threshold2=0.10,
    pose_iou_threshold2=0.02,
    # tracker View 2
    match_threshold2=0.55,
    iou_weight2=0.60,
    appearance_weight2=0.35,
    pose_weight2=0.05,
    ema_alpha2=0.5,
    max_velocity2=35.0,
    enable_view1_reseed_from_view2=True,
    enable_view2_reseed_from_view1=True,
    homography_path='homography_cam4_to_cam13.json',
    nested_overlap_thresh1=0.8,
    nested_overlap_thresh2=0.8,
    # line-based filtering + debug
    invalid_view1_line_ids=(12, 13, 14),
    court_lines_path1=None,
    court_corners_path1=None,
    # optional projection video onto a court template image
    save_court_projection=True,
    court_template_path='basketball_court_template_by_verasthebrujah_ddm06jb-fullview.png',
    homography_view1_to_court_path='homography_cam13_to_court.json',
    homography_view2_to_court_path='homography_cam4_to_court.json',
    output_court_projection_path='output_court_projection.mp4',
):
    """
    Multi-view tracking with single-view tracking + cross-view association.

    Flow:
    - Detect per-view and estimate pose
    - Track per-view independently
    - Associate tracks across views using homography-projected bottom centers
    - Optionally project both views onto a shared court template and save a video
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
        image_size=640,
        device='cuda:0'
    )
    pose_model = YOLO('yolo11x-pose.pt')
    print("Models loaded successfully")

    video_prefix1 = os.path.basename(source1).split('.')[0] if source1 else None
    video_prefix2 = os.path.basename(source2).split('.')[0] if source2 else None

    # Load first-frame annotations for both views (ground-truth bootstrap).
    annotations1 = load_annotations_from_dir(annotations_dir1, video_prefix1)
    annotations2 = load_annotations_from_dir(annotations_dir2, video_prefix2)

    if 0 not in annotations1 or 0 not in annotations2:
        raise ValueError(
            "Both views must provide first-frame annotations for fixed-slot tracking."
        )

    first_frame1, count1, gid_values1 = infer_fixed_roster_from_first_annotation_frame(annotations1)
    first_frame2, count2, gid_values2 = infer_fixed_roster_from_first_annotation_frame(annotations2)
    if count1 is None or count2 is None:
        raise ValueError("Failed to infer fixed roster from first-frame annotations in both views.")

    gid_pool = sorted(set(gid_values1) | set(gid_values2))
    if not gid_pool:
        raise ValueError("Could not derive any Global IDs from first-frame annotations.")

    fixed_slots_count = int(len(gid_pool))
    print(
        f"Fixed roster inferred from frame {first_frame1} (view1) and frame {first_frame2} (view2): "
        f"{fixed_slots_count} IDs -> {gid_pool}"
    )
    if set(gid_values1) != set(gid_values2):
        print(
            "Warning: first-frame annotation IDs differ between views. "
            "Using union of IDs for the fixed GID pool."
        )

    shared_id_manager = GlobalIDManager(
        gid_values=gid_pool,
        cooldown_frames=0,
    )

    # Create trackers with per-view matching settings and shared fixed GID pool.
    tracker1 = IOUTracker(
        match_threshold=match_threshold1,
        iou_weight=iou_weight1,
        appearance_weight=appearance_weight1,
        pose_weight=pose_weight1,
        ema_alpha=ema_alpha1,
        max_velocity=max_velocity1,
        global_id_manager=shared_id_manager,
        max_tracks=fixed_slots_count,
    )
    
    tracker2 = IOUTracker(
        match_threshold=match_threshold2,
        iou_weight=iou_weight2,
        appearance_weight=appearance_weight2,
        pose_weight=pose_weight2,
        ema_alpha=ema_alpha2,
        max_velocity=max_velocity2,
        global_id_manager=shared_id_manager,
        max_tracks=fixed_slots_count,
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

    court_template_img = None
    h_view1_to_court = None
    h_view2_to_court = None
    writer_court = None

    # Create CSV writers
    csv_file1 = open(output_csv_path1, 'w', newline='')
    csv_writer1 = csv.DictWriter(
        csv_file1,
        fieldnames=[
            'frame', 'track_id', 'event', 'global_id',
            'matched_with_other_view_track_id', 'new_global_id', 'old_global_id',
            'swapped_with_track_id',
            'detections_in_frame',
            'view_x', 'view_y', 'view_pos_json',
            'projected_other_x', 'projected_other_y', 'projected_other_pos_json',
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
            'frame', 'track_id', 'event', 'global_id',
            'matched_with_other_view_track_id', 'new_global_id', 'old_global_id',
            'swapped_with_track_id',
            'detections_in_frame',
            'view_x', 'view_y', 'view_pos_json',
            'projected_other_x', 'projected_other_y', 'projected_other_pos_json',
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
        if writer_court is not None:
            writer_court.release()
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

    if save_court_projection and output_court_projection_path:
        if not court_template_path or not os.path.exists(court_template_path):
            raise FileNotFoundError(f"Court template image not found: {court_template_path}")

        court_template_img = cv.imread(court_template_path)
        if court_template_img is None:
            raise RuntimeError(f"Failed to read court template image: {court_template_path}")

        court_h, court_w = court_template_img.shape[:2]
        h_view1_to_court_raw = load_homography_matrix(homography_view1_to_court_path, matrix_key='H')
        h_view2_to_court_raw = load_homography_matrix(homography_view2_to_court_path, matrix_key='H')

        # Homographies are estimated on original camera frames; scale them to processing size.
        h_view1_to_court = scale_homography_for_resized_views(
            h_view1_to_court_raw,
            src_orig_size=(orig_w1, orig_h1),
            dst_orig_size=(court_w, court_h),
            src_proc_size=process_size,
            dst_proc_size=(court_w, court_h),
        )
        h_view2_to_court = scale_homography_for_resized_views(
            h_view2_to_court_raw,
            src_orig_size=(orig_w2, orig_h2),
            dst_orig_size=(court_w, court_h),
            src_proc_size=process_size,
            dst_proc_size=(court_w, court_h),
        )

        fps_court = min(fps1, fps2) if fps1 > 0 and fps2 > 0 else (fps1 or fps2 or 25)
        writer_court = cv.VideoWriter(
            output_court_projection_path,
            cv.VideoWriter_fourcc(*'mp4v'),
            fps_court,
            (court_w, court_h),
        )
        if not writer_court.isOpened():
            raise RuntimeError(f"Could not open court projection writer: {output_court_projection_path}")
        print(f"Court projection video: {output_court_projection_path}")

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
        slice_overlap1, pose_model, pose_conf_threshold1,
        pose_iou_threshold1, world_homography=None,
        nested_overlap_thresh=nested_overlap_thresh1,
        invalid_if_above_lines=invalid_view1_lines,
        invalid_ref_pt=invalid_view1_ref_pt
    )

    initial_detections2 = detect_view(
        frame2, frame_id, annotations2, det_model,
        sahi_conf_threshold2, sahi_iou_threshold2, slice_h2, slice_w2,
        slice_overlap2, pose_model, pose_conf_threshold2,
        pose_iou_threshold2, world_homography=h_view2_to_view1,
        nested_overlap_thresh=nested_overlap_thresh2,
    )

    # Update trackers with initial detections
    if enable_view1_reseed_from_view2:
        inject_view1_reseed_hints_from_view2(
            initial_detections1,
            tracker2,
            tracker1,
            h_view2_to_view1,
            max_projected_dist_px=cross_view_max_dist_px1,
        )
    tracks1 = tracker1.update(
        initial_detections1, frame_id,
    )[0]
    if enable_view2_reseed_from_view1:
        inject_view2_reseed_hints_from_view1(
            initial_detections2,
            tracker1,
            tracker2,
            h_view1_to_view2,
            max_projected_dist_px=cross_view_max_dist_px1,
        )
    tracks2 = tracker2.update(
        initial_detections2, frame_id,
    )[0]
    # Associate tracks across views (one-way: View 1 -> View 2)
    cross_view_associations = associate_tracks_to_view1(
        tracker1, tracker2, frame_id,
        homography_anchor_to_other=h_view1_to_view2,
        homography_other_to_anchor=h_view2_to_view1,
        max_projected_dist_px=cross_view_max_dist_px1,
    )
    assoc_lookup_view1, assoc_lookup_view2 = _build_association_lookup(cross_view_associations)

    # Log initial events
    track_map1 = {t.get('track_id'): t for t in tracker1.tracks if t.get('track_id') is not None}
    track_map2 = {t.get('track_id'): t for t in tracker2.tracks if t.get('track_id') is not None}
    for event in tracker1.get_frame_events():
        event['detections_in_frame'] = len(initial_detections1)
        _augment_event_with_view_and_projected_pos(
            event, tracker1, h_view1_to_view2,
            association_info_by_track=assoc_lookup_view1,
            track_by_id=track_map1,
        )
        csv_writer1.writerow(event)
    for event in tracker2.get_frame_events():
        event['detections_in_frame'] = len(initial_detections2)
        _augment_event_with_view_and_projected_pos(
            event, tracker2, h_view2_to_view1,
            association_info_by_track=assoc_lookup_view2,
            track_by_id=track_map2,
        )
        csv_writer2.writerow(event)

    # Save first frame
    roi1 = frame1
    roi2 = frame2
    out1 = draw_tracks(roi1.copy(), tracks1, connections, tracker1)
    out2 = draw_tracks(roi2.copy(), tracks2, connections, tracker2)
    out1 = draw_cross_view_associations(out1, cross_view_associations, view_id=1)
    out2 = draw_cross_view_associations(out2, cross_view_associations, view_id=2)
    writer1.write(out1)
    writer2.write(out2)
    if writer_court is not None:
        court_frame = render_court_projection_frame(
            court_template_img,
            tracks1,
            tracks2,
            h_view1_to_court,
            h_view2_to_court,
            frame_id=frame_id,
        )
        if court_frame is not None:
            writer_court.write(court_frame)
    

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
            slice_overlap1, pose_model, pose_conf_threshold1,
            pose_iou_threshold1, world_homography=None,
            nested_overlap_thresh=nested_overlap_thresh1,
            invalid_if_above_lines=invalid_view1_lines,
            invalid_ref_pt=invalid_view1_ref_pt
        )
        detections2 = detect_view(
            frame2, frame_id, annotations2, det_model,
            sahi_conf_threshold2, sahi_iou_threshold2, slice_h2, slice_w2,
            slice_overlap2, pose_model, pose_conf_threshold2,
            pose_iou_threshold2, world_homography=h_view2_to_view1,
            nested_overlap_thresh=nested_overlap_thresh2,
        )
        # ===== STEP 4: UPDATE TRACKERS =====
        if enable_view1_reseed_from_view2:
            inject_view1_reseed_hints_from_view2(
                detections1,
                tracker2,
                tracker1,
                h_view2_to_view1,
                max_projected_dist_px=cross_view_max_dist_px1,
            )
        tracks1 = tracker1.update(
            detections1, frame_id,
        )[0]
        if enable_view2_reseed_from_view1:
            inject_view2_reseed_hints_from_view1(
                detections2,
                tracker1,
                tracker2,
                h_view1_to_view2,
                max_projected_dist_px=cross_view_max_dist_px1,
            )
        tracks2 = tracker2.update(
            detections2, frame_id,
        )[0]
        # ===== STEP 5: CROSS-VIEW ASSOCIATION (ONE-WAY: VIEW 1 -> VIEW 2) =====
        cross_view_associations = associate_tracks_to_view1(
            tracker1, tracker2, frame_id,
            homography_anchor_to_other=h_view1_to_view2,
            homography_other_to_anchor=h_view2_to_view1,
            max_projected_dist_px=cross_view_max_dist_px1,
        )
        assoc_lookup_view1, assoc_lookup_view2 = _build_association_lookup(cross_view_associations)

        # ===== LOG EVENTS =====
        track_map1 = {t.get('track_id'): t for t in tracker1.tracks if t.get('track_id') is not None}
        track_map2 = {t.get('track_id'): t for t in tracker2.tracks if t.get('track_id') is not None}
        for event in tracker1.get_frame_events():
            event['detections_in_frame'] = len(detections1)
            _augment_event_with_view_and_projected_pos(
                event, tracker1, h_view1_to_view2,
                association_info_by_track=assoc_lookup_view1,
                track_by_id=track_map1,
            )
            csv_writer1.writerow(event)
        
        for event in tracker2.get_frame_events():
            event['detections_in_frame'] = len(detections2)
            _augment_event_with_view_and_projected_pos(
                event, tracker2, h_view2_to_view1,
                association_info_by_track=assoc_lookup_view2,
                track_by_id=track_map2,
            )
            csv_writer2.writerow(event)

        # Refresh tracks after all operations
        tracks1 = tracker1.tracks
        tracks2 = tracker2.tracks

        # ===== DRAWING =====
        roi1 = frame1
        roi2 = frame2
        out1 = draw_tracks(roi1.copy(), tracks1, connections, tracker1)
        out2 = draw_tracks(roi2.copy(), tracks2, connections, tracker2)
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
        if writer_court is not None:
            court_frame = render_court_projection_frame(
                court_template_img,
                tracks1,
                tracks2,
                h_view1_to_court,
                h_view2_to_court,
                frame_id=frame_id,
            )
            if court_frame is not None:
                writer_court.write(court_frame)

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
    if writer_court is not None:
        writer_court.release()
    
    cv.destroyAllWindows()

    csv_file1.close()
    csv_file2.close()

    print("Processing complete!")


if __name__ == '__main__':
    try:
        yolo_sahi_pose_tracking(
            source1='Tracking/material4project/Rectified videos/tracking_12/out13.mp4',
            source2='Tracking/material4project/Rectified videos/tracking_12/out4.mp4',
            annotations_dir1='tracking_12.v4i.yolov11/train/labels/',
            annotations_dir2='tracking_12.v4i.yolov11/train/labels/',
            output_path1='output_view1.mp4',
            output_path2='output_view2.mp4',
            court_lines_path1='court_lines_cam13.json',
            court_corners_path1='cam13_img_corners_rectified.json',
            save_court_projection=True,
            court_template_path='basketball_court_template_by_verasthebrujah_ddm06jb-fullview.png',
            homography_view1_to_court_path='homography_cam13_to_court.json',
            homography_view2_to_court_path='homography_cam4_to_court.json',
            output_court_projection_path='output_court_projection.mp4',
        )
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
