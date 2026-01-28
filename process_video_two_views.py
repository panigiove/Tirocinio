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

from iou_tracker import IOUTracker, GlobalIDManager
from appearence_utils import compute_team_appearence, keypoints_to_pose_vec

# ================= ROI / MASK =================
# Mask for View 1
TRAPEZ_TOP_LEFT_1 = (0, 0)
TRAPEZ_TOP_RIGHT_1 = (1440, 0)
TRAPEZ_BOTTOM_LEFT_1 = (0, 810)
TRAPEZ_BOTTOM_RIGHT_1 = (1440, 810)
CURVE_HEIGHT_1 = 0

# Mask for View 2 (Placeholder values, adjust as needed)
TRAPEZ_TOP_LEFT_2 = (200, 330)
TRAPEZ_TOP_RIGHT_2 = (1240, 330)
TRAPEZ_BOTTOM_LEFT_2 = (0, 665)
TRAPEZ_BOTTOM_RIGHT_2 = (1440, 665)
CURVE_HEIGHT_2 = 65

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
    pose_kpts, pose_vec = None, None
    
    # Try first attempt (usually no padding, fastest)
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
    
    # Fallback to second attempt only if first failed
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

# new function to parse YOLO annotations
def parse_yolo_annotations(anno_file, img_width, img_height, frame, pose_model, pose_attempts, pose_conf_threshold, pose_iou_threshold, tracker=None):
    print(f"DEBUG: Attempting to parse {anno_file}")
    detections = []
    if not os.path.exists(anno_file):
        print(f"ERROR: Annotation file does not exist: {anno_file}")
        return detections
    try:
        lines_read = 0
        with open(anno_file, 'r') as f:
            for line in f:
                lines_read += 1
                parts = line.strip().split(' ')
                if len(parts) < 5:
                    print(f"DEBUG: Skipping short line: {line.strip()}")
                    continue  # Skip invalid lines
                class_id = int(parts[0])
                # Process all annotations (class_id can be any value - player IDs, object types, etc.)
                center_x, center_y, width, height = map(float, parts[1:])
                # Convert normalized YOLO format to pixel coordinates (xyxy)
                x1 = int((center_x - width / 2) * img_width)
                y1 = int((center_y - height / 2) * img_height)
                x2 = int((center_x + width / 2) * img_width)
                y2 = int((center_y + height / 2) * img_height)
                bbox = (x1, y1, x2, y2)

                appearance = compute_team_appearence(frame, bbox)
                pose_kpts, pose_vec = None, None

                # Perform pose estimation on the annotated bounding box (optimized)
                pose_kpts, pose_vec = process_detection_pose(
                    frame, bbox, pose_model, pose_attempts, pose_conf_threshold, pose_iou_threshold
                )

                world_pos = tracker.project_to_world(bbox) if tracker else None

                detections.append({
                    'bbox': bbox,
                    'appearance': appearance,
                    'keypoints': pose_kpts,
                    'pose_vec': pose_vec,
                    'world_pos': world_pos,
                    'score': 1.0, # Assign a high score for initial annotations
                    'global_id': class_id # Use class_id from annotation file
                })
        print(f"  Parsed {lines_read} lines from annotation file, created {len(detections)} detections")
    except FileNotFoundError:
        print(f"warning: annotation file not found at {anno_file}. skipping initial annotations for this view.")
    except Exception as e:
        print(f"error: failed to parse annotations from {anno_file}: {e}")
        import traceback
        traceback.print_exc()
    return detections

def draw_tracks(img, tracks, size, max_tracks, connections, tracker_self, other_tracks=None, other_view_name="V2"):
    """
    Helper function to draw tracks, keypoints and cross-view projections on a frame.
    """
    # Show any track that is currently tracked (within missed frame tolerance)
    valid_tracks = [t for t in tracks if t['missed'] <= tracker_self.max_missed]
    
    # Filter by FOV
    valid_tracks_filtered = []
    for t in valid_tracks:
        x1, y1, x2, y2 = t['bbox']
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # If the track is being deduced or is temporarily missed, we only draw it 
        # if its predicted/last-known center is within the current frame view.
        if t.get('was_deduced') or t['missed'] > 0:
            if not (0 <= cx < size[0] and 0 <= cy < size[1]):
                continue
        valid_tracks_filtered.append(t)
    
    # Sort by ID to keep colors consistent
    valid_tracks_filtered.sort(key=lambda x: x['global_id'])
    
    # Draw primary tracks for this view
    for t in valid_tracks_filtered:
        x1, y1, x2, y2 = map(int, t['bbox'])
        gid = t['global_id']
        col = (gid * 37 % 255, gid * 17 % 255, gid * 97 % 255)
        
        # Rectangle for player
        # Use dashed or thinner lines for deduced/missed tracks
        thickness = 1 if (t.get('was_deduced') or t['missed'] > 0) else 2
        cv.rectangle(img, (x1, y1), (x2, y2), col, thickness)
        
        if t.get('keypoints') is not None:
            kpts = t['keypoints'].astype(int)
            for a, b in connections:
                p1, p2 = kpts[a], kpts[b]
                if p1[0] > 0 and p2[0] > 0:
                    cv.line(img, tuple(p1), tuple(p2), col, thickness)
            for p in kpts:
                if p[0] > 0:
                    cv.circle(img, tuple(p), 3, col, -1)
                    cv.circle(img, tuple(p), 3, (255, 255, 255), 1)
            head = kpts[0]
            label = f'GID:{gid}' + (' (D)' if t.get('was_deduced') else '')
            cv.putText(img, label, (head[0], head[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # Draw projections from other view if applicable
    if other_tracks:
        for t_other in other_tracks:
            # Only project tracks that were actually detected (not deduced) in the other view
            if t_other['updated'] and not t_other.get('was_deduced'):
                gid = t_other['global_id']
                world_pos = t_other['world_pos']
                if world_pos is None: continue
                
                pixel_pos = tracker_self.project_to_pixel(world_pos)
                if pixel_pos is not None:
                    px, py = map(int, pixel_pos)
                    if 0 <= px < size[0] and 0 <= py < size[1]:
                        col = (gid * 37 % 255, gid * 17 % 255, gid * 97 % 255)
                        
                        # Use dimensions from local track if available, else default
                        tw, th = 40, 80
                        if gid in tracker_self.global_id_to_track:
                            lx1, ly1, lx2, ly2 = tracker_self.global_id_to_track[gid]['bbox']
                            tw, th = lx2 - lx1, ly2 - ly1
                        
                        cv.rectangle(img, (px - tw//2, py - th), (px + tw//2, py), col, 1)
                        cv.putText(img, f"{other_view_name}-GID:{gid}", (px - tw//2, py - th - 5), 
                                   cv.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    return img

# ================= MAIN =================

def load_calibration(calib_path):
    if not calib_path or not os.path.exists(calib_path):
        return None
    with open(calib_path, 'r') as f:
        return json.load(f)


def yolo_sahi_pose_tracking(
    source1,
    source2,
    calib_path1=None,
    calib_path2=None,
    initial_annotations_path1=None,
    initial_annotations_path2=None,
    output_path1='yolo_sahi_pose_tracking_view1.mp4',
    output_path2='yolo_sahi_pose_tracking_view2.mp4',
    output_csv_path1='track_events_view1.csv',
    output_csv_path2='track_events_view2.csv',
    size=(1440, 810),
    # detection View 1
    sahi_conf_threshold1=0.45,
    sahi_iou_threshold1=0.55,
    slice_h1=640,
    slice_w1=640,
    slice_overlap1=0.15,
    # pose View 1
    pose_conf_threshold1=0.15,
    pose_iou_threshold1=0.02,
    pose_attempts1=(
        {'pad': 0.0, 'conf': None, 'iou': None},
        {'pad': 0.25, 'conf': 0.1, 'iou': 0.005},
    ),
    # tracker View 1
    match_threshold1=0.25,
    iou_weight1=0.27,
    appearance_weight1=0.45,
    pose_weight1=0.33,
    max_missed_frames1=60,
    ema_alpha1=0.9,
    max_velocity1=400.0, # max units moved per frame for View 1
    # detection View 2
    sahi_conf_threshold2=0.25,
    sahi_iou_threshold2=0.50,
    slice_h2=640,
    slice_w2=640,
    slice_overlap2=0.37,
    # pose View 2
    pose_conf_threshold2=0.09,
    pose_iou_threshold2=0.02,
    pose_attempts2=(
        {'pad': 0.0, 'conf': None, 'iou': None},
        {'pad': 0.25, 'conf': 0.03, 'iou': 0.005},
    ),
    # tracker View 2
    match_threshold2=0.25,
    iou_weight2=0.30,
    appearance_weight2=0.45,
    pose_weight2=0.33,
    max_missed_frames2=60,
    ema_alpha2=0.9,
    max_velocity2=400.0 # max units moved per frame for View 2
):
    """
    Main pipeline for multi-view person tracking using YOLO, SAHI, and Pose estimation.
    
    Parameters:
        source1, source2: Paths to the input video files for view 1 and view 2.
        calib_path1, calib_path2: Paths to JSON files containing camera calibration parameters.
        initial_annotations_path1, 2: Paths to YOLO-format txt files for the first frame.
                                      Used to bootstrap tracking with known positions.
        output_path1, 2: Filenames for the processed output videos.
        output_csv_path1, 2: Filenames for saving track events (add, lost, refound).
        size: Tuple (width, height) to which input frames are resized.
        
        SAHI Parameters (sahi_conf, sahi_iou, slice_h, slice_w, slice_overlap):
            - sahi_conf_threshold: Confidence threshold for person detection in slices.
            - sahi_iou_threshold: IoU threshold for NMS after merging slice predictions.
            - slice_h, slice_w: Dimensions of the slices for SAHI.
            - slice_overlap: Overlap ratio between adjacent slices.
            
        Pose Parameters (pose_conf, pose_iou, pose_attempts):
            - pose_conf_threshold: Minimum confidence for keypoint detection.
            - pose_attempts: List of dictionaries defining padding and thresholds for multiple 
                             pose estimation retries if the first one fails.
                             
        Tracker Parameters (match_threshold, weights, max_missed, ema_alpha):
            - match_threshold: Minimum similarity to maintain a track.
            - iou_weight: Importance of bounding box overlap.
            - appearance_weight: Importance of visual similarity.
            - pose_weight: Importance of pose consistency.
            - max_missed_frames: Grace period for a track before it is considered lost.
            - ema_alpha: Update rate for the appearance feature vector.
            - max_velocity: Maximum physical distance a person can travel between frames.
                           Used to prevent physically impossible re-identifications.
    """
    # skeleton connections (YOLO format)
    connections = [
        (0,1),(0,2),(1,3),(2,4),
        (5,6),(5,7),(6,8),(7,9),(8,10),
        (5,11),(6,12),(11,12),
        (11,13),(12,14),(13,15),(14,16)
    ]

    # models
    det_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path='yolo11x.pt',
        confidence_threshold=min(sahi_conf_threshold1, sahi_conf_threshold2),
        image_size=round_to_multiple(640, 32),
        device='cuda:0'
    )
    pose_model = YOLO('yolo11x-pose.pt')

    calib1 = load_calibration(calib_path1)
    calib2 = load_calibration(calib_path2)

    shared_id_manager = GlobalIDManager()

    # Two trackers, one for each view
    tracker1 = IOUTracker(
        match_threshold=match_threshold1,
        max_missed_frames=max_missed_frames1,
        iou_weight=iou_weight1,
        appearance_weight=appearance_weight1,
        pose_weight=pose_weight1,
        ema_alpha=ema_alpha1,
        max_velocity=max_velocity1,
        camera_params=calib1,
        global_id_manager=shared_id_manager
    )
    tracker2 = IOUTracker( # new tracker for view 2
        match_threshold=match_threshold2,
        max_missed_frames=max_missed_frames2,
        iou_weight=iou_weight2,
        appearance_weight=appearance_weight2,
        pose_weight=pose_weight2,
        ema_alpha=ema_alpha2,
        max_velocity=max_velocity2,
        camera_params=calib2,
        global_id_manager=shared_id_manager
    )

    cap1 = cv.VideoCapture(source1)
    cap2 = cv.VideoCapture(source2) # new capture for view 2

    fps1 = int(cap1.get(cv.CAP_PROP_FPS)) or 25
    fps2 = int(cap2.get(cv.CAP_PROP_FPS)) or 25
    # For now, assuming both videos have the same FPS and are synchronized
    if fps1 != fps2:
        print("warning: video fps mismatch, assuming synchronized playback.")

    mask1, _ = create_trapezoid_mask((size[1], size[0]),
                                           TRAPEZ_TOP_LEFT_1, TRAPEZ_TOP_RIGHT_1, TRAPEZ_BOTTOM_LEFT_1, TRAPEZ_BOTTOM_RIGHT_1, CURVE_HEIGHT_1)
    mask2, _ = create_trapezoid_mask((size[1], size[0]),
                                           TRAPEZ_TOP_LEFT_2, TRAPEZ_TOP_RIGHT_2, TRAPEZ_BOTTOM_LEFT_2, TRAPEZ_BOTTOM_RIGHT_2, CURVE_HEIGHT_2)

    writer1 = cv.VideoWriter(
        output_path1,
        cv.VideoWriter_fourcc(*'mp4v'), fps1, size
    )
    writer2 = cv.VideoWriter( # new writer for view 2
        output_path2,
        cv.VideoWriter_fourcc(*'mp4v'), fps2, size
    )

    csv_file1 = open(output_csv_path1, 'w', newline='')
    csv_writer1 = csv.DictWriter(csv_file1, fieldnames=['frame', 'track_id', 'event', 'view', 'global_id', 'matched_with_other_view_track_id'])
    csv_writer1.writeheader()

    csv_file2 = open(output_csv_path2, 'w', newline='')
    csv_writer2 = csv.DictWriter(csv_file2, fieldnames=['frame', 'track_id', 'event', 'view', 'global_id', 'matched_with_other_view_track_id'])
    csv_writer2.writeheader()


    frame_id = 0
    fps_hist = deque(maxlen=30)  # Use deque for efficient FPS history (fixed size)

    # Process initial frame with annotations
    ok1, frame1 = cap1.read()
    ok2, frame2 = cap2.read()
    if not (ok1 and ok2):
        print(f"error: could not read initial frames. cap1: {ok1}, cap2: {ok2}")
        return

    frame1 = cv.resize(frame1, size)
    frame2 = cv.resize(frame2, size)
    frame_id += 1

    print(f"loading initial annotations for view 1 from {initial_annotations_path1}")
    print(f"  Frame size: {frame1.shape}, Processing size: {size[0]}x{size[1]}")
    initial_detections1 = parse_yolo_annotations(
        initial_annotations_path1, size[0], size[1], frame1, pose_model, pose_attempts1, pose_conf_threshold1, pose_iou_threshold1, tracker=tracker1
    )
    print(f"loading initial annotations for view 2 from {initial_annotations_path2}")
    print(f"  Frame size: {frame2.shape}, Processing size: {size[0]}x{size[1]}")
    initial_detections2 = parse_yolo_annotations(
        initial_annotations_path2, size[0], size[1], frame2, pose_model, pose_attempts2, pose_conf_threshold2, pose_iou_threshold2, tracker=tracker2
    )

    # Store initial counts and IDs
    all_initial_gids = set([d['global_id'] for d in initial_detections1] + [d['global_id'] for d in initial_detections2])
    max_total_players = len(all_initial_gids)
    
    if max_total_players == 0:
        print("warning: No players found in initial annotations. Relaxing max_tracks to 25.")
        max_total_players = 25 # Default fallback
    else:
        print(f"Total unique players in first frame: {max_total_players}")
    
    # Update trackers with max_tracks
    tracker1.max_tracks = max_total_players
    tracker2.max_tracks = max_total_players

    tracks1, _, _ = tracker1.update(initial_detections1, frame_id, finalize=True)
    tracks2, _, unmatched2 = tracker2.update(initial_detections2, frame_id, finalize=False)
    
    # Cross-view matching for the first frame to ensure shared GIDs from the start
    tracker2.assign_cross_view_tracks([t for t in tracks1 if t['updated']], unmatched2, frame_id)
    
    # Triangulate initial frame
    for gid in all_initial_gids:
        if gid in tracker1.global_id_to_track and gid in tracker2.global_id_to_track:
            t1 = tracker1.global_id_to_track[gid]
            t2 = tracker2.global_id_to_track[gid]
            if t1['updated'] and t2['updated']:
                p3d = IOUTracker.triangulate(tracker1, tracker2, t1['bbox'], t2['bbox'])
                if p3d is not None:
                    t1['world_pos'] = p3d
                    t2['world_pos'] = p3d

    # Log initial events to CSV
    for event in tracker1.get_frame_events():
        event['view'] = 1
        csv_writer1.writerow(event)
    for event in tracker2.get_frame_events():
        event['view'] = 2
        csv_writer2.writerow(event)

    # Refresh tracks2 after cross-view matching
    tracks2 = tracker2.tracks

    # Save first processed frame
    roi1 = apply_trapezoid_mask(frame1, mask1)
    roi2 = apply_trapezoid_mask(frame2, mask2)
    
    out1 = draw_tracks(roi1.copy(), tracks1, size, max_total_players, connections, tracker1, 
                       other_tracks=tracks2, other_view_name="V2")
    out2 = draw_tracks(roi2.copy(), tracks2, size, max_total_players, connections, tracker2, 
                       other_tracks=tracks1, other_view_name="V1")

    writer1.write(out1)
    writer2.write(out2)

    # Main loop for subsequent frames
    while cap1.isOpened() and cap2.isOpened():
        ok1, frame1 = cap1.read()
        ok2, frame2 = cap2.read()
        if not (ok1 and ok2):
            break

        frame1 = cv.resize(frame1, size)
        frame2 = cv.resize(frame2, size)
        frame_id += 1

        start_time = time.time()

        # --- Process View 1 ---
        roi1 = apply_trapezoid_mask(frame1, mask1)
        mask_1d = np.any(roi1 > 0, axis=2)
        detections1 = []
        if np.any(mask_1d):
            coords = np.where(mask_1d)
            min_y1, max_y1 = coords[0].min(), coords[0].max()
            min_x1, max_x1 = coords[1].min(), coords[1].max()
            cropped1 = roi1[min_y1:max_y1 + 1, min_x1:max_x1 + 1]

            preds1 = get_sliced_prediction(
                cropped1, det_model, slice_height=slice_h1, slice_width=slice_w1,
                overlap_height_ratio=slice_overlap1, overlap_width_ratio=slice_overlap1,
                postprocess_match_metric='IOU', postprocess_match_threshold=sahi_iou_threshold1, verbose=0
            )

            if preds1 and preds1.object_prediction_list:
                for p in preds1.object_prediction_list:
                    if p.category.id == 0 and p.score.value >= sahi_conf_threshold1:
                        bx1, by1, bx2, by2 = map(int, p.bbox.to_xyxy())
                        # CRITICAL: Map back to full frame coordinates correctly
                        bbox = (bx1 + min_x1, by1 + min_y1, bx2 + min_x1, by2 + min_y1)
                        
                        appearance = compute_team_appearence(frame1, bbox)
                        pose_kpts, pose_vec = process_detection_pose(
                            frame1, bbox, pose_model, pose_attempts1, pose_conf_threshold1, pose_iou_threshold1
                        )
                        world_pos = tracker1.project_to_world(bbox)

                        detections1.append({
                            'bbox': bbox, 'appearance': appearance, 'keypoints': pose_kpts, 'pose_vec': pose_vec, 'world_pos': world_pos, 'score': p.score.value
                        })
        
        tracks1, matched_ids1, unmatched1 = tracker1.update(detections1, frame_id, finalize=False)

        # --- Process View 2 ---
        roi2 = apply_trapezoid_mask(frame2, mask2)
        mask_2d = np.any(roi2 > 0, axis=2)
        detections2 = []
        if np.any(mask_2d):
            coords = np.where(mask_2d)
            min_y2, max_y2 = coords[0].min(), coords[0].max()
            min_x2, max_x2 = coords[1].min(), coords[1].max()
            cropped2 = roi2[min_y2:max_y2 + 1, min_x2:max_x2 + 1]

            preds2 = get_sliced_prediction(
                cropped2, det_model, slice_height=slice_h2, slice_width=slice_w2,
                overlap_height_ratio=slice_overlap2, overlap_width_ratio=slice_overlap2,
                postprocess_match_metric='IOU', postprocess_match_threshold=sahi_iou_threshold2, verbose=0
            )

            if preds2 and preds2.object_prediction_list:
                for p in preds2.object_prediction_list:
                    if p.category.id == 0 and p.score.value >= sahi_conf_threshold2:
                        bx1, by1, bx2, by2 = map(int, p.bbox.to_xyxy())
                        bbox = (bx1 + min_x2, by1 + min_y2, bx2 + min_x2, by2 + min_y2)
                        
                        appearance = compute_team_appearence(frame2, bbox)
                        pose_kpts, pose_vec = process_detection_pose(
                            frame2, bbox, pose_model, pose_attempts2, pose_conf_threshold2, pose_iou_threshold2
                        )
                        world_pos = tracker2.project_to_world(bbox)

                        detections2.append({
                            'bbox': bbox, 'appearance': appearance, 'keypoints': pose_kpts, 'pose_vec': pose_vec, 'world_pos': world_pos, 'score': p.score.value
                        })
        
        tracks2, matched_ids2, unmatched2 = tracker2.update(detections2, frame_id, finalize=False)
            
        # --- Cross-view synchronization ---
        # Try to match unmatched detections from View 1 with active tracks from View 2
        active_tracks2 = [t for t in tracks2 if t['updated']]
        cross_matches1 = tracker1.assign_cross_view_tracks(active_tracks2, unmatched1, frame_id)

        # Try to match unmatched detections from View 2 with active tracks from View 1
        active_tracks1 = [t for t in tracks1 if t['updated']]
        cross_matches2 = tracker2.assign_cross_view_tracks(active_tracks1, unmatched2, frame_id)

        # --- Cross-view deduction and triangulation ---
        for gid in all_initial_gids:
            t1 = tracker1.global_id_to_track.get(gid)
            t2 = tracker2.global_id_to_track.get(gid)
            
            # If updated in both, triangulate for precision
            if t1 and t2 and t1['updated'] and t2['updated'] and not t1.get('was_deduced') and not t2.get('was_deduced'):
                p3d = IOUTracker.triangulate(tracker1, tracker2, t1['bbox'], t2['bbox'])
                if p3d is not None:
                    t1['world_pos'] = p3d
                    t2['world_pos'] = p3d
            
            # If missed in one but active in other, deduce
            elif t1 and t2:
                if t2['updated'] and not t1['updated']:
                    tracker1.deduce_track_position(gid, t2['world_pos'], frame_id)
                elif t1['updated'] and not t2['updated']:
                    tracker2.deduce_track_position(gid, t1['world_pos'], frame_id)

        # Log all events for this frame to CSV (temporal, cross-view, and deduction)
        for event in tracker1.get_frame_events():
            if 'view' not in event: event['view'] = 1
            csv_writer1.writerow(event)
        for event in tracker2.get_frame_events():
            if 'view' not in event: event['view'] = 2
            csv_writer2.writerow(event)

        # Refresh tracks after cross-view matching and deduction
        tracks1 = tracker1.tracks
        tracks2 = tracker2.tracks


        # --- Drawing and Display ---
        roi1 = apply_trapezoid_mask(frame1, mask1)
        roi2 = apply_trapezoid_mask(frame2, mask2)

        out1 = draw_tracks(roi1.copy(), tracks1, size, max_total_players, connections, tracker1, 
                           other_tracks=tracks2, other_view_name="V2")
        out2 = draw_tracks(roi2.copy(), tracks2, size, max_total_players, connections, tracker2, 
                           other_tracks=tracks1, other_view_name="V1")

        dt = time.time() - start_time
        fps_hist.append(1 / dt if dt > 0 else 0)
        fps_avg = sum(fps_hist) / len(fps_hist) if fps_hist else 0
        draw_text_with_bg(out1, f'FPS {int(fps_avg)}', (30, 50))
        draw_text_with_bg(out1, f'Frame {frame_id}', (30, 100))
        draw_text_with_bg(out2, f'FPS {int(fps_avg)}', (30, 50))
        draw_text_with_bg(out2, f'Frame {frame_id}', (30, 100))

        writer1.write(out1)
        writer2.write(out2) # write output for view 2

        # Display both views in separate windows
        cv.imshow('Tracking View 1', out1)
        cv.imshow('Tracking View 2', out2)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    writer1.release()
    writer2.release()
    cv.destroyAllWindows()
    csv_file1.close() # Close csv_file1
    csv_file2.close() # Close csv_file2


if __name__ == '__main__':
    yolo_sahi_pose_tracking(
        source1='Tracking/material4project/Rectified videos/tracking_12/out4.mp4',
        source2='Tracking/material4project/Rectified videos/tracking_12/out13.mp4',
        calib_path1='Tracking/material4project/3D Tracking Material/camera_data_with_Rvecs/camera_data/cam_4/calib/camera_calib.json',
        calib_path2='Tracking/material4project/3D Tracking Material/camera_data_with_Rvecs/camera_data/cam_13/calib/camera_calib.json',
        initial_annotations_path1='tracking_12.v2i.yolov11/train/labels/out4_frame_0001_png.rf.13ad8a164866d2d0e8151ff7a71f7908_rectified.txt',
        initial_annotations_path2='tracking_12.v2i.yolov11/train/labels/out13_frame_0001_png.rf.68fc7ed57b749ab73105139ae9ec4f7e_rectified.txt',
        output_path1='output_view1.mp4',
        output_path2='output_view2.mp4',
    )