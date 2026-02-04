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
TRAPEZ_TOP_LEFT_1 = (0, 0)
TRAPEZ_TOP_RIGHT_1 = (1440, 0)
TRAPEZ_BOTTOM_LEFT_1 = (0, 810)
TRAPEZ_BOTTOM_RIGHT_1 = (1440, 810)
CURVE_HEIGHT_1 = 0

# Mask for View 2
TRAPEZ_TOP_LEFT_2 = (200, 330)
TRAPEZ_TOP_RIGHT_2 = (1240, 330)
TRAPEZ_BOTTOM_LEFT_2 = (0, 665)
TRAPEZ_BOTTOM_RIGHT_2 = (1440, 665)
CURVE_HEIGHT_2 = 65

# Mask for View 3 (assuming similar to view 2 for now)
TRAPEZ_TOP_LEFT_3 = (0, 0)
TRAPEZ_TOP_RIGHT_3 = (1440, 0)
TRAPEZ_BOTTOM_LEFT_3 = (0, 810)
TRAPEZ_BOTTOM_RIGHT_3 = (1440, 810)
CURVE_HEIGHT_3 = 0
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

def load_annotations_from_dir(annotations_dir, video_prefix, img_width, img_height, pose_model, pose_attempts, pose_conf_threshold, pose_iou_threshold, tracker=None):
    """
    Load all annotation files from a directory and organize by frame number.
    Filters for files starting with video_prefix and ending with '_rectified.txt'
    Assumes filenames contain 'frame_XXXX' where XXXX is the 1-based frame number.
    Converts to 0-based frame_id.
    Returns a dict frame_id -> list of anno_file paths
    """
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
            else:
                print(f"Warning: Could not extract frame number from {file}")
    return annotations

def draw_tracks(img, tracks, size, max_tracks, connections, tracker_self, other_tracks=None, other_view_name="V2"):
    """
    Helper function to draw tracks, keypoints and cross-view projections on a frame.
    """
    # Show any track that is currently tracked (within missed frame tolerance)
    valid_tracks = [t for t in tracks if t['missed'] <= tracker_self.max_missed]
    
    # Filter by FOV and only active tracks
    valid_tracks_filtered = []
    for t in valid_tracks:
        x1, y1, x2, y2 = t['bbox']
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Only draw tracks that are currently detected (missed == 0) and not deduced
        if t['missed'] == 0 and not t.get('was_deduced'):
            valid_tracks_filtered.append(t)
    
    # Sort by ID to keep colors consistent
    valid_tracks_filtered.sort(key=lambda x: x['global_id'])
    
    # Limit to max_tracks
    if max_tracks is not None:
        valid_tracks_filtered = valid_tracks_filtered[:max_tracks]
    
    # Draw primary tracks for this view
    for t in valid_tracks_filtered:
        x1, y1, x2, y2 = map(int, t['bbox'])
        gid = t['global_id']
        col = (gid * 37 % 255, gid * 17 % 255, gid * 97 % 255)
        
        # Rectangle for player
        # Use dashed or thinner lines for deduced/missed tracks
        thickness = 1 if (t.get('was_deduced') or t['missed'] > 0) else 2
        cv.rectangle(img, (x1, y1), (x2, y2), col, thickness)
        
        # Always display GID label at the top-center of the bbox
        cx = (x1 + x2) // 2
        cy = y1
        label = f'GID:{gid}' + (' (D)' if t.get('was_deduced') else '')
        cv.putText(img, label, (cx, cy - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        
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
            head = kpts[0]
            # Label is now displayed outside the pose check

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
    source3=None,
    calib_path1=None,
    calib_path2=None,
    calib_path3=None,
    annotations_dir1=None,
    annotations_dir2=None,
    annotations_dir3=None,
    output_path1='yolo_sahi_pose_tracking_view1.mp4',
    output_path2='yolo_sahi_pose_tracking_view2.mp4',
    output_path3='yolo_sahi_pose_tracking_view3.mp4',
    output_csv_path1='track_events_view1.csv',
    output_csv_path2='track_events_view2.csv',
    output_csv_path3='track_events_view3.csv',
    start_frame=0,
    enable_pose=True,
    size=(1440, 810),
    # detection View 1
    sahi_conf_threshold1=0.6,
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
    pose_weight1=0,
    max_missed_frames1=300,
    ema_alpha1=0.5,
    max_velocity1=1000.0, # max units moved per frame for View 1
    # detection View 2
    sahi_conf_threshold2=0.15,
    sahi_iou_threshold2=0.25,
    slice_h2=480,
    slice_w2=480,
    slice_overlap2=0.6,
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
    pose_weight2=0,
    max_missed_frames2=60,
    ema_alpha2=0.5,
    max_velocity2=1000.0, # max units moved per frame for View 2
    cross_view_match_threshold1=0.2,
    cross_view_match_threshold2=0.2,
    world_dist_weight1=0.9,
    world_dist_weight2=0.9,
    re_entry_proximity_threshold=700.0,
    # detection View 3
    sahi_conf_threshold3=0.6,
    sahi_iou_threshold3=0.55,
    slice_h3=640,
    slice_w3=640,
    slice_overlap3=0.15,
    # pose View 3
    pose_conf_threshold3=0.15,
    pose_iou_threshold3=0.02,
    pose_attempts3=(
        {'pad': 0.0, 'conf': None, 'iou': None},
        {'pad': 0.25, 'conf': 0.1, 'iou': 0.005},
    ),
    # tracker View 3
    match_threshold3=0.25,
    iou_weight3=0.27,
    appearance_weight3=0.45,
    pose_weight3=0,
    max_missed_frames3=300,
    ema_alpha3=0.5,
    max_velocity3=10000.0, # max units moved per frame for View 3
    cross_view_match_threshold3=0.2,
    world_dist_weight3=0.9
):
    """
    Main pipeline for multi-view person tracking using YOLO, SAHI, and Pose estimation.
    
    Parameters:
        source1, source2: Paths to the input video files for view 1 and view 2.
        calib_path1, calib_path2: Paths to JSON files containing camera calibration parameters.
        annotations_dir1, 2: Paths to directories containing YOLO-format txt files for multiple frames.
                              Filenames should contain 'frame_XXXX' where XXXX is the frame number (0-based).
                              Used to provide ground truth detections for specific frames.
        output_path1, 2: Filenames for the processed output videos.
        output_csv_path1, 2: Filenames for saving track events (add, lost, refound).
        start_frame: 0-based frame number to start processing from (default 0).
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
    print("Starting yolo_sahi_pose_tracking")
    # skeleton connections (YOLO format)
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

    calib1 = load_calibration(calib_path1)
    calib2 = load_calibration(calib_path2)
    calib3 = load_calibration(calib_path3) if source3 else None

    video_prefix1 = os.path.basename(source1).split('.')[0] if source1 else None
    video_prefix2 = os.path.basename(source2).split('.')[0] if source2 else None
    video_prefix3 = os.path.basename(source3).split('.')[0] if source3 else None

    shared_id_manager = GlobalIDManager()

    def match_detections(dets_a, dets_b, tracker_a, tracker_b, match_threshold, shared_id_manager):
        if not dets_a or not dets_b:
            return set(), set()
        cross_costs = np.zeros((len(dets_a), len(dets_b)), dtype=np.float32)
        for i, da in enumerate(dets_a):
            for j, db in enumerate(dets_b):
                sim = tracker_a._cross_view_similarity(da, db)
                cross_costs[i, j] = 1.0 - sim
        row_ind, col_ind = linear_sum_assignment(cross_costs)
        matched_a = set()
        matched_b = set()
        for r, c in zip(row_ind, col_ind):
            if cross_costs[r, c] <= 1.0 - match_threshold:
                da = dets_a[r]
                db = dets_b[c]
                p3d = IOUTracker.triangulate(tracker_a, tracker_b, da['bbox'], db['bbox'])
                if p3d is not None:
                    da['world_pos'] = p3d
                    db['world_pos'] = p3d
                if 'global_id' not in da and 'global_id' not in db:
                    gid = shared_id_manager.next_id()
                    da['global_id'] = gid
                    db['global_id'] = gid
                elif 'global_id' in da and 'global_id' not in db:
                    db['global_id'] = da['global_id']
                elif 'global_id' in db and 'global_id' not in da:
                    da['global_id'] = db['global_id']
                matched_a.add(r)
                matched_b.add(c)
        return matched_a, matched_b

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
        global_id_manager=shared_id_manager,
        cross_view_match_threshold=cross_view_match_threshold1,
        world_dist_weight=world_dist_weight1
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
        global_id_manager=shared_id_manager,
        cross_view_match_threshold=cross_view_match_threshold2,
        world_dist_weight=world_dist_weight2
    )
    tracker3 = IOUTracker( # new tracker for view 3
        match_threshold=match_threshold3,
        max_missed_frames=max_missed_frames3,
        iou_weight=iou_weight3,
        appearance_weight=appearance_weight3,
        pose_weight=pose_weight3,
        ema_alpha=ema_alpha3,
        max_velocity=max_velocity3,
        camera_params=calib3,
        global_id_manager=shared_id_manager,
        cross_view_match_threshold=cross_view_match_threshold3,
        world_dist_weight=world_dist_weight3
    ) if source3 else None

    cap1 = cv.VideoCapture(source1)
    cap2 = cv.VideoCapture(source2) # new capture for view 2
    cap3 = cv.VideoCapture(source3) if source3 else None

    fps1 = int(cap1.get(cv.CAP_PROP_FPS)) or 25
    fps2 = int(cap2.get(cv.CAP_PROP_FPS)) or 25
    fps3 = int(cap3.get(cv.CAP_PROP_FPS)) or 25 if cap3 else 25
    # For now, assuming both videos have the same FPS and are synchronized
    if fps1 != fps2:
        print("warning: video fps mismatch, assuming synchronized playback.")

    mask1, _ = create_trapezoid_mask((size[1], size[0]),
                                           TRAPEZ_TOP_LEFT_1, TRAPEZ_TOP_RIGHT_1, TRAPEZ_BOTTOM_LEFT_1, TRAPEZ_BOTTOM_RIGHT_1, CURVE_HEIGHT_1)
    mask2, _ = create_trapezoid_mask((size[1], size[0]),
                                           TRAPEZ_TOP_LEFT_2, TRAPEZ_TOP_RIGHT_2, TRAPEZ_BOTTOM_LEFT_2, TRAPEZ_BOTTOM_RIGHT_2, CURVE_HEIGHT_2)
    mask3, _ = create_trapezoid_mask((size[1], size[0]),
                                           TRAPEZ_TOP_LEFT_3, TRAPEZ_TOP_RIGHT_3, TRAPEZ_BOTTOM_LEFT_3, TRAPEZ_BOTTOM_RIGHT_3, CURVE_HEIGHT_3) if source3 else (None, None)

    writer1 = cv.VideoWriter(
        output_path1,
        cv.VideoWriter_fourcc(*'mp4v'), fps1, size
    )
    writer2 = cv.VideoWriter( # new writer for view 2
        output_path2,
        cv.VideoWriter_fourcc(*'mp4v'), fps2, size
    )
    writer3 = cv.VideoWriter( # new writer for view 3
        output_path3,
        cv.VideoWriter_fourcc(*'mp4v'), fps3, size
    ) if source3 else None

    csv_file1 = open(output_csv_path1, 'w', newline='')
    csv_writer1 = csv.DictWriter(csv_file1, fieldnames=['frame', 'track_id', 'event', 'view', 'global_id', 'matched_with_other_view_track_id', 'new_global_id', 'old_global_id'])
    csv_writer1.writeheader()

    csv_file2 = open(output_csv_path2, 'w', newline='')
    csv_writer2 = csv.DictWriter(csv_file2, fieldnames=['frame', 'track_id', 'event', 'view', 'global_id', 'matched_with_other_view_track_id', 'new_global_id', 'old_global_id'])
    csv_writer2.writeheader()

    csv_file3 = open(output_csv_path3, 'w', newline='') if source3 else None
    csv_writer3 = csv.DictWriter(csv_file3, fieldnames=['frame', 'track_id', 'event', 'view', 'global_id', 'matched_with_other_view_track_id', 'new_global_id', 'old_global_id']) if csv_file3 else None
    if csv_writer3: csv_writer3.writeheader()

    # Load annotations for all frames
    annotations1 = load_annotations_from_dir(annotations_dir1, video_prefix1, size[0], size[1], pose_model, pose_attempts1, pose_conf_threshold1, pose_iou_threshold1, tracker1)
    annotations2 = load_annotations_from_dir(annotations_dir2, video_prefix2, size[0], size[1], pose_model, pose_attempts2, pose_conf_threshold2, pose_iou_threshold2, tracker2)
    annotations3 = load_annotations_from_dir(annotations_dir3, video_prefix3, size[0], size[1], pose_model, pose_attempts3, pose_conf_threshold3, pose_iou_threshold3, tracker3) if source3 else {}

    frame_id = start_frame
    fps_hist = deque(maxlen=30)  # Use deque for efficient FPS history (fixed size)

    # Skip frames before start_frame
    for _ in range(start_frame):
        cap1.read()
        cap2.read()

    # Process initial frame with annotations
    ok1, frame1 = cap1.read()
    ok2, frame2 = cap2.read()
    ok3, frame3 = cap3.read() if cap3 else (False, None)
    if not (ok1 and ok2) or (cap3 and not ok3):
        print(f"error: could not read initial frames. cap1: {ok1}, cap2: {ok2}, cap3: {ok3 if cap3 else 'N/A'}")
        return

    frame1 = cv.resize(frame1, size)
    frame2 = cv.resize(frame2, size)
    if frame3 is not None:
        frame3 = cv.resize(frame3, size)

    print(f"loading initial annotations for frame {frame_id} from {annotations_dir1}")
    print(f"  Frame size: {frame1.shape}, Processing size: {size[0]}x{size[1]}")
    initial_detections1 = []
    if frame_id in annotations1:
        for anno_file in annotations1[frame_id]:
            initial_detections1.extend(parse_yolo_annotations(
                anno_file, size[0], size[1], frame1, pose_model, pose_attempts1, pose_conf_threshold1, pose_iou_threshold1, tracker=tracker1
            ))
    print(f"loading initial annotations for frame {frame_id} from {annotations_dir2}")
    print(f"  Frame size: {frame2.shape}, Processing size: {size[0]}x{size[1]}")
    initial_detections2 = []
    if frame_id in annotations2:
        for anno_file in annotations2[frame_id]:
            initial_detections2.extend(parse_yolo_annotations(
                anno_file, size[0], size[1], frame2, pose_model, pose_attempts2, pose_conf_threshold2, pose_iou_threshold2, tracker=tracker2
            ))

    print(f"loading initial annotations for frame {frame_id} from {annotations_dir3}")
    initial_detections3 = []
    if source3 and frame_id in annotations3:
        for anno_file in annotations3[frame_id]:
            initial_detections3.extend(parse_yolo_annotations(
                anno_file, size[0], size[1], frame3, pose_model, pose_attempts3, pose_conf_threshold3, pose_iou_threshold3, tracker=tracker3
            ))

    # Store initial counts and IDs
    all_initial_gids = set([d['global_id'] for d in initial_detections1] + [d['global_id'] for d in initial_detections2] + ([d['global_id'] for d in initial_detections3] if initial_detections3 else []))
    max_total_players = None # No limit on max detections/tracks
    print(f"Total unique players in initial annotations: {len(all_initial_gids)}. No max_tracks limit.")
    # Update trackers with max_tracks
    tracker1.max_tracks = max_total_players
    tracker2.max_tracks = max_total_players

    # Add world_pos to initial detections
    for d in initial_detections1:
        d['world_pos'] = tracker1.project_to_world(d['bbox'])
    for d in initial_detections2:
        d['world_pos'] = tracker2.project_to_world(d['bbox'])
    for d in initial_detections3:
        d['world_pos'] = tracker3.project_to_world(d['bbox'])

    # Cross-view match initial detections
    matched_12_1, matched_12_2 = match_detections(initial_detections1, initial_detections2, tracker1, tracker2, tracker1.cross_view_match_threshold, shared_id_manager)
    matched_13_1, matched_13_3 = match_detections(initial_detections1, initial_detections3, tracker1, tracker3, tracker1.cross_view_match_threshold, shared_id_manager)
    matched_23_2, matched_23_3 = match_detections(initial_detections2, initial_detections3, tracker2, tracker3, tracker2.cross_view_match_threshold, shared_id_manager)

    # Pose for initial
    for d in initial_detections1:
        kpts, vec = process_detection_pose(frame1, d['bbox'], pose_model, pose_attempts1, pose_conf_threshold1, pose_iou_threshold1)
        d['keypoints'] = kpts
        d['pose_vec'] = vec
    for d in initial_detections2:
        kpts, vec = process_detection_pose(frame2, d['bbox'], pose_model, pose_attempts2, pose_conf_threshold2, pose_iou_threshold2)
        d['keypoints'] = kpts
        d['pose_vec'] = vec
    for d in initial_detections3:
        kpts, vec = process_detection_pose(frame3, d['bbox'], pose_model, pose_attempts3, pose_conf_threshold3, pose_iou_threshold3)
        d['keypoints'] = kpts
        d['pose_vec'] = vec

    # Update trackers
    tracks1, _, _ = tracker1.update(initial_detections1, frame_id, finalize=True)
    tracks2, _, _ = tracker2.update(initial_detections2, frame_id, finalize=True)
    tracks3, _, _ = tracker3.update(initial_detections3, frame_id, finalize=True)

    # Log initial events
    for event in tracker1.get_frame_events():
        event['view'] = 1
        csv_writer1.writerow(event)
    for event in tracker2.get_frame_events():
        event['view'] = 2
        csv_writer2.writerow(event)
    for event in tracker3.get_frame_events():
        event['view'] = 3
        csv_writer3.writerow(event)

    # Save first frame
    roi1 = apply_trapezoid_mask(frame1, mask1)
    roi2 = apply_trapezoid_mask(frame2, mask2)
    out1 = draw_tracks(roi1.copy(), tracks1, size, 12, connections, tracker1, 
                       other_tracks=tracks2, other_view_name="V2")
    out2 = draw_tracks(roi2.copy(), tracks2, size, 12, connections, tracker2, 
                       other_tracks=tracks1, other_view_name="V1")
    writer1.write(out1)
    writer2.write(out2)
    if cap3:
        roi3 = apply_trapezoid_mask(frame3, mask3)
        out3 = draw_tracks(roi3.copy(), tracks3, size, 12, connections, tracker3, 
                           other_tracks=None, other_view_name="")
        writer3.write(out3)
    if cap3:
        roi3 = apply_trapezoid_mask(frame3, mask3)
        out3 = draw_tracks(roi3.copy(), tracks3, size, 12, connections, tracker3, 
                           other_tracks=None, other_view_name="")
        writer3.write(out3)

    frame_id += 1  # Increment to next frame for the loop

    def match_tracks(active_a, active_b, tracker_a, tracker_b, match_threshold):
        if not active_a or not active_b:
            return
        cross_costs = np.zeros((len(active_a), len(active_b)), dtype=np.float32)
        for i, ta in enumerate(active_a):
            for j, tb in enumerate(active_b):
                sim = tracker_a._cross_view_similarity(ta, tb)
                cross_costs[i, j] = 1.0 - sim
        row_ind, col_ind = linear_sum_assignment(cross_costs)
        for r, c in zip(row_ind, col_ind):
            if cross_costs[r, c] <= 1.0 - match_threshold:
                ta, tb = active_a[r], active_b[c]
                gid1, gid2 = ta['global_id'], tb['global_id']
                if gid1 != gid2:
                    # Merge using oldest ID
                    if ta['start_frame'] <= tb['start_frame']:
                        tracker_b.merge_global_ids(gid2, gid1, frame_id)
                        # Update world_pos with triangulation
                        p3d = IOUTracker.triangulate(tracker_a, tracker_b, ta['bbox'], tb['bbox'])
                        if p3d is not None:
                            if gid1 in tracker_a.global_id_to_track:
                                tracker_a.global_id_to_track[gid1]['world_pos'] = p3d
                            if gid1 in tracker_b.global_id_to_track:
                                tracker_b.global_id_to_track[gid1]['world_pos'] = p3d
                    else:
                        tracker_a.merge_global_ids(gid1, gid2, frame_id)
                        p3d = IOUTracker.triangulate(tracker_a, tracker_b, ta['bbox'], tb['bbox'])
                        if p3d is not None:
                            if gid2 in tracker_a.global_id_to_track:
                                tracker_a.global_id_to_track[gid2]['world_pos'] = p3d
                            if gid2 in tracker_b.global_id_to_track:
                                tracker_b.global_id_to_track[gid2]['world_pos'] = p3d

    # Main loop for subsequent frames
    while cap1.isOpened() and cap2.isOpened() and (not cap3 or cap3.isOpened()):
        ok1, frame1 = cap1.read()
        ok2, frame2 = cap2.read()
        ok3, frame3 = cap3.read() if cap3 else (False, None)
        if not (ok1 and ok2) or (cap3 and not ok3):
            break

        frame1 = cv.resize(frame1, size)
        frame2 = cv.resize(frame2, size)
        if frame3 is not None:
            frame3 = cv.resize(frame3, size)

        start_time = time.time()

        # --- Detection ---
        # View 1
        detections1 = []
        if frame_id in annotations1:
            # Use annotations for this frame
            for anno_file in annotations1[frame_id]:
                detections1.extend(parse_yolo_annotations(
                    anno_file, size[0], size[1], frame1, pose_model, pose_attempts1, pose_conf_threshold1, pose_iou_threshold1, tracker=tracker1
                ))
        else:
            # Use YOLO detection
            roi1 = apply_trapezoid_mask(frame1, mask1)
            mask_1d = np.any(roi1 > 0, axis=2)
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
                            bbox = (bx1 + min_x1, by1 + min_y1, bx2 + min_x1, by2 + min_y1)
                            appearance = compute_team_appearence(frame1, bbox)
                            world_pos = tracker1.project_to_world(bbox)
                            detections1.append({
                                'bbox': bbox, 'appearance': appearance, 'world_pos': world_pos, 'score': p.score.value
                            })

        # View 2
        detections2 = []
        if frame_id in annotations2:
            # Use annotations for this frame
            for anno_file in annotations2[frame_id]:
                detections2.extend(parse_yolo_annotations(
                    anno_file, size[0], size[1], frame2, pose_model, pose_attempts2, pose_conf_threshold2, pose_iou_threshold2, tracker=tracker2
                ))
        else:
            # Use YOLO detection
            roi2 = apply_trapezoid_mask(frame2, mask2)
            mask_2d = np.any(roi2 > 0, axis=2)
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
                            world_pos = tracker2.project_to_world(bbox)
                            detections2.append({
                                'bbox': bbox, 'appearance': appearance, 'world_pos': world_pos, 'score': p.score.value
                            })

        # View 3
        detections3 = []
        if annotations_dir3 and frame_id in annotations3:
            # Use annotations for this frame
            for anno_file in annotations3[frame_id]:
                detections3.extend(parse_yolo_annotations(
                    anno_file, size[0], size[1], frame3, pose_model, pose_attempts3, pose_conf_threshold3, pose_iou_threshold3, tracker=tracker3
                ))
        elif frame3 is not None:
            # Use YOLO detection
            roi3 = apply_trapezoid_mask(frame3, mask3)
            mask_3d = np.any(roi3 > 0, axis=2)
            if np.any(mask_3d):
                coords = np.where(mask_3d)
                min_y3, max_y3 = coords[0].min(), coords[0].max()
                min_x3, max_x3 = coords[1].min(), coords[1].max()
                cropped3 = roi3[min_y3:max_y3 + 1, min_x3:max_x3 + 1]

                preds3 = get_sliced_prediction(
                    cropped3, det_model, slice_height=slice_h3, slice_width=slice_w3,
                    overlap_height_ratio=slice_overlap3, overlap_width_ratio=slice_overlap3,
                    postprocess_match_metric='IOU', postprocess_match_threshold=sahi_iou_threshold3, verbose=0
                )

                if preds3 and preds3.object_prediction_list:
                    for p in preds3.object_prediction_list:
                        if p.category.id == 0 and p.score.value >= sahi_conf_threshold3:
                            bx1, by1, bx2, by2 = map(int, p.bbox.to_xyxy())
                            bbox = (bx1 + min_x3, by1 + min_y3, bx2 + min_x3, by2 + min_y3)
                            appearance = compute_team_appearence(frame3, bbox)
                            world_pos = tracker3.project_to_world(bbox)
                            detections3.append({
                                'bbox': bbox, 'appearance': appearance, 'world_pos': world_pos, 'score': p.score.value
                            })

        # --- Determine 3D positions using all views ---
        # Cross-view match detections for all pairs
        matched_12_1, matched_12_2 = match_detections(detections1, detections2, tracker1, tracker2, tracker1.cross_view_match_threshold, shared_id_manager)
        matched_13_1, matched_13_3 = match_detections(detections1, detections3, tracker1, tracker3, tracker1.cross_view_match_threshold, shared_id_manager)
        matched_23_2, matched_23_3 = match_detections(detections2, detections3, tracker2, tracker3, tracker2.cross_view_match_threshold, shared_id_manager)

        # For unmatched detections, assign global_id to allow single-view tracking
        for d in detections1:
            if 'global_id' not in d:
                d['global_id'] = shared_id_manager.next_id()
        for d in detections2:
            if 'global_id' not in d:
                d['global_id'] = shared_id_manager.next_id()
        for d in detections3:
            if 'global_id' not in d:
                d['global_id'] = shared_id_manager.next_id()

        # --- If detected only in one view, deduce in the other views ---
        # For detections in view1 not matched in any pair, deduce in view2 and view3
        for i, d1 in enumerate(detections1):
            if i not in matched_12_1 and i not in matched_13_1 and 'global_id' in d1:
                gid = d1['global_id']
                if detections2:
                    tracker2.deduce_track_position(gid, d1['world_pos'], frame_id, frame_shape=(size[1], size[0]), mask=mask2)
                if detections3:
                    tracker3.deduce_track_position(gid, d1['world_pos'], frame_id, frame_shape=(size[1], size[0]), mask=mask3)
        # For detections in view2 not matched in any pair, deduce in view1 and view3
        for j, d2 in enumerate(detections2):
            if j not in matched_12_2 and j not in matched_23_2 and 'global_id' in d2:
                gid = d2['global_id']
                if detections1:
                    tracker1.deduce_track_position(gid, d2['world_pos'], frame_id, frame_shape=(size[1], size[0]), mask=mask1)
                if detections3:
                    tracker3.deduce_track_position(gid, d2['world_pos'], frame_id, frame_shape=(size[1], size[0]), mask=mask3)
        # For detections in view3 not matched in any pair, deduce in view1 and view2
        for k, d3 in enumerate(detections3):
            if k not in matched_13_3 and k not in matched_23_3 and 'global_id' in d3:
                gid = d3['global_id']
                if detections1:
                    tracker1.deduce_track_position(gid, d3['world_pos'], frame_id, frame_shape=(size[1], size[0]), mask=mask1)
                if detections2:
                    tracker2.deduce_track_position(gid, d3['world_pos'], frame_id, frame_shape=(size[1], size[0]), mask=mask2)

        # --- Pose estimation ---
        for d in detections1:
            kpts, vec = process_detection_pose(frame1, d['bbox'], pose_model, pose_attempts1, pose_conf_threshold1, pose_iou_threshold1)
            d['keypoints'] = kpts
            d['pose_vec'] = vec
        for d in detections2:
            kpts, vec = process_detection_pose(frame2, d['bbox'], pose_model, pose_attempts2, pose_conf_threshold2, pose_iou_threshold2)
            d['keypoints'] = kpts
            d['pose_vec'] = vec
        for d in detections3:
            kpts, vec = process_detection_pose(frame3, d['bbox'], pose_model, pose_attempts3, pose_conf_threshold3, pose_iou_threshold3)
            d['keypoints'] = kpts
            d['pose_vec'] = vec

        # --- Update tracker ---
        tracks1, _, _ = tracker1.update(detections1, frame_id, finalize=True)
        tracks2, _, _ = tracker2.update(detections2, frame_id, finalize=True)
        tracks3, _, _ = tracker3.update(detections3, frame_id, finalize=True)

        # --- Cross-view matching of tracks to merge IDs ---
        active1 = [t for t in tracks1 if t['updated']]
        active2 = [t for t in tracks2 if t['updated']]
        active3 = [t for t in tracks3 if t['updated']]
        match_tracks(active1, active2, tracker1, tracker2, tracker1.cross_view_match_threshold)
        match_tracks(active1, active3, tracker1, tracker3, tracker1.cross_view_match_threshold)
        match_tracks(active2, active3, tracker2, tracker3, tracker2.cross_view_match_threshold)

        # --- Re-entry matching for lost tracks ---
        lost1 = [t for t in tracks1 if t['missed'] > 0 and not t['updated']]
        lost2 = [t for t in tracks2 if t['missed'] > 0 and not t['updated']]
        lost3 = [t for t in tracks3 if t['missed'] > 0 and not t['updated']]
        new1 = [t for t in tracks1 if t['start_frame'] == frame_id]
        new2 = [t for t in tracks2 if t['start_frame'] == frame_id]
        new3 = [t for t in tracks3 if t['start_frame'] == frame_id]

        # Match new tracks in view 1 to lost tracks in view 2 and view 3
        for nt in new1:
            if nt.get('world_pos') is None: continue
            min_dist = float('inf')
            closest_lost = None
            for lt in lost2 + lost3:
                if lt.get('world_pos') is None: continue
                if len(nt['world_pos']) != 3 or len(lt['world_pos']) != 3: continue
                dist = np.linalg.norm(np.array(nt['world_pos']) - np.array(lt['world_pos']))
                if dist < min_dist:
                    min_dist = dist
                    closest_lost = lt
            if closest_lost and min_dist < re_entry_proximity_threshold:
                tracker1.merge_global_ids(nt['global_id'], closest_lost['global_id'], frame_id)

        # Match new tracks in view 2 to lost tracks in view 1 and view 3
        for nt in new2:
            if nt.get('world_pos') is None: continue
            min_dist = float('inf')
            closest_lost = None
            for lt in lost1 + lost3:
                if lt.get('world_pos') is None: continue
                if len(nt['world_pos']) != 3 or len(lt['world_pos']) != 3: continue
                dist = np.linalg.norm(np.array(nt['world_pos']) - np.array(lt['world_pos']))
                if dist < min_dist:
                    min_dist = dist
                    closest_lost = lt
            if closest_lost and min_dist < re_entry_proximity_threshold:
                tracker2.merge_global_ids(nt['global_id'], closest_lost['global_id'], frame_id)

        # Match new tracks in view 3 to lost tracks in view 1 and view 2
        for nt in new3:
            if nt.get('world_pos') is None: continue
            min_dist = float('inf')
            closest_lost = None
            for lt in lost1 + lost2:
                if lt.get('world_pos') is None: continue
                if len(nt['world_pos']) != 3 or len(lt['world_pos']) != 3: continue
                dist = np.linalg.norm(np.array(nt['world_pos']) - np.array(lt['world_pos']))
                if dist < min_dist:
                    min_dist = dist
                    closest_lost = lt
            if closest_lost and min_dist < re_entry_proximity_threshold:
                tracker3.merge_global_ids(nt['global_id'], closest_lost['global_id'], frame_id)

        # Log all events for this frame to CSV
        for event in tracker1.get_frame_events():
            if 'view' not in event: event['view'] = 1
            csv_writer1.writerow(event)
        for event in tracker2.get_frame_events():
            if 'view' not in event: event['view'] = 2
            csv_writer2.writerow(event)
        for event in tracker3.get_frame_events():
            if 'view' not in event: event['view'] = 3
            csv_writer3.writerow(event)

        # Refresh tracks after cross-view matching and deduction
        tracks1 = tracker1.tracks
        tracks2 = tracker2.tracks
        tracks3 = tracker3.tracks

        # --- Drawing and Display ---
        roi1 = apply_trapezoid_mask(frame1, mask1)
        roi2 = apply_trapezoid_mask(frame2, mask2)
        roi3 = apply_trapezoid_mask(frame3, mask3) if frame3 is not None else None

        out1 = draw_tracks(roi1.copy(), tracks1, size, 12, connections, tracker1, 
                           other_tracks=tracks2, other_view_name="V2")
        out2 = draw_tracks(roi2.copy(), tracks2, size, 12, connections, tracker2, 
                           other_tracks=tracks1, other_view_name="V1")
        out3 = draw_tracks(roi3.copy(), tracks3, size, 12, connections, tracker3, 
                           other_tracks=None, other_view_name="") if roi3 is not None else None

        dt = time.time() - start_time
        fps_hist.append(1 / dt if dt > 0 else 0)
        fps_avg = sum(fps_hist) / len(fps_hist) if fps_hist else 0
        draw_text_with_bg(out1, f'FPS {int(fps_avg)}', (30, 50))
        draw_text_with_bg(out1, f'Frame {frame_id}', (30, 100))
        draw_text_with_bg(out2, f'FPS {int(fps_avg)}', (30, 50))
        draw_text_with_bg(out2, f'Frame {frame_id}', (30, 100))
        if out3 is not None:
            draw_text_with_bg(out3, f'FPS {int(fps_avg)}', (30, 50))
            draw_text_with_bg(out3, f'Frame {frame_id}', (30, 100))

        writer1.write(out1)
        writer2.write(out2) # write output for view 2
        if out3 is not None:
            writer3.write(out3)

        # Display both views in separate windows
        cv.imshow('Tracking View 1', out1)
        cv.imshow('Tracking View 2', out2)
        if out3 is not None:
            cv.imshow('Tracking View 3', out3)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap1.release()
    cap2.release()
    if cap3: cap3.release()
    writer1.release()
    writer2.release()
    writer3.release()
    cv.destroyAllWindows()

    csv_file1.close() # Close csv_file1
    csv_file2.close() # Close csv_file2
    if csv_file3: csv_file3.close()


if __name__ == '__main__':
    try:
        yolo_sahi_pose_tracking(
            source1='Tracking/material4project/Rectified videos/tracking_12/out4.mp4',
            source2='Tracking/material4project/Rectified videos/tracking_12/out13.mp4',
            source3='Tracking/material4project/Rectified videos/tracking_12/out2.mp4',
            calib_path1='Tracking/material4project/3D Tracking Material/camera_data_with_Rvecs/camera_data/cam_4/calib/camera_calib.json',
            calib_path2='Tracking/material4project/3D Tracking Material/camera_data_with_Rvecs/camera_data/cam_13/calib/camera_calib.json',
            calib_path3='Tracking/material4project/3D Tracking Material/camera_data_with_Rvecs/camera_data/cam_2/calib/camera_calib.json',
            annotations_dir1='tracking_12.v4i.yolov11/train/labels/',
            annotations_dir2='tracking_12.v4i.yolov11/train/labels/',
            output_path1='output_view1.mp4',
            output_path2='output_view2.mp4',
            output_path3='output_view3.mp4',
            start_frame=1,
            enable_pose=False,
        )
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()