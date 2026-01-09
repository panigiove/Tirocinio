# process_video.py
# Latest version
# - Pose estimation attempted for EVERY detected bounding box (multi-attempt, padded fallback)
# - Hungarian-based tracking (see iou_tracker.py)
# - Appearance tuned for team sports (see appearance_utils.py)
# - Only draw boxes for tracks UPDATED in the current frame (no thin/ghost boxes)

import cv2 as cv
import numpy as np
import time
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

from iou_tracker import IOUTracker
from appearence_utils import compute_team_appearance, keypoints_to_pose_vec

# ================= ROI / MASK =================
TRAPEZOID_TOP_LEFT = (210, 380)
TRAPEZ_TOP_RIGHT = (1240, 350)
TRAPEZ_BOTTOM_LEFT = (0, 665)
TRAPEZ_BOTTOM_RIGHT = (1440, 665)
CURVE_HEIGHT = 65

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


def draw_text_with_bg(img, text, pos, scale=0.6):
    font = cv.FONT_HERSHEY_SIMPLEX
    (tw, th), b = cv.getTextSize(text, font, scale, 2)
    x, y = pos
    cv.rectangle(img, (x, y - th - 10), (x + tw + 20, y + b + 10), (0, 0, 0), -1)
    cv.putText(img, text, (x + 10, y), font, scale, (255, 255, 255), 2)


# ================= MAIN =================

def yolo_sahi_pose_tracking(
    source,
    output_path='yolo_sahi_pose_tracking_latest.mp4',
    size=(1440, 810),
    # detection
    sahi_conf_threshold=0.28, #soglia di confidenza minima per considerare una detection valida
    sahi_iou_threshold=0.50, #soglia di iou per il postprocessing delle slice, più alto-> meno fusioni, più basso-> più fusioni
    slice_h=640,
    slice_w=640,
    slice_overlap=0.35, #sovrapposizione percentuale tra slice
    # pose
    pose_conf_threshold=0.08, #soglia di confidenza minima per considerare una pose valida
    pose_iou_threshold=0.01, #soglia di iou per NMS durante la stima della posa (riduce duplicati)
    pose_attempts=(
        {'pad': 0.0, 'conf': None, 'iou': None},
        {'pad': 0.25, 'conf': 0.03, 'iou': 0.005},
    ),
    # tracker
    match_threshold=0.30, #soglia minima di similarità combinate per accettare un abbinamento tra track e detection. Valore alto->meno switch ma più tracce nuove, più basso->meno tracce nuove ma rischio di accoppiare due persone per sbaglio
    iou_weight=0.27, #aumenta il peso della sovrapposizione spaziale (più alto->funziona meglio quando ci sono poche sovrapposizioni)
    appearance_weight=0.45, # matching robusto quando i vestiti sono distintivi
    pose_weight=0.30, #  aumenta il peso delle pose stimate, utile in sovrapposizioni prolungate
    max_missed_frames=50,
):
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
        model_path='yolov8x.pt',
        confidence_threshold=sahi_conf_threshold,
        image_size=round_to_multiple(640, 32),
        device='cuda:0'
    )
    pose_model = YOLO('yolo11x-pose.pt')

    tracker = IOUTracker(
        match_threshold=match_threshold,
        max_missed_frames=max_missed_frames,
        iou_weight=iou_weight,
        appearance_weight=appearance_weight,
        pose_weight=pose_weight
    )

    cap = cv.VideoCapture(source)
    fps = int(cap.get(cv.CAP_PROP_FPS)) or 25

    mask, _ = create_trapezoid_mask((size[1], size[0]),
                                           TRAPEZOID_TOP_LEFT,
                                           TRAPEZ_TOP_RIGHT,
                                           TRAPEZ_BOTTOM_LEFT,
                                           TRAPEZ_BOTTOM_RIGHT,
                                           CURVE_HEIGHT)

    writer = cv.VideoWriter(
        output_path,
        cv.VideoWriter_fourcc(*'mp4v'), fps, size
    )

    frame_id = 0
    fps_hist = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv.resize(frame, size)
        frame_id += 1

        roi = apply_trapezoid_mask(frame, mask)
        ys, xs = np.where(roi[:, :, 0] > 0)
        if len(xs) == 0:
            continue
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        cropped = roi[min_y:max_y + 1, min_x:max_x + 1]

        start = time.time()
        detections = []

        preds = get_sliced_prediction(
            cropped,
            det_model,
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=slice_overlap,
            overlap_width_ratio=slice_overlap,
            postprocess_match_metric='IOU',
            postprocess_match_threshold=sahi_iou_threshold, # adjusted here
            verbose=0
        )

        if preds and preds.object_prediction_list:
            for p in preds.object_prediction_list:
                if p.category.id != 0 or p.score.value < sahi_conf_threshold:
                    continue

                bx1, by1, bx2, by2 = map(int, p.bbox.to_xyxy())
                bbox = (bx1 + min_x, by1 + min_y, bx2 + min_x, by2 + min_y)

                appearance = compute_team_appearance(frame, bbox)
                pose_kpts, pose_vec = None, None

                for att in pose_attempts:
                    pb = pad_bbox(bbox, att['pad'], frame.shape)
                    x1, y1, x2, y2 = pb
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    res = pose_model.predict(
                        crop,
                        conf=att['conf'] or pose_conf_threshold,
                        iou=att['iou'] or pose_iou_threshold,
                        imgsz=round_to_multiple(max(crop.shape[:2]), 32),
                        device='cuda',
                        verbose=False
                    )

                    if res and len(res[0].keypoints.xy) > 0:
                        k = res[0].keypoints.xy.cpu().numpy()[0]
                        pose_kpts = k + np.array([x1, y1])
                        pose_vec = keypoints_to_pose_vec(pose_kpts, bbox)
                        break

                detections.append({
                    'bbox': bbox,
                    'appearance': appearance,
                    'keypoints': pose_kpts,
                    'pose_vec': pose_vec
                })

        tracks = tracker.update(detections)

        out = roi.copy()
        for t in tracks:
            if not t['updated']:
                continue
            x1, y1, x2, y2 = t['bbox']
            tid = t['track_id']
            col = (tid * 37 % 255, tid * 17 % 255, tid * 97 % 255)
            cv.rectangle(out, (x1, y1), (x2, y2), col, 2)
            if t.get('keypoints') is not None:
                for a, b in connections:
                    p1, p2 = t['keypoints'][a], t['keypoints'][b]
                    if p1[0] > 0 and p2[0] > 0:
                        cv.line(out, tuple(p1.astype(int)), tuple(p2.astype(int)), col, 2)
                for p in t['keypoints']:
                    if p[0] > 0:
                        cv.circle(out, tuple(p.astype(int)), 3, col, -1)
                        cv.circle(out, tuple(p.astype(int)), 3, (255, 255, 255), 1)
                head = t['keypoints'][0]
                cv.putText(out, f'ID:{tid}', (int(head[0]), int(head[1]) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        dt = time.time() - start
        fps_hist.append(1 / dt if dt > 0 else 0)
        fps_avg = sum(fps_hist[-30:]) / len(fps_hist[-30:])
        draw_text_with_bg(out, f'FPS {int(fps_avg)}', (30, 50))
        draw_text_with_bg(out, f'Frame {frame_id}', (30, 100))

        writer.write(out)
        cv.imshow('Tracking', out)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    yolo_sahi_pose_tracking('video raw/test_clip1.mp4')
