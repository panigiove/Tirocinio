import cv2 as cv
import numpy as np
import time
from ultralytics import YOLO

# Import SAHI components
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from iou_tracker import IOUTracker # Import our new tracker

avg_fps = []
video_writer = None

# Parametri della maschera trapezoidale con base curva
# Facilmente regolabili per adattarsi alla propria scena
TRAPEZOID_TOP_LEFT = (210, 380)      # Angolo superiore sinistro
TRAPEZ_TOP_RIGHT = (1240, 350)    # Angolo superiore destro
TRAPEZ_BOTTOM_LEFT = (0, 665)     # Angolo inferiore sinistro
TRAPEZ_BOTTOM_RIGHT = (1440, 665) # Angolo inferiore destro
CURVE_HEIGHT = 65                   # Altezza della curva della base (0 = retta, più alto = più curva)

# Helper function to round up to the nearest multiple of 'multiple'
def round_to_multiple(value, multiple):
    return multiple * ((value + multiple - 1) // multiple)

def draw_text_with_bg(img, text, pos, font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv.FONT_HERSHEY_SIMPLEX
    thickness = 2
    padding = 20  # Equal padding on all sides
    (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, thickness)
    rect_width = text_width + 2 * padding
    rect_height = text_height + baseline + 2 * padding
    x, y = pos  # Calculate rectangle position
    rect_x1 = x
    rect_y1 = y - text_height
    rect_x2 = rect_x1 + rect_width
    rect_y2 = rect_y1 + rect_height
    cv.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)  # Draw background rectangle
    text_x = rect_x1 + padding  # Center text in rectangle with equal padding
    text_y = rect_y1 + padding + text_height  # Properly centered vertically
    cv.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness)


def apply_roi_mask(frame, x1, y1, x2, y2):
    """Estrae la regione di interesse (ROI) dal frame e la visualizza solo quella"""
    roi = frame[y1:y2, x1:x2].copy()
    return roi


def draw_roi_rectangle(frame, x1, y1, x2, y2, color=(0, 255, 0), thickness=2):
    """Disegna il rettangolo della ROI sul frame"""
    cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def create_trapezoid_mask(frame_shape, tl, tr, bl, br, curve_height=0):
    """
    Crea una maschera trapezoidale con base curva
    
    Args:
        frame_shape: (height, width) del frame
        tl: (x, y) angolo top-left
        tr: (x, y) angolo top-right
        bl: (x, y) angolo bottom-left
        br: (x, y) angolo bottom-right
        curve_height: altezza della curva della base (0 = retta)
    
    Returns:
        mask: maschera binaria e punti del poligono
    """
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Punti del trapezio
    points = [np.array([tl, tr, br, bl], dtype=np.int32)]
    
    # Se curve_height > 0, aggiungi punti per curvare solo la base inferiore
    if curve_height > 0:
        # Crea una curva parabolica solo sulla base inferiore (da bl a br)
        points_curve = [tl, tr]
        
        # Lato destro: dritto da tr a br
        points_curve.append(br)
        
        # Base inferiore: curva da br a bl (con concavità verso il basso)
        for i in range(1, 20):
            t = i / 20.0
            x = br[0] + (bl[0] - br[0]) * t
            y = br[1] + (bl[1] - br[1]) * t + curve_height * (4 * t * (1 - t))  # Plus per curvatura verso il basso
            points_curve.append((int(x), int(y)))
        
        # Lato sinistro: dritto da bl a tl
        points_curve.append(bl)
        
        points = [np.array(points_curve, dtype=np.int32)]
    
    # Riempi il poligono con bianco
    cv.fillPoly(mask, points, 255)
    
    return mask, points[0]


def apply_trapezoid_mask(frame, mask):
    """Applica la maschera trapezoidale al frame"""
    masked_frame = cv.bitwise_and(frame, frame, mask=mask)
    return masked_frame


def draw_trapezoid_on_frame(frame, points, color=(0, 255, 0), thickness=2):
    """Disegna il contorno del trapezio sul frame"""
    cv.polylines(frame, [points], isClosed=True, color=color, thickness=2)


def yolo_pose_tracking(source=0, size=(1440, 810), conf_threshold=0.3, iou_threshold=0.7):
    """ 
    Performs YOLO-based pose tracking on video frames.
    
    Args:
        source: video source (e.g., 0 for webcam, "path/to/video.mp4").
        size: tuple (width, height) for resizing frames.
        conf_threshold: confidence threshold for object detection.
        iou_threshold: IOU threshold for non-maximum suppression (NMS).
    """

    # YOLO connections
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]

    yolo_model = YOLO("yolo11x-pose.pt")  # Initialize YOLO with larger model

    cap = cv.VideoCapture(source)

    orig_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv.CAP_PROP_FPS))
    
    # Crea la maschera trapezoidale con le dimensioni del frame ridimensionato (size)
    trap_mask, trap_points = create_trapezoid_mask(
        (size[1], size[0]),  # (height, width) - dimensioni ridimensionate
        TRAPEZOID_TOP_LEFT, 
        TRAPEZ_TOP_RIGHT, 
        TRAPEZ_BOTTOM_LEFT, 
        TRAPEZ_BOTTOM_RIGHT, 
        curve_height=CURVE_HEIGHT
    )
    
    # Calcola i bounds della maschera per il video writer
    x_coords = trap_points[:, 0]
    y_coords = trap_points[:, 1]
    roi_width = int(np.max(x_coords) - np.min(x_coords))
    roi_height = int(np.max(y_coords) - np.min(y_coords))
    
    # Inizializza video_writer UNA SOLA VOLTA con le dimensioni del frame ridimensionato
    video_writer = cv.VideoWriter("yolo-pose-tracking.avi", cv.VideoWriter_fourcc(*"mp4v"), original_fps, (size[0], size[1]))
    
    if not video_writer.isOpened():
        print("ERRORE: Impossibile creare il video writer!")
        cap.release()
        return

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame, size)
        frame_count += 1

        # Applica la maschera trapezoidale al frame
        roi_frame = apply_trapezoid_mask(frame, trap_mask)
        
        # Ensure imgsz_roi is a multiple of 32
        imgsz_roi = round_to_multiple(max(orig_width, orig_height), 32)

        start_time = time.time()
        # Usa track() con parametri per tracking più consistente
        results = yolo_model.track(
            roi_frame,
            device='cuda',
            tracker='bytetrack.yaml',
            show=False,
            conf=conf_threshold,           # Use parameter
            iou=iou_threshold,             # Use parameter
            imgsz=imgsz_roi, 
            max_det=300,
            persist=True,                  # Mantieni gli ID tra i frame
            verbose=False
        )
        process_time = time.time() - start_time

        # Copia il frame originale per visualizzare
        output_frame = frame.copy()
        roi_output = roi_frame.copy()
        num_persons = 0
        num_detections = len(results[0].boxes) if results else 0

        if results and len(results) > 0:
            if results[0].keypoints is not None:
                keypoints_list = results[0].keypoints.xy.cpu().numpy()
                track_ids = results[0].boxes.id if results[0].boxes.id is not None else None
                num_persons = len(keypoints_list)
                print(f"Rilevamenti: {num_detections} | Persone Tracciate: {num_persons}")  # DEBUG

                for person_idx, person_kpts in enumerate(keypoints_list):
                    # Usa l'ID di tracciamento se disponibile, altrimenti usa l'indice
                    if track_ids is not None:
                        track_id = int(track_ids[person_idx])
                        color_line = (int(50 * track_id) % 256, int(100 * track_id) % 256, int(150 * track_id) % 256)
                    else:
                        track_id = person_idx
                        color_line = (int(50 * person_idx) % 256, int(100 * person_idx) % 256, int(150 * person_idx) % 256)

                    for start, end in connections:  # Draw connections
                        if start < len(person_kpts) and end < len(person_kpts):
                            pt1, pt2 = person_kpts[start], person_kpts[end]
                            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                                # Disegna sulla ROI
                                cv.line(roi_output, (int(pt1[0]), int(pt1[1])),
                                         (int(pt2[0]), int(pt2[1])), color_line, 2)

                    for pt in person_kpts:  # Draw keypoints
                        if pt[0] > 0 and pt[1] > 0:
                            # Disegna sulla ROI
                            cv.circle(roi_output, (int(pt[0]), int(pt[1])), 3, color_line, -1)
                            cv.circle(roi_output, (int(pt[0]), int(pt[1])), 3, (255, 255, 255), 1)
                    
                    # Disegna l'ID di tracciamento sopra la testa (keypoint 0)
                    head_kpt = person_kpts[0]
                    if head_kpt[0] > 0 and head_kpt[1] > 0:
                        cv.putText(roi_output, f'ID: {track_id}', (int(head_kpt[0]) - 20, int(head_kpt[1]) - 15),
                                  cv.FONT_HERSHEY_SIMPLEX, 0.7, color_line, 2)
            else:
                print(f"Rilevamenti trovati ma nessun keypoint")  # DEBUG

        # Disegna il trapezio della ROI sul frame originale
        draw_trapezoid_on_frame(output_frame, trap_points, color=(0, 255, 0), thickness=2)

        fps_val = 1.0 / process_time if process_time > 0 else 0  # Calculate FPS
        avg_fps.append(fps_val)
        fps_avg = sum(avg_fps[-30:]) / len(avg_fps[-30:]) if avg_fps else 0  # Media degli ultimi 30 frame

        draw_text_with_bg(output_frame, f'YOLO-Pose Tracking FPS: {int(fps_avg)}', (50, 60), bg_color=(255, 27, 108), font_scale=1.2)
        draw_text_with_bg(output_frame, f'Time: {process_time * 1000:.1f}ms', (50, 145), bg_color=(255, 27, 108),
                          font_scale=1.2)
        draw_text_with_bg(output_frame, f'Frame: {frame_count} | Persone Tracciate: {num_persons} | Rilevamenti: {num_detections}', (50, 230), bg_color=(255, 27, 108),
                          font_scale=1.2)

        draw_text_with_bg(roi_output, f'ROI - Persone Tracciate: {num_persons}', (10, 40), bg_color=(255, 27, 108),
                          font_scale=1.0)

        # Visualizza sia il frame completo che la ROI
        cv.imshow('YOLO Pose - Frame Completo (ROI evidenziata)', output_frame)
        cv.imshow('YOLO Pose - Solo ROI (rilevamento)', roi_output)

        # Salva solo la ROI nel video
        video_writer.write(roi_output)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv.destroyAllWindows()


def yolo_pose_predict(source=0, size=(1440, 810), sahi_conf_threshold=0.4, sahi_iou_threshold=0.7, pose_conf_threshold=0.5, pose_iou_threshold=0.7):
    """
    Versione con rilevamento di persone tramite SAHI e stima pose sulle ROI delle persone.
    
    Args:
        source: video source (e.g., 0 for webcam, "path/to/video.mp4").
        size: tuple (width, height) for resizing frames.
        sahi_conf_threshold: confidence threshold for initial SAHI person detection.
        sahi_iou_threshold: IOU threshold for SAHI's NMS.
        pose_conf_threshold: confidence threshold for keypoints detection by the pose model.
        pose_iou_threshold: IOU threshold for NMS in the pose model.
    """

    # YOLO connections for drawing keypoints
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]

    # Initialize SAHI model for sliced detection (e.g., yolov8x.pt for general object detection)
    # This model will detect 'person' objects.
    sahi_detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolov8x.pt", # Using a general detection model like YOLOv8x
        confidence_threshold=sahi_conf_threshold, # Use parameter
        image_size=round_to_multiple(640, 32), # Ensure image_size is a multiple of 32
        device="cuda:0" # Ensure it runs on GPU
    )

    # Initialize a separate YOLO pose estimation model
    # This model will be used on cropped regions to find keypoints.
    pose_estimation_model = YOLO("yolo11x-pose.pt") 

    cap = cv.VideoCapture(source)

    orig_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv.CAP_PROP_FPS))
    
    # Crea la maschera trapezoidale con le dimensioni del frame ridimensionato (size)
    trap_mask, trap_points = create_trapezoid_mask(
        (size[1], size[0]),  # (height, width) - dimensioni ridimensionate
        TRAPEZOID_TOP_LEFT, 
        TRAPEZ_TOP_RIGHT, 
        TRAPEZ_BOTTOM_LEFT, 
        TRAPEZ_BOTTOM_RIGHT, 
        curve_height=CURVE_HEIGHT
    )
    
    # Inizializza video_writer UNA SOLA VOLTA con le dimensioni del frame ridimensionato
    video_writer = cv.VideoWriter("yolo-pose-predict-sahi.avi", cv.VideoWriter_fourcc(*"mp4v"), original_fps, (size[0], size[1]))
    
    if not video_writer.isOpened():
        print("ERRORE: Impossibile creare il video writer!")
        cap.release()
        return

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame, size)
        frame_count += 1

        # Applica la maschera trapezoidale al frame
        roi_frame_with_black = apply_trapezoid_mask(frame, trap_mask)
        
        # Calculate bounding box of the non-black (trapezoid) region in roi_frame_with_black
        y_nonzero, x_nonzero, _ = np.where(roi_frame_with_black > 0)
        
        cropped_roi_frame = np.zeros_like(frame) # Default to empty if no valid crop
        min_x, min_y = 0, 0 # Default offsets

        if y_nonzero.size > 0 and x_nonzero.size > 0:
            min_y, max_y = np.min(y_nonzero), np.max(y_nonzero)
            min_x, max_x = np.min(x_nonzero), np.max(x_nonzero)
            
            if max_y > min_y and max_x > min_x:
                cropped_roi_frame = roi_frame_with_black[min_y:max_y+1, min_x:max_x+1].copy()
            
        start_time = time.time()
        
        all_keypoints = []
        num_persons = 0
        num_detections = 0
        all_person_bboxes_on_full_frame = [] # New list to store adjusted bounding boxes

        if cropped_roi_frame.size > 0 and np.any(cropped_roi_frame):
            # Stage 1: Person detection using SAHI
            # Filtering for 'person' class (class_id 0 in COCO for YOLOv8)
            detection_results = get_sliced_prediction(
                cropped_roi_frame,
                sahi_detection_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                postprocess_match_metric="IOU",
                postprocess_match_threshold=sahi_iou_threshold, # Use parameter
                # slice_conf_threshold=0.05, # Removed due to potential TypeError
                verbose=0,
            )
            
            person_boxes = []
            if detection_results and detection_results.object_prediction_list:
                for prediction in detection_results.object_prediction_list:
                    # Assuming 'person' has class_id 0 in the model used for SAHI detection (e.g. COCO trained YOLOv8)
                    if prediction.category.id == 0 and prediction.score.value > sahi_conf_threshold: # Filter for persons with reasonable confidence
                        bbox = prediction.bbox.to_xyxy()
                        # Adjust bbox to be relative to the original frame's coordinate system
                        adjusted_bbox = (int(bbox[0] + min_x), int(bbox[1] + min_y), int(bbox[2] + min_x), int(bbox[3] + min_y))
                        person_boxes.append((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                        all_person_bboxes_on_full_frame.append(adjusted_bbox) # Store adjusted bbox
            
            num_detections = len(person_boxes)

            # Stage 2: Pose estimation for each detected person
            for person_bbox in person_boxes:
                p_x1, p_y1, p_x2, p_y2 = person_bbox
                
                # Crop the person region from the cropped_roi_frame
                person_crop = cropped_roi_frame[p_y1:p_y2, p_x1:p_x2]

                if person_crop.size > 0 and np.any(person_crop):
                    # Round up the image size for pose estimation to a multiple of 32
                    pose_imgsz = round_to_multiple(max(person_crop.shape[:2]), 32)
                    
                    # Run pose estimation on the person crop
                    pose_results = pose_estimation_model.predict(
                        person_crop,
                        conf=pose_conf_threshold, # Use parameter
                        iou=pose_iou_threshold,   # Use parameter
                        imgsz=pose_imgsz, # Use appropriate image size for pose model
                        device='cuda',
                        verbose=False
                    )

                    if pose_results and len(pose_results[0].keypoints.xy) > 0:
                        # Extract keypoints and adjust their coordinates
                        kpts = pose_results[0].keypoints.xy.cpu().numpy()[0]
                        # Adjust keypoint coordinates from person_crop -> cropped_roi_frame -> original_frame
                        adjusted_kpts = kpts + [min_x + p_x1, min_y + p_y1]
                        all_keypoints.append(adjusted_kpts)
                        num_persons += 1
        
        process_time = time.time() - start_time

        # Copia il frame originale per visualizzare
        output_frame = frame.copy()
        
        # Initialize roi_output with the masked frame, not a black canvas
        roi_output = roi_frame_with_black.copy()

        # Draw bounding boxes for detected people first
        for bbox_idx, bbox in enumerate(all_person_bboxes_on_full_frame):
            x1, y1, x2, y2 = bbox
            color_bbox = (0, 255, 255) # Cyan color for bounding box
            cv.rectangle(roi_output, (x1, y1), (x2, y2), color_bbox, 2)
            cv.putText(roi_output, f'Det: {bbox_idx}', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, color_bbox, 2)


        if num_persons > 0:
            for person_idx, person_kpts in enumerate(all_keypoints):
                color_line = (int(50 * person_idx) % 256, int(100 * person_idx) % 256, int(150 * person_idx) % 256)

                # Draw connections
                for start, end in connections:
                    if start < len(person_kpts) and end < len(person_kpts):
                        pt1, pt2 = person_kpts[start], person_kpts[end]
                        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                            cv.line(roi_output, (int(pt1[0]), int(pt1[1])),
                                     (int(pt2[0]), int(pt2[1])), color_line, 2)

                # Draw keypoints
                for pt in person_kpts:
                    if pt[0] > 0 and pt[1] > 0:
                        cv.circle(roi_output, (int(pt[0]), int(pt[1])), 3, color_line, -1)
                        cv.circle(roi_output, (int(pt[0]), int(pt[1])), 3, (255, 255, 255), 1)
                
                # Draw person index
                # Use keypoint 0 (nose) for text placement, or average if not available
                head_kpt = person_kpts[0] if len(person_kpts) > 0 else (0,0)
                if head_kpt[0] > 0 and head_kpt[1] > 0:
                    cv.putText(roi_output, f'P: {person_idx}', (int(head_kpt[0]) - 20, int(head_kpt[1]) - 15),
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, color_line, 2)

        # Draw the ROI trapezoid outline on the full frame
        draw_trapezoid_on_frame(output_frame, trap_points, color=(0, 255, 0), thickness=2)

        fps_val = 1.0 / process_time if process_time > 0 else 0  # Calculate FPS
        avg_fps.append(fps_val)
        fps_avg = sum(avg_fps[-30:]) / len(avg_fps[-30:]) if avg_fps else 0  # Media degli ultimi 30 frame

        draw_text_with_bg(output_frame, f'YOLO-Pose Predict (SAHI) FPS: {int(fps_avg)}', (50, 60), bg_color=(0, 150, 255), font_scale=1.2)
        draw_text_with_bg(output_frame, f'Time: {process_time * 1000:.1f}ms', (50, 145), bg_color=(0, 150, 255),
                          font_scale=1.2)
        draw_text_with_bg(output_frame, f'Frame: {frame_count} | Persone: {num_persons} | Detections: {num_detections}', (50, 230), bg_color=(0, 150, 255),
                          font_scale=1.2)

        draw_text_with_bg(roi_output, f'ROI - Persone: {num_persons}', (10, 40), bg_color=(0, 150, 255),
                          font_scale=1.0)

        # Visualizza sia il frame completo che la ROI
        cv.imshow('YOLO Pose Predict - Frame Completo (ROI evidenziata)', output_frame)
        cv.imshow('YOLO Pose Predict - Solo ROI (rilevamento)', roi_output)

        # Salva solo la ROI nel video
        video_writer.write(roi_output)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv.destroyAllWindows()


def yolo_sahi_pose_tracking(source=0, size=(1440, 810), sahi_conf_threshold=0.4, sahi_iou_threshold=0.7, pose_conf_threshold=0.5, pose_iou_threshold=0.7, tracker_iou_threshold=0.5):
    """
    Combines SAHI detection, YOLO pose estimation, and IOU-based tracking.
    
    Args:
        source: video source (e.g., 0 for webcam, "path/to/video.mp4").
        size: tuple (width, height) for resizing frames.
        sahi_conf_threshold: confidence threshold for initial SAHI person detection.
        sahi_iou_threshold: IOU threshold for SAHI's NMS.
        pose_conf_threshold: confidence threshold for keypoints detection by the pose model.
        pose_iou_threshold: IOU threshold for NMS in the pose model.
        tracker_iou_threshold: IOU threshold for our custom IOUTracker.
    """
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]

    sahi_detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolov8x.pt",
        confidence_threshold=sahi_conf_threshold,
        image_size=round_to_multiple(640, 32),
        device="cuda:0"
    )

    pose_estimation_model = YOLO("yolo11x-pose.pt")
    iou_tracker = IOUTracker(iou_threshold=tracker_iou_threshold)

    cap = cv.VideoCapture(source)

    orig_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv.CAP_PROP_FPS))
    
    trap_mask, trap_points = create_trapezoid_mask(
        (size[1], size[0]),
        TRAPEZOID_TOP_LEFT, 
        TRAPEZ_TOP_RIGHT, 
        TRAPEZ_BOTTOM_LEFT, 
        TRAPEZ_BOTTOM_RIGHT, 
        curve_height=CURVE_HEIGHT
    )
    
    video_writer = cv.VideoWriter("yolo-sahi-pose-tracking.avi", cv.VideoWriter_fourcc(*"mp4v"), original_fps, (size[0], size[1]))
    
    if not video_writer.isOpened():
        print("ERRORE: Impossibile creare il video writer!")
        cap.release()
        return

    frame_count = 0
    global avg_fps # Using the global list for FPS calculation

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame, size)
        frame_count += 1

        roi_frame_with_black = apply_trapezoid_mask(frame, trap_mask)
        
        y_nonzero, x_nonzero, _ = np.where(roi_frame_with_black > 0)
        
        cropped_roi_frame = np.zeros_like(frame)
        min_x, min_y = 0, 0

        if y_nonzero.size > 0 and x_nonzero.size > 0:
            min_y, max_y = np.min(y_nonzero), np.max(y_nonzero)
            min_x, max_x = np.min(x_nonzero), np.max(x_nonzero)
            
            if max_y > min_y and max_x > min_x:
                cropped_roi_frame = roi_frame_with_black[min_y:max_y+1, min_x:max_x+1].copy()
            
        start_time = time.time()
        
        current_frame_detections = [] # Store all detections with keypoints for current frame
        num_raw_detections = 0 # Count of SAHI person detections

        if cropped_roi_frame.size > 0 and np.any(cropped_roi_frame):
            # Stage 1: Person detection using SAHI
            detection_results = get_sliced_prediction(
                cropped_roi_frame,
                sahi_detection_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                postprocess_match_metric="IOU",
                postprocess_match_threshold=sahi_iou_threshold,
                verbose=0,
            )
            
            if detection_results and detection_results.object_prediction_list:
                for prediction in detection_results.object_prediction_list:
                    if prediction.category.id == 0 and prediction.score.value > sahi_conf_threshold:
                        bbox = prediction.bbox.to_xyxy()
                        # Adjust bbox to be relative to the original frame's coordinate system for tracking and drawing
                        adjusted_bbox_global = (int(bbox[0] + min_x), int(bbox[1] + min_y), int(bbox[2] + min_x), int(bbox[3] + min_y))
                        num_raw_detections += 1

                        p_x1, p_y1, p_x2, p_y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        person_crop = cropped_roi_frame[p_y1:p_y2, p_x1:p_x2]

                        if person_crop.size > 0 and np.any(person_crop):
                            pose_imgsz = round_to_multiple(max(person_crop.shape[:2]), 32)
                            pose_results = pose_estimation_model.predict(
                                person_crop,
                                conf=pose_conf_threshold,
                                iou=pose_iou_threshold,
                                imgsz=pose_imgsz,
                                device='cuda',
                                verbose=False
                            )

                            if pose_results and len(pose_results[0].keypoints.xy) > 0:
                                kpts = pose_results[0].keypoints.xy.cpu().numpy()[0]
                                adjusted_kpts_global = kpts + [min_x + p_x1, min_y + p_y1]
                                current_frame_detections.append({'bbox': adjusted_bbox_global, 'keypoints': adjusted_kpts_global})
        
        # Stage 3: Update tracker with current frame's detections
        tracked_objects = iou_tracker.update(current_frame_detections)
        num_persons = len(tracked_objects) # Number of unique tracked persons

        process_time = time.time() - start_time

        output_frame = frame.copy()
        roi_output = roi_frame_with_black.copy()
        
        # Draw tracked objects
        if num_persons > 0:
            for tracked_obj in tracked_objects:
                track_id = tracked_obj['track_id']
                bbox = tracked_obj['bbox']
                person_kpts = tracked_obj['keypoints']

                x1, y1, x2, y2 = bbox
                
                # Generate a color based on track_id for consistency
                color_line = (int(50 * track_id) % 256, int(100 * track_id) % 256, int(150 * track_id) % 256)

                # Draw bounding box
                cv.rectangle(roi_output, (x1, y1), (x2, y2), color_line, 2)
                
                # Draw connections
                for start, end in connections:
                    if start < len(person_kpts) and end < len(person_kpts):
                        pt1, pt2 = person_kpts[start], person_kpts[end]
                        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                            cv.line(roi_output, (int(pt1[0]), int(pt1[1])),
                                     (int(pt2[0]), int(pt2[1])), color_line, 2)

                # Draw keypoints
                for pt in person_kpts:
                    if pt[0] > 0 and pt[1] > 0:
                        cv.circle(roi_output, (int(pt[0]), int(pt[1])), 3, color_line, -1)
                        cv.circle(roi_output, (int(pt[0]), int(pt[1])), 3, (255, 255, 255), 1)
                
                # Draw track ID
                head_kpt = person_kpts[0] if len(person_kpts) > 0 else (0,0)
                if head_kpt[0] > 0 and head_kpt[1] > 0:
                    cv.putText(roi_output, f'ID: {track_id}', (int(head_kpt[0]) - 20, int(head_kpt[1]) - 15),
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, color_line, 2)

        # Draw the ROI trapezoid outline on the full frame
        draw_trapezoid_on_frame(output_frame, trap_points, color=(0, 255, 0), thickness=2)

        fps_val = 1.0 / process_time if process_time > 0 else 0
        avg_fps.append(fps_val)
        fps_avg = sum(avg_fps[-30:]) / len(avg_fps[-30:]) if avg_fps else 0

        draw_text_with_bg(output_frame, f'YOLO-SAHI-Pose Tracking FPS: {int(fps_avg)}', (50, 60), bg_color=(0, 100, 200), font_scale=1.2)
        draw_text_with_bg(output_frame, f'Time: {process_time * 1000:.1f}ms', (50, 145), bg_color=(0, 100, 200),
                          font_scale=1.2)
        draw_text_with_bg(output_frame, f'Frame: {frame_count} | Tracked: {num_persons} | Raw Detections: {num_raw_detections}', (50, 230), bg_color=(0, 100, 200),
                          font_scale=1.2)

        draw_text_with_bg(roi_output, f'ROI - Tracked Persons: {num_persons}', (10, 40), bg_color=(0, 100, 200),
                          font_scale=1.0)

        cv.imshow('YOLO SAHI Pose Tracking - Full Frame', output_frame)
        cv.imshow('YOLO SAHI Pose Tracking - ROI', roi_output)

        video_writer.write(roi_output)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # Usage examples:
    
    # Scegli quale versione usare:
    # 1. track() - Tracking consistente ma meno rilevamenti
    # You can now specify conf_threshold and iou_threshold
    # Try increasing iou_threshold if people are being merged, e.g., 0.7 or 0.8
    # yolo_pose_tracking("video raw/test_clip4.mp4", conf_threshold=0.3, iou_threshold=0.7)      # YOLO with tracking
    
    # 2. predict() - Massimi rilevamenti ma senza tracking persistente
    # You can now specify sahi_conf_threshold, sahi_iou_threshold, pose_conf_threshold, pose_iou_threshold
    # For SAHI detection (yolov8x.pt), increase sahi_iou_threshold if detections are merging, e.g., 0.7 or 0.8
    # For pose estimation (yolo11x-pose.pt), similarly adjust pose_iou_threshold
    # yolo_pose_predict("video raw/test_clip1.mp4", sahi_conf_threshold=0.2, sahi_iou_threshold=0.7, pose_conf_threshold=0.01, pose_iou_threshold=0.2)  # YOLO predict with direct method
    # yolo_pose_predict(0, sahi_conf_threshold=0.2, sahi_iou_threshold=0.7, pose_conf_threshold=0.3, pose_iou_threshold=0.7)           # YOLO predict with webcam and direct method

    # 3. yolo_sahi_pose_tracking() - SAHI detection + YOLO pose estimation + IOU-based tracking
    # Adjust `tracker_iou_threshold` for how aggressively detections are assigned to existing tracks.
    yolo_sahi_pose_tracking(
        "video raw/test_clip1.mp4",
        sahi_conf_threshold=0.2,
        sahi_iou_threshold=0.7,
        pose_conf_threshold=0.01,
        pose_iou_threshold=0.2,
        tracker_iou_threshold=0.5 # Adjust this threshold for tracking consistency
    )
    # yolo_sahi_pose_tracking(
    #     0, # for webcam
    #     sahi_conf_threshold=0.2,
    #     sahi_iou_threshold=0.7,
    #     pose_conf_threshold=0.01,
    #     pose_iou_threshold=0.2,
    #     tracker_iou_threshold=0.5
    # )          # YOLO predict with webcam and direct method