import cv2 as cv
import numpy as np
import time

avg_fps = []
video_writer = None

# Parametri della maschera trapezoidale con base curva
# Facilmente regolabili per adattarsi alla propria scena
TRAPEZOID_TOP_LEFT = (210, 380)      # Angolo superiore sinistro
TRAPEZOID_TOP_RIGHT = (1240, 350)    # Angolo superiore destro
TRAPEZOID_BOTTOM_LEFT = (0, 665)     # Angolo inferiore sinistro
TRAPEZOID_BOTTOM_RIGHT = (1440, 665) # Angolo inferiore destro
CURVE_HEIGHT = 65                   # Altezza della curva della base (0 = retta, più alto = più curva)

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
            # Interpolazione parabolica sulla base inferiore
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
    cv.polylines(frame, [points], isClosed=True, color=color, thickness=thickness)


def yolo_pose_tracking(source=0, size=(1440, 810)):
    from ultralytics import YOLO

    # Usa il modello più grande per miglior accuratezza
    yolo_model = YOLO("yolo11x-pose.pt")  # Initialize YOLO with larger model

    """ results = yolo_model.train(
        data = "coco8-pose.yaml",
        epochs = 100,
        imgsz = 640,
        batch = 16,
        lr0 = 0.001,
        lrf = 0.01,
        momentum = 0.9,
        weight_decay = 0.0005,
        warmup_epochs = 5.0,
        augment = False,
        degrees = 10.0,
        translate = 0.1,
        scale = 0.2,
        flipud = 0.0,
        fliplr = 0.7,
        mosaic = 0.0,
        mixup = 1.0,
        device = "cpu",
        resume = False,
        save_period = 5,
        name = "pose_experiment",
        exist_ok = True
    ) """

    # YOLO connections
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]

    cap = cv.VideoCapture(source)

    orig_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv.CAP_PROP_FPS))
    
    # Crea la maschera trapezoidale con le dimensioni del frame ridimensionato (size)
    trap_mask, trap_points = create_trapezoid_mask(
        (size[1], size[0]),  # (height, width) - dimensioni ridimensionate
        TRAPEZOID_TOP_LEFT, 
        TRAPEZOID_TOP_RIGHT, 
        TRAPEZOID_BOTTOM_LEFT, 
        TRAPEZOID_BOTTOM_RIGHT, 
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
        imgsz_roi = max(orig_width, orig_height)

        start_time = time.time()
        # Usa track() con parametri per tracking più consistente
        # conf=0.005: confidence ragionevole per evitare troppi falsi positivi
        # iou=0.25: IoU moderato per bilanciare rilevamenti e soppressioni
        # max_det=300: numero massimo di rilevamenti per frame
        results = yolo_model.track(
            roi_frame,
            device='cuda',
            tracker='bytetrack.yaml',
            show=False,
            conf=0.005,                    # Confidence leggermente più alto per stabilità
            iou=0.25,                      # IoU moderato
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


def yolo_pose_predict(source=0, size=(1440, 810)):
    """Versione senza tracking - usa predict() per massimizzare i rilevamenti"""
    from ultralytics import YOLO

    # Usa il modello più grande per miglior accuratezza
    yolo_model = YOLO("yolo11x-pose.pt")  # Initialize YOLO with larger model

    # YOLO connections
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]

    cap = cv.VideoCapture(source)

    orig_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv.CAP_PROP_FPS))
    
    # Crea la maschera trapezoidale con le dimensioni del frame ridimensionato (size)
    trap_mask, trap_points = create_trapezoid_mask(
        (size[1], size[0]),  # (height, width) - dimensioni ridimensionate
        TRAPEZOID_TOP_LEFT, 
        TRAPEZOID_TOP_RIGHT, 
        TRAPEZOID_BOTTOM_LEFT, 
        TRAPEZOID_BOTTOM_RIGHT, 
        curve_height=CURVE_HEIGHT
    )
    
    # Calcola i bounds della maschera per il video writer
    x_coords = trap_points[:, 0]
    y_coords = trap_points[:, 1]
    roi_width = int(np.max(x_coords) - np.min(x_coords))
    roi_height = int(np.max(y_coords) - np.min(y_coords))
    
    # Inizializza video_writer UNA SOLA VOLTA con le dimensioni del frame ridimensionato
    video_writer = cv.VideoWriter("yolo-pose-predict.avi", cv.VideoWriter_fourcc(*"mp4v"), original_fps, (size[0], size[1]))
    
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
        imgsz_roi = max(orig_width, orig_height)

        start_time = time.time()
        # Usa predict() per massimizzare i rilevamenti senza tracking
        # conf=0.001: confidence molto basso per rilevare persone piccole/lontane
        # iou=0.2: IoU basso per ridurre le soppressioni di box
        # max_det=300: numero massimo di rilevamenti per frame
        results = yolo_model.predict(
            roi_frame,
            device='cuda',
            conf=0.01,                    # Confidence molto basso
            iou=0.2,                       # IoU basso
            imgsz=imgsz_roi, 
            max_det=300,
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
                num_persons = len(keypoints_list)
                print(f"Rilevamenti: {num_detections} | Persone: {num_persons}")  # DEBUG

                for person_idx, person_kpts in enumerate(keypoints_list):
                    # Colore diverso per ogni persona
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
                    
                    # Disegna il numero della persona (index)
                    head_kpt = person_kpts[0]
                    if head_kpt[0] > 0 and head_kpt[1] > 0:
                        cv.putText(roi_output, f'P: {person_idx}', (int(head_kpt[0]) - 20, int(head_kpt[1]) - 15),
                                  cv.FONT_HERSHEY_SIMPLEX, 0.7, color_line, 2)
            else:
                print(f"Rilevamenti trovati ma nessun keypoint")  # DEBUG

        # Disegna il trapezio della ROI sul frame originale
        draw_trapezoid_on_frame(output_frame, trap_points, color=(0, 255, 0), thickness=2)

        fps_val = 1.0 / process_time if process_time > 0 else 0  # Calculate FPS
        avg_fps.append(fps_val)
        fps_avg = sum(avg_fps[-30:]) / len(avg_fps[-30:]) if avg_fps else 0  # Media degli ultimi 30 frame

        draw_text_with_bg(output_frame, f'YOLO-Pose Predict (No Tracking) FPS: {int(fps_avg)}', (50, 60), bg_color=(0, 150, 255), font_scale=1.2)
        draw_text_with_bg(output_frame, f'Time: {process_time * 1000:.1f}ms', (50, 145), bg_color=(0, 150, 255),
                          font_scale=1.2)
        draw_text_with_bg(output_frame, f'Frame: {frame_count} | Persone: {num_persons} | Rilevamenti: {num_detections}', (50, 230), bg_color=(0, 150, 255),
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


if __name__ == "__main__":
    # Usage examples:
    
    # Scegli quale versione usare:
    # 1. track() - Tracking consistente ma meno rilevamenti
    #yolo_pose_tracking("video raw/test_clip4.mp4")      # YOLO with tracking
    
    # 2. predict() - Massimi rilevamenti ma senza tracking persistente
    yolo_pose_predict("video raw/test_clip2.mp4")  # YOLO senza tracking

    #yolo_pose(0)                   # YOLO with webcam
    #yolo_pose_predict(0)           # YOLO predict with webcam
    