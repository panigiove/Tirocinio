import cv2 as cv
import numpy as np
import open3d as o3d
from ultralytics import YOLO
import time
from pathlib import Path

# Parametri della maschera trapezoidale
TRAPEZOID_TOP_LEFT = (210, 380)
TRAPEZOID_TOP_RIGHT = (1240, 350)
TRAPEZOID_BOTTOM_LEFT = (0, 665)
TRAPEZOID_BOTTOM_RIGHT = (1440, 665)
CURVE_HEIGHT = 65

# YOLO connections per scheletro umano
CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (12, 14), (13, 15), (14, 16)
]

def create_pose_mesh(keypoints_2d, image_height, image_width):
    """
    creates a simple 3d wireframe mesh from 2d keypoints.
    
    this is a simplified representation. for true 3d reconstruction,
    more advanced techniques (e.g., multi-view, smpl models) are needed.
    we assume a fixed depth for visualization here.
    """
    if len(keypoints_2d) == 0:
        return None

    # normalize keypoints to a 0-1 range based on image dimensions
    normalized_keypoints = np.array(keypoints_2d) / np.array([image_width, image_height, 1])

    # create a simple 3d representation:
    # use normalized x, y and assign a fixed small depth for z for visualization
    # let's assume 'z' value is proportional to the y-coordinate (height in image)
    # this will give a slight perspective, making "further" parts seem lower.
    points_3d = []
    for kp in normalized_keypoints:
        x, y, visibility = kp
        # simple depth mapping: points higher on the screen (smaller y) are further (larger z)
        # scale y from 0-1 to something like 0.5-1.5 for z
        z = 1.0 - y + 0.5 # invert y, add offset
        points_3d.append([x, y, z])
    points_3d = np.array(points_3d)

    # define connections for a human skeleton (based on common keypoint indices)
    # yolo's 17 keypoints:
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

    connections = [
        (0, 1), (1, 3), (0, 2), (2, 4),  # head
        (5, 6), (5, 11), (6, 12),  # torso (shoulders to hips)
        (11, 12),  # hip line
        (5, 7), (7, 9),  # left arm
        (6, 8), (8, 10), # right arm
        (11, 13), (13, 15), # left leg
        (12, 14), (14, 16)  # right leg
    ]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_3d),
        lines=o3d.utility.Vector2iVector(connections),
    )
    # assign a color to the lines (e.g., blue)
    line_set.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (len(connections), 1)))

    return line_set

def draw_text_with_bg(img, text, pos, font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Disegna testo con sfondo"""
    font = cv.FONT_HERSHEY_SIMPLEX
    thickness = 2
    padding = 20
    (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, thickness)
    rect_width = text_width + 2 * padding
    rect_height = text_height + baseline + 2 * padding
    x, y = pos
    rect_x1 = x
    rect_y1 = y - text_height
    rect_x2 = rect_x1 + rect_width
    rect_y2 = rect_y1 + rect_height
    cv.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
    text_x = rect_x1 + padding
    text_y = rect_y1 + padding + text_height
    cv.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness)

def create_trapezoid_mask(frame_shape, tl, tr, bl, br, curve_height=0):
    """Crea una maschera trapezoidale con base curva"""
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    points = [np.array([tl, tr, br, bl], dtype=np.int32)]
    
    if curve_height > 0:
        points_curve = [tl, tr]
        points_curve.append(br)
        
        for i in range(1, 20):
            t = i / 20.0
            x = br[0] + (bl[0] - br[0]) * t
            y = br[1] + (bl[1] - br[1]) * t + curve_height * (4 * t * (1 - t))
            points_curve.append((int(x), int(y)))
        
        points_curve.append(bl)
        points = [np.array(points_curve, dtype=np.int32)]
    
    cv.fillPoly(mask, points, 255)
    return mask, points[0]

def apply_trapezoid_mask(frame, mask):
    """Applica la maschera trapezoidale al frame"""
    masked_frame = cv.bitwise_and(frame, frame, mask=mask)
    return masked_frame

def draw_trapezoid_on_frame(frame, points, color=(0, 255, 0), thickness=2):
    """Disegna il contorno del trapezio sul frame"""
    cv.polylines(frame, [points], isClosed=True, color=color, thickness=thickness)

def save_mesh_to_ply(mesh, filepath):
    """Salva una mesh in formato PLY"""
    o3d.io.write_line_set(filepath, mesh)

def process_video(video_path="video raw/test_clip1.mp4", size=(1440, 810)):
    """
    Processa il video con YOLO pose tracking, salva il video con keypoint e mesh,
    e salva le mesh dei giocatori in file separati.
    """
    # Crea cartella per i risultati
    results_dir = Path("Results/meshes")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load a pre-trained YOLOv11x-pose model
    model = YOLO("yolo11x-pose.pt")

    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("error: could not open video.")
        return

    # Prendi le dimensioni originali del video
    orig_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv.CAP_PROP_FPS))
    
    # Crea la maschera trapezoidale
    trap_mask, trap_points = create_trapezoid_mask(
        (size[1], size[0]),
        TRAPEZOID_TOP_LEFT,
        TRAPEZOID_TOP_RIGHT,
        TRAPEZOID_BOTTOM_LEFT,
        TRAPEZOID_BOTTOM_RIGHT,
        curve_height=CURVE_HEIGHT
    )
    
    # Inizializza video writer per salvare il video processato
    output_video_path = "Results/yolo-pose-with-mesh.avi"
    video_writer = cv.VideoWriter(
        output_video_path,
        cv.VideoWriter_fourcc(*"mp4v"),
        original_fps,
        (size[0], size[1])
    )
    
    if not video_writer.isOpened():
        print("ERRORE: Impossibile creare il video writer!")
        cap.release()
        return
    
    # Inizializza open3d visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3d pose visualization', width=800, height=600)
    
    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    render_option.line_width = 3.0
    render_option.background_color = np.asarray([0, 0, 0])

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vis.add_geometry(coord_frame)

    frame_count = 0
    all_meshes = []  # Lista per salvare tutte le mesh
    track_ids = set()  # Per tracciare gli ID unici dei giocatori
    
    avg_fps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv.resize(frame, size)
        frame_count += 1
        
        # Applica la maschera trapezoidale
        roi_frame = apply_trapezoid_mask(frame, trap_mask)
        imgsz_roi = max(orig_width, orig_height)

        start_time = time.time()
        
        # Usa track() dalla logica di yolo11_pose.py
        results = model.track(
            roi_frame,
            device='cuda',
            tracker='bytetrack.yaml',
            show=False,
            conf=0.005,
            iou=0.25,
            imgsz=imgsz_roi,
            max_det=300,
            persist=True,
            verbose=False
        )
        
        process_time = time.time() - start_time

        # Copia il frame per visualizzare
        output_frame = frame.copy()
        roi_output = roi_frame.copy()
        num_persons = 0
        current_frame_meshes = []
        
        if results and len(results) > 0:
            num_detections = len(results[0].boxes) if results[0].boxes else 0
            
            if results[0].keypoints is not None:
                keypoints_list = results[0].keypoints.xy.cpu().numpy()
                keypoints_data = results[0].keypoints.data.cpu().numpy()  # Include confidence
                track_ids_frame = results[0].boxes.id if results[0].boxes.id is not None else None
                num_persons = len(keypoints_list)
                
                print(f"Frame {frame_count} | Rilevamenti: {num_detections} | Persone: {num_persons}")

                for person_idx, person_kpts in enumerate(keypoints_list):
                    # Ottieni ID di tracciamento
                    if track_ids_frame is not None:
                        track_id = int(track_ids_frame[person_idx])
                        track_ids.add(track_id)
                        color_line = (
                            int(50 * track_id) % 256,
                            int(100 * track_id) % 256,
                            int(150 * track_id) % 256
                        )
                    else:
                        track_id = person_idx
                        color_line = (
                            int(50 * person_idx) % 256,
                            int(100 * person_idx) % 256,
                            int(150 * person_idx) % 256
                        )

                    # Disegna connessioni e keypoint sul video
                    for start, end in CONNECTIONS:
                        if start < len(person_kpts) and end < len(person_kpts):
                            pt1, pt2 = person_kpts[start], person_kpts[end]
                            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                                cv.line(
                                    roi_output,
                                    (int(pt1[0]), int(pt1[1])),
                                    (int(pt2[0]), int(pt2[1])),
                                    color_line,
                                    2
                                )

                    for pt in person_kpts:
                        if pt[0] > 0 and pt[1] > 0:
                            cv.circle(roi_output, (int(pt[0]), int(pt[1])), 3, color_line, -1)
                            cv.circle(roi_output, (int(pt[0]), int(pt[1])), 3, (255, 255, 255), 1)

                    # Disegna ID di tracciamento
                    head_kpt = person_kpts[0]
                    if head_kpt[0] > 0 and head_kpt[1] > 0:
                        cv.putText(
                            roi_output,
                            f'ID: {track_id}',
                            (int(head_kpt[0]) - 20, int(head_kpt[1]) - 15),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color_line,
                            2
                        )

                    # Crea la mesh 3D usando i keypoint con confidence
                    if keypoints_data.shape[0] > person_idx:
                        person_kpts_3d = keypoints_data[person_idx]
                        mesh_line_set = create_pose_mesh(person_kpts_3d, size[1], size[0])
                        if mesh_line_set:
                            current_frame_meshes.append((track_id, mesh_line_set))
                            all_meshes.append((frame_count, track_id, mesh_line_set))

        # Disegna il trapezio della ROI
        draw_trapezoid_on_frame(output_frame, trap_points, color=(0, 255, 0), thickness=2)

        # Calcola FPS
        fps_val = 1.0 / process_time if process_time > 0 else 0
        avg_fps.append(fps_val)
        fps_avg = sum(avg_fps[-30:]) / len(avg_fps[-30:]) if avg_fps else 0

        # Disegna info
        draw_text_with_bg(
            output_frame,
            f'YOLO-Pose Tracking FPS: {int(fps_avg)}',
            (50, 60),
            bg_color=(255, 27, 108),
            font_scale=1.2
        )
        draw_text_with_bg(
            output_frame,
            f'Time: {process_time * 1000:.1f}ms',
            (50, 145),
            bg_color=(255, 27, 108),
            font_scale=1.2
        )
        draw_text_with_bg(
            output_frame,
            f'Frame: {frame_count} | Persone: {num_persons}',
            (50, 230),
            bg_color=(255, 27, 108),
            font_scale=1.2
        )

        draw_text_with_bg(
            roi_output,
            f'ROI - Persone: {num_persons}',
            (10, 40),
            bg_color=(255, 27, 108),
            font_scale=1.0
        )

        # Aggiorna visualizzazione open3d
        vis.clear_geometries()
        vis.add_geometry(coord_frame)
        
        for track_id, mesh in current_frame_meshes:
            vis.add_geometry(mesh)
        
        vis.poll_events()
        vis.update_renderer()

        # Salva il video processato
        video_writer.write(roi_output)

        # Visualizza i frame
        cv.imshow("YOLO Pose - Frame Completo (ROI evidenziata)", output_frame)
        cv.imshow("YOLO Pose - Solo ROI (rilevamento)", roi_output)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

        if frame_count % 100 == 0:
            print(f"Elaborati {frame_count} frame.")

    cap.release()
    video_writer.release()
    cv.destroyAllWindows()
    vis.destroy_window()
    
    # Salva tutte le mesh in un file PLY unico
    if all_meshes:
        print(f"\nSalvataggio mesh dei {len(track_ids)} giocatori rilevati...")
        
        # Combina tutte le mesh in un'unica struttura
        all_points = []
        all_lines = []
        point_offset = 0
        
        for frame_idx, track_id, mesh in all_meshes:
            # Estrai punti e linee dalla mesh
            if isinstance(mesh, o3d.geometry.LineSet):
                points = np.asarray(mesh.points)
                lines = np.asarray(mesh.lines)
                
                all_points.append(points)
                
                # Aggiusta gli indici delle linee in base all'offset
                adjusted_lines = lines + point_offset
                all_lines.append(adjusted_lines)
                
                point_offset += len(points)
        
        if all_points:
            combined_points = np.vstack(all_points)
            combined_lines = np.vstack(all_lines)
            
            combined_mesh = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(combined_points),
                lines=o3d.utility.Vector2iVector(combined_lines)
            )
            
            mesh_output_path = str(results_dir / "all_players_mesh.ply")
            save_mesh_to_ply(combined_mesh, mesh_output_path)
            print(f"Mesh combinata salvata in: {mesh_output_path}")
        
        # Salva mesh per ogni giocatore separatamente
        for track_id in track_ids:
            player_meshes = [mesh for frame_idx, tid, mesh in all_meshes if tid == track_id]
            
            if player_meshes:
                # Combina le mesh di un singolo giocatore
                player_points = []
                player_lines = []
                player_point_offset = 0
                
                for mesh in player_meshes:
                    if isinstance(mesh, o3d.geometry.LineSet):
                        points = np.asarray(mesh.points)
                        lines = np.asarray(mesh.lines)
                        
                        player_points.append(points)
                        adjusted_lines = lines + player_point_offset
                        player_lines.append(adjusted_lines)
                        player_point_offset += len(points)
                
                if player_points:
                    combined_player_points = np.vstack(player_points)
                    combined_player_lines = np.vstack(player_lines)
                    
                    player_mesh = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(combined_player_points),
                        lines=o3d.utility.Vector2iVector(combined_player_lines)
                    )
                    
                    player_mesh_path = str(results_dir / f"player_ID_{track_id}_mesh.ply")
                    save_mesh_to_ply(player_mesh, player_mesh_path)
                    print(f"Mesh giocatore ID {track_id} salvata in: {player_mesh_path}")
    
    print(f"\nVideo processato salvato in: {output_video_path}")
    print(f"Frame totali elaborati: {frame_count}")
    print(f"Giocatori rilevati: {len(track_ids)}")


if __name__ == "__main__":
    print("Avvio processamento video. Premi 'q' per uscire.")
    process_video(video_path="video raw/test_clip1.mp4")