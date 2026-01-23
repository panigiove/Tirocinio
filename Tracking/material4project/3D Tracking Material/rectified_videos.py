import cv2
import numpy as np
import json
import os
import glob
import re

# ---------------- CONFIG MINIMA ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # dove si trova lo script (es. .../material4project/3D Camera Tracking)
# project root = material4project
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# la cartella video (fratella di "3D Camera Tracking")
VIDEOS_ROOT = os.path.join(PROJECT_ROOT, "video")

# cartella di output (verrà creata se non esiste)
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "Rectified videos")

# pattern file video
VIDEO_PATTERN = "out*.mp4"
# ------------------------------------------------

def load_calibration(calib_path):
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist

def find_calibration_for_cam(cam_index):
    """
    Prova vari percorsi sotto PROJECT_ROOT dove potrebbe esserci camera_calib.json.
    Se non trova, fa una ricerca ricorsiva.
    """
    candidates = [
        os.path.join(PROJECT_ROOT, "3D Tracking Material", "camera_data", f"cam_{cam_index}", "calib", "camera_calib.json"),
        os.path.join(PROJECT_ROOT, "3D Tracking Material", "camera_data", f"cam_{cam_index}", "camera_calib.json"),
        os.path.join(PROJECT_ROOT, "camera_data", f"cam_{cam_index}", "calib", "camera_calib.json"),
        os.path.join(PROJECT_ROOT, "camera_data", f"cam_{cam_index}", "camera_calib.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # fallback: cerca ricorsivamente sotto PROJECT_ROOT
    pattern = os.path.join(PROJECT_ROOT, "**", f"cam_{cam_index}", "calib", "camera_calib.json")
    found = glob.glob(pattern, recursive=True)
    if found:
        return found[0]

    # altro fallback: qualunque file chiamato camera_calib.json in una cartella cam_<index>
    pattern2 = os.path.join(PROJECT_ROOT, "**", f"cam_{cam_index}", "*camera_calib.json")
    found2 = glob.glob(pattern2, recursive=True)
    if found2:
        return found2[0]

    return None

def process_video(video_path, calib_path, output_path):
    try:
        mtx, dist = load_calibration(calib_path)
    except Exception as e:
        print(f"Errore caricamento calibrazione ({calib_path}): {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # assicurati che la cartella di output esista prima di aprire il writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # precompute undistortion map
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    pts = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    pts = pts.reshape(-1, 1, 2)

    undistorted_pts = cv2.undistortPoints(pts, mtx, dist, P=mtx)
    undistorted_map = undistorted_pts.reshape(height, width, 2)
    map_x = undistorted_map[:, :, 0].astype(np.float32)
    map_y = undistorted_map[:, :, 1].astype(np.float32)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rectified_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        out.write(rectified_frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames for {video_path}")

    cap.release()
    out.release()
    print(f"Finished: {video_path} -> {output_path}")

def main():
    # verifica che la cartella video esista
    if not os.path.exists(VIDEOS_ROOT):
        print(f"Cartella video non trovata: {VIDEOS_ROOT}")
        print("Controlla la variabile VIDEOS_ROOT nello script.")
        return

    # trova tutti i video ricorsivamente sotto VIDEOS_ROOT
    pattern = os.path.join(VIDEOS_ROOT, "**", VIDEO_PATTERN)
    video_files = glob.glob(pattern, recursive=True)
    if not video_files:
        print("Nessun video trovato con pattern:", pattern)
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for video_path in video_files:
        # percorso relativo rispetto alla cartella video -> mantiene struttura
        rel_path = os.path.relpath(video_path, VIDEOS_ROOT)
        basename = os.path.basename(video_path)

        # estrai indice camera dal nome file (outN.mp4)
        match = re.search(r'out(\d+)\.mp4', basename)
        if not match:
            print("Impossibile estrarre indice camera (skipping):", video_path)
            continue
        cam_index = match.group(1)

        # trova calibrazione
        calib_path = find_calibration_for_cam(cam_index)
        if not calib_path:
            print(f"Calibration file per cam_{cam_index} NON trovato. Skipping {video_path}")
            continue

        # costruisci output path replicando la struttura
        output_path = os.path.join(OUTPUT_ROOT, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"Processing: {video_path}")
        print(f"  calib: {calib_path}")
        print(f"  out:   {output_path}")
        process_video(video_path, calib_path, output_path)

if __name__ == "__main__":
    main()


