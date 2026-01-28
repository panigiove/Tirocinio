import cv2
import numpy as np
import os
from ultralytics import YOLO

# Paths
rectified_video = r'Tracking/material4project/Rectified videos/tracking_12/out13.mp4'
rectified_labels = r'tracking_12.v2i.yolov11/train/labels/out13_frame_0001_png.rf.68fc7ed57b749ab73105139ae9ec4f7e_rectified.txt'
model_path = 'yolo11x.pt'

def load_yolo_labels(path, img_width, img_height):
    labels = []
    if not os.path.exists(path):
        return labels
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x1 = (cx - w/2) * img_width
            y1 = (cy - h/2) * img_height
            x2 = (cx + w/2) * img_width
            y2 = (cy + h/2) * img_height
            labels.append({'cls': cls, 'bbox': [x1, y1, x2, y2]})
    return labels

def main():
    # 1. Load first frame of rectified video
    cap = cv2.VideoCapture(rectified_video)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read video")
        return
    
    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    # 2. Run YOLO detection
    model = YOLO(model_path)
    results = model(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        if int(box.cls) == 0: # Person
            detections.append(box.xyxy[0].cpu().numpy().tolist())
    
    print(f"YOLO found {len(detections)} persons")

    # 3. Load rectified labels
    anno_labels = load_yolo_labels(rectified_labels, w, h)
    print(f"Loaded {len(anno_labels)} rectified annotations")

    # 4. Compare
    # For each annotation, find the closest YOLO detection
    for i, anno in enumerate(anno_labels):
        ab = anno['bbox']
        center_anno = [(ab[0]+ab[2])/2, (ab[1]+ab[3])/2]
        
        min_dist = float('inf')
        best_match = None
        for db in detections:
            center_det = [(db[0]+db[2])/2, (db[1]+db[3])/2]
            dist = np.sqrt((center_anno[0]-center_det[0])**2 + (center_anno[1]-center_det[1])**2)
            if dist < min_dist:
                min_dist = dist
                best_match = db
        
        if best_match:
            print(f"Anno {i}: Dist to best YOLO match = {min_dist:.2f} pixels")
            print(f"  Anno bbox: {ab}")
            print(f"  YOLO bbox: {best_match}")
        else:
            print(f"Anno {i}: No YOLO match found")

if __name__ == '__main__':
    main()
