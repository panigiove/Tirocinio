import cv2
import numpy as np
import json
import os
import glob
import re

def load_calibration(calib_path):
    # Load the camera calibration parameters from a JSON file.
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist

def process_video(video_path, calib_path, output_path):
    mtx, dist = load_calibration(calib_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    pts = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    pts = pts.reshape(-1, 1, 2)

    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.25, (w, h))
        undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        out.write(undistorted_frame)
        frame_count += 1
        print(f"Processed {frame_count} frame for {video_path}")
    
    cap.release()
    out.release()
    print(f"Finished processing video: {video_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    material4project_dir = os.path.dirname(script_dir)
    
    video_dir = os.path.join(material4project_dir, "video")
    output_base_dir = os.path.join(material4project_dir, "Rectified videos")
    
    video_files = glob.glob(os.path.join(video_dir, "**", "out*.mp4"), recursive=True)
    
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    for video_path in video_files:
        
        basename = os.path.basename(video_path)
        match = re.search(r'out(\d+)\.mp4', basename)
        if match:
            cam_index = match.group(1)
            
            calib_path = os.path.join(script_dir, "camera_data", f"cam_{cam_index}", "calib", "camera_calib.json")
        else:
            print("Could not extract camera index from filename:", video_path)
            continue
        
        ## Create one folder for each sample e.g. tracking_01, mocap_1, hpe_1
        rel_subdir = os.path.relpath(os.path.dirname(video_path), video_dir)
        output_dir = os.path.join(output_base_dir, rel_subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path = os.path.join(output_dir, basename)
            
        print(f"Processing {video_path} using calibration file {calib_path}...")
        process_video(video_path, calib_path, output_path)

if __name__ == "__main__":
    main()
