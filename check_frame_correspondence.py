import os
import cv2
import glob

# Paths
folder = 'tracking_12.v4i.yolov11/train/labels/'
image_folder = 'tracking_12.v4i.yolov11/train/images/'
video1 = 'Tracking/material4project/video/tracking_12/out4.mp4'
video2 = 'Tracking/material4project/video/tracking_12/out13.mp4'

# Get list of annotation files
txt_files = glob.glob(os.path.join(folder, '*.txt'))

correspondence = {}
found_all = False

for txt_file in txt_files:
    base = os.path.basename(txt_file).replace('.txt', '')
    
    # Find corresponding image
    img_path = os.path.join(image_folder, base + '.png')
    if not os.path.exists(img_path):
        img_path = os.path.join(image_folder, base + '.jpg')
    if not os.path.exists(img_path):
        continue
    
    # Extract frame number from filename
    # Expected format: out4_frame_0001_png.rf...
    parts = base.split('_')
    frame_num = None
    if 'frame' in parts:
        idx = parts.index('frame')
        if idx + 1 < len(parts):
            frame_str = parts[idx + 1]
            try:
                frame_num = int(frame_str) - 1  # Convert to 0-based
            except ValueError:
                pass
    
    if frame_num is None:
        continue
    
    # Determine which video
    if 'out4' in base:
        video_path = video1
    elif 'out13' in base:
        video_path = video2
    else:
        continue
    
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    # Load the video frame at the expected position
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        continue
    
    # Check if they match exactly
    if img.shape == frame.shape and (img == frame).all():
        correspondence[base] = frame_num
        continue
    
    # If not exact match, search nearby frames (+/- 50)
    best_frame = None
    min_diff = float('inf')
    for offset in range(-50, 51):
        fnum = frame_num + offset
        if fnum < 0:
            continue
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ret, fr = cap.read()
        cap.release()
        
        if ret and fr.shape == img.shape:
            # Use L2 norm for difference
            diff = cv2.norm(img, fr, cv2.NORM_L2)
            if diff < min_diff:
                min_diff = diff
                best_frame = fnum
    
    if best_frame is not None:
        correspondence[base] = best_frame

# Print results
print("Image to Frame Correspondence:")
for img_name, frame_num in sorted(correspondence.items(), key=lambda x: x[1]):
    print(f"{img_name} -> Frame {frame_num} (0-based)")

print(f"\nTotal correspondences found: {len(correspondence)}")
print(f"Total annotation files processed: {len(txt_files)}")

if len(correspondence) == len(txt_files):
    print("All images have been matched to frames.")
else:
    print(f"Some images could not be matched. Found {len(correspondence)} out of {len(txt_files)}.")