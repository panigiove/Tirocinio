import os
path = 'tracking_12.v2i.yolov11/train/labels'
if os.path.exists(path):
    files = os.listdir(path)
    print(f"Files in {path}: {files}")
    for f in files:
        if '_rectified' in f:
            os.remove(os.path.join(path, f))
            print(f"Removed {f}")
else:
    print(f"Path {path} not found")
