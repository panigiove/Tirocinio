import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load calibration data
def load_calib(path):
    with open(path, 'r') as f:
        return json.load(f)

# Camera calibration paths (adjust if needed)
calib_paths = {
    'cam_4': r'Tracking/material4project/3D Tracking Material/camera_data_with_Rvecs/camera_data/cam_4/calib/camera_calib_real.json',
    'cam_13': r'Tracking/material4project/3D Tracking Material/camera_data_with_Rvecs/camera_data/cam_13/calib/camera_calib_real.json',
    'cam_2': r'Tracking/material4project/3D Tracking Material/camera_data_with_Rvecs/camera_data/cam_2/calib/camera_calib_real.json'
}

# Load positions
positions = {}
for cam, path in calib_paths.items():
    calib = load_calib(path)
    positions[cam] = np.array(calib['tvecs']).flatten()

# Print absolute positions
print("Absolute Camera Positions (mm):")
for cam, pos in positions.items():
    print(f"{cam}: {pos}")

# Print relative to cam_13
cam13_pos = positions['cam_13']
print("\nRelative to Cam_13 (mm):")
for cam, pos in positions.items():
    if cam != 'cam_13':
        rel_pos = pos - cam13_pos
        print(f"{cam}: {rel_pos}")

# 3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = {'cam_4': 'red', 'cam_13': 'blue', 'cam_2': 'green'}
for cam, pos in positions.items():
    ax.scatter(pos[0], pos[1], pos[2], color=colors[cam], s=100, label=f'{cam} ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Camera Positions in World Coordinates')
ax.legend()
plt.show()