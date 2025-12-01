import cv2
import os
import numpy as np
import open3d as o3d
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R_scipy

# Define the connections for a 17-point pose skeleton
# These are standard COCO keypoint connections
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head and eyes/ears
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Shoulders, elbows, wrists
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Hips, knees, ankles
    # Torso connections (added)
    (5, 11),  # Left shoulder to left hip
    (6, 12),  # Right shoulder to right hip
    (11, 12)  # Left hip to right hip
]

# Standard COCO 17 keypoint indices:
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,\
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,\
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,\
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

# Colors for keypoints and connections (B, G, R)
KEYPOINT_COLOR = (0, 255, 0)  # Green
CONNECTION_COLOR = (0, 0, 255) # Blue

def create_3d_skeleton(keypoints_data, frame_width, frame_height, z_offset=0, confidence_threshold=0.5):
    """
    Creates a simplified 3D mesh representation from 2D keypoints using Open3D.
    Each keypoint is represented by a sphere, and each connection by a cylinder.
    Assumes a fixed Z-coordinate for all points for basic 3D visualization.
    Only includes keypoints and connections above a certain confidence threshold.
    """
    points_3d = {} # Store 3D points as a dictionary to easily access by original index
    
    for i, kp_data in enumerate(keypoints_data):
        if kp_data[2].item() > confidence_threshold:
            x, y = kp_data[0].item(), kp_data[1].item()
            # Scale x and y to be within a reasonable range (e.g., -1 to 1) for 3D visualization
            # Also, invert y-axis for typical 3D coordinate systems (Open3D's Y is up)
            x_norm = (x / frame_width) * 2 - 1
            y_norm = (y / frame_height) * -2 + 1 # Invert Y for typical 3D view (Y up)
            points_3d[i] = np.array([x_norm, y_norm, z_offset])

    geometries = []
    
    # Create spheres for keypoints
    for i, p in points_3d.items():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03) # Adjust radius as needed
        sphere.translate(p)
        sphere.paint_uniform_color([c / 255.0 for c in KEYPOINT_COLOR])
        geometries.append(sphere)

    # Create cylinders for connections (bones)
    for connection in POSE_CONNECTIONS:
        idx1, idx2 = connection
        # Only draw connection if both keypoints are present and met confidence threshold
        if idx1 in points_3d and idx2 in points_3d:
            p1 = points_3d[idx1]
            p2 = points_3d[idx2]

            # Calculate direction vector and distance
            direction = p2 - p1
            distance = np.linalg.norm(direction)

            if distance == 0: # Avoid division by zero if keypoints are at the same location
                continue

            # Create cylinder, default is along Z-axis
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=distance) # Adjust radius
            
            # Position the cylinder at the midpoint between p1 and p2
            midpoint = (p1 + p2) / 2

            # Target direction for the cylinder (normalized)
            target_direction = direction / distance 
            
            # Initial direction of the cylinder (along Z-axis)
            initial_direction = np.array([0., 0., 1.])

            # Calculate rotation to align initial_direction with target_direction
            rotation_vec, _ = R_scipy.align_vectors(target_direction[np.newaxis, :], initial_direction[np.newaxis, :])
            # Open3D's rotate method uses a rotation matrix. Convert scipy.Rotation to matrix.
            rotation_matrix = rotation_vec.as_matrix()

            # Apply rotation and then translate
            cylinder.rotate(rotation_matrix, center=np.array([0.,0.,0.])) # Rotate around its own center
            cylinder.translate(midpoint) # Translate to the midpoint
            
            cylinder.paint_uniform_color([c / 255.0 for c in CONNECTION_COLOR])
            geometries.append(cylinder)

    return geometries


def process_image_for_pose_and_3d(image_path="images.jpg", output_dir="output"):
    """
    Detects human poses in an image using YOLOv8-pose, extracts keypoints,
    creates a 3D mesh representation for each person, and saves them as .obj files.
    Also displays the annotated image and saves it.
    """
    print("loading yolov8x-pose model for keypoint detection...")
    model = YOLO("yolov8x-pose.pt")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"error: could not open image file {image_path}")
        return

    frame_height, frame_width, _ = frame.shape
    
    output_image_path = os.path.join(output_dir, "output_pose_image.jpg")

    print(f"processing image: {image_path}")
    
    results = model(frame, verbose=False)
    annotated_frame = None

    # Process results and save geometries
    if results:
        for r in results:
            if r.keypoints is not None:
                for i, kpts_data_tensor in enumerate(r.keypoints.data):
                    person_id = f"person_{i}" # Simple ID for each person in the image

                    # Pass all keypoint data (including confidence)
                    meshes = create_3d_skeleton(kpts_data_tensor, frame_width, frame_height, z_offset=0)
                    
                    if meshes:
                        # Combine all meshes for this person into a single mesh for saving
                        combined_mesh = o3d.geometry.TriangleMesh()
                        for m in meshes:
                            combined_mesh += m # Use += to combine meshes

                        output_obj_path = os.path.join(output_dir, f"{person_id}_pose.obj")
                        o3d.io.write_triangle_mesh(output_obj_path, combined_mesh)
                        print(f"saved 3D pose for {person_id} to {output_obj_path}")

            # Draw results on the frame
            annotated_frame = r.plot() # This generates the annotated frame once per result object

    if annotated_frame is not None:
        cv2.imwrite(output_image_path, annotated_frame)
        cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)
        cv2.waitKey(0) # Wait indefinitely until a key is pressed, then close
        cv2.destroyAllWindows()
        print(f"annotated image saved to {output_image_path}.")
    else:
        print("no poses detected or annotated frame not generated.")

    print(f"image processing complete!")

if __name__ == "__main__":
    if not os.path.exists("images.jpg"):
        print("error: 'images.jpg' not found. please place an image file in the current directory.")
        print("you can drag and drop it using the vscode plugin on the right, or upload it to your lightning drive.")
    else:
        process_image_for_pose_and_3d()