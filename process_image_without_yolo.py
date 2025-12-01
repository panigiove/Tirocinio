import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN_ResNet50_FPN_Weights
import open3d as o3d
import numpy as np
import os

# Create an output directory for meshes if it doesn't exist
output_dir = "output_meshes"
os.makedirs(output_dir, exist_ok=True)

# Load Keypoint R-CNN model
# For a headless environment, ensure weights are downloaded beforehand if possible or handle potential network issues.
weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
model = keypointrcnn_resnet50_fpn(weights=weights, progress=True)
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = T.Compose([T.ToTensor()])

def create_mesh_from_keypoints(keypoints, image_size):
    """
    Creates a simple 3D mesh (e.g., spheres for keypoints and cylinders for connections)
    from 2D keypoints. This is a placeholder for actual 3D pose reconstruction.
    For a proper 3D mesh, a full 3D pose estimation model or depth information would be needed.
    Here, we'll project 2D keypoints into a simple 3D space.
    """
    if keypoints.ndim == 2:
        keypoints_2d = keypoints.cpu().numpy()
    else:
        keypoints_2d = keypoints.squeeze(0).cpu().numpy() # Remove batch dimension for single person

    # Normalize keypoints to [0, 1] range based on image size
    normalized_keypoints = keypoints_2d[:, :2] / np.array([image_size[1], image_size[0]])

    # Simple depth estimation: assume points closer to the center are 'further' back
    # This is a very rough approximation, for real 3D, actual depth is needed.
    center_x, center_y = 0.5, 0.5
    depth = 1 - np.sqrt((normalized_keypoints[:, 0] - center_x)**2 + (normalized_keypoints[:, 1] - center_y)**2)
    # Scale depth to a reasonable range, e.g., 0 to 1
    depth = depth * 0.5 + 0.5 

    # Create 3D keypoints (x, y, z)
    keypoints_3d = np.hstack((normalized_keypoints, depth[:, np.newaxis]))

    # Create Open3D PointCloud for keypoints
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(keypoints_3d)

    # Simple sphere geometry for each keypoint
    geometries = []
    for kp in keypoints_3d:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(kp)
        sphere.paint_uniform_color([0, 0.7, 0.7]) # Cyan color
        geometries.append(sphere)
    
    # Define connections (simplified human skeleton)
    # Using COCO keypoint format for connections
    # This is a general connection set, adjust based on actual keypoint indices and desired skeleton
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head and shoulders
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11), # Arms
        (5, 11), (6, 12), (11, 12), # Torso (simplified)
        (13, 14), (13, 15), (14, 16), (15, 17), (16, 18), (17, 19) # Legs
    ]

    for start_idx, end_idx in connections:
        if start_idx < len(keypoints_3d) and end_idx < len(keypoints_3d):
            start_point = keypoints_3d[start_idx]
            end_point = keypoints_3d[end_idx]

            # Create a cylinder connecting two points
            vec = end_point - start_point
            length = np.linalg.norm(vec)
            if length > 0: # Avoid division by zero
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=length)
                
                # Align cylinder with vector
                R = get_rotation_matrix_from_vectors(np.array([0, 0, 1]), vec / length)
                cylinder.rotate(R, center=(0,0,0))
                
                # Translate to midpoint
                mid_point = (start_point + end_point) / 2
                cylinder.translate(mid_point)
                
                cylinder.paint_uniform_color([0.8, 0.2, 0.2]) # Red color
                geometries.append(cylinder)

    # Combine all geometries into a single mesh
    combined_mesh = o3d.geometry.TriangleMesh()
    for geom in geometries:
        combined_mesh += geom # Combine meshes

    return combined_mesh

def get_rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix between two vectors. """
    a, b = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0: return np.identity(3) # No rotation needed
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.identity(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def process_image_and_save_meshes(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"error: could not load image {image_path}")
        return

    # Convert to RGB (model expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)

    if len(prediction) > 0:
        # Filter detections by score threshold if needed
        scores = prediction[0]['scores'].cpu().numpy()
        keep = scores > 0.7 # Example threshold
        keypoints_filtered = prediction[0]['keypoints'][keep]
        
        # Ensure keypoints have at least 17 points for COCO format
        keypoints_valid = [kp for kp in keypoints_filtered if kp.shape[0] >= 17]

        if len(keypoints_valid) > 0:
            print(f"found {len(keypoints_valid)} people in {image_path}")
            for i, kps in enumerate(keypoints_valid):
                # kps shape: [num_keypoints, 3] (x, y, confidence)
                # Ensure the keypoints are for a single person (unsqueeze(0) if it was batched)
                person_mesh = create_mesh_from_keypoints(kps, image.shape[:2]) # Pass image height, width
                
                mesh_filename = os.path.join(output_dir, f"person_{i+1}_mesh.obj")
                o3d.io.write_triangle_mesh(mesh_filename, person_mesh)
                print(f"saved mesh for person {i+1} to {mesh_filename}")
        else:
            print(f"no valid poses found for tracking in {image_path}")
    else:
        print(f"no detections found in {image_path}")

if __name__ == "__main__":
    input_image_file = "images.jpg" # Assuming this file exists from previous context
    
    # Create a dummy images.jpg file if it doesn't exist for testing
    if not os.path.exists(input_image_file):
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_image, "Placeholder Image", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(input_image_file, dummy_image)
        print(f"created a dummy {input_image_file} for demonstration.")

    process_image_and_save_meshes(input_image_file)