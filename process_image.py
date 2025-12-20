import cv2
import os
import numpy as np
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R_scipy
import json
import open3d as o3d # new import for 3D mesh operations

# Define the connections for a 17-point pose skeleton
# These are standard COCO keypoint connections
# Added connections for a more complete humanoid look.
POSE_CONNECTIONS = [
    (0, 1), (0, 2), # Nose to eyes
    (1, 2),         # Left eye to Right eye (new)
    (1, 3), (2, 4), # Eyes to ears
    (3, 4),         # Left ear to Right ear (new)
    
    (5, 6),         # Shoulders connection
    (0, 5), (0, 6), # Nose to shoulders (implied neck) (new)
    (5, 7), (6, 8), # Shoulders to elbows
    (7, 9), (8, 10), # Elbows to wrists
    
    (11, 12),       # Hips connection
    (11, 13), (12, 14), # Hips to knees
    (13, 15), (14, 16), # Knees to ankles
    
    # Torso connections
    (5, 11),  # Left shoulder to left hip
    (6, 12),  # Right shoulder to right hip
    (11, 12)  # Left hip to right hip (already there, but good to keep explicit)
]

# Standard COCO 17 keypoint indices:
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,\
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,\
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,\
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: left_ankle

# Mapping keypoint indices to names for better readability in bone data
KEYPOINT_NAMES = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
}


def create_3d_skeleton(keypoints_data, bbox, frame_width, frame_height, z_range=(-0.5, 0.5), confidence_threshold=0.5):
    """
    Calculates 3D keypoint coordinates from 2D keypoints using a heuristic for Z-depth.
    Returns a dictionary of 3D keypoints (index -> [x, y, z]) and the connection list.
    
    Args:
        keypoints_data: A list of 2D keypoints (x, y, confidence) for a single person.
        bbox: Bounding box of the person (x_min, y_min, x_max, y_max).
        frame_width: Width of the original image frame.
        frame_height: Height of the original image frame.
        z_range: Min and max Z-depth values for mapping.
        confidence_threshold: Minimum confidence for a keypoint to be included.

    Returns:
        tuple: (dict of 3D keypoints, list of connections).
               The dict maps keypoint index to a list [x, y, z].
               The list of connections is POSE_CONNECTIONS.
    """
    points_3d = {} # Store 3D points as a dictionary to easily access by original index
    
    x_min, y_min, x_max, y_max = bbox
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    # bbox_area = bbox_width * bbox_height # not used currently

    # Heuristic for base Z-depth for the person:
    # Use normalized y_max (bottom of bbox) for general depth and scale by bbox height
    normalized_y_base = y_max / frame_height # 0 (top) to 1 (bottom)

    # Deeper in the image (smaller normalized_y_base) should map to more negative Z
    # Closer in the image (larger normalized_y_base) should map to more positive Z
    base_z_from_y = z_range[0] + normalized_y_base * (z_range[1] - z_range[0])
    
    # Also factor in bbox height: taller people seem closer.
    height_influence_factor = 0.5 # How much bbox height impacts base_z
    base_z_from_height = (bbox_height / frame_height - 0.5) * (z_range[1] - z_range[0]) * height_influence_factor
    
    # Combine these two influences.
    base_z = base_z_from_y + base_z_from_height
    
    # Ensure base_z stays within a reasonable range
    base_z = np.clip(base_z, z_range[0], z_range[1])

    for i, kp_data in enumerate(keypoints_data):
        if kp_data[2].item() > confidence_threshold:
            x, y = kp_data[0].item(), kp_data[1].item()
            
            # Scale x and y to be within a reasonable range (e.g., -1 to 1) for 3D visualization
            # Also, invert y-axis for typical 3D coordinate systems (Y up)
            x_norm = (x / frame_width) * 2 - 1
            y_norm = (y / frame_height) * -2 + 1 # Invert Y for typical 3D view (Y up)

            # Refine Z-coordinate using base_z and keypoint's vertical position within the bbox
            if bbox_height > 0:
                relative_y_in_bbox = (y - y_min) / bbox_height # 0 at top of bbox, 1 at bottom
                
                # Adjust z based on relative_y_in_bbox: head further (more negative offset), feet closer (more positive offset)
                z_offset_scale = 0.2 # Total range of z offset
                z_offset = (relative_y_in_bbox - 0.5) * z_offset_scale # Shifts from -0.1 to +0.1
                
                # Apply additional small, keypoint-specific offsets for visual plausibility (heuristic)
                kp_depth_priors = {
                    0: 0.05,  # nose: slightly forward
                    1: 0.02,  # left_eye: slightly forward
                    2: 0.02,  # right_eye: slightly forward
                    3: -0.01, # left_ear: slightly backward
                    4: -0.01, # right_ear: slightly backward
                    # Add more specific offsets if desired for other keypoints
                }
                
                z_kp_prior = kp_depth_priors.get(i, 0.0) # Get prior for current keypoint index
                z = base_z + z_offset + z_kp_prior
            else:
                z = base_z # Fallback if bbox_height is 0

            points_3d[i] = [x_norm, y_norm, z] # Store as list for JSON serialization

    # Filter connections to only include those where both keypoints exist (met confidence threshold)
    valid_connections = []
    for connection in POSE_CONNECTIONS:
        idx1, idx2 = connection
        if idx1 in points_3d and idx2 in points_3d:
            valid_connections.append(connection)

    return points_3d, valid_connections

def get_hierarchical_bone_data(keypoints_3d, connections):
    """
    Generates bone data with hierarchical information from 3D keypoints and connections.
    
    Args:
        keypoints_3d (dict): Dictionary of 3D keypoints (index -> [x, y, z]).
        connections (list): List of valid connections [(idx1, idx2), ...].

    Returns:
        list: A list of dictionaries, each representing a bone.
              Each bone dict includes 'name', 'start_joint_index', 'end_joint_index',
              'start_position', 'end_position', 'length', and 'direction'.
    """
    bones = []
    for idx1, idx2 in connections:
        start_pos = np.array(keypoints_3d[idx1])
        end_pos = np.array(keypoints_3d[idx2])
        
        vector = end_pos - start_pos
        length = np.linalg.norm(vector)
        direction = (vector / length).tolist() if length > 0 else [0.0, 0.0, 0.0]

        bone_name = f"{KEYPOINT_NAMES.get(idx1, str(idx1))}_to_{KEYPOINT_NAMES.get(idx2, str(idx2))}"
        
        bones.append({
            "name": bone_name,
            "start_joint_index": idx1,
            "end_joint_index": idx2,
            "start_position": start_pos.tolist(),
            "end_position": end_pos.tolist(),
            "length": length,
            "direction": direction
        })
    return bones


def generate_humanoid_mesh(keypoints_3d, connections, base_radius=0.015): # Adjusted base_radius
    """
    Generates a simple humanoid mesh from 3D keypoints and connections using open3d.
    Joints are spheres, and limbs are cylinders, with varied radii for more detail.
    """
    if not keypoints_3d:
        return o3d.geometry.TriangleMesh() # Return empty mesh if no keypoints

    # Define specific radii for different joints and limbs relative to base_radius
    # Adjusted these values to make the skeleton appear more distinct
    joint_radii = {
        0: base_radius * 1.0,  # Nose
        1: base_radius * 0.7,  # Left eye
        2: base_radius * 0.7,  # Right eye
        3: base_radius * 0.7,  # Left ear
        4: base_radius * 0.7,  # Right ear
        5: base_radius * 1.5,  # Left shoulder (larger)
        6: base_radius * 1.5,  # Right shoulder (larger)
        7: base_radius * 1.0,  # Left elbow
        8: base_radius * 1.0,  # Right elbow
        9: base_radius * 0.8,  # Left wrist (smaller)
        10: base_radius * 0.8, # Right wrist (smaller)
        11: base_radius * 1.5, # Left hip (larger)
        12: base_radius * 1.5, # Right hip (larger)
        13: base_radius * 1.2, # Left knee
        14: base_radius * 1.2, # Right knee
        15: base_radius * 1.0, # Left ankle
        16: base_radius * 1.0, # Right ankle
    }

    limb_radii = {
        # Torso
        (5, 11): base_radius * 1.8,  # Left shoulder to left hip
        (6, 12): base_radius * 1.8,  # Right shoulder to right hip
        (11, 12): base_radius * 1.8, # Hips connection
        (5, 6): base_radius * 1.5,   # Shoulder connection
        # Head (approximate)
        (0, 1): base_radius * 0.6, # Nose to eye
        (0, 2): base_radius * 0.6, # Nose to eye
        (1, 2): base_radius * 0.5, # Eye to eye (new)
        (1, 3): base_radius * 0.5, # Eye to ear
        (2, 4): base_radius * 0.5, # Eye to ear
        (3, 4): base_radius * 0.5, # Ear to ear (new)
        (0, 5): base_radius * 1.2, # Nose to left shoulder (new)
        (0, 6): base_radius * 1.2, # Nose to right shoulder (new)
        # Upper Arms
        (5, 7): base_radius * 1.2,   # Left shoulder to elbow
        (6, 8): base_radius * 1.2,   # Right shoulder to elbow
        # Forearms
        (7, 9): base_radius * 0.9,   # Left elbow to wrist
        (8, 10): base_radius * 0.9,  # Right elbow to wrist
        # Upper Legs
        (11, 13): base_radius * 1.3, # Left hip to knee
        (12, 14): base_radius * 1.3, # Right hip to knee
        # Lower Legs
        (13, 15): base_radius * 1.0, # Left knee to ankle
        (14, 16): base_radius * 1.0, # Right knee to ankle
    }

    combined_mesh = o3d.geometry.TriangleMesh()

    # Create spheres for joints
    for kp_idx, coords in keypoints_3d.items():
        current_radius = joint_radii.get(kp_idx, base_radius) # Use specific radius or default
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=current_radius)
        sphere.translate(np.array(coords))
        sphere.paint_uniform_color([0.1, 0.7, 0.1]) # Green joints
        combined_mesh += sphere

    # Create cylinders for limbs
    for c_idx1, c_idx2 in connections:
        kp1 = np.array(keypoints_3d[c_idx1])
        kp2 = np.array(keypoints_3d[c_idx2])

        # Get specific limb radius, or calculate an average if not explicitly defined
        connection_key = tuple(sorted((c_idx1, c_idx2))) # Ensure consistent key order
        current_limb_radius = limb_radii.get(connection_key, base_radius * 0.7) # Default if not specified

        # Calculate vector between keypoints
        vec = kp2 - kp1
        length = np.linalg.norm(vec)

        if length > 1e-6: # Avoid division by zero for identical points
            # Create cylinder
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=current_limb_radius, height=length)
            
            # Orient cylinder: default is along Y-axis, need to rotate to align with vec
            # First, normalize the vector
            vec_norm = vec / length
            
            # The default cylinder's axis in Open3D is along the Y-axis (0, 1, 0)
            Y_AXIS = np.array([0., 1., 0.])
            
            # Calculate rotation axis and angle
            rotation_axis = np.cross(Y_AXIS, vec_norm)
            rotation_axis_norm = np.linalg.norm(rotation_axis)
            
            if rotation_axis_norm < 1e-6: # Vectors are parallel or anti-parallel (very small cross product magnitude)
                if np.dot(Y_AXIS, vec_norm) > 0: # Parallel, no rotation needed
                    R_matrix = np.eye(3) 
                else: # Anti-parallel, rotate 180 degrees around X-axis
                    R_matrix = R_scipy.from_euler('x', np.pi).as_matrix()
            else:
                rotation_axis = rotation_axis / rotation_axis_norm # Normalize rotation axis
                dot_product = np.dot(Y_AXIS, vec_norm)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) # Clip for numerical stability
                R_matrix = R_scipy.from_rotvec(angle * rotation_axis).as_matrix()
            
            cylinder.rotate(R_matrix, center=(0,0,0))
            
            # Translate cylinder to midpoint between kp1 and kp2
            midpoint = (kp1 + kp2) / 2
            cylinder.translate(midpoint)
            
            cylinder.paint_uniform_color([0.8, 0.1, 0.1]) # Red limbs
            combined_mesh += cylinder
    
    return combined_mesh


def process_image_for_pose_and_3d(image_path="images.jpg", output_dir="output"):
    """
    Detects human poses in an image using YOLOv8-pose, extracts keypoints and bounding boxes,
    calculates 3D coordinates for each person, and saves them to a JSON file.
    Also displays the annotated image and saves it.
    Generates a 3D mesh for each detected person and saves it as an OBJ file.
    """
    print("loading yolov8x-pose model for keypoint detection...")
    model = YOLO("yolov8x-pose.pt") # 'x' for extra large, more accurate but slower

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

    all_persons_data = [] # Changed name to reflect more comprehensive data

    # Process results and save 3D skeletal data
    if results:
        for r in results:
            annotated_frame = r.plot() # This generates the annotated frame once per result object
            
            if r.keypoints is not None and r.boxes is not None:
                for i, (kpts_data_tensor, bbox_data_tensor) in enumerate(zip(r.keypoints.data, r.boxes.data)):
                    person_id = f"person_{i}" # Simple ID for each person in the image

                    # Get bounding box coordinates for the current person
                    x_min, y_min, x_max, y_max = bbox_data_tensor[:4].cpu().numpy().astype(int)
                    bbox = (x_min, y_min, x_max, y_max)

                    # Get 3D keypoints and valid connections
                    person_3d_keypoints, person_connections = create_3d_skeleton(
                        kpts_data_tensor, bbox, frame_width, frame_height, z_range=(-0.5, 0.5)
                    ) 
                    
                    if person_3d_keypoints:
                        # Generate bone data for hierarchical skeleton
                        person_bone_data = get_hierarchical_bone_data(person_3d_keypoints, person_connections)

                        person_data = {
                            "person_id": person_id,
                            "keypoints_3d": person_3d_keypoints, # Dict: {index: [x,y,z]}
                            "connections": person_connections, # List of tuples: [(idx1, idx2), ...]
                            "bone_data": person_bone_data # New: List of detailed bone dictionaries
                        }
                        all_persons_data.append(person_data)
                        print(f"extracted 3d skeleton data for {person_id}:")
                        for kp_idx, coords in person_3d_keypoints.items():
                            print(f"  keypoint {kp_idx}: {coords}")
                        print(f"  connections: {person_connections}")
                        print(f"  generated {len(person_bone_data)} bones for rigging.")
                        print("-" * 30)

                        # Generate and save the 3D mesh
                        person_mesh = generate_humanoid_mesh(person_3d_keypoints, person_connections)
                        mesh_output_path = os.path.join(output_dir, f"{person_id}_humanoid.obj")
                        o3d.io.write_triangle_mesh(mesh_output_path, person_mesh)
                        print(f"generated and saved 3D mesh for {person_id} to {mesh_output_path}")

    if all_persons_data:
        output_json_path = os.path.join(output_dir, "3d_skeletons_and_bones.json") # Updated filename
        with open(output_json_path, 'w') as f:
            json.dump(all_persons_data, f, indent=4)
        print(f"all 3d skeleton and bone data saved to {output_json_path}")
    else:
        print("no 3d skeleton data extracted.")


    if annotated_frame is not None:
        cv2.imwrite(output_image_path, annotated_frame)
        # Removed cv2.imshow as it may not work in cloud environments.
        # cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)
        # cv2.waitKey(0) # Wait indefinitely until a key is pressed, then close
        # cv2.destroyAllWindows()
        print(f"annotated image saved to {output_image_path}. please check the 'output' folder.")
    else:
        print("no poses detected or annotated frame not generated.")

    print(f"image processing complete!")

if __name__ == "__main__":
    if not os.path.exists("images.jpg"):
        print("error: 'images.jpg' not found. please place an image file in the current directory.")
        print("you can drag and drop it using the vscode plugin on the right, or upload it to your lightning drive.")
        print("\nfor example, try putting a file named `images.jpg` into your studio's root directory.")
    else:
        process_image_for_pose_and_3d()