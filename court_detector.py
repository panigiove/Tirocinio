# court_detector.py
# Deep learning-based basketball court detection and homography estimation

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCourtDetector(nn.Module):
    """
    Lightweight neural network for detecting basketball court keypoints.
    This is a simplified version that can be trained on your specific court views.
    """
    def __init__(self, num_keypoints=8):
        super(SimpleCourtDetector, self).__init__()
        
        # Encoder (feature extraction)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Global average pooling + fully connected
        self.fc1 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_keypoints * 2)  # (x, y) for each keypoint
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class DeepCourtDetector:
    """
    Deep learning-based court detector with fallback to classical methods.
    """
    def __init__(self, model_path=None, use_pretrained=False, device='cuda'):
        """
        Args:
            model_path: Path to trained model weights (optional)
            use_pretrained: Use a pretrained model (not yet implemented)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = SimpleCourtDetector(num_keypoints=8)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model_loaded = True
        else:
            self.model_loaded = False
            print("Warning: No trained model found. Using classical fallback method.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Basketball court dimensions (in meters)
        self.court_length = 28.0  # FIBA standard
        self.court_width = 15.0
        
        # Define standard court keypoints in world coordinates
        # Order: [top-left, top-right, bottom-right, bottom-left, 
        #         half-court-top, half-court-bottom, three-point-left, three-point-right]
        self.world_keypoints = np.array([
            [0, 0],                          # top-left corner
            [self.court_length, 0],          # top-right corner
            [self.court_length, self.court_width],  # bottom-right corner
            [0, self.court_width],           # bottom-left corner
            [self.court_length/2, 0],        # half-court top
            [self.court_length/2, self.court_width],  # half-court bottom
            [6.75, self.court_width/2],      # three-point line left
            [self.court_length-6.75, self.court_width/2],  # three-point line right
        ], dtype=np.float32)
        
    def preprocess_frame(self, frame):
        """Preprocess frame for neural network input"""
        # Resize to network input size
        img = cv.resize(frame, (640, 480))
        
        # Convert to tensor and normalize
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        
        return img.to(self.device)
    
    def detect_keypoints_nn(self, frame):
        """Detect court keypoints using neural network"""
        if not self.model_loaded:
            return None
        
        h, w = frame.shape[:2]
        
        # Preprocess
        img_tensor = self.preprocess_frame(frame)
        
        # Inference
        with torch.no_grad():
            keypoints = self.model(img_tensor)
        
        # Convert to numpy and denormalize
        keypoints = keypoints.cpu().numpy().reshape(-1, 2)
        keypoints[:, 0] *= w  # scale x to original width
        keypoints[:, 1] *= h  # scale y to original height
        
        return keypoints
    
    def detect_keypoints_classical(self, frame):
        """
        Classical computer vision fallback for detecting court keypoints.
        Uses improved line detection and corner finding.
        """
        # Enhanced line mask detection
        mask = self._enhanced_line_mask(frame)
        
        # Detect lines
        lines = cv.HoughLinesP(
            mask, 
            rho=1, 
            theta=np.pi/180, 
            threshold=80,
            minLineLength=100, 
            maxLineGap=27
        )
        
        if lines is None or len(lines) < 8:
            return None
        
        # Cluster lines into horizontal and vertical
        horizontals = []
        verticals = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            line_length = np.sqrt(dx**2 + dy**2)
            
            if line_length < 80:
                continue
            
            angle = np.degrees(np.arctan2(dy, dx)) % 180
            
            # Horizontal lines (±15 degrees)
            if angle < 15 or angle > 165:
                y_pos = (y1 + y2) / 2
                horizontals.append((y_pos, (x1, y1, x2, y2), line_length))
            # Vertical lines (75-105 degrees)
            elif 75 < angle < 105:
                x_pos = (x1 + x2) / 2
                verticals.append((x_pos, (x1, y1, x2, y2), line_length))
        
        if len(horizontals) < 2 or len(verticals) < 2:
            return None
        
        # Sort by length and get the most prominent lines
        horizontals.sort(key=lambda x: x[2], reverse=True)
        verticals.sort(key=lambda x: x[2], reverse=True)
        
        # Get extreme lines
        h_lines = [h[1] for h in horizontals[:min(4, len(horizontals))]]
        v_lines = [v[1] for v in verticals[:min(4, len(verticals))]]
        
        # Find top and bottom horizontal lines
        top_h = min(h_lines, key=lambda l: (l[1] + l[3]) / 2)
        bottom_h = max(h_lines, key=lambda l: (l[1] + l[3]) / 2)
        
        # Find left and right vertical lines
        left_v = min(v_lines, key=lambda l: (l[0] + l[2]) / 2)
        right_v = max(v_lines, key=lambda l: (l[0] + l[2]) / 2)
        
        # Compute intersections (corners)
        def line_intersection(line1, line2):
            # Convert to float64 to prevent overflow
            x1, y1, x2, y2 = map(float, line1)
            x3, y3, x4, y4 = map(float, line2)
            
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                return None
            
            # Use float64 arithmetic to prevent overflow
            num_x = (x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)
            num_y = (x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)
            
            px = num_x / denom
            py = num_y / denom
            
            # Sanity check: intersection should be within reasonable bounds
            h, w = frame.shape[:2]
            if px < -w or px > 2*w or py < -h or py > 2*h:
                return None
            
            return (float(px), float(py))
        
        # Compute 4 corners
        tl = line_intersection(top_h, left_v)
        tr = line_intersection(top_h, right_v)
        br = line_intersection(bottom_h, right_v)
        bl = line_intersection(bottom_h, left_v)
        
        if None in [tl, tr, br, bl]:
            return None
        
        # Try to find half-court line (middle vertical)
        middle_verticals = [v for v in verticals if abs(v[0] - frame.shape[1]/2) < frame.shape[1]*0.3]
        
        if middle_verticals:
            middle_v = middle_verticals[0][1]
            half_top = line_intersection(top_h, middle_v)
            half_bottom = line_intersection(bottom_h, middle_v)
        else:
            # Estimate from corners
            half_top = ((tl[0] + tr[0])/2, (tl[1] + tr[1])/2)
            half_bottom = ((bl[0] + br[0])/2, (bl[1] + br[1])/2)
        
        # Estimate three-point line positions (approximation)
        three_left = ((tl[0] + bl[0])/2 + (tr[0] + br[0])/2 * 0.25) / 1.25, (tl[1] + bl[1] + tr[1] + br[1])/4
        three_right = ((tl[0] + bl[0])/2 * 0.25 + (tr[0] + br[0])/2) / 1.25, (tl[1] + bl[1] + tr[1] + br[1])/4
        
        keypoints = np.array([
            tl, tr, br, bl,
            half_top, half_bottom,
            three_left, three_right
        ], dtype=np.float32)
        
        return keypoints
    
    def _enhanced_line_mask(self, frame):
        """Enhanced line detection combining multiple approaches"""
        h, w = frame.shape[:2]
        
        # Method 1: HSV white detection
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask_hsv = cv.inRange(hsv, (0, 0, 180), (180, 50, 255))
        
        # Method 2: LAB bright detection
        lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        _, mask_lab = cv.threshold(l_channel, 190, 255, cv.THRESH_BINARY)
        
        # Method 3: Enhanced edge detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        # Bilateral filter to reduce noise while preserving edges
        gray_smooth = cv.bilateralFilter(gray_enhanced, 9, 75, 75)
        edges = cv.Canny(gray_smooth, 40, 120)
        
        # Combine masks
        mask_combined = cv.bitwise_or(mask_hsv, mask_lab)
        mask_combined = cv.bitwise_or(mask_combined, edges)
        
        # Morphological operations
        kernel_small = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        mask_combined = cv.morphologyEx(mask_combined, cv.MORPH_CLOSE, kernel_small)
        
        # Remove small noise
        kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        mask_combined = cv.morphologyEx(mask_combined, cv.MORPH_OPEN, kernel_open)
        
        # Connected components filtering
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask_combined)
        min_area = 40
        for i in range(1, num_labels):
            if stats[i, cv.CC_STAT_AREA] < min_area:
                mask_combined[labels == i] = 0
        
        return mask_combined
    
    def compute_homography(self, frame):
        """
        Detect keypoints and compute homography matrix.
        Returns H_inv (pixel to world coordinates).
        """
        # Try neural network first
        keypoints = None
        if self.model_loaded:
            keypoints = self.detect_keypoints_nn(frame)
        
        # Fallback to classical method
        if keypoints is None:
            keypoints = self.detect_keypoints_classical(frame)
        
        if keypoints is None:
            return None
        
        # Use minimum 4 corners for homography
        # Ensure we have at least 4 valid keypoints
        valid_kpts = keypoints[:4]  # Use the 4 corners
        valid_world = self.world_keypoints[:4]
        
        if len(valid_kpts) < 4:
            return None
        
        # Compute homography (pixel -> world)
        H_inv, mask = cv.findHomography(
            valid_kpts, 
            valid_world, 
            cv.RANSAC, 
            5.0
        )
        
        return H_inv


class DeepCourtRefiner:
    """
    Dynamic court homography refiner using deep learning detection.
    Integrates with your existing tracking system.
    """
    def __init__(self, detector, refresh_interval=30, use_temporal_smoothing=True, static_mode=False):
        """
        Args:
            detector: DeepCourtDetector instance
            refresh_interval: Frames between full re-detection (0 = every frame)
            use_temporal_smoothing: Apply EMA smoothing to homography
            static_mode: If True, compute homography once and reuse for all frames
        """
        self.detector = detector
        self.refresh_interval = refresh_interval
        self.use_temporal_smoothing = use_temporal_smoothing
        self.static_mode = static_mode
        
        self.H_inv = None
        self.frame_count = 0
        self.ema_alpha = 0.7  # Weight for current frame
        
    def update(self, frame):
        """
        Update homography for current frame.
        
        Args:
            frame: Current video frame
            
        Returns:
            H_inv: Homography matrix (pixel to world), or None if detection failed
        """
        self.frame_count += 1

        if self.static_mode and self.H_inv is not None:
            return self.H_inv
        
        # Decide whether to recompute
        should_compute = (
            self.H_inv is None or 
            self.refresh_interval == 0 or 
            self.frame_count % self.refresh_interval == 0
        )
        
        if should_compute:
            H_new = self.detector.compute_homography(frame)
            
            if H_new is not None:
                if self.H_inv is None or not self.use_temporal_smoothing:
                    self.H_inv = H_new
                else:
                    # Apply exponential moving average for smooth transitions
                    self.H_inv = self.ema_alpha * H_new + (1 - self.ema_alpha) * self.H_inv
        
        return self.H_inv


# For backwards compatibility / testing
import os

def test_court_detector(video_path, output_path=None, frame_index=None, image_output_path=None):
    """
    Test the court detector on a video file.
    Draws detected keypoints and court boundaries.
    """
    detector = DeepCourtDetector()
    refiner = DeepCourtRefiner(detector, refresh_interval=10)
    
    cap = cv.VideoCapture(video_path)
    
    if output_path:
        fps = int(cap.get(cv.CAP_PROP_FPS))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), 
                                fps, (width, height))
    
    if frame_index is not None:
        cap.set(cv.CAP_PROP_POS_FRAMES, int(frame_index))
        ret, frame = cap.read()
        if not ret:
            cap.release()
            if output_path:
                writer.release()
            return
        frames = [(int(frame_index), frame)]
    else:
        frames = []
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append((frame_id, frame))
            frame_id += 1
    for frame_id, frame in frames:
        # Get homography
        H_inv = refiner.update(frame)
        
        # Visualize
        vis_frame = frame.copy()
        
        if H_inv is not None:
            # Draw detected court boundary
            world_corners = np.array([
                [0, 0],
                [detector.court_length, 0],
                [detector.court_length, detector.court_width],
                [0, detector.court_width]
            ], dtype=np.float32)
            
            # Project world corners back to image
            H = np.linalg.inv(H_inv)
            corners_homog = np.column_stack([world_corners, np.ones(4)])
            img_corners_homog = (H @ corners_homog.T).T
            img_corners = img_corners_homog[:, :2] / img_corners_homog[:, 2:3]
            
            # Draw court boundary
            img_corners = img_corners.astype(np.int32)
            cv.polylines(vis_frame, [img_corners], True, (0, 255, 0), 3)
            
            # Draw half-court line
            half_top = np.array([[detector.court_length/2, 0, 1]])
            half_bottom = np.array([[detector.court_length/2, detector.court_width, 1]])
            
            img_half_top = (H @ half_top.T).T
            img_half_top = (img_half_top[:, :2] / img_half_top[:, 2:3]).astype(np.int32)[0]
            
            img_half_bottom = (H @ half_bottom.T).T
            img_half_bottom = (img_half_bottom[:, :2] / img_half_bottom[:, 2:3]).astype(np.int32)[0]
            
            cv.line(vis_frame, tuple(img_half_top), tuple(img_half_bottom), 
                   (0, 255, 255), 2)
            
            cv.putText(vis_frame, "Court Detected", (30, 50),
                      cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv.putText(vis_frame, "No Court Detected", (30, 50),
                      cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv.putText(vis_frame, f"Frame: {frame_id}", (30, 100),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if output_path:
            writer.write(vis_frame)
        if image_output_path and frame_index is not None:
            cv.imwrite(image_output_path, vis_frame)
        
        cv.imshow('Court Detection', vis_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if output_path:
        writer.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # Test on your video
    test_court_detector(
        'Tracking/material4project/Rectified videos/tracking_12/out13.mp4',
        'court_detection_test.mp4',
        frame_index=None,
        image_output_path='court_detection_frame.png'
    )
