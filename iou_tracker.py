import numpy as np

class IOUTracker:
    def __init__(self, iou_threshold=0.5, max_missed_frames=5):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        # stores {track_id: {'bbox': (x1,y1,x2,y2), 'keypoints': array, 'missed_frames_count': int}}
        self.tracks = {}
        self.next_id = 0

    def _iou(self, box1, box2):
        """Calculates Intersection over Union (IoU) of two bounding boxes."""
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xi1 = max(x1, x1g)
        yi1 = max(y1, y1g)
        xi2 = min(x2, x2g)
        yi2 = min(y2, y2g)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def update(self, current_detections):
        """
        Updates tracks with current detections.
        current_detections: list of dictionaries, each with 'bbox' (x1, y1, x2, y2)
                            and 'keypoints' for pose estimation.
        Returns:
            list of dictionaries, each with 'bbox', 'keypoints', and 'track_id' for active tracks.
        """
        matched_track_ids = set()
        matched_det_indices = set()
        active_tracks_this_frame = [] # To store detections that are either new or matched to existing tracks

        # Step 1: Try to match current detections with existing tracks
        for det_idx, current_det in enumerate(current_detections):
            best_iou = -1
            best_track_id = -1
            current_bbox = current_det['bbox']

            for track_id, track_info in self.tracks.items():
                # Only consider tracks that haven't exceeded missed_frames_count
                if track_info['missed_frames_count'] <= self.max_missed_frames:
                    track_bbox = track_info['bbox']
                    iou = self._iou(current_bbox, track_bbox)
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_track_id = track_id
            
            if best_track_id != -1:
                # Match found: update existing track
                self.tracks[best_track_id]['bbox'] = current_bbox
                self.tracks[best_track_id]['keypoints'] = current_det['keypoints']
                self.tracks[best_track_id]['missed_frames_count'] = 0
                self.tracks[best_track_id]['track_id'] = best_track_id # Ensure track_id is present
                matched_track_ids.add(best_track_id)
                matched_det_indices.add(det_idx)
                active_tracks_this_frame.append(self.tracks[best_track_id])
            
        # Step 2: Add unmatched detections as new tracks
        for det_idx, current_det in enumerate(current_detections):
            if det_idx not in matched_det_indices:
                new_track_info = {
                    'bbox': current_det['bbox'],
                    'keypoints': current_det['keypoints'],
                    'missed_frames_count': 0,
                    'track_id': self.next_id # Assign new ID
                }
                self.tracks[self.next_id] = new_track_info
                active_tracks_this_frame.append(new_track_info)
                self.next_id += 1
        
        # Step 3: Update missed_frames_count for tracks that were not matched in this frame
        # and remove tracks that have exceeded max_missed_frames
        tracks_to_delete = []
        for track_id in self.tracks:
            if track_id not in matched_track_ids:
                self.tracks[track_id]['missed_frames_count'] += 1
                if self.tracks[track_id]['missed_frames_count'] > self.max_missed_frames:
                    tracks_to_delete.append(track_id)
            # If a track was missed but not yet deleted, and it's still below the threshold,
            # it's considered active for potential future matching.
            elif self.tracks[track_id]['missed_frames_count'] <= self.max_missed_frames:
                # This track was matched, so its missed_frames_count was reset to 0
                # It's already in active_tracks_this_frame
                pass

        for track_id in tracks_to_delete:
            del self.tracks[track_id]

        return active_tracks_this_frame