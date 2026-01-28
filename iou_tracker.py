# Hungarian-based multi-cue tracker (IoU + appearance + pose)

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


def iou(b1, b2):
    x1, y1, x2, y2 = b1
    x1g, y1g, x2g, y2g = b2
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = max(0, x2 - x1) * max(0, y2 - y1)
    a2 = max(0, x2g - x1g) * max(0, y2g - y1g)
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0.0


class GlobalIDManager:
    def __init__(self):
        self._next_id = 0

    def next_id(self):
        id_to_return = self._next_id
        self._next_id += 1
        return id_to_return


class IOUTracker:
    def __init__(self,
                 match_threshold=0.35,
                 cross_view_match_threshold=0.5, # new: threshold for cross-view matching
                 max_missed_frames=10,
                 iou_weight=0.5,
                 appearance_weight=0.35,
                 pose_weight=0.15,
                 world_dist_weight=0.7, # Increased weight for spatial distance
                 ema_alpha=0.9,
                 camera_params=None,
                 max_tracks=None,
                 global_id_manager=None): # new: shared ID manager
        self.match_threshold = match_threshold
        self.cross_view_match_threshold = cross_view_match_threshold # new
        self.max_missed = max_missed_frames
        self.iou_w = iou_weight
        self.app_w = appearance_weight
        self.pose_w = pose_weight
        self.world_w = world_dist_weight
        self.ema_alpha = ema_alpha
        self.max_tracks = max_tracks # new
        self.id_manager = global_id_manager if global_id_manager else GlobalIDManager()
        
        # Camera parameters for spatial projection
        self.camera_params = camera_params
        if camera_params:
            self.mtx = np.array(camera_params['mtx'], dtype=np.float32)
            self.dist = np.array(camera_params['dist'], dtype=np.float32)
            self.rvec = np.array(camera_params['rvecs'], dtype=np.float32)
            self.tvec = np.array(camera_params['tvecs'], dtype=np.float32)
            
            # Compute homography to ground plane (Z=0)
            R, _ = cv2.Rodrigues(self.rvec)
            # P = K * [R1 R2 t] (since Z=0, R3 is not needed)
            H = self.mtx @ np.hstack([R[:, 0:2], self.tvec])
            self.H_inv = np.linalg.inv(H)

        self.tracks = []
        self.next_id = 0
        self.current_frame_events = [] # new: store events for the current frame
        self.total_refound_tracks = 0
        self.total_temporary_missed_frames = 0
        self.total_active_tracks = 0
        self.current_frame_statistics = {
            'added_tracks': 0,
            'lost_tracks': 0,
            'refound_tracks': 0,
            'temporary_missed_tracks': 0,
            'active_tracks': 0,
        }
        self.track_to_global_id = {} # maps local track_id to a global_id
        self.global_id_to_track = {} # maps global_id to actual track objects

    def project_to_world(self, bbox):
        """Project the bottom center of the bounding box to the ground plane (Z=0)."""
        if not hasattr(self, 'H_inv'):
            return None
        
        x1, y1, x2, y2 = bbox
        # Use the bottom center of the bounding box as the feet position
        pixel_point = np.array([(x1 + x2) / 2.0, y2, 1.0], dtype=np.float32).reshape(3, 1)
        
        world_point = self.H_inv @ pixel_point
        world_point /= world_point[2] # Normalize
        
        return world_point[:2].flatten() # Return (X, Y)

    def project_to_pixel(self, world_pos):
        """Project world point (X, Y) back to pixel coordinates (x, y)."""
        if not hasattr(self, 'mtx') or self.mtx is None:
            return None
        
        # Assume Z=0 for ground plane
        pts_3d = np.array([[world_pos[0], world_pos[1], 0.0]], dtype=np.float32)
        img_pts, _ = cv2.projectPoints(pts_3d, self.rvec, self.tvec, self.mtx, self.dist)
        return img_pts[0][0] # Returns [x, y]

    def deduce_track_position(self, global_id, world_pos, frame_id):
        """
        Deduce the position of a missed track using world coordinates from another view.
        """
        if global_id not in self.global_id_to_track:
            return False
        
        track = self.global_id_to_track[global_id]
        if track['updated']:
            return False # Already updated in this view
            
        pixel_pos = self.project_to_pixel(world_pos)
        if pixel_pos is None:
            return False
            
        # Check if projected position is within reasonably "visible" range
        # (even if View 1 doesn't cover all, we might want to keep the virtual bbox)
        px, py = map(int, pixel_pos)
        
        # Maintain current bbox size but move center to projected point
        x1, y1, x2, y2 = track['bbox']
        w, h = x2 - x1, y2 - y1
        new_bbox = (px - w//2, py - h, px + w//2, py) # Bottom center at px, py
        
        track['bbox'] = new_bbox
        track['world_pos'] = world_pos
        track['updated'] = True # Mark as updated (deduced)
        track['missed'] = 0
        track['was_deduced'] = True
        
        self.current_frame_events.append({
            'frame': frame_id, 
            'track_id': track['track_id'], 
            'event': 'deduced_from_other_view', 
            'global_id': global_id
        })
        return True

    def _appearance_sim(self, a, b):
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))  # cosine similarity (vectors are normalized)

    def _pose_sim(self, p1, p2):
        if p1 is None or p2 is None:
            return 0.0
        # for pose, a smaller distance means higher similarity.
        # we can convert distance to similarity using an exponential decay.
        d = np.linalg.norm(p1 - p2)
        # adjust the decay factor (e.g., 0.1) based on expected pose variations
        return float(np.exp(-0.1 * d))

    def _cost_matrix(self, detections):
        n_t = len(self.tracks)
        n_d = len(detections)
        cost = np.zeros((n_t, n_d), dtype=np.float32)

        for i, t in enumerate(self.tracks):
            for j, d in enumerate(detections):
                iou_score = iou(t['bbox'], d['bbox'])
                app_score = self._appearance_sim(t.get('appearance'), d.get('appearance'))
                pose_score = self._pose_sim(t.get('pose_vec'), d.get('pose_vec'))
                spatial_sim = self._world_pos_sim(t.get('world_pos'), d.get('world_pos'))

                if t.get('world_pos') is not None and d.get('world_pos') is not None:
                    # Spatial information is pivotal
                    w_spatial = self.world_w
                    w_other = (1.0 - w_spatial)
                    # Use spatial similarity as primary, others as tie-breakers
                    sim = (w_spatial * spatial_sim + 
                           w_other * (0.4 * iou_score + 
                                      0.4 * app_score + 
                                      0.2 * pose_score))
                else:
                    # Fallback to standard cues if spatial is missing
                    norm = self.iou_w + self.app_w + self.pose_w
                    sim = (self.iou_w / norm * iou_score + 
                           self.app_w / norm * app_score + 
                           self.pose_w / norm * pose_score)
                
                # Give a boost to tracks that were missed to encourage re-identification (Re-ID)
                if t['missed'] > 0:
                    sim += 0.15 # Increased boost for Re-ID
                
                cost[i, j] = 1.0 - sim
        return cost

    def _world_pos_sim(self, wp1, wp2):
        if wp1 is None or wp2 is None:
            return 0.0
        # Euclidean distance in world units (e.g. millimeters or meters depending on calib)
        dist = np.linalg.norm(wp1 - wp2)
        # Convert distance to similarity using exponential decay
        # Sigma should be tuned based on units and expected accuracy (e.g. 500-1000mm)
        sigma = 1000.0 
        return float(np.exp(-dist / sigma))

    # new: method to compute cross-view similarity
    def _cross_view_similarity(self, track_a, track_b):
        # combine appearance, pose, and spatial similarities
        app_sim = self._appearance_sim(track_a.get('appearance'), track_b.get('appearance'))
        pose_sim = self._pose_sim(track_a.get('pose_vec'), track_b.get('pose_vec'))
        
        # Spatial similarity if both have world positions
        spatial_sim = self._world_pos_sim(track_a.get('world_pos'), track_b.get('world_pos'))

        if track_a.get('world_pos') is not None and track_b.get('world_pos') is not None:
            # Adjust weights: spatial position is highly reliable
            w_spatial = self.world_w
            w_other = (1.0 - w_spatial) / 2.0
            cross_sim = (w_other * app_sim + w_other * pose_sim + w_spatial * spatial_sim)
        else:
            cross_sim = (0.6 * app_sim + 0.4 * pose_sim)

        return cross_sim
    
    def _start_track(self, det, frame_id, global_id=None):
        if self.max_tracks is not None and len(self.tracks) >= self.max_tracks:
            return None # do not start new track if cap reached

        if global_id is None:
            global_id = self.id_manager.next_id()
            
        new_track = {
            'track_id': self.next_id,
            'global_id': global_id, # associate with a global ID
            'bbox': det['bbox'],
            'appearance': det.get('appearance'),
            'pose_vec': det.get('pose_vec'),
            'keypoints': det.get('keypoints'),
            'world_pos': det.get('world_pos'),
            'missed': 0,
            'updated': True,
            'was_refound_in_this_frame': False # new: initial state
        }
        self.tracks.append(new_track)
        self.track_to_global_id[self.next_id] = global_id
        self.global_id_to_track[global_id] = new_track # keep a reference to the track object
        self.current_frame_events.append({'frame': frame_id, 'track_id': self.next_id, 'event': 'added', 'global_id': global_id})
        self.current_frame_statistics['added_tracks'] += 1
        self.next_id += 1
        return new_track # return the new track object


    def _update_track(self, track, det, frame_id): # new: accept frame_id
        # if this track was previously missed, it's now refound
        if track['missed'] > 0:
            self.current_frame_events.append({'frame': frame_id, 'track_id': track['track_id'], 'event': 'refound', 'global_id': track['global_id']})
            self.current_frame_statistics['refound_tracks'] += 1
            self.total_refound_tracks += 1
            track['was_refound_in_this_frame'] = True # mark as refound

        track['bbox'] = det['bbox']
        track['keypoints'] = det.get('keypoints')
        track['pose_vec'] = det.get('pose_vec')
        track['world_pos'] = det.get('world_pos')

        # EMA update for appearance
        if track.get('appearance') is not None and det.get('appearance') is not None:
            track['appearance'] = (
                self.ema_alpha * track['appearance'] +
                (1.0 - self.ema_alpha) * det['appearance']
            )
        else:
            track['appearance'] = det.get('appearance')

        track['missed'] = 0
        track['updated'] = True

    def _prune(self, frame_id):
        retained_tracks = []
        for t in self.tracks:
            if t['missed'] <= self.max_missed:
                retained_tracks.append(t)
            else:
                self.current_frame_events.append({'frame': frame_id, 'track_id': t['track_id'], 'event': 'lost', 'global_id': t['global_id']})
                self.current_frame_statistics['lost_tracks'] += 1
                # Remove from global_id mappings if the track is truly lost
                if t['global_id'] in self.global_id_to_track:
                    del self.global_id_to_track[t['global_id']]
                if t['track_id'] in self.track_to_global_id:
                    del self.track_to_global_id[t['track_id']]
        self.tracks = retained_tracks


    def get_frame_events(self):
        return self.current_frame_events

    def get_frame_statistics(self):
        return self.current_frame_statistics

    def get_statistics(self):
        return {
            'total_refound_tracks': self.total_refound_tracks,
            'total_temporary_missed_frames': self.total_temporary_missed_frames,
            'total_active_tracks_last_frame': self.total_active_tracks,
            'current_tracked_objects': len(self.tracks)
        }

    # Main update logic
    def update(self, detections, frame_id, finalize=True): # new: finalize parameter
        self.current_frame_events = [] # clear events for the new frame
        self.current_frame_statistics = { # reset per-frame statistics
            'added_tracks': 0,
            'lost_tracks': 0,
            'refound_tracks': 0,
            'temporary_missed_tracks': 0,
            'active_tracks': 0,
        }
        
        for t in self.tracks:
            t['updated'] = False
            t['was_refound_in_this_frame'] = False # new: track if refound in this frame
            t['was_deduced'] = False # reset deduction flag

        if len(detections) == 0:
            for t in self.tracks:
                t['missed'] += 1
                if t['missed'] > 0 and t['missed'] <= self.max_missed:
                    self.current_frame_events.append({'frame': frame_id, 'track_id': t['track_id'], 'event': 'temporary_missed', 'global_id': t['global_id']})
                    self.current_frame_statistics['temporary_missed_tracks'] += 1
                    self.total_temporary_missed_frames += 1
            self._prune(frame_id)
            self.current_frame_statistics['active_tracks'] = len(self.tracks)
            self.total_active_tracks = len(self.tracks)
            return self.tracks, [], detections # Return tracks, matched_global_ids, unmatched_detections

        cost = self._cost_matrix(detections)
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks_indices = set()
        matched_dets_indices = set()
        matched_global_ids = []

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= (1.0 - self.match_threshold):
                self._update_track(self.tracks[r], detections[c], frame_id) # new: pass frame_id
                matched_tracks_indices.add(r)
                matched_dets_indices.add(c)
                matched_global_ids.append(self.tracks[r]['global_id'])


        # unmatched tracks (potential lost or temporary missed)
        for i, t in enumerate(self.tracks):
            if i not in matched_tracks_indices:
                t['missed'] += 1
                if t['missed'] > 0 and t['missed'] <= self.max_missed:
                    self.current_frame_events.append({'frame': frame_id, 'track_id': t['track_id'], 'event': 'temporary_missed', 'global_id': t['global_id']})
                    self.current_frame_statistics['temporary_missed_tracks'] += 1
                    self.total_temporary_missed_frames += 1


        # unmatched detections (potential new tracks or cross-view matches)
        unmatched_detections = []
        for j, d in enumerate(detections):
            if j not in matched_dets_indices:
                unmatched_detections.append(d)

        if finalize:
            for d in unmatched_detections:
                self._start_track(d, frame_id)
            unmatched_detections = [] # clear them since they are now tracks

        self._prune(frame_id)
        
        self.current_frame_statistics['active_tracks'] = len(self.tracks)
        self.total_active_tracks = len(self.tracks)
        return self.tracks, matched_global_ids, unmatched_detections

    def assign_cross_view_tracks(self, other_tracker_active_tracks, current_view_unmatched_detections, frame_id):
        # build a list of tracks from the other view that are not yet matched
        # (i.e. tracks that might represent an object also seen in current view's unmatched detections)
        
        cross_view_matches = []
        matched_other_view_tracks_indices = set()
        matched_current_dets_indices = set()

        if not other_tracker_active_tracks or not current_view_unmatched_detections:
            return current_view_unmatched_detections # No new matches if no tracks or no unmatched detections

        cross_cost_matrix = np.zeros((len(other_tracker_active_tracks), len(current_view_unmatched_detections)), dtype=np.float32)

        for i, other_track in enumerate(other_tracker_active_tracks):
            for j, current_det in enumerate(current_view_unmatched_detections):
                # use a similarity based on appearance and pose between tracks from different views
                sim = self._cross_view_similarity(other_track, current_det) # Assuming detection has track-like attributes
                cross_cost_matrix[i, j] = 1.0 - sim

        row_ind, col_ind = linear_sum_assignment(cross_cost_matrix)

        newly_matched_current_view_detections = []
        for r, c in zip(row_ind, col_ind):
            if cross_cost_matrix[r, c] <= (1.0 - self.cross_view_match_threshold):
                other_track = other_tracker_active_tracks[r]
                current_det = current_view_unmatched_detections[c]
                
                # Assign the global_id from the other track to the new track created from current_det
                new_track_obj = self._start_track(current_det, frame_id, global_id=other_track['global_id'])
                if new_track_obj:
                    cross_view_matches.append({
                        'current_view_track_id': new_track_obj['track_id'],
                        'other_view_track_id': other_track['track_id'],
                        'global_id': other_track['global_id']
                    })
                    matched_other_view_tracks_indices.add(r)
                    matched_current_dets_indices.add(c)
                    self.current_frame_events.append({
                        'frame': frame_id,
                        'track_id': new_track_obj['track_id'],
                        'event': 'cross_view_match_found',
                        'global_id': new_track_obj['global_id'],
                        'matched_with_other_view_track_id': other_track['track_id']
                    })

        # detections that were unmatched even after cross-view matching will start new tracks
        remaining_unmatched_detections = []
        for j, det in enumerate(current_view_unmatched_detections):
            if j not in matched_current_dets_indices:
                remaining_unmatched_detections.append(det)

        for det in remaining_unmatched_detections:
            self._start_track(det, frame_id)

        return cross_view_matches