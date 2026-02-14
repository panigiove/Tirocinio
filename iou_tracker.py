# Hungarian-based multi-cue tracker (IoU + appearance + pose)
# FIXED VERSION with corrections for multi-view tracking

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
    def __init__(self, pool_size=None, gid_values=None, cooldown_frames=90):
        self.cooldown_frames = int(max(0, cooldown_frames))
        self._active_counts = {}
        self._last_released_frame = {}
        self._next_id = 0

        # Fixed-pool mode when pool_size/gid_values are provided.
        self._fixed_pool = (gid_values is not None) or (pool_size is not None)
        if self._fixed_pool:
            if gid_values is None:
                gid_values = list(range(int(pool_size)))
            gids = sorted({int(g) for g in gid_values})
            if not gids:
                raise ValueError("GlobalIDManager requires at least one GID in the pool")
            for gid in gids:
                self._active_counts[gid] = 0
                self._last_released_frame[gid] = -10**9
            self._gids = gids
        else:
            self._gids = []

    def update_max_id(self, current_id):
        """
        Backward compatibility helper.
        """
        if self._fixed_pool:
            return
        current_id = int(current_id)
        if current_id >= self._next_id:
            self._next_id = current_id + 1

    def claim_id(self, gid, _frame_id=None):
        gid = int(gid)
        if gid not in self._active_counts:
            if self._fixed_pool:
                return False
            self._active_counts[gid] = 0
            self._last_released_frame[gid] = -10**9
            if gid >= self._next_id:
                self._next_id = gid + 1
        self._active_counts[gid] += 1
        return True

    def release_id(self, gid, frame_id):
        gid = int(gid)
        if gid not in self._active_counts:
            return False
        if self._active_counts[gid] > 0:
            self._active_counts[gid] -= 1
        if self._active_counts[gid] == 0:
            self._last_released_frame[gid] = int(frame_id)
        return True

    def next_id(self, frame_id=0):
        if not self._fixed_pool:
            gid = self._next_id
            self._next_id += 1
            self._active_counts[gid] = self._active_counts.get(gid, 0)
            self._last_released_frame[gid] = self._last_released_frame.get(gid, -10**9)
            return gid

        frame_id = int(frame_id)
        free_and_cooled = []

        for gid in self._gids:
            if self._active_counts[gid] != 0:
                continue
            if (frame_id - self._last_released_frame[gid]) >= self.cooldown_frames:
                free_and_cooled.append(gid)

        if free_and_cooled:
            return min(free_and_cooled)

        # Pool exhausted (all active).
        return None




class IOUTracker:
    """
    Multi-cue tracker that combines IoU, Appearance, Pose, and World coordinates.
    Uses the Hungarian algorithm (linear_sum_assignment) for optimal matching.
    """
    def __init__(self,
                 match_threshold=0.35,
                 max_missed_frames=10,
                 iou_weight=0.5,
                 appearance_weight=0.35,
                 pose_weight=0.15,
                 world_dist_weight=0.7,
                 world_dist_tolerance=0.0,
                 ema_alpha=0.9,
                 camera_params=None,
                 max_tracks=None,
                 max_velocity=1000.0,
                 global_id_manager=None,
                 is_rectified=True,  # NEW: flag for rectified videos
                 prune_tracks=True,
                 fixed_slots_mode=False,
                 suppress_nested_tracks=True,
                 nested_contain_thresh=0.9,
                 nested_area_ratio=0.6,
                 nested_app_sim_thresh=0.7):
        """
        Initializes the tracker with various weights and thresholds.

        Parameters:
            match_threshold (float): Minimum similarity [0,1] required for a temporal match.
            cross_view_match_threshold (float): Minimum similarity [0,1] for cross-view matching.
            max_missed_frames (int): Number of frames to keep a track alive without detections.
            iou_weight (float): Weight for Intersection over Union (spatial overlap in 2D).
            appearance_weight (float): Weight for visual appearance similarity (cosine similarity).
            pose_weight (float): Weight for pose estimation similarity.
            world_dist_weight (float): Weight for Euclidean distance in 3D/world coordinates. 
                                     When available, this cue becomes dominant for tracking.
            ema_alpha (float): Momentum for Exponential Moving Average of appearance vectors.
            camera_params (dict): Calibration data (K, R, t) for 3D projection.
            max_tracks (int): Limit the number of active tracks (useful for fixed scenarios).
            max_velocity (float): Maximum physical distance a person can travel between frames.
            global_id_manager (GlobalIDManager): Shared object to coordinate IDs across views.
            is_rectified (bool): Whether input videos are already rectified (no distortion).
        """
        self.match_threshold = match_threshold
        self.max_missed = max_missed_frames
        self.iou_w = iou_weight
        self.app_w = appearance_weight
        self.pose_w = pose_weight
        self.world_w = world_dist_weight
        self.world_dist_tol = world_dist_tolerance
        self.ema_alpha = ema_alpha
        self.max_tracks = max_tracks
        self.max_velocity = max_velocity
        self.id_manager = global_id_manager if global_id_manager else GlobalIDManager()
        self.is_rectified = is_rectified
        self.prune_tracks = bool(prune_tracks)
        self.fixed_slots_mode = bool(fixed_slots_mode)
        self.suppress_nested_tracks = suppress_nested_tracks
        self.nested_contain_thresh = nested_contain_thresh
        self.nested_area_ratio = nested_area_ratio
        self.nested_app_sim_thresh = nested_app_sim_thresh
        
        # Camera parameters for spatial projection
        self.camera_params = camera_params
        if camera_params:
            self.mtx = np.array(camera_params['mtx'], dtype=np.float32)
            # FIX: For rectified videos, set dist to None
            if is_rectified:
                self.dist = None
            else:
                self.dist = np.array(camera_params['dist'], dtype=np.float32)
            
            self.rvec = np.array(camera_params['rvecs'], dtype=np.float32)
            self.tvec = np.array(camera_params['tvecs'], dtype=np.float32)
            
            # FIX: Ensure tvec is column vector (3,1)
            if self.tvec.ndim == 1:
                self.tvec = self.tvec.reshape(3, 1)
            
            # Compute rotation matrix
            R = cv2.Rodrigues(self.rvec)[0]
            
            # Compute projection matrix P = K * [R | t]
            self.P = self.mtx @ np.hstack([R, self.tvec])

            # FIX: Correct homography to ground plane (Z=0)
            # H = K * [r1 r2 t] where r1, r2 are first two columns of R
            H = self.mtx @ np.column_stack([R[:, 0], R[:, 1], self.tvec.flatten()])
            self.H_inv = np.linalg.inv(H)
        self.H_inv_dynamic = None

        self.tracks = []
        self.next_id = 0
        self.current_frame_events = []
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
        self.track_to_global_id = {}
        self.global_id_to_track = {}

    @staticmethod
    def _bbox_area(b):
        x1, y1, x2, y2 = b
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _bbox_intersection(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        return max(0, ix2 - ix1) * max(0, iy2 - iy1)

    def _suppress_nested_tracks(self, frame_id):
        if len(self.tracks) < 2:
            return
        to_remove = set()
        for i in range(len(self.tracks)):
            ti = self.tracks[i]
            if not ti.get('updated'):
                continue
            bi = ti.get('bbox')
            if bi is None:
                continue
            ai = self._bbox_area(bi)
            if ai <= 0:
                continue
            for j in range(len(self.tracks)):
                if i == j:
                    continue
                tj = self.tracks[j]
                if not tj.get('updated'):
                    continue
                bj = tj.get('bbox')
                if bj is None:
                    continue
                aj = self._bbox_area(bj)
                if aj <= 0 or aj >= ai:
                    continue
                inter = self._bbox_intersection(bi, bj)
                contain = inter / aj if aj > 0 else 0.0
                area_ratio = aj / ai
                if contain < self.nested_contain_thresh or area_ratio > self.nested_area_ratio:
                    continue
                if tj.get('gid_source') == 'annotation':
                    continue
                if self.nested_app_sim_thresh is not None:
                    sim = self._appearance_sim(ti.get('appearance'), tj.get('appearance'))
                    if sim < self.nested_app_sim_thresh:
                        continue
                to_remove.add(tj['track_id'])

        if not to_remove:
            return
        retained = []
        for t in self.tracks:
            if t['track_id'] in to_remove:
                self.current_frame_events.append({
                    'frame': frame_id,
                    'track_id': t['track_id'],
                    'event': 'duplicate_suppressed',
                    'global_id': t['global_id']
                })
                self.id_manager.release_id(t['global_id'], frame_id)
                if t['global_id'] in self.global_id_to_track and self.global_id_to_track[t['global_id']] is t:
                    del self.global_id_to_track[t['global_id']]
                if t['track_id'] in self.track_to_global_id:
                    del self.track_to_global_id[t['track_id']]
                continue
            retained.append(t)
        self.tracks = retained

    def project_to_world(self, bbox):
        """Project the bottom center of the bounding box to the ground plane (Z=0)."""
        if not hasattr(self, 'H_inv'):
            return None
        H_inv = self.H_inv_dynamic if self.H_inv_dynamic is not None else self.H_inv
        
        x1, x2, y2 = bbox[0], bbox[2], bbox[3]
        # Use the bottom center of the bounding box as the feet position
        pixel_point = np.array([(x1 + x2) / 2.0, y2, 1.0], dtype=np.float32).reshape(3, 1)
        
        world_point = H_inv @ pixel_point
        world_point /= world_point[2]  # Normalize
        
        # FIX: Return 3D point with Z=0 for consistency
        return np.array([world_point[0, 0], world_point[1, 0], 0.0], dtype=np.float32)

    def set_dynamic_hinv(self, H_inv):
        self.H_inv_dynamic = H_inv

    def project_to_pixel(self, world_pos):
        """Project world point (X, Y, Z) back to pixel coordinates (x, y)."""
        if not hasattr(self, 'mtx') or self.mtx is None:
            return None
        
        # world_pos can be (X, Y) or (X, Y, Z)
        if len(world_pos) == 2:
            pts_3d = np.array([[world_pos[0], world_pos[1], 0.0]], dtype=np.float32)
        else:
            pts_3d = np.array([world_pos], dtype=np.float32)
        
        # FIX: Use dist=None for rectified videos
        img_pts = cv2.projectPoints(pts_3d, self.rvec, self.tvec, self.mtx, self.dist)[0]
        return img_pts[0][0]  # Returns [x, y]


    @staticmethod
    def _appearance_sim(a, b):
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))  # cosine similarity (vectors are normalized)

    @staticmethod
    def _pose_sim(p1, p2):
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
                
                # Spatial gating logic
                spatial_sim = 0.0
                physically_possible = True
                
                if t.get('world_pos') is not None and d.get('world_pos') is not None:
                    # FIX: Use only X,Y for distance (ignore Z which is always 0)
                    twp = t['world_pos'][:2]
                    dwp = d['world_pos'][:2]
                    dist = np.linalg.norm(twp - dwp)
                    # Gate: max distance allowed is velocity * frames since last update
                    max_dist = self.max_velocity * (t['missed'] + 1)
                    
                    if dist > (max_dist + self.world_dist_tol):
                        physically_possible = False
                        spatial_sim = 0.0
                    else:
                        spatial_sim = self._world_pos_sim(t['world_pos'], d['world_pos'], dist=dist)

                if t.get('world_pos') is not None and d.get('world_pos') is not None:
                    # Use provided weights for non-spatial cues, normalized to sum to 1.0
                    w_norm = self.iou_w + self.app_w + self.pose_w
                    if w_norm == 0:
                        non_spatial_sim = 0
                    else:
                        non_spatial_sim = (self.iou_w / w_norm * iou_score + 
                                          self.app_w / w_norm * app_score + 
                                          self.pose_w / w_norm * pose_score)

                    if not physically_possible:
                        # Impossible jump: heavily penalize
                        spatial_sim = 0.0
                        w_spatial = self.world_w
                        w_other = (1.0 - w_spatial)
                        sim = w_other * non_spatial_sim * 0.5  # Severe penalty
                    else:
                        # Spatial information is pivotal
                        w_spatial = self.world_w
                        w_other = (1.0 - w_spatial)
                        sim = (w_spatial * spatial_sim + 
                               w_other * non_spatial_sim)
                else:
                    # Fallback to standard cues if spatial is missing
                    norm = self.iou_w + self.app_w + self.pose_w
                    sim = (self.iou_w / norm * iou_score + 
                           self.app_w / norm * app_score + 
                           self.pose_w / norm * pose_score)
                
                # Give a boost to tracks that were missed to encourage re-identification (Re-ID)
                if t['missed'] > 0 and physically_possible:
                    reid_boost = 0.15
                    # Additional boost if the detection is spatially very near
                    if spatial_sim > 0.8:
                        reid_boost += 0.10
                    sim += reid_boost
                
                cost[i, j] = 1.0 - sim
        return cost

    def _world_pos_sim(self, wp1, wp2, dist=None):
        """
        Calculates similarity between two world positions using exponential decay.
        
        Parameters:
            wp1 (np.array): World coordinates (X, Y, Z).
            wp2 (np.array): World coordinates (X, Y, Z).
            dist (float, optional): Pre-calculated Euclidean distance.
            
        Returns:
            float: Similarity score in range [0, 1].
        """
        if wp1 is None or wp2 is None:
            return 0.0
        # Use pre-calculated distance if available
        if dist is None:
            # Use only X,Y for distance
            dist = np.linalg.norm(wp1[:2] - wp2[:2])
        # Allow a tolerance buffer for calibration error / minor drift
        if self.world_dist_tol > 0:
            dist = max(0.0, dist - self.world_dist_tol)
        # Convert distance to similarity using exponential decay
        sigma = 1000.0 
        return float(np.exp(-dist / sigma))

    
    def _start_track(self, det, frame_id, global_id=None):
        """
        Starts a new track or reactivates an existing one.
        
        FIX: Improved Re-ID logic and global_id handling.
        """
        # 1. If global_id is provided, try to find that specific track first
        if global_id is not None:
            if global_id in self.global_id_to_track:
                existing_track = self.global_id_to_track[global_id]
                # If it's already updated, we can't update it again
                if existing_track['updated']:
                    return None
                self._update_track(existing_track, det, frame_id)
                existing_track['updated'] = True
                return existing_track
        
        # 2. Try to Re-ID into any MISSED track based on appearance/pose/world_pos
        best_sim = -1
        best_t = None
        missed_tracks = [t for t in self.tracks if not t['updated']]
        
        for t in missed_tracks:
            app_sim = self._appearance_sim(t.get('appearance'), det.get('appearance'))
            pose_sim = self._pose_sim(t.get('pose_vec'), det.get('pose_vec'))
            spatial_sim = self._world_pos_sim(t.get('world_pos'), det.get('world_pos'))
            
            # FIX: Include spatial similarity in Re-ID
            w_norm = self.app_w + self.pose_w
            if w_norm > 0 and spatial_sim > 0:
                # Combine appearance, pose, and spatial
                sim = (self.app_w / w_norm * app_sim + 
                       self.pose_w / w_norm * pose_sim) * 0.5 + spatial_sim * 0.5
            elif w_norm > 0:
                sim = (self.app_w / w_norm * app_sim + self.pose_w / w_norm * pose_sim)
            else:
                sim = spatial_sim
            
            if sim > best_sim:
                best_sim = sim
                best_t = t
        
        # Threshold for Re-ID
        if best_t is not None and (best_sim > 0.4 or self.fixed_slots_mode):
            self._update_track(best_t, det, frame_id)
            best_t['updated'] = True
            return best_t

        # 3. If we can't Re-ID, only start a new track if we are under the limit
        active_in_frame = len([t for t in self.tracks if t['updated']])
        if self.max_tracks is not None:
            if active_in_frame >= self.max_tracks:
                return None
            if global_id is None and len(self.global_id_to_track) >= self.max_tracks:
                return None

        # FIX: Use fixed-pool GIDs with cooldown-aware allocation.
        if global_id is None:
            global_id = self.id_manager.next_id(frame_id)
            if global_id is None:
                return None
            if not self.id_manager.claim_id(global_id, frame_id):
                return None
        else:
            global_id = int(global_id)
            if not self.id_manager.claim_id(global_id, frame_id):
                fallback_gid = self.id_manager.next_id(frame_id)
                if fallback_gid is None or not self.id_manager.claim_id(fallback_gid, frame_id):
                    return None
                global_id = fallback_gid
            
        new_track = {
            'track_id': self.next_id,
            'global_id': global_id,
            'gid_source': det.get('gid_source'),
            'bbox': det['bbox'],
            'appearance': det.get('appearance'),
            'pose_vec': det.get('pose_vec'),
            'keypoints': det.get('keypoints'),
            'world_pos': det.get('world_pos'),
            'start_frame': frame_id,
            'missed': 0,
            'updated': True,
            'was_refound_in_this_frame': False,
            'was_deduced': False
        }
        self.tracks.append(new_track)
        self.track_to_global_id[self.next_id] = global_id
        self.global_id_to_track[global_id] = new_track 
        self.current_frame_events.append({
            'frame': frame_id, 
            'track_id': self.next_id, 
            'event': 'added', 
            'global_id': global_id
        })
        self.current_frame_statistics['added_tracks'] += 1
        self.next_id += 1
        return new_track

    def _find_track_by_gid(self, gid, exclude_track=None):
        if gid is None:
            return None
        t = self.global_id_to_track.get(gid)
        if t is not None and t is not exclude_track and t in self.tracks:
            return t
        for cand in self.tracks:
            if cand is exclude_track:
                continue
            if cand.get('global_id') == gid:
                return cand
        return None

    def reassign_track_global_id(
        self,
        track,
        new_gid,
        frame_id,
        allow_target_conflict=False,
        event_name='merged_id',
    ):
        """
        Reassign a specific track to a new global ID.
        If allow_target_conflict is False, reassignment is refused when new_gid
        is already occupied by a different track in this view.
        """
        if track is None or new_gid is None:
            return False

        old_gid = track.get('global_id')
        tid = track.get('track_id')
        new_gid = int(new_gid)

        if old_gid == new_gid:
            if tid is not None:
                self.track_to_global_id[tid] = new_gid
            self.global_id_to_track[new_gid] = track
            return True

        occupied = self._find_track_by_gid(new_gid, exclude_track=track)
        if occupied is not None and not allow_target_conflict:
            return False

        if occupied is not None and allow_target_conflict:
            # In fixed-slot settings we often want GIDs to follow the cross-view
            # association result, even if the target GID is currently occupied.
            # Resolve this by swapping GIDs between the two tracks.
            occupied_tid = occupied.get('track_id')
            occupied_gid = occupied.get('global_id')

            track['global_id'] = new_gid
            if tid is not None:
                self.track_to_global_id[tid] = new_gid
            self.global_id_to_track[new_gid] = track

            if occupied_gid is not None:
                occupied['global_id'] = old_gid
                if occupied_tid is not None:
                    self.track_to_global_id[occupied_tid] = old_gid
                if old_gid is not None:
                    self.global_id_to_track[old_gid] = occupied

            self.current_frame_events.append({
                'frame': frame_id,
                'track_id': tid,
                'event': event_name,
                'old_global_id': old_gid,
                'new_global_id': new_gid,
                'swapped_with_track_id': occupied_tid
            })
            return True

        if not self.id_manager.claim_id(new_gid, frame_id):
            return False

        if old_gid in self.global_id_to_track and self.global_id_to_track[old_gid] is track:
            del self.global_id_to_track[old_gid]

        track['global_id'] = new_gid
        if tid is not None:
            self.track_to_global_id[tid] = new_gid
        self.global_id_to_track[new_gid] = track
        if old_gid is not None:
            self.id_manager.release_id(old_gid, frame_id)

        self.current_frame_events.append({
            'frame': frame_id,
            'track_id': tid,
            'event': event_name,
            'old_global_id': old_gid,
            'new_global_id': new_gid
        })
        return True

    def merge_global_ids(self, old_gid, new_gid, frame_id):
        """
        Merges two global IDs. All references to old_gid will be updated to new_gid.
        """
        if old_gid == new_gid:
            return True

        track = self._find_track_by_gid(old_gid)
        if track is None:
            return False
        return self.reassign_track_global_id(
            track, new_gid, frame_id, allow_target_conflict=False
        )

    def resolve_gid_conflicts(self, frame_id):
        """
        Resolve duplicate global_id assignments within the same view.
        Keeps the oldest track (earliest start_frame, then annotated) and reassigns others.
        """
        gid_to_tracks = {}
        for t in self.tracks:
            gid = t.get('global_id')
            if gid is None:
                continue
            gid_to_tracks.setdefault(gid, []).append(t)

        for gid, tracks in gid_to_tracks.items():
            if len(tracks) <= 1:
                continue

            # Keep oldest track, prefer annotated if start_frame ties
            def _keep_key(t):
                annot = 0 if t.get('gid_source') == 'annotation' else 1
                return (t.get('start_frame', 0), annot, t.get('track_id', 0))

            tracks_sorted = sorted(tracks, key=_keep_key)
            keeper = tracks_sorted[0]

            for t in tracks_sorted[1:]:
                new_gid = self.id_manager.next_id(frame_id)
                if new_gid is None:
                    continue
                moved = self.reassign_track_global_id(
                    t,
                    new_gid,
                    frame_id,
                    allow_target_conflict=False,
                    event_name='gid_conflict_resolved',
                )
                if moved:
                    t['gid_source'] = None

            # Ensure keeper mapping is correct
            self.global_id_to_track[keeper['global_id']] = keeper
            self.track_to_global_id[keeper['track_id']] = keeper['global_id']

    def _update_track(self, track, det, frame_id):
        # if this track was previously missed, it's now refound
        if track['missed'] > 0:
            self.current_frame_events.append({
                'frame': frame_id, 
                'track_id': track['track_id'], 
                'event': 'refound', 
                'global_id': track['global_id']
            })
            self.current_frame_statistics['refound_tracks'] += 1
            self.total_refound_tracks += 1
            track['was_refound_in_this_frame'] = True

        if det.get('gid_source') == 'annotation' and det.get('global_id') is not None:
            if det['global_id'] != track['global_id']:
                self.reassign_track_global_id(
                    track, det['global_id'], frame_id, allow_target_conflict=False
                )

        track['bbox'] = det['bbox']
        track['keypoints'] = det.get('keypoints')
        track['pose_vec'] = det.get('pose_vec')
        track['world_pos'] = det.get('world_pos')
        if det.get('gid_source') is not None:
            track['gid_source'] = det.get('gid_source')

        # EMA update for appearance
        if track.get('appearance') is not None and det.get('appearance') is not None:
            updated_app = (
                self.ema_alpha * track['appearance'] +
                (1.0 - self.ema_alpha) * det['appearance']
            )
            # Re-normalize to maintain unit vector for cosine similarity
            norm = np.linalg.norm(updated_app)
            if norm > 0:
                updated_app /= norm
            track['appearance'] = updated_app
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
                self.current_frame_events.append({
                    'frame': frame_id, 
                    'track_id': t['track_id'], 
                    'event': 'lost', 
                    'global_id': t['global_id']
                })
                self.current_frame_statistics['lost_tracks'] += 1
                self.id_manager.release_id(t['global_id'], frame_id)
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

    def update(self, detections, frame_id, finalize=True, resolve_conflicts=True):
        """
        Main update logic.
        
        Parameters:
            detections: List of detection dictionaries
            frame_id: Current frame number
            finalize: If True, create new tracks for unmatched detections.
                     If False, only update existing tracks.
        """
        self.current_frame_events = []
        self.current_frame_statistics = {
            'added_tracks': 0,
            'lost_tracks': 0,
            'refound_tracks': 0,
            'temporary_missed_tracks': 0,
            'active_tracks': 0,
        }
        
        for t in self.tracks:
            t['updated'] = False
            t['was_refound_in_this_frame'] = False
            t['was_deduced'] = False

        if len(detections) == 0:
            for t in self.tracks:
                t['missed'] += 1
                if t['missed'] > 0 and (not self.prune_tracks or t['missed'] <= self.max_missed):
                    self.current_frame_events.append({
                        'frame': frame_id, 
                        'track_id': t['track_id'], 
                        'event': 'temporary_missed', 
                        'global_id': t['global_id']
                    })
                    self.current_frame_statistics['temporary_missed_tracks'] += 1
                    self.total_temporary_missed_frames += 1
            if self.prune_tracks:
                self._prune(frame_id)
            self.current_frame_statistics['active_tracks'] = len(self.tracks)
            self.total_active_tracks = len(self.tracks)
            return self.tracks, [], detections

        cost = self._cost_matrix(detections)
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks_indices = set()
        matched_dets_indices = set()
        matched_global_ids = []

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= (1.0 - self.match_threshold):
                self._update_track(self.tracks[r], detections[c], frame_id)
                matched_tracks_indices.add(r)
                matched_dets_indices.add(c)
                matched_global_ids.append(self.tracks[r]['global_id'])

        # unmatched tracks
        for i, t in enumerate(self.tracks):
            if i not in matched_tracks_indices:
                t['missed'] += 1
                if t['missed'] > 0 and (not self.prune_tracks or t['missed'] <= self.max_missed):
                    self.current_frame_events.append({
                        'frame': frame_id, 
                        'track_id': t['track_id'], 
                        'event': 'temporary_missed', 
                        'global_id': t['global_id']
                    })
                    self.current_frame_statistics['temporary_missed_tracks'] += 1
                    self.total_temporary_missed_frames += 1

        # unmatched detections
        unmatched_detections = []
        for j, d in enumerate(detections):
            if j not in matched_dets_indices:
                unmatched_detections.append(d)

        if finalize:
            for d in unmatched_detections:
                self._start_track(d, frame_id, global_id=d.get('global_id'))
            unmatched_detections = []

        if self.suppress_nested_tracks:
            self._suppress_nested_tracks(frame_id)

        if resolve_conflicts:
            self.resolve_gid_conflicts(frame_id)
        if self.prune_tracks:
            self._prune(frame_id)
        
        self.current_frame_statistics['active_tracks'] = len(self.tracks)
        self.total_active_tracks = len(self.tracks)
        return self.tracks, matched_global_ids, unmatched_detections
