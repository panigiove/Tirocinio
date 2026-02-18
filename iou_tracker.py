# Hungarian-based multi-cue tracker (IoU + appearance + pose)
# FIXED VERSION with corrections for multi-view tracking

import numpy as np
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
        self._active_counts = {}

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
            self._gids = gids
        else:
            self._gids = []

    def claim_id(self, gid, _frame_id=None):
        gid = int(gid)
        if gid not in self._active_counts:
            if self._fixed_pool:
                return False
            self._active_counts[gid] = 0
        self._active_counts[gid] += 1
        return True




class IOUTracker:
    """
    Multi-cue tracker that combines IoU, Appearance, Pose, and World coordinates.
    Uses the Hungarian algorithm (linear_sum_assignment) for optimal matching.
    """
    def __init__(self,
                 match_threshold=0.35,
                 iou_weight=0.5,
                 appearance_weight=0.35,
                 pose_weight=0.15,
                 world_dist_weight=0.7,
                 ema_alpha=0.9,
                 max_tracks=None,
                 max_velocity=140.0,
                 global_id_manager=None):
        """
        Initializes the tracker with various weights and thresholds.

        Parameters:
            match_threshold (float): Minimum similarity [0,1] required for a temporal match.
            cross_view_match_threshold (float): Minimum similarity [0,1] for cross-view matching.
            iou_weight (float): Weight for Intersection over Union (spatial overlap in 2D).
            appearance_weight (float): Weight for visual appearance similarity (cosine similarity).
            pose_weight (float): Weight for pose estimation similarity.
            world_dist_weight (float): Weight for Euclidean distance in 3D/world coordinates. 
                                     When available, this cue becomes dominant for tracking.
            ema_alpha (float): Momentum for Exponential Moving Average of appearance vectors.
            max_tracks (int): Limit the number of active tracks (useful for fixed scenarios).
            max_velocity (float): Maximum physical distance a person can travel between frames.
            global_id_manager (GlobalIDManager): Shared object to coordinate IDs across views.
        """
        self.match_threshold = match_threshold
        self.iou_w = iou_weight
        self.app_w = appearance_weight
        self.pose_w = pose_weight
        self.world_w = world_dist_weight
        self.ema_alpha = ema_alpha
        self.max_tracks = max_tracks
        self.max_velocity = max_velocity
        self.id_manager = global_id_manager if global_id_manager else GlobalIDManager()

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

    def _pair_similarity(self, track, det):
        iou_score = iou(track['bbox'], det['bbox'])
        app_score = self._appearance_sim(track.get('appearance'), det.get('appearance'))
        pose_score = self._pose_sim(track.get('pose_vec'), det.get('pose_vec'))

        # Non-spatial cues (IoU + appearance + pose), normalized.
        cue_norm = self.iou_w + self.app_w + self.pose_w
        if cue_norm <= 0:
            non_spatial_sim = 0.0
        else:
            non_spatial_sim = (
                self.iou_w / cue_norm * iou_score
                + self.app_w / cue_norm * app_score
                + self.pose_w / cue_norm * pose_score
            )

        track_wp = track.get('world_pos')
        det_wp = det.get('world_pos')
        has_world = (track_wp is not None) and (det_wp is not None)

        if has_world:
            # Use only X,Y for distance (ignore Z which is always 0).
            dist = np.linalg.norm(track_wp[:2] - det_wp[:2])
            max_dist = self.max_velocity * (track['missed'] + 1)
            w_spatial = self.world_w
            w_other = (1.0 - w_spatial)

            if dist > max_dist:
                # Impossible jump: heavily penalize non-spatial cues.
                sim = w_other * non_spatial_sim * 0.5
            else:
                spatial_sim = self._world_pos_sim(track_wp, det_wp, dist=dist)
                sim = (w_spatial * spatial_sim + w_other * non_spatial_sim)
        else:
            # Fallback when spatial information is unavailable.
            sim = non_spatial_sim

        return float(sim)

    def _cost_matrix(self, detections, track_indices=None, det_indices=None):
        if track_indices is None:
            track_indices = list(range(len(self.tracks)))
        if det_indices is None:
            det_indices = list(range(len(detections)))

        n_t = len(track_indices)
        n_d = len(det_indices)
        cost = np.zeros((n_t, n_d), dtype=np.float32)
        if n_t == 0 or n_d == 0:
            return cost

        for row_i, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]
            for col_i, det_idx in enumerate(det_indices):
                det = detections[det_idx]
                det_gid = det.get('global_id')
                track_gid = track.get('global_id')
                if (
                    det_gid is not None
                    and track_gid is not None
                    and int(det_gid) != int(track_gid)
                ):
                    cost[row_i, col_i] = 1.0
                    continue
                sim = self._pair_similarity(track, det)
                cost[row_i, col_i] = 1.0 - sim
        return cost

    def _match_track_subset(self, track_indices, detections, det_indices, frame_id):
        if not track_indices or not det_indices:
            return set(), set(), []

        cost = self._cost_matrix(
            detections=detections,
            track_indices=track_indices,
            det_indices=det_indices,
        )
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_track_indices = set()
        matched_det_indices = set()
        matched_global_ids = []

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > (1.0 - self.match_threshold):
                continue
            track_idx = track_indices[r]
            det_idx = det_indices[c]
            self._update_track(self.tracks[track_idx], detections[det_idx], frame_id)
            matched_track_indices.add(track_idx)
            matched_det_indices.add(det_idx)
            matched_global_ids.append(self.tracks[track_idx]['global_id'])

        return matched_track_indices, matched_det_indices, matched_global_ids

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
        # Convert distance to similarity using exponential decay
        sigma = 220.0 
        return float(np.exp(-dist / sigma))

    
    def _start_track(self, det, frame_id, global_id=None):
        """
        Start a new track. If a target global_id is provided and already present
        in this view, update that track slot (used for first-frame annotation bootstrap).
        """
        # If global_id is provided, try to update that specific existing track first.
        if global_id is not None:
            if global_id in self.global_id_to_track:
                existing_track = self.global_id_to_track[global_id]
                # If it's already updated, we can't update it again in this frame.
                if existing_track['updated']:
                    return None
                self._update_track(existing_track, det, frame_id)
                existing_track['updated'] = True
                return existing_track

        # Strict slot cap: never exceed max_tracks.
        if self.max_tracks is not None:
            if len(self.tracks) >= self.max_tracks:
                return None

        # In this pipeline tracks are bootstrapped from annotated IDs.
        if global_id is None:
            return None
        global_id = int(global_id)
        if not self.id_manager.claim_id(global_id, frame_id):
            return None
            
        new_track = {
            'track_id': self.next_id,
            'global_id': global_id,
            'bbox': det['bbox'],
            'appearance': det.get('appearance'),
            'pose_vec': det.get('pose_vec'),
            'keypoints': det.get('keypoints'),
            'world_pos': det.get('world_pos'),
            'start_frame': frame_id,
            'missed': 0,
            'updated': True,
            'was_refound_in_this_frame': False,
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
        event_name='merged_id',
    ):
        """
        Reassign a specific track to a new global ID.
        If new_gid is already occupied by a different track in this view,
        swap GIDs between the two tracks to keep the fixed-slot assignment valid.
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
        if occupied is not None:
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

        self.current_frame_events.append({
            'frame': frame_id,
            'track_id': tid,
            'event': event_name,
            'old_global_id': old_gid,
            'new_global_id': new_gid
        })
        return True

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

        track['bbox'] = det['bbox']
        track['keypoints'] = det.get('keypoints')
        track['pose_vec'] = det.get('pose_vec')
        track['world_pos'] = det.get('world_pos')

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

    def get_frame_events(self):
        return self.current_frame_events

    def update(
        self,
        detections,
        frame_id,
    ):
        """
        Main update logic with strict two-stage assignment:
        1) match detections to active tracks (missed == 0)
        2) match remaining detections to missed tracks (missed > 0)
        
        Parameters:
            detections: List of detection dictionaries
            frame_id: Current frame number
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

        if len(detections) == 0:
            for t in self.tracks:
                t['missed'] += 1
                if t['missed'] > 0:
                    self.current_frame_events.append({
                        'frame': frame_id, 
                        'track_id': t['track_id'], 
                        'event': 'temporary_missed', 
                        'global_id': t['global_id']
                    })
                    self.current_frame_statistics['temporary_missed_tracks'] += 1
                    self.total_temporary_missed_frames += 1
            self.current_frame_statistics['active_tracks'] = len(self.tracks)
            self.total_active_tracks = len(self.tracks)
            return self.tracks, [], detections

        all_det_indices = list(range(len(detections)))
        active_track_indices = [i for i, t in enumerate(self.tracks) if t.get('missed', 0) == 0]
        missed_track_indices = [i for i, t in enumerate(self.tracks) if t.get('missed', 0) > 0]

        # Stage 1: active tracks first.
        matched_tracks_indices, matched_dets_indices, matched_global_ids = self._match_track_subset(
            active_track_indices, detections, all_det_indices, frame_id
        )

        # Stage 2: remaining detections to previously missed tracks.
        remaining_det_indices = [j for j in all_det_indices if j not in matched_dets_indices]
        matched_tracks_missed, matched_dets_missed, matched_global_ids_missed = self._match_track_subset(
            missed_track_indices, detections, remaining_det_indices, frame_id
        )
        matched_tracks_indices.update(matched_tracks_missed)
        matched_dets_indices.update(matched_dets_missed)
        matched_global_ids.extend(matched_global_ids_missed)

        # Tracks still unmatched after both stages become missed.
        for i, t in enumerate(self.tracks):
            if i in matched_tracks_indices:
                continue
            t['missed'] += 1
            if t['missed'] > 0:
                self.current_frame_events.append({
                    'frame': frame_id,
                    'track_id': t['track_id'],
                    'event': 'temporary_missed',
                    'global_id': t['global_id']
                })
                self.current_frame_statistics['temporary_missed_tracks'] += 1
                self.total_temporary_missed_frames += 1

        unmatched_detections = [d for j, d in enumerate(detections) if j not in matched_dets_indices]

        for d in unmatched_detections:
            self._start_track(d, frame_id, global_id=d.get('global_id'))
        unmatched_detections = []

        # Nested duplicate suppression is intentionally done at detection stage
        # before assignment to tracks (see detect_view in process_video_two_views.py).
        
        self.current_frame_statistics['active_tracks'] = len(self.tracks)
        self.total_active_tracks = len(self.tracks)
        return self.tracks, matched_global_ids, unmatched_detections
