# iou_tracker.py
# Hungarian-based multi-cue tracker (IoU + appearance + pose)

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


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


class IOUTracker:
    def __init__(self,
                 match_threshold=0.35,
                 max_missed_frames=10,
                 iou_weight=0.5,
                 appearance_weight=0.35,
                 pose_weight=0.15,
                 ema_alpha=0.9):
        self.match_threshold = match_threshold
        self.max_missed = max_missed_frames
        self.iou_w = iou_weight
        self.app_w = appearance_weight
        self.pose_w = pose_weight
        self.ema_alpha = ema_alpha

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

    def _appearance_sim(self, a, b):
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))  # cosine similarity (vectors are normalized)

    def _pose_sim(self, p1, p2):
        if p1 is None or p2 is None:
            return 0.0
        d = np.linalg.norm(p1 - p2)
        return float(np.exp(-d))

    def _cost_matrix(self, detections):
        n_t = len(self.tracks)
        n_d = len(detections)
        cost = np.zeros((n_t, n_d), dtype=np.float32)

        for i, t in enumerate(self.tracks):
            for j, d in enumerate(detections):
                iou_score = iou(t['bbox'], d['bbox'])
                app_score = self._appearance_sim(t.get('appearance'), d.get('appearance'))
                pose_score = self._pose_sim(t.get('pose_vec'), d.get('pose_vec'))

                sim = (self.iou_w * iou_score +
                       self.app_w * app_score +
                       self.pose_w * pose_score)
                cost[i, j] = 1.0 - sim
        return cost

    def update(self, detections, frame_id): # new: accept frame_id
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

        if len(self.tracks) == 0:
            for d in detections:
                self._start_track(d, frame_id)
            self._prune(frame_id) # ensure pruning happens even for first frame
            self.current_frame_statistics['active_tracks'] = len(self.tracks)
            self.total_active_tracks = len(self.tracks)
            return self.tracks

        if len(detections) == 0:
            for t in self.tracks:
                t['missed'] += 1
                if t['missed'] > 0 and t['missed'] <= self.max_missed:
                    self.current_frame_events.append({'frame': frame_id, 'track_id': t['track_id'], 'event': 'temporary_missed'})
                    self.current_frame_statistics['temporary_missed_tracks'] += 1
                    self.total_temporary_missed_frames += 1
            self._prune(frame_id)
            self.current_frame_statistics['active_tracks'] = len(self.tracks)
            self.total_active_tracks = len(self.tracks)
            return self.tracks

        cost = self._cost_matrix(detections)
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= (1.0 - self.match_threshold):
                self._update_track(self.tracks[r], detections[c], frame_id) # new: pass frame_id
                matched_tracks.add(r)
                matched_dets.add(c)

        # unmatched tracks
        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t['missed'] += 1
                if t['missed'] > 0 and t['missed'] <= self.max_missed:
                    self.current_frame_events.append({'frame': frame_id, 'track_id': t['track_id'], 'event': 'temporary_missed'})
                    self.current_frame_statistics['temporary_missed_tracks'] += 1
                    self.total_temporary_missed_frames += 1


        # new tracks
        for j, d in enumerate(detections):
            if j not in matched_dets:
                self._start_track(d, frame_id)

        self._prune(frame_id)
        
        self.current_frame_statistics['active_tracks'] = len(self.tracks)
        self.total_active_tracks = len(self.tracks)
        return self.tracks

    def _start_track(self, det, frame_id):
        new_track = {
            'track_id': self.next_id,
            'bbox': det['bbox'],
            'appearance': det.get('appearance'),
            'pose_vec': det.get('pose_vec'),
            'keypoints': det.get('keypoints'),
            'missed': 0,
            'updated': True,
            'was_refound_in_this_frame': False # new: initial state
        }
        self.tracks.append(new_track)
        self.current_frame_events.append({'frame': frame_id, 'track_id': self.next_id, 'event': 'added'})
        self.current_frame_statistics['added_tracks'] += 1
        self.next_id += 1

    def _update_track(self, track, det, frame_id): # new: accept frame_id
        # if this track was previously missed, it's now refound
        if track['missed'] > 0:
            self.current_frame_events.append({'frame': frame_id, 'track_id': track['track_id'], 'event': 'refound'})
            self.current_frame_statistics['refound_tracks'] += 1
            self.total_refound_tracks += 1
            track['was_refound_in_this_frame'] = True # mark as refound

        track['bbox'] = det['bbox']
        track['keypoints'] = det.get('keypoints')
        track['pose_vec'] = det.get('pose_vec')

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
                self.current_frame_events.append({'frame': frame_id, 'track_id': t['track_id'], 'event': 'lost'})
                self.current_frame_statistics['lost_tracks'] += 1
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