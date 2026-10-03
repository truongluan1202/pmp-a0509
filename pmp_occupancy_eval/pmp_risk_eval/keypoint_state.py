"""Per-keypoint hold state.

A keypoint that stops being reported has not moved. It has stopped being
observed. This tracker keeps its last position and counts how long ago that
was, which is the input the risk field needs to grow its uncertainty radius.
"""
import numpy as np
from .types import State, N_KP


class KeypointTracker:
    def __init__(self, d_contact):
        d = np.asarray(d_contact, dtype=float)
        if d.shape != (N_KP,):
            raise ValueError(f"d_contact must be ({N_KP},), got {d.shape}")
        self.d_contact = d
        self.positions = np.full((N_KP, 3), np.nan)
        self.last_seen_t = np.full(N_KP, np.nan)
        self.ever_seen = np.zeros(N_KP, dtype=bool)
        self._t = None

    def update(self, t, body, subject_id=None, id_changed=False):
        """body may be None (subject not detected this frame)."""
        seen_now = np.zeros(N_KP, dtype=bool)
        if body is not None:
            seen_now = body.valid.copy()
            self.positions[seen_now] = body.keypoints[seen_now]
            self.last_seen_t[seen_now] = t
            self.ever_seen |= seen_now

        ages = np.where(self.ever_seen, t - self.last_seen_t, np.inf)
        self._t = t
        return State(t=t, positions=self.positions.copy(), ages=ages,
                     d_contact=self.d_contact.copy(), seen_now=seen_now,
                     ever_seen=self.ever_seen.copy(),
                     subject_id=subject_id, id_changed=id_changed)
