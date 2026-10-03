"""Pick the patient body out of a frame and log ID changes.

Track IDs are not stable: session 310826 observed 0 -> 318 -> 389 within two
consecutive recordings. Selection therefore cannot rely on a fixed ID. Two
rules are offered:

  fixed   -- a body ID configured by eye from the recording
  bed     -- torso centroid inside a configured bed volume  (default)

The bed rule survives ID resets, which the fixed rule does not.
"""
import numpy as np
from .types import TORSO


class SubjectFilter:
    def __init__(self, mode="bed", subject_id=None, bed_bounds=None,
                 hysteresis=0.0, seed=None):
        if mode not in ("fixed", "bed"):
            raise ValueError("mode must be 'fixed' or 'bed'")
        if mode == "fixed" and subject_id is None:
            raise ValueError("mode 'fixed' needs subject_id")
        if mode == "bed" and bed_bounds is None:
            raise ValueError("mode 'bed' needs bed_bounds "
                             "((xmin,xmax),(ymin,ymax),(zmin,zmax))")
        self.mode = mode
        self.subject_id = subject_id
        self.bed_bounds = bed_bounds
        self.hysteresis = float(hysteresis)
        # Continuity: when several bodies lie in the bed volume, keep the one
        # nearest the previous pick. A phantom cannot move 0.3-0.8 m in one
        # frame; an operator leaning in can appear there. Seeded from the
        # static-phantom torso position when given, else the first pick
        # falls back to the most-keypoints rule.
        self.last_c = None if seed is None else np.asarray(seed, float)
        self.last_id = None
        self.id_changes = []      # (t, old_id, new_id)

    def _in_bed(self, body, margin=0.0):
        """Acquisition uses the strict bounds; a body already held as the
        subject is retained inside bounds expanded by `hysteresis`, so a
        centroid jittering on the boundary does not oscillate in and out."""
        v = body.valid[TORSO]
        if not v.any():
            return False, None
        c = np.nanmean(body.keypoints[TORSO][v], axis=0)
        (x0, x1), (y0, y1), (z0, z1) = self.bed_bounds
        m = margin
        inside = (x0 - m <= c[0] <= x1 + m) and \
                 (y0 - m <= c[1] <= y1 + m) and \
                 (z0 - m <= c[2] <= z1 + m)
        return inside, c

    def select(self, frame):
        """Return (Body or None, id_changed)."""
        pick = None
        if self.mode == "fixed":
            for b in frame.bodies:
                if b.id == self.subject_id:
                    pick = b
                    break
        else:
            best = None
            for b in frame.bodies:
                margin = self.hysteresis if (
                    self.last_id is not None and b.id == self.last_id) else 0.0
                inside, c = self._in_bed(b, margin=margin)
                if not inside:
                    continue
                if self.last_c is not None:
                    score = -float(np.linalg.norm(c - self.last_c))
                else:
                    score = float(b.valid.sum())
                if best is None or score > best[0]:
                    best = (score, b, c)
            pick = best[1] if best else None
            if best is not None:
                self.last_c = np.asarray(best[2], float)

        changed = False
        if pick is not None:
            if self.last_id is not None and pick.id != self.last_id:
                self.id_changes.append((frame.t, self.last_id, pick.id))
                changed = True
            self.last_id = pick.id
        return pick, changed

    def report(self):
        return {"n_id_changes": len(self.id_changes),
                "id_changes": self.id_changes}
