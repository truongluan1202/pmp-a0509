"""Estimate v_hat_max, the patient limb speed bound, from the fast-motion bag.

Three filters, all necessary:
  invalid keypoints   -- a NaN either side gives no displacement
  ID resets           -- a track break renumbers the body; differencing across
                         it produces a jump that is not motion. Session 310826
                         saw 0 -> 318 -> 389, so this is not hypothetical.
  anatomical ceiling  -- displacements above MAX_PLAUSIBLE are detector error,
                         not limb speed, and would inflate the bound.
"""
import numpy as np
from .types import N_KP

MAX_PLAUSIBLE = 5.0     # m/s, above this it is a detection artefact


def speeds(frames, subject_filter, percentile=95.0, per_keypoint=False):
    prev_pos, prev_t, prev_id = None, None, None
    samples = [[] for _ in range(N_KP)]
    n_breaks = 0

    for f in frames:
        body, changed = subject_filter.select(f)
        if body is None:
            prev_pos, prev_t, prev_id = None, None, None
            continue
        if prev_pos is not None and (changed or body.id != prev_id):
            n_breaks += 1
            prev_pos, prev_t, prev_id = None, None, None

        if prev_pos is not None:
            dt = f.t - prev_t
            if dt > 1e-6:
                both = body.valid & np.isfinite(prev_pos).all(axis=1)
                disp = np.linalg.norm(body.keypoints - prev_pos, axis=1)
                v = disp / dt
                ok = both & np.isfinite(v) & (v <= MAX_PLAUSIBLE)
                for i in np.where(ok)[0]:
                    samples[i].append(float(v[i]))

        prev_pos = body.keypoints.copy()
        prev_t = f.t
        prev_id = body.id

    allv = np.array([v for s in samples for v in s])
    result = {
        "n_samples": int(allv.size),
        "n_track_breaks": n_breaks,
        "v_mean": float(allv.mean()) if allv.size else float("nan"),
        "v_p95": float(np.percentile(allv, percentile)) if allv.size else float("nan"),
        "v_p99": float(np.percentile(allv, 99)) if allv.size else float("nan"),
        "v_max": float(allv.max()) if allv.size else float("nan"),
    }
    if per_keypoint:
        result["per_keypoint_p95"] = [
            float(np.percentile(s, percentile)) if s else float("nan")
            for s in samples]
    return result
