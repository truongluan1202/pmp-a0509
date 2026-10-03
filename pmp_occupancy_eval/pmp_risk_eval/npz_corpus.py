"""Read an exported .npz corpus. No ROS, no sqlite, no camera.

Yields the same Frame objects as bag_reader.read_frames, so every consumer
downstream, subject selection included, is unchanged. This is what makes the
analysis portable: the container is needed once, to decode, and never again.
"""
import numpy as np

from .types import Body, Frame, N_KP


def read_frames_npz(path, max_frames=None):
    """Yield Frame objects from a file written by scripts/export_npz.py."""
    z = np.load(path, allow_pickle=False)
    times = z["times"]
    t_abs = z["t_abs"]
    counts = z["body_counts"]
    kp = z["keypoints"]
    valid = z["valid"]
    conf = z["confidence"]
    ids = z["body_ids"]
    tstate = z["tracking_state"]

    cursor = 0
    for f in range(len(times)):
        n = int(counts[f])
        bodies = []
        for b in range(cursor, cursor + n):
            bodies.append(Body(
                id=int(ids[b]),
                keypoints=kp[b].astype(float),
                valid=valid[b].copy(),
                confidence=conf[b].astype(float),
                tracking_state=int(tstate[b]),
                label_id=int(ids[b]),
            ))
        cursor += n
        yield Frame(t=float(times[f]), bodies=bodies, t_abs=float(t_abs[f]))
        if max_frames and f + 1 >= max_frames:
            break


def summarise(path):
    """Quick shape check without building Frame objects."""
    z = np.load(path, allow_pickle=False)
    counts = z["body_counts"]
    valid = z["valid"]
    times = z["times"]
    dur = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    return {
        "frames": int(len(times)),
        "duration_s": dur,
        "rate_hz": (len(times) - 1) / dur if dur > 0 else float("nan"),
        "body_records": int(counts.sum()),
        "frames_with_no_body": int((counts == 0).sum()),
        "frames_multi_body": int((counts > 1).sum()),
        "mean_keypoint_yield": float(valid.mean()) if valid.size else 0.0,
    }
