"""Acausal reference.

The estimator under test is causal: at time t it knows only frames up to t.
The reference is allowed to look forward. A keypoint missing at t but present
at t-k and t+m is interpolated between them, giving the position the keypoint
almost certainly had. Running the risk field over the filled data yields the
field a perfect observer would have reported.

This is a reference, not ground truth. It inherits the SDK's own 0.5 s
prediction, which could not be disabled on this rig. State that in the paper.
"""
import numpy as np
from .types import N_KP


def fill_acausal(times, positions, valid, max_gap=None):
    """positions (F,38,3), valid (F,38) -> filled positions, filled mask."""
    t = np.asarray(times, dtype=float)
    P = np.array(positions, dtype=float, copy=True)
    V = np.asarray(valid, dtype=bool)
    filled = V.copy()

    for i in range(N_KP):
        idx = np.where(V[:, i])[0]
        if idx.size < 2:
            continue
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a <= 1:
                continue
            gap = t[b] - t[a]
            if max_gap is not None and gap > max_gap:
                continue
            for k in range(a + 1, b):
                w = (t[k] - t[a]) / gap if gap > 0 else 0.0
                P[k, i] = (1.0 - w) * P[a, i] + w * P[b, i]
                filled[k, i] = True
    return P, filled


def reference_field(times, positions, valid, risk_field, tracker_factory,
                    queries, max_gap=None):
    """Run the estimator over acausally filled data. Returns (F,N) array."""
    P, F = fill_acausal(times, positions, valid, max_gap=max_gap)
    tracker = tracker_factory()
    out = np.zeros((len(times), len(queries)))

    class _B:
        pass

    for k, t in enumerate(times):
        b = _B()
        b.keypoints = P[k]
        b.valid = F[k]
        b.id = -1
        st = tracker.update(t, b if F[k].any() else None)
        out[k] = risk_field.evaluate(st, queries)
    return out


# ---------------------------------------------------------------------------
# Paper v2 additions: second reference scheme and binary reference occupancy.
# ---------------------------------------------------------------------------

def fill_hold(times, positions, valid, max_gap=None):
    """Hold-forward reference: a missing keypoint is assumed not to have
    moved, held at its pre-gap position until the next observation.

    Fills the same interior gaps as fill_acausal (same bounding rule, same
    max_gap), so the two schemes disagree only in WHERE they place the
    keypoint, never in WHICH query-frames carry a claim. Linear
    interpolation and hold-forward bracket a slowly moving supine subject
    from opposite sides; claims are restricted to where they agree.
    """
    t = np.asarray(times, dtype=float)
    P = np.array(positions, dtype=float, copy=True)
    V = np.asarray(valid, dtype=bool)
    filled = V.copy()

    for i in range(N_KP):
        idx = np.where(V[:, i])[0]
        if idx.size < 2:
            continue
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a <= 1:
                continue
            gap = t[b] - t[a]
            if max_gap is not None and gap > max_gap:
                continue
            for k in range(a + 1, b):
                P[k, i] = P[a, i]
                filled[k, i] = True
    return P, filled


def reference_occupancy(times, positions, valid, d_contact, queries,
                        scheme="linear", max_gap=None):
    """Binary reference field: where the filled skeleton places the patient.

    Returns (F, N) uint8. A frame with nothing filled or observed makes no
    claim (all zeros), so it can never charge the estimator with a
    free-space error there.
    """
    from .occupancy_field import occupancy_of_positions
    if scheme == "linear":
        P, F = fill_acausal(times, positions, valid, max_gap=max_gap)
    elif scheme == "hold":
        P, F = fill_hold(times, positions, valid, max_gap=max_gap)
    else:
        raise ValueError("scheme must be 'linear' or 'hold'")
    out = np.zeros((len(times), len(np.atleast_2d(queries))), dtype=np.uint8)
    for k in range(len(times)):
        out[k] = occupancy_of_positions(P[k], F[k], d_contact)(queries)
    return out
