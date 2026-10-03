"""Binary occupancy estimator: the paper's Equation (4), both arms.

This module is the ACRA paper's model, kept separate from risk_field.py,
which remains the thesis-track continuous field. The two differ on purpose:

  * Occupancy here is binary. Region sensitivity is deferred to the
    controller and does not enter the estimate.
  * The proposed arm's radius is UNCAPPED. risk_field.py clips age at a_max;
    the paper (Remark 2) proves any finite horizon in the occupancy test
    reproduces the hold-last-position defect, so no horizon appears here.
  * Coverage is a union of capsules over skeleton edges, not balls alone.
    A ball union leaves the mid-segment of a limb, which lies farther than
    d_contact from either joint, in reported free space. The capsule over
    edge (i, j) is the segment [x_i, x_j] swollen by max(r_i, r_j); taking
    the larger radius keeps the error in the safe direction. Keypoints whose
    every neighbour is unseen still contribute their ball.

Arms
  hold      fixed radius r0, keypoint removed once age > t_hold, empty
            covering set reported FREE. This reproduces hold-last-position,
            including the empty-set behaviour of the original
            implementation (Proposition 1).
  proposed  r_i = d_contact_i + k * a_i, no cap, no expiry, empty E
            reported OCCUPIED (seeing nothing is the most dangerous state).

Both arms share the capsule geometry, so the comparison isolates the
temporal policy rather than mixing it with a geometry change.

Monotonicity of the proposed arm (Proposition 2): between observations
every a_i is non-decreasing, so every r_i is non-decreasing; endpoint
positions do not change; E and the edge set only grow. The union of balls
and capsules is therefore non-shrinking, and U cannot fall.
"""
import numpy as np

from .types import N_KP, EDGES

ARMS = ("hold", "proposed")


def _point_segment_distance(q, a, b):
    """Distances from query points to segments.

    q (N,3); a, b (M,3) segment endpoints. Returns (N,M). Degenerate
    segments (a == b) fall back to point distance.
    """
    ab = b - a                                          # (M,3)
    l2 = np.einsum("md,md->m", ab, ab)                  # (M,)
    l2_safe = np.where(l2 > 0.0, l2, 1.0)
    ap = q[:, None, :] - a[None, :, :]                  # (N,M,3)
    t = np.einsum("nmd,md->nm", ap, ab) / l2_safe[None, :]
    t = np.clip(np.where(l2[None, :] > 0.0, t, 0.0), 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.linalg.norm(q[:, None, :] - proj, axis=2)


class OccupancyEstimator:
    def __init__(self, d_contact, k=0.8, arm="proposed",
                 r0=0.20, t_hold=0.5, edges=EDGES):
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        d = np.asarray(d_contact, dtype=float)
        if d.shape != (N_KP,):
            raise ValueError(f"d_contact must be ({N_KP},), got {d.shape}")
        self.d_contact = d
        self.k = float(k)
        self.arm = arm
        # r0 = None or "d_contact": hold uses the same per-region base radius
        # as the proposed arm, so the arms differ only in temporal policy.
        self.r0 = None if r0 in (None, "d_contact") else float(r0)
        self.t_hold = float(t_hold)
        self.edges = np.asarray(edges, dtype=int)

    # -- radius law and membership --------------------------------------

    def radii(self, state):
        if self.arm == "hold":
            if self.r0 is None:
                return self.d_contact.copy()
            return np.full(N_KP, self.r0)
        a = np.where(np.isfinite(state.ages), state.ages, 0.0)
        return self.d_contact + self.k * a          # UNCAPPED

    def active_mask(self, state):
        finite = np.isfinite(state.positions).all(axis=1)
        if self.arm == "hold":
            return state.ever_seen & finite & (state.ages <= self.t_hold)
        return state.ever_seen & finite

    # -- Equation (4) ----------------------------------------------------

    def evaluate(self, state, queries):
        """queries (N,3) -> U (N,) uint8 in {0, 1}."""
        q = np.atleast_2d(np.asarray(queries, dtype=float))
        active = self.active_mask(state)

        if not active.any():
            # proposed: E empty (or nothing usable) is the most dangerous
            # state. hold: the original implementation reported free.
            fill = 0 if self.arm == "hold" else 1
            return np.full(len(q), fill, dtype=np.uint8)

        return self._cover(state, q, active)

    def covered_by_subset(self, state, queries, keep):
        """Coverage from only those elements whose keypoints are all in
        `keep`: balls of kept keypoints, capsules with both endpoints kept.
        No empty-set fill: if nothing is kept, nothing is covered."""
        q = np.atleast_2d(np.asarray(queries, dtype=float))
        active = self.active_mask(state) & np.asarray(keep, dtype=bool)
        if not active.any():
            return np.zeros(len(q), dtype=np.uint8)
        return self._cover(state, q, active)

    def _cover(self, state, q, active):
        r = self.radii(state)
        x = state.positions

        # balls
        d_ball = np.linalg.norm(
            q[:, None, :] - x[active][None, :, :], axis=2)
        covered = (d_ball <= r[active][None, :]).any(axis=1)

        # capsules over edges with both endpoints active
        e = self.edges
        both = active[e[:, 0]] & active[e[:, 1]]
        if both.any():
            i, j = e[both, 0], e[both, 1]
            d_seg = _point_segment_distance(q, x[i], x[j])
            r_seg = np.maximum(r[i], r[j])
            covered |= (d_seg <= r_seg[None, :]).any(axis=1)

        return covered.astype(np.uint8)


def occupancy_of_positions(positions, valid, d_contact, edges=EDGES):
    """One-frame occupancy of a body at zero observation age.

    Used for the reference: 'the reference places the patient at p' means p
    lies inside the d_contact envelope of the (acausally filled) skeleton.
    Returns a closure frame_occupancy(queries) built per call site instead?
    No: keep it simple -- callers pass one frame at a time.

    positions (38,3), valid (38,) bool -> function of queries (N,3).
    """
    d = np.asarray(d_contact, dtype=float)
    x = np.asarray(positions, dtype=float)
    v = np.asarray(valid, dtype=bool) & np.isfinite(x).all(axis=1)

    def _eval(queries):
        q = np.atleast_2d(np.asarray(queries, dtype=float))
        if not v.any():
            return np.zeros(len(q), dtype=np.uint8)   # no claim, not danger
        d_ball = np.linalg.norm(q[:, None, :] - x[v][None, :, :], axis=2)
        covered = (d_ball <= d[v][None, :]).any(axis=1)
        e = np.asarray(edges, dtype=int)
        both = v[e[:, 0]] & v[e[:, 1]]
        if both.any():
            i, j = e[both, 0], e[both, 1]
            d_seg = _point_segment_distance(q, x[i], x[j])
            r_seg = np.maximum(d[i], d[j])
            covered |= (d_seg <= r_seg[None, :]).any(axis=1)
        return covered.astype(np.uint8)

    return _eval
