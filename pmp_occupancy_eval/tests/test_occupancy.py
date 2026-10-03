#!/usr/bin/env python3
"""Paper v2 checks. No bag, no ROS, no camera.

1. Capsules cover the mid-segment of a limb that a ball union misses.
2. The proposed arm is monotone through a dropout; the hold arm frees a
   covered point at t_hold (Propositions 1 and 2).
3. Empty-state behaviour: proposed reports occupied, hold reports free.
4. Structured injection removes chains as units, in contiguous bursts, at
   roughly the requested rate.
5. The two reference schemes fill identical query-frames and disagree only
   on position.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmp_risk_eval.types import Body, N_KP, CHAINS
from pmp_risk_eval.keypoint_state import KeypointTracker
from pmp_risk_eval.occupancy_field import OccupancyEstimator
from pmp_risk_eval.occlusion_injector import (
    inject_structured, natural_gap_stats, realised_rate)
from pmp_risk_eval.reference_builder import fill_acausal, fill_hold
from pmp_risk_eval.metrics import monotonicity_violations

D = np.full(N_KP, 0.08)
HIP_R, KNEE_R = 19, 21          # RIGHT_HIP, RIGHT_KNEE: a skeleton edge


def _state(tracker, t, kp_idx, kp_pos):
    valid = np.zeros(N_KP, dtype=bool)
    kp = np.full((N_KP, 3), np.nan)
    for i, p in zip(kp_idx, kp_pos):
        valid[i] = True
        kp[i] = p
    body = Body(0, kp, valid, np.full(N_KP, 90.0)) if valid.any() else None
    return tracker.update(t, body)


def test_capsule_covers_midsegment():
    tr = KeypointTracker(D)
    st = _state(tr, 0.0, [HIP_R, KNEE_R], [[0, 0, 0], [0.4, 0, 0]])
    est = OccupancyEstimator(D, k=0.8, arm="proposed")
    mid = np.array([[0.2, 0.05, 0.0]])       # 0.2 m from either joint
    assert est.evaluate(st, mid)[0] == 1, "capsule must cover mid-thigh"
    # sanity: a ball-only test at this point would fail (0.2 > d_contact)
    d_to_joints = min(0.2, np.hypot(0.2, 0.05))
    assert d_to_joints > D[HIP_R]
    print("capsule mid-segment: PASS")


def test_monotone_vs_hold():
    q = np.array([[0.15, 0.0, 0.0]])        # near the knee only
    dt = 1 / 30.0
    seq = []
    t = 0.0
    while t < 1.4:
        if t < 0.2:
            seq.append((t, [HIP_R, KNEE_R], [[-0.5, 0, 0], [0.0, 0, 0]]))
        else:
            seq.append((t, [HIP_R], [[-0.5, 0, 0]]))   # knee lost for good
        t += dt

    for arm, expect_viol in (("proposed", 0), ("hold", 1)):
        tr = KeypointTracker(D)
        est = OccupancyEstimator(D, k=0.8, arm=arm, r0=0.20, t_hold=0.5)
        U, seen = [], []
        for t, idx, pos in seq:
            st = _state(tr, t, idx, pos)
            U.append(est.evaluate(st, q)[0])
            seen.append(st.seen_now.copy())
        U = np.array(U, dtype=float)
        m = monotonicity_violations(U, np.array(seen))
        if arm == "proposed":
            assert m["violations"] == 0, "proposed arm must be monotone"
            assert U[-1] == 1, "once covered, never freed"
        else:
            assert m["violations"] >= expect_viol, "hold arm must violate"
            assert U[0] == 1 and U[-1] == 0, "hold frees the covered point"
        print(f"monotonicity {arm}: PASS ({m['violations']} violations)")


def test_empty_state():
    q = np.array([[0.0, 0.0, 0.0]])
    for arm, expect in (("proposed", 1), ("hold", 0)):
        tr = KeypointTracker(D)
        st = tr.update(0.0, None)
        got = OccupancyEstimator(D, arm=arm).evaluate(st, q)[0]
        assert got == expect, f"{arm} empty-state must report {expect}"
    print("empty state: PASS")


def test_structured_injection():
    rng = np.random.default_rng(0)
    F = 3000
    times = np.arange(F) / 30.0
    valid = np.ones((F, N_KP), dtype=bool)
    V = inject_structured(times, valid, 0.3, 0.4, CHAINS, rng=rng)
    rate = realised_rate(valid, V)
    assert 0.03 < rate < 0.6, f"realised rate {rate:.3f} implausible"
    # chain unity: within any frame, a chain is all-present or all-absent
    for cols in CHAINS.values():
        block = V[:, cols]
        assert ((block.all(axis=1)) | (~block.any(axis=1))).all(), \
            "chain must vanish as a unit"
    # bursts: suppressed runs of the right length (12 frames at 30 Hz)
    leg = V[:, CHAINS["right_leg"][0]]
    runs = np.diff(np.flatnonzero(np.diff(
        np.concatenate([[1], leg.view(np.int8), [1]]))))[::2]
    if runs.size:
        assert np.median(runs) >= 10, "bursts must be contiguous, not flicker"
    print(f"structured injection: PASS (realised rate {rate:.3f})")


def test_reference_schemes_agree_on_mask():
    F = 60
    times = np.arange(F) / 30.0
    pos = np.zeros((F, N_KP, 3))
    pos[:, KNEE_R, 0] = np.linspace(0.0, 0.6, F)   # knee moves
    valid = np.zeros((F, N_KP), dtype=bool)
    valid[:, KNEE_R] = True
    valid[20:40, KNEE_R] = False                    # interior gap
    P_lin, F_lin = fill_acausal(times, pos, valid)
    P_hold, F_hold = fill_hold(times, pos, valid)
    assert (F_lin == F_hold).all(), "schemes must fill identical frames"
    assert not np.allclose(P_lin[25, KNEE_R], P_hold[25, KNEE_R]), \
        "schemes must disagree on position inside the gap"
    assert np.allclose(P_hold[25, KNEE_R], pos[19, KNEE_R]), \
        "hold-forward must hold the pre-gap position"
    print("reference schemes: PASS")


def test_natural_gaps():
    F = 300
    times = np.arange(F) / 30.0
    valid = np.ones((F, N_KP), dtype=bool)
    valid[100:112, 5] = False                       # one 12-frame gap
    g = natural_gap_stats(times, valid)
    assert g["n_gaps"] == 1
    # duration spans the bounding observations: 13 frames at 30 Hz
    assert abs(g["median_s"] - 13 / 30.0) < 1e-6
    print("natural gaps: PASS")


def test_per_point_monotonicity():
    """The paper's per-point definition, as computed in the runner.

    Jitter: a keypoint seen every frame moves back and forth. Points at the
    edge of its ball fall 1 -> 0, but the keypoint covering them was
    re-observed, so these are legitimate updates: falls > 0, violations = 0,
    for BOTH arms. (The frame-level v1 check counts these as violations.)

    Expiry: a keypoint lost for good. Hold frees its neighbourhood after
    t_hold with no new observation of it: violations > 0. Proposed never
    frees it.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    from run_paper_analysis import run_arm

    F = 60
    times = np.arange(F) / 30.0
    q = np.stack(np.meshgrid(np.linspace(-0.3, 0.3, 13),
                             np.linspace(-0.3, 0.3, 13), [0.0]),
                 -1).reshape(-1, 3)

    # jitter: one keypoint, always observed, alternating +-3 cm
    pos = np.full((F, N_KP, 3), np.nan)
    val = np.zeros((F, N_KP), dtype=bool)
    pos[:, KNEE_R] = [[0.03 * (-1) ** f, 0, 0] for f in range(F)]
    val[:, KNEE_R] = True
    for arm in ("hold", "proposed"):
        _, m = run_arm(times, pos, val, D, q, arm, 0.8, "d_contact", 0.5)
        assert m["falls"] > 0, "jitter must move the ball edge"
        assert m["violations"] == 0, f"{arm}: re-observation is evidence"
    old = monotonicity_violations(
        run_arm(times, pos, val, D, q, "proposed", 0.8, "d_contact", 0.5
                )[0].astype(float), val)
    assert old["violations"] > 0, "v1 frame-level check miscounts jitter"

    # expiry: hip always seen, knee seen for 0.2 s then lost for good
    pos = np.full((F, N_KP, 3), np.nan)
    val = np.zeros((F, N_KP), dtype=bool)
    pos[:, HIP_R] = [-0.25, 0, 0]
    val[:, HIP_R] = True
    pos[:6, KNEE_R] = [0.2, 0, 0]
    val[:6, KNEE_R] = True
    pos[6:, KNEE_R] = pos[5, KNEE_R]      # tracker holds; value unused
    _, mh = run_arm(times, pos, val, D, q, "hold", 0.8, "d_contact", 0.5)
    _, mp = run_arm(times, pos, val, D, q, "proposed", 0.8, "d_contact", 0.5)
    assert mh["violations"] > 0, "hold must free the lost knee"
    assert mp["violations"] == 0, "proposed must never free it"
    print(f"per-point monotonicity: PASS (jitter falls "
          f"{m['falls']} -> 0 violations; v1 check counted "
          f"{old['violations']}; hold expiry {mh['violations']} violations, "
          f"proposed {mp['violations']})")


def test_radius_parity():
    """r0 = 'd_contact' gives the hold arm the proposed arm's base radius."""
    tr = KeypointTracker(D)
    st = _state(tr, 0.0, [KNEE_R], [[0.0, 0, 0]])
    h = OccupancyEstimator(D, arm="hold", r0="d_contact").radii(st)
    p = OccupancyEstimator(D, arm="proposed", k=0.8).radii(st)
    assert np.allclose(h, D) and np.allclose(p[KNEE_R], D[KNEE_R])
    print("radius parity: PASS")


def test_injection_duty_cycle():
    """Realised rate tracks target across the full grid, short bursts
    included. The earlier trigger formula saturated at 1.0 by rho = 0.8."""
    from pmp_risk_eval.occlusion_injector import inject_independent
    F = 6000
    times = np.arange(F) / 15.0                     # stride-2 spacing
    valid = np.ones((F, N_KP), dtype=bool)
    rng = np.random.default_rng(1)
    for rho in (0.3, 0.6, 0.8, 0.9):
        V = inject_independent(times, valid, rho, 0.2, rng=rng)   # T = 3
        got = realised_rate(valid, V)
        assert abs(got - rho) < 0.02, f"rho {rho}: realised {got:.3f}"
    print("injection duty cycle: PASS (0.3-0.9 within 0.02)")


if __name__ == "__main__":
    test_capsule_covers_midsegment()
    test_monotone_vs_hold()
    test_empty_state()
    test_structured_injection()
    test_reference_schemes_agree_on_mask()
    test_natural_gaps()
    test_per_point_monotonicity()
    test_radius_parity()
    test_injection_duty_cycle()
    print("\nall paper v2 checks pass")
