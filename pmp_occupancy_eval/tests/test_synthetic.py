#!/usr/bin/env python3
"""Verifies the estimator on synthetic data. No bag, no ROS, no camera.

Scenario: one keypoint visible, hidden for 0.7 s, then reacquired. Baseline
expiry deletes it at 0.5 s and the reported risk collapses although nothing
moved. Corrected and probabilistic hold it and the risk does not fall.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmp_risk_eval.types import Body, Frame, N_KP
from pmp_risk_eval.keypoint_state import KeypointTracker
from pmp_risk_eval.risk_field import RiskField
from pmp_risk_eval.metrics import monotonicity_violations

KP = 22                                  # LEFT_ANKLE, the one that vanishes
KP_TORSO = 0                             # PELVIS, stays visible throughout
SIGMA = np.full(N_KP, 0.3); SIGMA[KP] = 1.0; SIGMA[KP_TORSO] = 0.3
D_CONTACT = np.full(N_KP, 0.08)

QUERY = np.array([[0.15, 0.0, 0.0]])     # 0.15 m from the ankle
TORSO_AT = [0.30, 0.0, 0.0]              # also 0.15 m from the query


def make_frames(dt=1 / 30., hide_from=0.2, hide_to=0.9, total=1.2):
    """Ankle disappears for 0.7 s. Pelvis is visible in every frame.

    The pelvis matters: without it the ankle's expiry empties the active set
    and the R_max fallback fires, hiding the defect. With it, the baseline has
    something left to report and its maximum quietly drops instead.
    """
    frames, t = [], 0.0
    while t < total:
        valid = np.zeros(N_KP, dtype=bool)
        kp = np.full((N_KP, 3), np.nan)
        valid[KP_TORSO] = True
        kp[KP_TORSO] = TORSO_AT
        if not (hide_from <= t < hide_to):
            valid[KP] = True
            kp[KP] = [0.0, 0.0, 0.0]
        frames.append(Frame(t, [Body(0, kp, valid, np.ones(N_KP))]))
        t += dt
    return frames


def run(mode, v_hat_max=0.5, a_max=1.0):
    tracker = KeypointTracker(D_CONTACT)
    rf = RiskField(SIGMA, v_hat_max, a_max, lam=0.05, mode=mode,
                   baseline_expiry=0.5)
    R, seen = [], []
    for f in make_frames():
        body = f.bodies[0] if f.bodies[0].valid.any() else None
        st = tracker.update(f.t, body)
        R.append(rf.evaluate(st, QUERY)[0])
        seen.append(st.seen_now.copy())
    return np.array(R), np.array(seen)


print(f"{'mode':<16}{'R min':>9}{'R max':>9}{'viol/1000 pairs':>18}")
print("-" * 52)
results = {}
for mode in ("baseline", "corrected", "probabilistic"):
    R, seen = run(mode)
    m = monotonicity_violations(R, seen)
    results[mode] = m
    print(f"{mode:<16}{R.min():>9.3f}{R.max():>9.3f}"
          f"{m['violations_per_1000_pairs']:>18.1f}")

print()
assert results["baseline"]["violation_frames"] > 0, \
    "baseline should violate monotonicity"
assert results["corrected"]["violation_frames"] == 0, \
    "corrected must not violate monotonicity"
assert results["probabilistic"]["violation_frames"] == 0, \
    "probabilistic must not violate monotonicity"
print("PASS: baseline violates, corrected and probabilistic do not.")

tracker = KeypointTracker(D_CONTACT)
rf = RiskField(SIGMA, 0.5, 1.0, mode="probabilistic")
st = tracker.update(0.0, None)
r = rf.evaluate(st, np.array([[0.0, 0.0, 0.0]]))
assert r[0] == 1.0, f"empty set must return R_max, got {r[0]}"
print("PASS: empty set returns R_max, not None.")

R, seen = run("corrected")
no_new = np.array([not (seen[k] & ~seen[k - 1]).any() for k in range(1, len(R))])
diffs = np.diff(R)[no_new]
assert (diffs >= -1e-12).all(), "corrected fell with no new evidence"
print("PASS: corrected is non-decreasing with no new evidence.")
print("\nALL TESTS PASSED")
