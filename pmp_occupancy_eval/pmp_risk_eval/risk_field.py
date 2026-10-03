"""Risk field estimator.

    R(p) = max_i  sigma_i * exp( -max(0, ||p - x_i|| - r_i) / lambda )
    r_i  = d_contact_i + v_hat_max * a_i          (a_i clipped at a_max)

Monotonicity: with no new observation every a_i is non-decreasing, so every
r_i is non-decreasing, so the exponent argument is non-increasing, so R cannot
fall. Past a_max the keypoint freezes at its a_max radius and its sigma is
raised to SIGMA_UNKNOWN, because the body part can no longer be identified and
the worst case must be assumed.

Empty set returns R_MAX, not None. Seeing nothing is the most dangerous state,
not the safest.

Modes
  baseline       fixed r0, expiry deletes the keypoint   (the defect)
  corrected      growing radius, hard boundary           (monotonicity fix)
  probabilistic  growing radius, soft boundary decay     (the proposal)
"""
import numpy as np

SIGMA_UNKNOWN = 1.0
R_MAX = 1.0
MODES = ("baseline", "corrected", "probabilistic")


class RiskField:
    def __init__(self, sigma, v_hat_max, a_max, lam=0.05,
                 mode="probabilistic", baseline_expiry=0.5, r0=0.20):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.sigma = np.asarray(sigma, dtype=float)
        self.v_hat_max = float(v_hat_max)
        self.a_max = float(a_max)
        self.lam = float(lam)
        self.mode = mode
        self.baseline_expiry = float(baseline_expiry)
        self.r0 = float(r0)

    def radii(self, state):
        if self.mode == "baseline":
            return np.full(len(state.ages), self.r0)
        a = np.minimum(state.ages, self.a_max)
        a = np.where(np.isfinite(a), a, self.a_max)
        return state.d_contact + self.v_hat_max * a

    def active_mask(self, state):
        if self.mode == "baseline":
            # expiry deletes the keypoint entirely: this is the defect
            return state.ever_seen & (state.ages <= self.baseline_expiry)
        return state.ever_seen

    def effective_sigma(self, state):
        s = self.sigma.copy()
        s[state.ages > self.a_max] = SIGMA_UNKNOWN
        return s

    def evaluate(self, state, queries):
        """queries: (N,3) -> (N,) risk in [0,1]."""
        q = np.atleast_2d(np.asarray(queries, dtype=float))
        active = self.active_mask(state) & np.isfinite(state.positions).all(axis=1)
        if not active.any():
            return np.full(len(q), R_MAX)

        x = state.positions[active]                 # (M,3)
        r = self.radii(state)[active]               # (M,)
        s = self.effective_sigma(state)[active]     # (M,)

        d = np.linalg.norm(q[:, None, :] - x[None, :, :], axis=2)   # (N,M)
        excess = np.maximum(0.0, d - r[None, :])

        if self.mode == "probabilistic":
            contrib = s[None, :] * np.exp(-excess / self.lam)
        else:
            contrib = np.where(excess <= 0.0, s[None, :], 0.0)

        return contrib.max(axis=1)

    def argmax_keypoint(self, state, queries):
        """Index of the keypoint setting the score, and its age. -1 if empty."""
        q = np.atleast_2d(np.asarray(queries, dtype=float))
        active = self.active_mask(state) & np.isfinite(state.positions).all(axis=1)
        if not active.any():
            return np.full(len(q), -1), np.full(len(q), np.inf)
        idx = np.where(active)[0]
        x = state.positions[active]
        r = self.radii(state)[active]
        s = self.effective_sigma(state)[active]
        d = np.linalg.norm(q[:, None, :] - x[None, :, :], axis=2)
        excess = np.maximum(0.0, d - r[None, :])
        if self.mode == "probabilistic":
            contrib = s[None, :] * np.exp(-excess / self.lam)
        else:
            contrib = np.where(excess <= 0.0, s[None, :], 0.0)
        j = contrib.argmax(axis=1)
        return idx[j], state.ages[idx[j]]
