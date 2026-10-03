"""Synthetic dropout at a controlled rate and duration.

The natural occlusion in the bags is uncontrolled, so it cannot produce a
dose-response curve. Injection can: drop keypoints at a known rate for a known
duration, measure field error against the reference, and plot error against
rate. That curve is the paper's central contribution.
"""
import numpy as np


def inject(times, valid, drop_fraction, drop_duration, rng=None, keypoints=None):
    """Return a copy of `valid` with dropouts applied.

    drop_fraction  expected fraction of keypoint-frames suppressed
    drop_duration  seconds each dropout lasts once triggered
    keypoints      restrict dropouts to these indices (None = all)
    """
    rng = np.random.default_rng() if rng is None else rng
    t = np.asarray(times, dtype=float)
    V = np.array(valid, dtype=bool, copy=True)
    F, K = V.shape
    cols = range(K) if keypoints is None else keypoints

    dt = float(np.median(np.diff(t))) if F > 1 else 1 / 30.0
    frames_per_drop = max(1, int(round(drop_duration / dt)))
    # trigger rate chosen so expected suppressed fraction matches the target
    p_trigger = np.clip(drop_fraction / frames_per_drop, 0.0, 1.0)

    for i in cols:
        k = 0
        while k < F:
            if V[k, i] and rng.random() < p_trigger:
                V[k:k + frames_per_drop, i] = False
                k += frames_per_drop
            else:
                k += 1
    return V


def realised_rate(valid_orig, valid_dropped):
    """Fraction of originally valid keypoint-frames actually suppressed."""
    o = np.asarray(valid_orig, dtype=bool)
    d = np.asarray(valid_dropped, dtype=bool)
    denom = o.sum()
    return float((o & ~d).sum() / denom) if denom else 0.0


# ---------------------------------------------------------------------------
# Paper v2 additions: structured (chain) injection and natural gap statistics.
# inject() above is unchanged and remains the independent per-keypoint mode.
# ---------------------------------------------------------------------------

def natural_gap_stats(times, valid):
    """Durations of naturally occurring interior gaps, pooled over keypoints.

    A gap is a run of missing frames between two observations of the same
    keypoint. Leading and trailing absence is not a gap: it has no second
    observation to bound it. The median is the data-derived burst length for
    injection, so the injected loss has the temporal structure of real loss
    instead of per-frame flicker.
    """
    t = np.asarray(times, dtype=float)
    V = np.asarray(valid, dtype=bool)
    durations = []
    for i in range(V.shape[1]):
        idx = np.where(V[:, i])[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > 1:
                durations.append(t[b] - t[a])
    d = np.asarray(durations, dtype=float)
    if not d.size:
        return {"n_gaps": 0, "median_s": float("nan"),
                "p25_s": float("nan"), "p75_s": float("nan"),
                "max_s": float("nan")}
    return {"n_gaps": int(d.size),
            "median_s": float(np.median(d)),
            "p25_s": float(np.percentile(d, 25)),
            "p75_s": float(np.percentile(d, 75)),
            "max_s": float(d.max())}


def inject_structured(times, valid, drop_fraction, drop_duration, chains,
                      rng=None):
    """Chain-structured dropout: whole limb chains vanish together, in
    contiguous bursts.

    drop_fraction  expected fraction of chain keypoint-frames suppressed
    drop_duration  seconds each burst lasts once triggered
    chains         {name: [keypoint indices]} removed as units
    """
    rng = np.random.default_rng() if rng is None else rng
    t = np.asarray(times, dtype=float)
    V = np.array(valid, dtype=bool, copy=True)
    F = V.shape[0]

    dt = float(np.median(np.diff(t))) if F > 1 else 1 / 30.0
    frames_per_drop = max(1, int(round(drop_duration / dt)))
    # A burst blocks re-triggering for its own duration, so the duty cycle
    # of trigger probability p is pT/(1+pT). Solving for duty = rho gives
    # pT = rho/(1-rho); the naive p = rho/T undershoots by that factor.
    rho = float(np.clip(drop_fraction, 0.0, 0.99))
    # Renewal duty cycle: a trigger starts a T-frame burst; between bursts the
    # number of non-triggering frames is geometric with mean (1-p)/p. The
    # suppressed fraction T*p / (T*p + 1 - p) equals rho when
    # p = rho / (T*(1-rho) + rho). The earlier rho/((1-rho)*T) omitted the
    # +rho term, overshot from rho ~0.5 and saturated at 1.0 by rho = 0.8.
    p_trigger = np.clip(rho / ((1.0 - rho) * frames_per_drop + rho),
                        0.0, 1.0)

    for cols in chains.values():
        cols = np.asarray(cols, dtype=int)
        k = 0
        while k < F:
            if V[k, cols].any() and rng.random() < p_trigger:
                V[k:k + frames_per_drop][:, cols] = False
                k += frames_per_drop
            else:
                k += 1
    return V


def inject_independent(times, valid, drop_fraction, drop_duration, rng=None,
                       keypoints=None):
    """Per-keypoint burst dropout with an exact duty cycle.

    Same semantics as inject(), but the trigger probability is corrected for
    burst blocking so the expected suppressed fraction equals drop_fraction
    rather than drop_fraction/(1+drop_fraction). inject() is left unchanged
    for continuity with earlier thesis-track runs.
    """
    rng = np.random.default_rng() if rng is None else rng
    t = np.asarray(times, dtype=float)
    V = np.array(valid, dtype=bool, copy=True)
    F, K = V.shape
    cols = range(K) if keypoints is None else keypoints

    dt = float(np.median(np.diff(t))) if F > 1 else 1 / 30.0
    frames_per_drop = max(1, int(round(drop_duration / dt)))
    rho = float(np.clip(drop_fraction, 0.0, 0.99))
    # Renewal duty cycle: a trigger starts a T-frame burst; between bursts the
    # number of non-triggering frames is geometric with mean (1-p)/p. The
    # suppressed fraction T*p / (T*p + 1 - p) equals rho when
    # p = rho / (T*(1-rho) + rho). The earlier rho/((1-rho)*T) omitted the
    # +rho term, overshot from rho ~0.5 and saturated at 1.0 by rho = 0.8.
    p_trigger = np.clip(rho / ((1.0 - rho) * frames_per_drop + rho),
                        0.0, 1.0)

    for i in cols:
        k = 0
        while k < F:
            if V[k, i] and rng.random() < p_trigger:
                V[k:k + frames_per_drop, i] = False
                k += frames_per_drop
            else:
                k += 1
    return V
