"""Metrics comparing the causal field to the acausal reference.

Three say whether the estimator is safe:
    monotonicity_violations, underestimation, roc_auc
Three say what that safety costs:
    held_fraction, argmax_staleness, volume_above
Report both. That is what makes it characterisation rather than advocacy.
"""
import numpy as np


def monotonicity_violations(R, seen, tol=1e-9):
    """R (F,) or (F,N); seen (F,K) bool mask, or (F,) count as a fallback.

    A violation is a fall in R between consecutive frames in which no new
    evidence arrived. "No new evidence" means the set of keypoints observed at
    frame k is a subset of those observed at k-1: nothing was learned that was
    not already known.

    The count fallback is coarser -- it treats any observation as new evidence
    -- and will under-report violations whenever some keypoints stay visible
    while others are lost. That is precisely the baseline's failure mode, so
    pass the mask wherever it is available.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim == 1:
        R = R[:, None]
    s = np.asarray(seen)
    mask_mode = s.ndim == 2
    pairs = 0
    viol = 0
    viol_frames = 0
    drops = []
    for k in range(1, len(R)):
        if mask_mode:
            # new evidence iff some keypoint is seen now that was not before
            if bool((s[k] & ~s[k - 1]).any()):
                continue
        elif s[k] > 0:
            continue
        pairs += R.shape[1]
        d = R[k - 1] - R[k]
        bad = d > tol
        viol += int(bad.sum())
        if bad.any():
            viol_frames += 1
            drops.extend(d[bad].tolist())
    return {
        "no_evidence_pairs": pairs,
        "violations": viol,
        "violation_frames": viol_frames,
        "violations_per_1000_pairs": 1000.0 * viol / pairs if pairs else 0.0,
        "mean_drop": float(np.mean(drops)) if drops else 0.0,
        "max_drop": float(np.max(drops)) if drops else 0.0,
    }


def underestimation(R, R_ref, tol=1e-9):
    """Fraction of query-frames where the estimator sits below the reference."""
    a = np.asarray(R, dtype=float)
    b = np.asarray(R_ref, dtype=float)
    under = (b - a) > tol
    return {
        "underestimation_rate": float(under.mean()),
        "mean_deficit": float((b - a)[under].mean()) if under.any() else 0.0,
        "max_deficit": float((b - a).max()),
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "mae": float(np.mean(np.abs(a - b))),
    }


def roc_auc(R, R_ref, ref_threshold=0.5):
    """AUC for detecting reference-unsafe states from the reported field.

    Trapezoidal integration over the sorted score, no sklearn dependency.
    """
    y = (np.asarray(R_ref).ravel() >= ref_threshold).astype(int)
    s = np.asarray(R).ravel()
    P, N = int(y.sum()), int((1 - y).sum())
    if P == 0 or N == 0:
        return {"auc": float("nan"), "n_pos": P, "n_neg": N}
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tpr = np.cumsum(y) / P
    fpr = np.cumsum(1 - y) / N
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return {"auc": float(np.trapezoid(tpr, fpr)), "n_pos": P, "n_neg": N}


def held_fraction(argmax_ages, tol=1e-9):
    """Frames where the score is set by a keypoint not observed that frame.

    This measures instability, not coverage loss. Report it separately from
    keypoint yield; they are independent.
    """
    a = np.asarray(argmax_ages, dtype=float)
    finite = np.isfinite(a)
    if not finite.any():
        return {"held_fraction": float("nan"), "n": 0}
    held = a[finite] > tol
    return {"held_fraction": float(held.mean()), "n": int(finite.sum())}


def argmax_staleness(argmax_ages):
    a = np.asarray(argmax_ages, dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"mean_staleness": float("nan"), "p95_staleness": float("nan")}
    return {"mean_staleness": float(a.mean()),
            "p95_staleness": float(np.percentile(a, 95)),
            "max_staleness": float(a.max())}


def volume_above(R, threshold=0.5):
    """Fraction of the query volume at or above threshold.

    The usability number. An estimator that marks most of the bed high is safe
    and useless.
    """
    return {"fraction_above": float((np.asarray(R) >= threshold).mean()),
            "threshold": threshold}


def keypoint_yield(valid, index_sets):
    """{region: mean fraction of that region's keypoints present per frame}."""
    V = np.asarray(valid, dtype=bool)
    out = {}
    for name, idx in index_sets.items():
        if not len(idx):
            continue
        out[name] = float(V[:, idx].mean())
    out["all"] = float(V.mean())
    return out


# ---------------------------------------------------------------------------
# Paper v2 additions: binary occupancy metrics. The continuous-field metrics
# above are unchanged and remain in use for the thesis-track evaluation.
# ---------------------------------------------------------------------------

def free_space_error(U, U_ref):
    """Fraction of query-frames reported free while the reference places the
    patient there. The only error counted as a safety failure.

    U, U_ref: (F, N) binary. Denominator is all query-frames, matching the
    paper's definition; the reference-positive-conditioned rate is reported
    alongside for interpretation.
    """
    u = np.asarray(U).astype(bool)
    r = np.asarray(U_ref).astype(bool)
    err = (~u) & r
    n_ref = int(r.sum())
    return {
        "free_space_error": float(err.mean()),
        "free_space_error_given_ref": float(err.sum() / n_ref) if n_ref else 0.0,
        "n_ref_positive": n_ref,
    }


def free_space_error_agreed(U, U_ref_a, U_ref_b):
    """Free-space error counted only where both reference schemes agree the
    patient is present. The claim-carrying figure of the paper."""
    u = np.asarray(U).astype(bool)
    r = np.asarray(U_ref_a).astype(bool) & np.asarray(U_ref_b).astype(bool)
    err = (~u) & r
    n_ref = int(r.sum())
    return {
        "free_space_error_agreed": float(err.mean()),
        "free_space_error_agreed_given_ref":
            float(err.sum() / n_ref) if n_ref else 0.0,
        "n_ref_agreed": n_ref,
    }


def occupied_fraction(U):
    """Proportion of query-frames reported occupied. The workspace price."""
    return {"occupied_fraction": float(np.asarray(U).astype(bool).mean())}


def monotonicity_per_point(falls, stale_falls, n_frames, n_points):
    """Summarise the paper's per-point monotonicity check.

    A fall is any query point going 1 -> 0 between consecutive frames. It is a
    VIOLATION only if, at the earlier frame, the point was covered by an
    element (ball, or capsule) none of whose keypoints were observed at the
    later frame: the point lost coverage with no new observation of anything
    covering it. Falls caused by re-observing a covering keypoint (it moved,
    or its age reset) are legitimate updates, not violations.

    falls, stale_falls: per-frame-pair counts, computed in the runner.
    """
    falls = int(falls)
    viol = int(stale_falls)
    return {
        "falls": falls,
        "violations": viol,
        "violations_per_1000_falls": 1000.0 * viol / falls if falls else 0.0,
        "violations_per_1e6_query_frames":
            1e6 * viol / max(1, (n_frames - 1) * n_points),
    }
