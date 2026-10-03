#!/usr/bin/env python3
"""Section 5.3 numbers from the v2.1 pipeline itself: yield, held fraction,
oldest-keypoint age and identity, and the frame-by-frame correlation between
occupied fraction and oldest age. Proposed arm, rho = 0, k = k_phys.

    python3 scripts/run_mechanism.py --bags npz_replay_trim/310826_B*.npz \
        --out paper_replay/mechanism.csv

Same subject filter, confidence gate, grid and stride as run_paper_analysis.
"""
import _boot  # noqa
import argparse
import csv
import json
import os
import numpy as np

from pmp_risk_eval import config_io
from pmp_risk_eval.subject_filter import SubjectFilter
from pmp_risk_eval.keypoint_state import KeypointTracker
from pmp_risk_eval.occupancy_field import OccupancyEstimator
from pmp_risk_eval.query_set import bed_grid, grid_bounds
from pmp_risk_eval.types import N_KP, KP_NAMES, LOWER_LIMB, UPPER_LIMB, TORSO, HEAD
from run_paper_analysis import load_bag

RIGHT_FOOT = [KP_NAMES.index(n) for n in
              ("RIGHT_BIG_TOE", "RIGHT_SMALL_TOE", "RIGHT_HEEL")]


def held_fraction(val):
    """Fraction of keypoint-frames not observed, each keypoint counted from
    its first observation."""
    first = np.cumsum(val, axis=0) > 0
    return float((first & ~val).sum() / max(first.sum(), 1))


def walk(times, pos, val, d_contact, queries, k):
    est = OccupancyEstimator(d_contact, k=k, arm="proposed")
    tr = KeypointTracker(d_contact)

    class _B:
        pass

    occ, oldest, who = [], [], []
    for f, t in enumerate(times):
        b = _B()
        b.keypoints, b.valid, b.id = pos[f], val[f], -1
        st = tr.update(t, b if val[f].any() else None)
        occ.append(est.evaluate(st, queries).mean())
        a = np.where(np.isfinite(st.ages), st.ages, -1.0)
        if a.max() >= 0:
            oldest.append(a.max())
            who.append(int(a.argmax()))
        else:
            oldest.append(np.nan)
            who.append(-1)
    return np.array(occ), np.array(oldest), np.array(who)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", required=True)
    ap.add_argument("--out", default="paper_replay/mechanism.csv")
    ap.add_argument("--config", default=None)
    ap.add_argument("--d-contact", default=None)
    a = ap.parse_args()

    cfg = config_io.load(a.config or os.path.join(
        config_io.default_config_dir(), "eval_params.yaml"))
    pp = cfg["paper"]
    bb = config_io.bed_bounds(cfg)
    d_contact = (np.array(json.load(open(a.d_contact))["d_contact"])
                 if a.d_contact else np.full(N_KP, 0.08))
    queries = bed_grid(grid_bounds(bb, pp.get("grid_margin", 0.0)),
                       res=pp["grid_res"])
    stride = int(pp.get("frame_stride", 1))
    k = pp["k_phys"]

    rows = []
    for bag in a.bags:
        name = os.path.basename(bag).replace(".npz", "")
        sf = SubjectFilter("bed", bed_bounds=bb, hysteresis=pp["hysteresis"],
                           seed=pp.get("subject_seed"))
        times_f, pos_f, val_f = load_bag(bag, sf, pp["kp_conf_min"])
        times, pos, val = times_f[::stride], pos_f[::stride], val_f[::stride]

        occ, oldest, who = walk(times, pos, val, d_contact, queries, k)
        ok = np.isfinite(oldest)
        r = float(np.corrcoef(occ[ok], oldest[ok])[0, 1])
        u, c = np.unique(who[ok], return_counts=True)
        order = np.argsort(-c)
        top = [(KP_NAMES[u[i]], float(c[i] / ok.sum())) for i in order[:3]]
        top_kp = u[order[0]]

        val_nl = val.copy()
        val_nl[:, LOWER_LIMB] = False
        occ_nl, _, _ = walk(times, pos, val_nl, d_contact, queries, k)

        row = {
            "bag": name,
            "frames_full": len(times_f), "frames_eval": len(times),
            "yield": float(val.mean()), "held": held_fraction(val),
            "yield_head": float(val_f[:, HEAD].mean()),
            "yield_torso": float(val_f[:, TORSO].mean()),
            "yield_upper": float(val_f[:, UPPER_LIMB].mean()),
            "yield_lower": float(val_f[:, LOWER_LIMB].mean()),
            "rfoot_frames_full": int(val_f[:, RIGHT_FOOT].any(axis=1).sum()),
            "occ_prop": float(occ.mean()), "occ_no_legs": float(occ_nl.mean()),
            "oldest_age_median": float(np.median(oldest[ok])),
            "oldest_age_mean": float(np.mean(oldest[ok])),
            "corr_occ_oldest": r,
            "top_kp": top[0][0], "top_share": top[0][1],
            "top_kp_eval_frames_seen": int(val[:, top_kp].sum()),
            "second_kp": top[1][0] if len(top) > 1 else "",
            "second_share": top[1][1] if len(top) > 1 else 0.0,
        }
        rows.append(row)
        print(f"{name[7:]:<26} yield {row['yield']:.3f} held {row['held']:.3f} "
              f"occ {row['occ_prop']:.3f} (no legs {row['occ_no_legs']:.3f}) "
              f"oldest med {row['oldest_age_median']:.2f}s r {r:.2f} | "
              + ", ".join(f"{n} {s:.0%}" for n, s in top)
              + f" | top seen {row['top_kp_eval_frames_seen']}/{len(times)}"
              f" | lower {row['yield_lower']:.3f} rfoot {row['rfoot_frames_full']}/{len(times_f)}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
