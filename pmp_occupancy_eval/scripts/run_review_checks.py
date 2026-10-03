#!/usr/bin/env python3
"""Extra analyses requested by the pre-submission review (v20).

Same subject filter, confidence gate, grid, d_contact and k as the final
run (config/eval_params_final.yaml, config/d_contact_tape.json), rho = 0,
every frame.

  A. Finger and toe keypoints excluded (the manikin has neither), with and
     without enlarging the wrist and ankle radii to cover the hand and foot.
  B. Legs kept but their age forced to zero (separates staleness from the
     size of the leg tubes in the no-legs ablation).
  C. Static-recording ground truth: the manikin does not move, so each
     keypoint's median seen position is its true position. Free-space error
     of both arms against it, for gaps of any length.
  D. How often head keypoints were seen.

    python3 scripts/run_review_checks.py --out review/
"""
import _boot  # noqa
import argparse
import csv
import glob
import json
import os
import time
import numpy as np

from pmp_risk_eval import config_io
from pmp_risk_eval.subject_filter import SubjectFilter
from pmp_risk_eval.keypoint_state import KeypointTracker
from pmp_risk_eval.occupancy_field import OccupancyEstimator
from pmp_risk_eval.query_set import bed_grid, grid_bounds
from pmp_risk_eval.types import N_KP, KP_NAMES, LOWER_LIMB, HEAD, EDGES
from run_paper_analysis import load_bag

FINGERS = [i for i, n in enumerate(KP_NAMES) if "HAND_" in n]
TOES = [i for i, n in enumerate(KP_NAMES) if "TOE" in n]
WRISTS = [KP_NAMES.index("LEFT_WRIST"), KP_NAMES.index("RIGHT_WRIST")]
ANKLES = [KP_NAMES.index("LEFT_ANKLE"), KP_NAMES.index("RIGHT_ANKLE")]


class _B:
    pass


def walk(times, pos, val, d_contact, queries, arm, k, t_hold=0.5,
         age_zero=None, keep_U=False):
    """Occupied fraction per frame (and U if keep_U). age_zero: keypoint
    indices whose age is forced to 0 once seen."""
    est = OccupancyEstimator(d_contact, k=k, arm=arm, r0="d_contact",
                             t_hold=t_hold)
    tr = KeypointTracker(d_contact)
    occ = np.zeros(len(times))
    U = np.zeros((len(times), len(queries)), dtype=np.uint8) if keep_U else None
    for f, t in enumerate(times):
        b = _B()
        b.keypoints, b.valid, b.id = pos[f], val[f], -1
        st = tr.update(t, b if val[f].any() else None)
        if age_zero is not None:
            idx = np.array(age_zero)
            m = st.ever_seen[idx]
            st.ages[idx[m]] = 0.0
        u = est.evaluate(st, queries)
        occ[f] = u.mean()
        if keep_U:
            U[f] = u
    return occ, U


def extent(pos, val, a_idx, b_idx):
    """Median distance from joints a_idx to keypoints b_idx when both seen."""
    d = []
    for a in a_idx:
        side = "LEFT" if "LEFT" in KP_NAMES[a] else "RIGHT"
        for b in b_idx:
            if side not in KP_NAMES[b]:
                continue
            m = val[:, a] & val[:, b]
            if m.any():
                d.extend(np.linalg.norm(pos[m, a] - pos[m, b], axis=1))
    return float(np.median(d)) if d else float("nan"), len(d)


def gt_occupancy(pos, val, d_contact, queries, min_frames):
    """Ground-truth occupancy of a still body: median seen position of every
    keypoint seen in at least min_frames frames, tubes at d_contact."""
    n_seen = val.sum(axis=0)
    real = n_seen >= min_frames
    x = np.full((N_KP, 3), np.nan)
    for i in np.where(real)[0]:
        x[i] = np.median(pos[val[:, i], i], axis=0)
    return x, real


def cover(queries, x, active, r):
    q = queries
    d_ball = np.linalg.norm(q[:, None, :] - x[active][None, :, :], axis=2)
    c = (d_ball <= r[active][None, :]).any(axis=1)
    e = np.asarray(EDGES)
    both = active[e[:, 0]] & active[e[:, 1]]
    if both.any():
        from pmp_risk_eval.occupancy_field import _point_segment_distance
        i, j = e[both, 0], e[both, 1]
        ds = _point_segment_distance(q, x[i], x[j])
        c |= (ds <= np.maximum(r[i], r[j])[None, :]).any(axis=1)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/eval_params_final.yaml")
    ap.add_argument("--d-contact", default="config/d_contact_tape.json")
    ap.add_argument("--bags", nargs="+",
                    default=sorted(glob.glob("npz_replay_trim/310826_B*.npz")))
    ap.add_argument("--out", default="review/")
    ap.add_argument("--gt-min-frames", type=int, default=30)
    # pooled over all six recordings (wrist->finger, ankle->toe medians)
    ap.add_argument("--hand-ext", type=float, default=0.108)
    ap.add_argument("--foot-ext", type=float, default=0.147)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = config_io.load(a.config)
    pp = cfg["paper"]
    bb = config_io.bed_bounds(cfg)
    d_contact = np.array(json.load(open(a.d_contact))["d_contact"])
    queries = bed_grid(grid_bounds(bb, pp.get("grid_margin", 0.0)),
                       res=pp["grid_res"])
    k = pp["k_phys"]
    print(f"{len(queries)} grid points, k = {k}")

    data = {}
    for bag in a.bags:
        name = os.path.basename(bag).replace(".npz", "").replace("310826_", "")
        sf = SubjectFilter("bed", bed_bounds=bb, hysteresis=pp["hysteresis"],
                           seed=pp.get("subject_seed"))
        data[name] = load_bag(bag, sf, pp["kp_conf_min"])

    # hand and foot extent measured on the manikin, pooled over recordings
    allpos = np.concatenate([d[1] for d in data.values()])
    allval = np.concatenate([d[2] for d in data.values()])
    h_bag, nh = extent(allpos, allval, WRISTS, FINGERS)
    f_bag, nf = extent(allpos, allval, ANKLES, TOES)
    print(f"this bag: wrist->finger {h_bag:.3f} m (n={nh}), ankle->toe "
          f"{f_bag:.3f} m (n={nf}); using pooled {a.hand_ext}, {a.foot_ext}")
    hand_ext, foot_ext = a.hand_ext, a.foot_ext
    d_big = d_contact.copy()
    d_big[WRISTS] = np.maximum(d_big[WRISTS], hand_ext + 0.05)
    d_big[ANKLES] = np.maximum(d_big[ANKLES], foot_ext + 0.05)

    rows_a, rows_c = [], []
    for name, (times, pos, val) in data.items():
        t0 = time.time()
        base, _ = walk(times, pos, val, d_contact, queries, "proposed", k)
        v_nf = val.copy()
        v_nf[:, FINGERS + TOES] = False
        nofin, _ = walk(times, pos, v_nf, d_contact, queries, "proposed", k)
        nofin_big, _ = walk(times, pos, v_nf, d_big, queries, "proposed", k)
        legs0, _ = walk(times, pos, val, d_contact, queries, "proposed", k,
                        age_zero=LOWER_LIMB)
        v_nl = val.copy()
        v_nl[:, LOWER_LIMB] = False
        nolegs, _ = walk(times, pos, v_nl, d_contact, queries, "proposed", k)
        head_frames = int(val[:, HEAD].any(axis=1).sum())
        fin_frames = int(val[:, FINGERS].any(axis=1).sum())
        row = dict(bag=name, frames=len(times),
                   occ_base=base.mean(), occ_no_fingers_toes=nofin.mean(),
                   occ_no_fingers_toes_big_wrist_ankle=nofin_big.mean(),
                   occ_legs_age_zero=legs0.mean(), occ_no_legs=nolegs.mean(),
                   head_frames_seen=head_frames, finger_frames_seen=fin_frames)
        rows_a.append(row)
        print(f"{name:<24} base {row['occ_base']:.3f}  no fing/toe "
              f"{row['occ_no_fingers_toes']:.3f}  +big {row['occ_no_fingers_toes_big_wrist_ankle']:.3f}"
              f"  legs age0 {row['occ_legs_age_zero']:.3f}  no legs {row['occ_no_legs']:.3f}"
              f"  head seen {head_frames}/{len(times)}  fingers seen {fin_frames}"
              f"  ({time.time()-t0:.0f}s)")

        if "static" not in name:
            continue
        # C. ground truth for the still manikin
        x_gt, real = gt_occupancy(pos, val, d_contact, queries, a.gt_min_frames)
        tr_seen = np.cumsum(val, axis=0) > 0     # keypoint in E at frame f
        runs = [("proposed", None)] + [("hold", th) for th in (0.5, 1, 2, 5)]
        U_all = {}
        for arm, th in runs:
            _, U = walk(times, pos, val, d_contact, queries, arm, k,
                        t_hold=th or 0.5, keep_U=True)
            U_all[(arm, th)] = U
        miss = {key: 0 for key in U_all}
        tot = 0
        gt_cache = {}
        for f in range(len(times)):
            act = real & tr_seen[f]
            if not act.any():
                continue
            key_act = act.tobytes()
            if key_act not in gt_cache:
                gt_cache[key_act] = cover(queries, x_gt, act, d_contact)
            g = gt_cache[key_act]
            tot += int(g.sum())
            for key, U in U_all.items():
                miss[key] += int((g & (U[f] == 0)).sum())
        for (arm, th), m in miss.items():
            rows_c.append(dict(bag=name, arm=arm, t_hold=th if th else "",
                               gt_keypoints=int(real.sum()),
                               gt_point_frames=tot, missed=m,
                               fse_gt=m / max(tot, 1)))
            print(f"   GT {arm:<8} t_hold={th}  FSE_gt {100*m/max(tot,1):.3f}%"
                  f"  ({int(real.sum())} GT keypoints)")

    for fn, rows in (("ablations.csv", rows_a), ("static_ground_truth.csv", rows_c)):
        with open(os.path.join(a.out, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    json.dump(dict(hand_extent_m=hand_ext, foot_extent_m=foot_ext,
                   wrist_radius_big=float(d_big[WRISTS[0]]),
                   ankle_radius_big=float(d_big[ANKLES[0]])),
              open(os.path.join(a.out, "extents.json"), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
