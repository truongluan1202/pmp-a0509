#!/usr/bin/env python3
"""Baseline vs corrected vs probabilistic over one or more bags.

Produces the five-column safety/cost table plus an a_max sweep.

    python3 scripts/run_evaluation.py --bags ~/mnt/rosbag/310826_B* \
        --out eval.csv --per-frame-dir per_frame/
"""
import _boot  # noqa
import argparse
import csv
import json
import os
import numpy as np
from pmp_risk_eval import bag_reader, config_io, metrics
from pmp_risk_eval.subject_filter import SubjectFilter
from pmp_risk_eval.keypoint_state import KeypointTracker
from pmp_risk_eval.risk_field import RiskField
from pmp_risk_eval.query_set import sweep_path, bed_grid
from pmp_risk_eval.reference_builder import reference_field
from pmp_risk_eval.types import N_KP


def load_bag(bag, sf):
    times, pos, val, nseen = [], [], [], []
    for f in bag_reader.read_frames(bag):
        body, _ = sf.select(f)
        times.append(f.t)
        if body is None:
            pos.append(np.full((N_KP, 3), np.nan))
            val.append(np.zeros(N_KP, dtype=bool))
            nseen.append(0)
        else:
            pos.append(body.keypoints)
            val.append(body.valid)
            nseen.append(int(body.valid.sum()))
    return (np.array(times), np.array(pos), np.array(val), np.array(nseen))


def run_mode(times, pos, val, d_contact, sigma, queries, mode, cfg_e, a_max):
    rf = RiskField(sigma, cfg_e["v_hat_max"], a_max, lam=cfg_e["lam"],
                   mode=mode, baseline_expiry=cfg_e["baseline_expiry"],
                   r0=cfg_e["r0"])
    tr = KeypointTracker(d_contact)
    R = np.zeros((len(times), len(queries)))
    ages = np.zeros((len(times), len(queries)))

    class _B:
        pass

    for k, t in enumerate(times):
        b = _B()
        b.keypoints = pos[k]
        b.valid = val[k]
        b.id = -1
        st = tr.update(t, b if val[k].any() else None)
        R[k] = rf.evaluate(st, queries)
        _, ages[k] = rf.argmax_keypoint(st, queries)
    return R, ages, rf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--sensitivity", default=None)
    ap.add_argument("--d-contact", default=None,
                    help="d_contact.json from run_dcontact.py")
    ap.add_argument("--subject-id", type=int, default=None)
    ap.add_argument("--queries", choices=("sweep", "grid"), default="sweep")
    ap.add_argument("--sweep-amax", action="store_true")
    ap.add_argument("--out", default="eval.csv")
    ap.add_argument("--per-frame-dir", default=None)
    a = ap.parse_args()

    cdir = config_io.default_config_dir()
    cfg = config_io.load(a.config or os.path.join(cdir, "eval_params.yaml"))
    sens = config_io.load(a.sensitivity or os.path.join(
        cdir, "region_sensitivity.yaml"))
    sigma = config_io.sigma_vector(sens)
    bb = config_io.bed_bounds(cfg)
    cfg_e = cfg["estimator"]

    if a.d_contact:
        d_contact = np.array(json.load(open(a.d_contact))["d_contact"])
    else:
        d_contact = np.full(N_KP, 0.08)
        print("no --d-contact given, using 0.08 m everywhere")

    queries = (sweep_path(bb, n=cfg["query"]["sweep_n"]) if a.queries == "sweep"
               else bed_grid(bb, res=cfg["query"]["grid_res"]))
    print(f"{len(queries)} query points ({a.queries})")

    a_maxes = cfg["a_max_sweep"] if a.sweep_amax else [cfg_e["a_max"]]
    rows = []

    for bag in a.bags:
        if bag.endswith(".svo2") or not os.path.isdir(bag):
            continue
        name = os.path.basename(bag.rstrip("/"))
        sf = (SubjectFilter("fixed", subject_id=a.subject_id)
              if a.subject_id is not None else SubjectFilter("bed", bed_bounds=bb))
        try:
            times, pos, val, nseen = load_bag(bag, sf)
        except Exception as e:
            print(f"  SKIP {name}: {e}")
            continue
        if val.sum() == 0:
            print(f"  SKIP {name}: subject never selected. Check bed_bounds.")
            continue
        print(f"\n{name}: {len(times)} frames, "
              f"{sf.report()['n_id_changes']} ID changes")

        for a_max in a_maxes:
            ref_rf = RiskField(sigma, cfg_e["v_hat_max"], a_max,
                               lam=cfg_e["lam"], mode="probabilistic")
            R_ref = reference_field(
                times, pos, val, ref_rf,
                lambda: KeypointTracker(d_contact), queries)

            for mode in ("baseline", "corrected", "probabilistic"):
                R, ages, _ = run_mode(times, pos, val, d_contact, sigma,
                                      queries, mode, cfg_e, a_max)
                m = {}
                m.update(metrics.monotonicity_violations(R, val))
                m.update(metrics.underestimation(R, R_ref))
                m.update(metrics.roc_auc(R, R_ref))
                m.update(metrics.held_fraction(ages))
                m.update(metrics.argmax_staleness(ages))
                m.update(metrics.volume_above(R))
                rows.append({"bag": name, "mode": mode, "a_max": a_max,
                             "frames": len(times),
                             "id_changes": sf.report()["n_id_changes"], **m})

                if a.per_frame_dir and a_max == cfg_e["a_max"]:
                    os.makedirs(a.per_frame_dir, exist_ok=True)
                    np.savetxt(
                        os.path.join(a.per_frame_dir, f"{name}_{mode}.csv"),
                        np.column_stack([times, R.max(axis=1), R.mean(axis=1),
                                         R_ref.max(axis=1), nseen,
                                         np.nanmean(np.where(
                                             np.isfinite(ages), ages, np.nan),
                                             axis=1)]),
                        delimiter=",", header="t,R_max,R_mean,R_ref_max,"
                                              "n_seen,mean_argmax_age",
                        comments="")

    if not rows:
        print("\nnothing evaluated")
        return

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 96)
    print("SAFETY (first three) and COST (last three)")
    print("=" * 96)
    hdr = (f"{'bag':<34}{'mode':<15}{'viol/1k':>9}{'under':>8}"
           f"{'auc':>7}{'held':>8}{'stale':>8}{'vol>0.5':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["a_max"] != cfg_e["a_max"]:
            continue
        print(f"{r['bag'][:33]:<34}{r['mode']:<15}"
              f"{r['violations_per_1000_pairs']:>9.1f}"
              f"{r['underestimation_rate']:>8.3f}{r['auc']:>7.3f}"
              f"{r['held_fraction']:>8.3f}{r['mean_staleness']:>8.3f}"
              f"{r['fraction_above']:>9.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
