#!/usr/bin/env python3
"""Occlusion dose-response: field error against injection rate.

The paper's central contribution. Natural occlusion in the bags is
uncontrolled and cannot produce a curve; injection can.

    python3 scripts/run_occlusion_study.py --bag ~/mnt/rosbag/310826_B1_neither_static \
        --out occlusion.csv
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
from pmp_risk_eval.query_set import sweep_path
from pmp_risk_eval.reference_builder import reference_field
from pmp_risk_eval.occlusion_injector import inject, realised_rate
from pmp_risk_eval.types import N_KP


def evaluate(times, pos, val, d_contact, sigma, queries, mode, cfg_e, a_max):
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
    return R, ages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--sensitivity", default=None)
    ap.add_argument("--d-contact", default=None)
    ap.add_argument("--subject-id", type=int, default=None)
    ap.add_argument("--out", default="occlusion.csv")
    a = ap.parse_args()

    cdir = config_io.default_config_dir()
    cfg = config_io.load(a.config or os.path.join(cdir, "eval_params.yaml"))
    sens = config_io.load(a.sensitivity or os.path.join(
        cdir, "region_sensitivity.yaml"))
    sigma = config_io.sigma_vector(sens)
    bb = config_io.bed_bounds(cfg)
    cfg_e = cfg["estimator"]
    occ = cfg["occlusion"]
    a_max = cfg_e["a_max"]

    d_contact = (np.array(json.load(open(a.d_contact))["d_contact"])
                 if a.d_contact else np.full(N_KP, 0.08))

    sf = (SubjectFilter("fixed", subject_id=a.subject_id)
          if a.subject_id is not None else SubjectFilter("bed", bed_bounds=bb))

    times, pos, val = [], [], []
    for f in bag_reader.read_frames(a.bag):
        body, _ = sf.select(f)
        times.append(f.t)
        pos.append(body.keypoints if body else np.full((N_KP, 3), np.nan))
        val.append(body.valid if body else np.zeros(N_KP, dtype=bool))
    times, pos, val = np.array(times), np.array(pos), np.array(val)
    if val.sum() == 0:
        print("subject never selected. Check bed_bounds or pass --subject-id.")
        return
    print(f"{len(times)} frames, base yield {val.mean():.3f}")

    queries = sweep_path(bb, n=cfg["query"]["sweep_n"])
    ref_rf = RiskField(sigma, cfg_e["v_hat_max"], a_max, lam=cfg_e["lam"],
                       mode="probabilistic")
    R_ref = reference_field(times, pos, val, ref_rf,
                            lambda: KeypointTracker(d_contact), queries)

    rows = []
    rng = np.random.default_rng(occ["seed"])
    for frac in occ["drop_fractions"]:
        for rep in range(occ["n_repeats"] if frac > 0 else 1):
            V = (val if frac == 0 else
                 inject(times, val, frac, occ["drop_duration"], rng=rng))
            rate = realised_rate(val, V)
            for mode in ("baseline", "corrected", "probabilistic"):
                R, ages = evaluate(times, pos, V, d_contact, sigma,
                                   queries, mode, cfg_e, a_max)
                m = {}
                m.update(metrics.underestimation(R, R_ref))
                m.update(metrics.monotonicity_violations(R, V))
                m.update(metrics.held_fraction(ages))
                m.update(metrics.volume_above(R))
                rows.append({"bag": os.path.basename(a.bag.rstrip("/")),
                             "target_rate": frac, "realised_rate": round(rate, 4),
                             "rep": rep, "mode": mode, **m})
        print(f"  rate {frac:.2f} done")

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\nDOSE-RESPONSE (mean over repeats)")
    hdr = f"{'rate':>7}{'mode':<16}{'rmse':>9}{'under':>9}{'viol/1k':>10}{'held':>8}"
    print(hdr)
    print("-" * len(hdr))
    for frac in occ["drop_fractions"]:
        for mode in ("baseline", "corrected", "probabilistic"):
            sub = [r for r in rows if r["target_rate"] == frac and r["mode"] == mode]
            if not sub:
                continue
            print(f"{frac:>7.2f}{mode:<16}"
                  f"{np.mean([r['rmse'] for r in sub]):>9.4f}"
                  f"{np.mean([r['underestimation_rate'] for r in sub]):>9.4f}"
                  f"{np.mean([r['violations_per_1000_pairs'] for r in sub]):>10.1f}"
                  f"{np.mean([r['held_fraction'] for r in sub]):>8.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
