#!/usr/bin/env python3
"""All ACRA paper numbers from the archived bags. One command, three CSVs.

    python3 scripts/run_paper_analysis.py --bags ~/mnt/rosbag/310826_B* \
        --d-contact d_contact.json --out-prefix paper/

Outputs
  <prefix>q1_doseresponse.csv   both arms x both injection modes x rho grid,
                                at k = k_phys, both reference schemes plus
                                the agreed (claim-carrying) figure
  <prefix>q2_ksweep.csv         occupied fraction against k at the rho
                                subset, both arms, both injection modes
  <prefix>gaps.csv              natural gap statistics per bag, and the
                                burst length used

Defaults are sized for an overnight run (grid_res 0.10, frame_stride 2).
For the submission run set grid_res 0.05 and frame_stride 1 in the config.
"""
import _boot  # noqa
import argparse
import csv
import json
import os
import numpy as np

from pmp_risk_eval import bag_reader, config_io, metrics, npz_corpus
from pmp_risk_eval.subject_filter import SubjectFilter
from pmp_risk_eval.keypoint_state import KeypointTracker
from pmp_risk_eval.occupancy_field import OccupancyEstimator
from pmp_risk_eval.reference_builder import reference_occupancy
from pmp_risk_eval.occlusion_injector import (
    inject_independent, inject_structured, natural_gap_stats, realised_rate)
from pmp_risk_eval.query_set import bed_grid, grid_bounds
from pmp_risk_eval.types import N_KP, CHAINS


CHAIN_COLS = sorted({i for c in CHAINS.values() for i in c})


def _realised(val, V_now, inj_mode):
    """Realised suppression rate over the keypoint-frames the mode can
    touch: all keypoints for independent, chain keypoints for structured.
    rho is defined on that denominator in both modes."""
    if inj_mode == "structured":
        return realised_rate(val[:, CHAIN_COLS], V_now[:, CHAIN_COLS])
    return realised_rate(val, V_now)


def _frames(source):
    """Bag directory (needs ROS) or exported .npz (needs nothing)."""
    if str(source).endswith(".npz"):
        return npz_corpus.read_frames_npz(source)
    return bag_reader.read_frames(source)


def load_bag(bag, sf, conf_min):
    times, pos, val = [], [], []
    for f in _frames(bag):
        body, _ = sf.select(f)
        times.append(f.t)
        if body is None:
            pos.append(np.full((N_KP, 3), np.nan))
            val.append(np.zeros(N_KP, dtype=bool))
        else:
            pos.append(body.keypoints)
            val.append(bag_reader.observed_mask(body, conf_min))
    return np.array(times), np.array(pos), np.array(val)


def run_arm(times, pos, val, d_contact, queries, arm, k, r0, t_hold):
    est = OccupancyEstimator(d_contact, k=k, arm=arm, r0=r0, t_hold=t_hold)
    tr = KeypointTracker(d_contact)
    U = np.zeros((len(times), len(queries)), dtype=np.uint8)

    class _B:
        pass

    falls = viol = viol_frames = 0
    prev = None
    for f, t in enumerate(times):
        b = _B()
        b.keypoints = pos[f]
        b.valid = val[f]
        b.id = -1
        st = tr.update(t, b if val[f].any() else None)
        U[f] = est.evaluate(st, queries)
        if prev is not None:
            fell = (U[f - 1] == 1) & (U[f] == 0)
            nf = int(fell.sum())
            if nf:
                falls += nf
                # coverage at f-1 from elements with no keypoint observed at f
                stale = est.covered_by_subset(prev, queries[fell], ~val[f])
                nv = int(stale.sum())
                viol += nv
                viol_frames += int(nv > 0)
        prev = st
    mono = metrics.monotonicity_per_point(falls, viol, len(times), len(queries))
    mono["violation_frames"] = viol_frames
    return U, mono


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", required=True,
                    help="bag directories, or .npz files from export_npz.py")
    ap.add_argument("--config", default=None)
    ap.add_argument("--d-contact", default=None)
    ap.add_argument("--out-prefix", default="paper/")
    ap.add_argument("--skip-ksweep", action="store_true")
    a = ap.parse_args()

    cdir = config_io.default_config_dir()
    cfg = config_io.load(a.config or os.path.join(cdir, "eval_params.yaml"))
    pp = cfg["paper"]
    bb = config_io.bed_bounds(cfg)

    d_contact = (np.array(json.load(open(a.d_contact))["d_contact"])
                 if a.d_contact else np.full(N_KP, 0.08))
    if not a.d_contact:
        print("no --d-contact given, using 0.08 m everywhere")

    gb = grid_bounds(bb, pp.get("grid_margin", 0.0))
    queries = bed_grid(gb, res=pp["grid_res"])
    stride = int(pp.get("frame_stride", 1))
    print(f"{len(queries)} grid points at {pp['grid_res']} m, "
          f"frame stride {stride}, grid bounds {gb}")

    os.makedirs(a.out_prefix or ".", exist_ok=True)
    q1_rows, q2_rows, gap_rows = [], [], []
    rng = np.random.default_rng(pp["seed"])

    for bag in a.bags:
        if bag.endswith(".svo2"):
            continue
        if not (bag.endswith(".npz") or os.path.isdir(bag)):
            continue
        name = os.path.basename(bag.rstrip("/")).replace(".npz", "")
        sf = SubjectFilter("bed", bed_bounds=bb,
                           hysteresis=pp["hysteresis"],
                           seed=pp.get("subject_seed"))
        try:
            times, pos, val = load_bag(bag, sf, pp["kp_conf_min"])
        except Exception as e:
            print(f"SKIP {name}: {e}")
            continue
        if val.sum() == 0:
            print(f"SKIP {name}: subject never selected. Check bed_bounds.")
            continue
        times, pos, val = times[::stride], pos[::stride], val[::stride]
        print(f"\n{name}: {len(times)} frames (strided), "
              f"yield {val.mean():.3f}, "
              f"{sf.report()['n_id_changes']} ID changes")

        gaps = natural_gap_stats(times, val)
        burst = (gaps["median_s"] if pp["burst"] == "auto"
                 else float(pp["burst"]))
        if not np.isfinite(burst):
            burst = 0.4
            print("  no natural gaps found; burst falls back to 0.4 s")
        gap_rows.append({"bag": name, **gaps, "burst_used_s": burst})
        print(f"  natural gaps: n={gaps['n_gaps']}, "
              f"median {gaps['median_s']:.3f} s -> burst {burst:.3f} s")

        # references, computed once per bag from the UNINJECTED stream
        refs = {s: reference_occupancy(times, pos, val, d_contact, queries,
                                       scheme=s, max_gap=pp["max_gap_ref"])
                for s in ("linear", "hold")}

        def measure(U, mono, tag):
            m = {}
            for s in ("linear", "hold"):
                r = metrics.free_space_error(U, refs[s])
                m[f"fse_{s}"] = r["free_space_error"]
                m[f"fse_{s}_given_ref"] = r["free_space_error_given_ref"]
            m.update(metrics.free_space_error_agreed(
                U, refs["linear"], refs["hold"]))
            m.update(metrics.occupied_fraction(U))
            m.update({f"mono_{k2}": v for k2, v in mono.items()})
            return {**tag, **m}

        # ---------------- Q1: dose-response at k_phys ----------------
        for inj_mode in ("independent", "structured"):
            for rho in pp["rho_grid"]:
                reps = pp["n_repeats"] if rho > 0 else 1
                for rep in range(reps):
                    if rho == 0:
                        V_now = val
                    elif inj_mode == "independent":
                        V_now = inject_independent(times, val, rho, burst,
                                                   rng=rng)
                    else:
                        V_now = inject_structured(times, val, rho, burst,
                                                  CHAINS, rng=rng)
                    rate = _realised(val, V_now, inj_mode)
                    for arm in ("hold", "proposed"):
                        U, mono = run_arm(times, pos, V_now, d_contact,
                                          queries, arm, pp["k_phys"],
                                          pp["r0"], pp["t_hold"])
                        q1_rows.append(measure(U, mono, {
                            "bag": name, "arm": arm, "inj_mode": inj_mode,
                            "k": pp["k_phys"], "target_rho": rho,
                            "realised_rho": round(rate, 4), "rep": rep,
                            "burst_s": burst}))
                print(f"  Q1 {inj_mode} rho {rho:.1f} done")

        # ---------------- Q2: k sweep ----------------
        if not a.skip_ksweep:
            for k in pp["k_sweep"]:
                for inj_mode in ("independent", "structured"):
                    for rho in pp["rho_for_k_sweep"]:
                        reps = pp["n_repeats"] if rho > 0 else 1
                        for rep in range(reps):
                            if rho == 0:
                                V_now = val
                            elif inj_mode == "independent":
                                V_now = inject_independent(
                                    times, val, rho, burst, rng=rng)
                            else:
                                V_now = inject_structured(
                                    times, val, rho, burst, CHAINS, rng=rng)
                            rate = _realised(val, V_now, inj_mode)
                            for arm in ("hold", "proposed"):
                                U, mono = run_arm(times, pos, V_now,
                                                  d_contact, queries, arm, k,
                                                  pp["r0"], pp["t_hold"])
                                q2_rows.append(measure(U, mono, {
                                    "bag": name, "arm": arm,
                                    "inj_mode": inj_mode, "k": k,
                                    "target_rho": rho,
                                    "realised_rho": round(rate, 4),
                                    "rep": rep, "burst_s": burst}))
                print(f"  Q2 k {k:.2f} done")

    for fname, rows in (("q1_doseresponse.csv", q1_rows),
                        ("q2_ksweep.csv", q2_rows),
                        ("gaps.csv", gap_rows)):
        if not rows:
            continue
        path = os.path.join(a.out_prefix, fname)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")

    if q1_rows:
        print("\nQ1 SUMMARY at rho=0 (estimator-attributable error)")
        hdr = (f"{'bag':<32}{'arm':<10}{'fse_agreed':>11}{'fse|ref':>9}"
               f"{'occupied':>10}{'falls':>9}{'mono_viol':>10}")
        print(hdr)
        print("-" * len(hdr))
        for r in q1_rows:
            if r["target_rho"] == 0 and r["inj_mode"] == "independent":
                print(f"{r['bag'][:31]:<32}{r['arm']:<10}"
                      f"{r['free_space_error_agreed']:>11.4f}"
                      f"{r['free_space_error_agreed_given_ref']:>9.4f}"
                      f"{r['occupied_fraction']:>10.3f}"
                      f"{r['mono_falls']:>9d}"
                      f"{r['mono_violations']:>10d}")


if __name__ == "__main__":
    main()
