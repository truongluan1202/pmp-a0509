#!/usr/bin/env python3
"""Keypoint yield across conditions. Answers the lower-limb question.

Session 310826 observed no lower-limb keypoints in any static or slow-dynamic
condition, but lower limbs present under fast motion. If motion rather than
surface treatment drives recovery, the condition comparison is a null and the
paper's Experiment 1 result changes shape. Run this before anything else.

    python3 scripts/run_keypoint_yield.py --bags ~/mnt/rosbag/310826_* \
        --out yield.csv
"""
import _boot  # noqa
import argparse
import csv
import os
import numpy as np
from pmp_risk_eval import bag_reader
from pmp_risk_eval.config_io import YIELD_REGIONS
from pmp_risk_eval.types import N_KP, KP_NAMES


def parse_name(path):
    b = os.path.basename(path.rstrip("/"))
    cond = "unknown"
    for c in ("B1", "B2", "B3", "B4", "fastmotion"):
        if c in b:
            cond = c
            break
    if "fastmotion" in b:
        phase = "fast"
    elif "dynamic" in b:
        phase = "dynamic"
    elif "static" in b:
        phase = "static"
    else:
        phase = "unknown"
    return b, cond, phase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", required=True)
    ap.add_argument("--out", default="keypoint_yield.csv")
    ap.add_argument("--per-keypoint-out", default="keypoint_yield_detail.csv")
    a = ap.parse_args()

    rows, detail = [], []
    for bag in a.bags:
        if bag.endswith(".svo2") or not os.path.isdir(bag):
            continue
        name, cond, phase = parse_name(bag)
        V, n_bodies = [], []
        try:
            for f in bag_reader.read_frames(bag):
                n_bodies.append(len(f.bodies))
                if f.bodies:
                    best = max(f.bodies, key=lambda b: b.valid.sum())
                    V.append(best.valid)
                else:
                    V.append(np.zeros(N_KP, dtype=bool))
        except Exception as e:
            print(f"  SKIP {name}: {e}")
            continue
        if not V:
            continue
        V = np.array(V)
        row = {"bag": name, "condition": cond, "phase": phase,
               "frames": len(V),
               "mean_bodies": round(float(np.mean(n_bodies)), 3)}
        for rname, idx in YIELD_REGIONS.items():
            row[rname] = round(float(V[:, idx].mean()), 4)
        row["all"] = round(float(V.mean()), 4)
        rows.append(row)
        for i in range(N_KP):
            detail.append({"bag": name, "condition": cond, "phase": phase,
                           "kp_index": i, "kp_name": KP_NAMES[i],
                           "yield": round(float(V[:, i].mean()), 4)})

    if not rows:
        print("no bags read")
        return

    cols = ["bag", "condition", "phase", "frames", "mean_bodies"] \
        + list(YIELD_REGIONS) + ["all"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(a.per_keypoint_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0]))
        w.writeheader()
        w.writerows(detail)

    hdr = f"{'bag':<38}{'phase':<9}" + "".join(f"{r:>12}" for r in YIELD_REGIONS)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['bag']:<38}{r['phase']:<9}"
              + "".join(f"{r[k]:>12.3f}" for k in YIELD_REGIONS))

    print("\nPOOLED BY PHASE (the question)")
    print(f"{'phase':<12}{'lower_limb':>12}{'upper_limb':>12}{'torso':>12}")
    for ph in ("static", "dynamic", "fast"):
        sub = [r for r in rows if r["phase"] == ph]
        if not sub:
            continue
        print(f"{ph:<12}" + "".join(
            f"{np.mean([r[k] for r in sub]):>12.3f}"
            for k in ("lower_limb", "upper_limb", "torso")))

    print("\nPOOLED BY CONDITION")
    print(f"{'condition':<12}{'lower_limb':>12}{'upper_limb':>12}{'torso':>12}")
    for c in ("B1", "B2", "B3", "B4"):
        sub = [r for r in rows if r["condition"] == c and r["phase"] != "fast"]
        if not sub:
            continue
        print(f"{c:<12}" + "".join(
            f"{np.mean([r[k] for r in sub]):>12.3f}"
            for k in ("lower_limb", "upper_limb", "torso")))

    print(f"\nwrote {a.out} and {a.per_keypoint_out}")


if __name__ == "__main__":
    main()
