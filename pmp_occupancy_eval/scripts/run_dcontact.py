#!/usr/bin/env python3
"""Derive d_contact per keypoint from depth. 310826 bags only.

    python3 scripts/run_dcontact.py --bag ~/mnt/rosbag/310826_B1_neither_static \
        --out d_contact.json
"""
import _boot  # noqa
import argparse
import json
import os
import numpy as np
from pmp_risk_eval import bag_reader, config_io, depth_contact
from pmp_risk_eval.subject_filter import SubjectFilter
from pmp_risk_eval.types import N_KP, KP_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--subject-id", type=int, default=None)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--out", default="d_contact.json")
    a = ap.parse_args()

    cfg = config_io.load(a.config or os.path.join(
        config_io.default_config_dir(), "eval_params.yaml"))

    ci = bag_reader.read_camera_info(a.bag)
    if ci is None:
        print("No depth camera_info in this bag. Use tape measurements:")
        print("  from pmp_risk_eval.depth_contact import d_contact_from_tape")
        return
    fx, fy, cx, cy = ci
    print(f"intrinsics fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    depth = list(bag_reader.read_depth(a.bag, max_frames=a.max_frames))
    if not depth:
        print("No depth images in this bag.")
        return
    dt = np.array([t for t, _ in depth])
    print(f"{len(depth)} depth frames")

    sf = (SubjectFilter("fixed", subject_id=a.subject_id)
          if a.subject_id is not None
          else SubjectFilter("bed", bed_bounds=config_io.bed_bounds(cfg)))

    samples = []
    n = 0
    for f in bag_reader.read_frames(a.bag):
        body, _ = sf.select(f)
        if body is None:
            continue
        j = int(np.argmin(np.abs(dt - f.t)))
        if abs(dt[j] - f.t) > 0.10:
            continue
        img = depth[j][1]
        uv, rng = depth_contact.project(body.keypoints, fx, fy, cx, cy)
        surf = depth_contact.surface_depth(img, uv, win=a.window)
        off = np.abs(surf - rng)
        off[~body.valid] = np.nan
        samples.append(off)
        n += 1
        if n >= a.max_frames:
            break

    if not samples:
        print("No matched skeleton/depth frames. Check the bed volume.")
        return

    d = depth_contact.accumulate(samples)
    print(f"\nmatched {n} frames\n")
    print(f"{'idx':>4} {'keypoint':<22}{'d_contact (m)':>14}")
    for i in range(N_KP):
        print(f"{i:>4} {KP_NAMES[i]:<22}{d[i]:>14.4f}")

    with open(a.out, "w") as f:
        json.dump({"bag": os.path.basename(a.bag.rstrip('/')),
                   "n_frames": n, "d_contact": d.tolist(),
                   "kp_names": KP_NAMES}, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
