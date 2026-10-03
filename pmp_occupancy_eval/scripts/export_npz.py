#!/usr/bin/env python3
"""Dump each bag's skeleton stream to a portable .npz. Run in the container.

Decoding ZED messages needs rclpy and zed_msgs typesupport, which exist only
inside pmp-ros2-base. Everything downstream of decoding is plain NumPy. This
script draws the line in the right place: run it once on the machine that can
read bags, and the analysis then runs anywhere.

Deliberately dumps ALL detected bodies per frame, before subject selection.
Selection depends on bed_bounds, which is still unmeasured; baking it in here
would force a return to this machine to change it. With every body kept, the
bed volume can be set and re-set offline.

    python3 scripts/export_npz.py --bags ~/mnt/rosbag/310826_* --out-dir npz/

Depth and SVO are not exported. The paper analysis does not use them:
d_contact comes from tape measurements, not point-cloud depth.
"""
import _boot  # noqa
import argparse
import os
import numpy as np

from pmp_risk_eval import bag_reader
from pmp_risk_eval.types import N_KP


def export_bag(bag, out_path):
    times, t_abs, counts = [], [], []
    kp, valid, conf, ids, tstate = [], [], [], [], []

    for f in bag_reader.read_frames(bag):
        times.append(f.t)
        t_abs.append(f.t_abs)
        counts.append(len(f.bodies))
        for b in f.bodies:
            kp.append(b.keypoints)
            valid.append(b.valid)
            conf.append(b.confidence)
            ids.append(b.id)
            tstate.append(b.tracking_state)

    if not times:
        raise RuntimeError("no frames read")

    n_bodies = len(kp)
    np.savez_compressed(
        out_path,
        times=np.asarray(times, dtype=np.float64),
        t_abs=np.asarray(t_abs, dtype=np.float64),
        body_counts=np.asarray(counts, dtype=np.int32),
        keypoints=(np.asarray(kp, dtype=np.float32) if n_bodies
                   else np.zeros((0, N_KP, 3), np.float32)),
        valid=(np.asarray(valid, dtype=bool) if n_bodies
               else np.zeros((0, N_KP), bool)),
        confidence=(np.asarray(conf, dtype=np.float32) if n_bodies
                    else np.zeros((0, N_KP), np.float32)),
        body_ids=np.asarray(ids, dtype=np.int32),
        tracking_state=np.asarray(tstate, dtype=np.int32),
        bag_name=np.array(os.path.basename(bag.rstrip("/"))),
    )
    return len(times), n_bodies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", required=True)
    ap.add_argument("--out-dir", default="npz")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    ok = 0
    for bag in a.bags:
        if bag.endswith(".svo2") or not os.path.isdir(bag):
            continue
        name = os.path.basename(bag.rstrip("/"))
        out = os.path.join(a.out_dir, f"{name}.npz")
        try:
            n_frames, n_bodies = export_bag(bag, out)
        except Exception as e:
            print(f"SKIP {name}: {e}")
            continue
        size_mb = os.path.getsize(out) / 1e6
        print(f"{name}: {n_frames} frames, {n_bodies} body records, "
              f"{size_mb:.1f} MB -> {out}")
        ok += 1

    print(f"\nexported {ok} bags to {a.out_dir}/")
    if ok:
        print("Zip that directory and send it. Nothing else is needed.")


if __name__ == "__main__":
    main()
