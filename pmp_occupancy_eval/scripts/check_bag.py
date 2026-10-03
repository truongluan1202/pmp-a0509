#!/usr/bin/env python3
"""Bag inventory: topics, body IDs, ID persistence, keypoint yield by region.

    python3 scripts/check_bag.py --bag ~/mnt/rosbag/310826_B1_neither_static
"""
import _boot  # noqa
import argparse
import numpy as np
from pmp_risk_eval import bag_reader
from pmp_risk_eval.config_io import YIELD_REGIONS
from pmp_risk_eval.types import N_KP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()

    print("=" * 62)
    print("TOPICS")
    print("=" * 62)
    tt = bag_reader.topic_table(a.bag)
    for name, (ty, count) in sorted(tt.items()):
        print(f"  {count:>7}  {name}\n           {ty}")

    has_depth = bag_reader.DEPTH_TOPIC in tt and tt[bag_reader.DEPTH_TOPIC][1] > 0
    print(f"\n  depth present: {has_depth}  "
          f"-> d_contact {'from depth' if has_depth else 'from tape only'}")

    ci = bag_reader.read_camera_info(a.bag)
    print(f"  depth intrinsics: {ci}")

    print("\n" + "=" * 62)
    print("BODIES")
    print("=" * 62)
    counts, per_id_valid, order = {}, {}, []
    valid_all = []
    n_frames = 0
    for f in bag_reader.read_frames(a.bag, max_frames=a.max_frames):
        n_frames += 1
        for b in f.bodies:
            counts[b.id] = counts.get(b.id, 0) + 1
            per_id_valid.setdefault(b.id, []).append(int(b.valid.sum()))
            if not order or order[-1] != b.id:
                order.append(b.id)
        if f.bodies:
            best = max(f.bodies, key=lambda b: b.valid.sum())
            valid_all.append(best.valid)
        else:
            valid_all.append(np.zeros(N_KP, dtype=bool))

    print(f"  frames: {n_frames}")
    for bid, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        kp = np.mean(per_id_valid[bid])
        print(f"  id {bid:>5}: {c:>5} frames ({100*c/n_frames:5.1f}%), "
              f"mean {kp:4.1f}/38 keypoints")

    switches = sum(1 for x, y in zip(order[:-1], order[1:]) if x != y)
    print(f"\n  ID sequence: {order[:12]}{' ...' if len(order) > 12 else ''}")
    print(f"  ID switches: {switches}"
          + ("  <- track breaks, velocity must not difference across these"
             if switches else "  <- stable"))

    print("\n" + "=" * 62)
    print("KEYPOINT YIELD BY REGION (fraction of frames present)")
    print("=" * 62)
    V = np.array(valid_all)
    for name, idx in YIELD_REGIONS.items():
        print(f"  {name:<12} {V[:, idx].mean():6.3f}")
    print(f"  {'all':<12} {V.mean():6.3f}")
    zero = [i for i in range(N_KP) if not V[:, i].any()]
    print(f"\n  never detected: {len(zero)}/38 keypoints -> {zero}")


if __name__ == "__main__":
    main()
