#!/usr/bin/env python3
"""Estimate v_hat_max from the fast-motion bag.

    python3 scripts/run_velocity.py --bag ~/mnt/rosbag/310826_fastmotion_bare

Write the p95 into config/eval_params.yaml as estimator.v_hat_max.
Then check a_max <= D / v_hat_max, with D the A0509 stopping distance.
"""
import _boot  # noqa
import argparse
import json
import os
from pmp_risk_eval import bag_reader, config_io
from pmp_risk_eval.subject_filter import SubjectFilter
from pmp_risk_eval.velocity_estimator import speeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--subject-id", type=int, default=None,
                    help="use fixed-ID selection instead of bed volume")
    ap.add_argument("--out", default="velocity.json")
    a = ap.parse_args()

    cfg_path = a.config or os.path.join(config_io.default_config_dir(),
                                        "eval_params.yaml")
    cfg = config_io.load(cfg_path)

    sf = (SubjectFilter("fixed", subject_id=a.subject_id)
          if a.subject_id is not None
          else SubjectFilter("bed", bed_bounds=config_io.bed_bounds(cfg)))

    res = speeds(bag_reader.read_frames(a.bag), sf, per_keypoint=True)
    res["id_changes"] = sf.report()["n_id_changes"]

    print(json.dumps({k: v for k, v in res.items()
                      if k != "per_keypoint_p95"}, indent=2))

    if res["n_samples"] == 0:
        print("\nNO SAMPLES. The bed volume in eval_params.yaml is almost "
              "certainly wrong. Re-run with --subject-id from check_bag.py.")
        return

    print(f"\n  v_hat_max (p95) = {res['v_p95']:.3f} m/s")
    print(f"  set estimator.v_hat_max to this in {cfg_path}")
    D = 0.20
    print(f"  then a_max <= D / v_hat_max = {D:.2f} / {res['v_p95']:.3f} "
          f"= {D / res['v_p95']:.2f} s   (D is a placeholder, "
          f"use the A0509 datasheet figure)")

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
