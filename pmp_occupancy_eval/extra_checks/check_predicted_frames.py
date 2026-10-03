#!/usr/bin/env python3
"""Check for Section 3.1: do SEARCHING or predicted outputs pass the 'seen' test?

Run from the package root:  python3 extra_checks/check_predicted_frames.py

What it does
  1. For the selected subject, counts frames by SDK tracking state
     (0 OFF, 1 OK, 2 SEARCHING, 3 TERMINATE) and the seen keypoint readings in each.
  2. Shows that in the 0.5 s before any body switches to SEARCHING the SDK repeats the
     previous confidence values exactly (the prediction window), so such frames can be
     recognised by a confidence vector identical to the previous frame's.
  3. Counts those predicted frames for the subject, and the seen readings in them.
"""
import sys, glob, os, collections
sys.path.insert(0, '.')
import numpy as np
from pmp_risk_eval import npz_corpus, config_io, bag_reader
from pmp_risk_eval.subject_filter import SubjectFilter

cfg = config_io.load('config/eval_params_final.yaml'); pp = cfg['paper']; bb = config_io.bed_bounds(cfg)


def frozen(b0, b1):
    v = b0.valid & b1.valid
    return bool(v.any() and (b0.valid == b1.valid).all()
                and np.array_equal(b0.confidence[v], b1.confidence[v]))


T = collections.Counter()
for path in sorted(glob.glob('npz_replay_trim/*.npz')):
    name = os.path.basename(path)[7:-4]
    frames = list(npz_corpus.read_frames_npz(path))
    series = collections.defaultdict(list)
    for f in frames:
        for b in f.bodies:
            series[b.id].append(b)
    pre_tot = pre_frozen = 0
    for rows in series.values():
        for a, b in enumerate(rows):
            if b.tracking_state == 2 and a > 0 and rows[a - 1].tracking_state == 1:
                win = rows[max(0, a - 15):a]
                for b0, b1 in zip(win[:-1], win[1:]):
                    pre_tot += 1; pre_frozen += frozen(b0, b1)
    sf = SubjectFilter('bed', bed_bounds=bb, hysteresis=pp['hysteresis'], seed=pp.get('subject_seed'))
    prev = None; n = pred = seen = seen_pred = 0; states = collections.Counter(); seen_not_ok = 0; bad = 0
    for f in frames:
        body, _ = sf.select(f)
        if body is None:
            prev = None; continue
        n += 1; states[body.tracking_state] += 1
        obs = bag_reader.observed_mask(body, pp['kp_conf_min']); seen += int(obs.sum())
        bad += int((obs & ~(np.isfinite(body.keypoints).all(axis=1) & np.isfinite(body.confidence))).sum())
        if body.tracking_state != 1: seen_not_ok += int(obs.sum())
        if prev is not None and prev.id == body.id and frozen(prev, body):
            pred += 1; seen_pred += int(obs.sum())
        prev = body
    print(f"{name:26s} subject frames {n:5d} by state {dict(sorted(states.items()))} | seen readings {seen} "
          f"(non-finite: {bad}; in frames not OK: {seen_not_ok}) | pairs before SEARCHING with repeated confidence "
          f"{pre_frozen}/{pre_tot} | predicted frames {pred} ({100*pred/n:.2f}%), seen readings in them {seen_pred} ({100*seen_pred/seen:.2f}%)")
    T['n'] += n; T['pred'] += pred; T['seen'] += seen; T['seen_pred'] += seen_pred; T['not_ok'] += seen_not_ok
    T['pre'] += pre_tot; T['pre_f'] += pre_frozen; T['searching'] += states[2]
print(f"\nALL: subject frames {T['n']}, of them SEARCHING {T['searching']}; predicted frames {T['pred']} "
      f"({100*T['pred']/T['n']:.2f}%); seen readings {T['seen']}, in predicted frames {T['seen_pred']} "
      f"({100*T['seen_pred']/T['seen']:.2f}%), in frames not OK {T['not_ok']}; signature check {T['pre_f']}/{T['pre']}")
