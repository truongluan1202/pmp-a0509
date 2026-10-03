#!/usr/bin/env python3
"""Check for Section 3.4: do both endpoint contact radii bound the half-thickness
of the body segment each bone stands for?

Run from the package root:  python3 extra_checks/check_contact_radii.py
Half-thickness follows scripts/tape_to_dcontact.py: half the larger of width and height,
and half the height for chest and hips.
"""
import sys, json
sys.path.insert(0, '.')
import yaml
from pmp_risk_eval.types import KP_NAMES, EDGES

d = json.load(open('config/d_contact_tape.json'))['d_contact']
tape = yaml.safe_load(open('config/tape_measurements.yaml'))['body_cm']


def segment(i, j):
    s = KP_NAMES[i] + ' ' + KP_NAMES[j]
    if 'KNEE' in s and 'HIP' in s: return 'thigh'
    if 'KNEE' in s and 'ANKLE' in s: return 'lower_leg'
    if 'ANKLE' in s: return 'foot'
    if 'SHOULDER' in s and 'ELBOW' in s: return 'upper_arm'
    if 'ELBOW' in s and 'WRIST' in s: return 'lower_arm'
    if 'WRIST' in s: return 'hand'
    if any(k in s for k in ('NOSE', 'EYE', 'EAR')): return 'head'
    if 'PELVIS' in s and 'HIP' in s: return 'hips'
    return 'chest'


fail = 0
print(f"{'bone':34s} {'d_i':>6} {'d_j':>6} {'segment':>10} {'half':>6}  both ends  larger end")
for i, j in EDGES:
    r = segment(i, j); w, h = tape[r]['wide'] / 100, tape[r]['tall'] / 100
    half = h / 2 if r in ('chest', 'hips') else max(w, h) / 2
    both = min(d[i], d[j]) >= half - 1e-9; big = max(d[i], d[j]) >= half - 1e-9
    fail += not both
    print(f"{KP_NAMES[i] + ' - ' + KP_NAMES[j]:34s} {d[i]:6.3f} {d[j]:6.3f} {r:>10} {half:6.3f}  "
          f"{'yes' if both else 'NO':^9}  {'yes' if big else 'NO'}")
print(f"\nboth ends bound the half-thickness on {len(EDGES) - fail} of {len(EDGES)} bones; "
      f"the larger end bounds it on all {len(EDGES)}" if all(
          max(d[i], d[j]) >= (tape[segment(i, j)]['tall'] / 200 if segment(i, j) in ('chest', 'hips')
                              else max(tape[segment(i, j)]['wide'], tape[segment(i, j)]['tall']) / 200) - 1e-9
          for i, j in EDGES) else f"\nboth ends bound it on {len(EDGES) - fail} of {len(EDGES)} bones")
