#!/usr/bin/env python3
"""Tape widths and heights -> per-keypoint d_contact JSON for --d-contact.
Single-line segments: d = max(w, h)/2. Torso (chest, hips): d = h/2, since
paired left/right keypoints already span the width."""
import _boot  # noqa
import json, os, yaml
from pmp_risk_eval.types import KP_NAMES, N_KP

REGION_KP = {
    "head": ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR"],
    "neck": ["NECK"],
    "chest": ["SPINE_1", "SPINE_2", "SPINE_3", "LEFT_CLAVICLE", "RIGHT_CLAVICLE"],
    "hips": ["PELVIS", "LEFT_HIP", "RIGHT_HIP"],
    "upper_arm": ["LEFT_SHOULDER", "RIGHT_SHOULDER"],
    "lower_arm": ["LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST"],
    "hand": [n for n in KP_NAMES if "HAND" in n],
    "thigh": ["LEFT_KNEE", "RIGHT_KNEE"],
    "lower_leg": ["LEFT_ANKLE", "RIGHT_ANKLE"],
    "foot": [n for n in KP_NAMES if "TOE" in n or "HEEL" in n],
}
TORSO = {"chest", "hips"}
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
body = yaml.safe_load(open(os.path.join(here, "config", "tape_measurements.yaml")))["body_cm"]
radius = {}
for r, m in body.items():
    w, h = float(m["wide"]) / 100, float(m["tall"]) / 100
    radius[r] = round(h / 2 if r in TORSO else max(w, h) / 2, 4)
d = [None] * N_KP
for r, names in REGION_KP.items():
    for n in names:
        d[KP_NAMES.index(n)] = radius[r]
assert all(v is not None for v in d)
out = os.path.join(here, "config", "d_contact_tape.json")
json.dump({"d_contact": d, "source": "tape_measurements.yaml", "regions_m": radius}, open(out, "w"), indent=1)
for k, v in radius.items():
    print(f"  {k:<10} {v:.3f} m")
print(f"wrote {out}")
