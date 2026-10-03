"""Config loading and the region -> per-keypoint sigma expansion."""
import os
import numpy as np
import yaml
from .types import N_KP, LOWER_LIMB, UPPER_LIMB, TORSO, HEAD, KP_NAMES

EXTREMITY = [i for i, n in enumerate(KP_NAMES)
             if any(t in n for t in ("HAND", "TOE", "HEEL"))]

REGION_INDEX = {
    "head": HEAD,
    "torso": TORSO,
    "upper_limb": [i for i in UPPER_LIMB if i not in EXTREMITY],
    "lower_limb": [i for i in LOWER_LIMB if i not in EXTREMITY],
    "extremity": EXTREMITY,
}

YIELD_REGIONS = {"head": HEAD, "torso": TORSO,
                 "upper_limb": UPPER_LIMB, "lower_limb": LOWER_LIMB}


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def sigma_vector(sensitivity_cfg):
    sig = np.full(N_KP, float(sensitivity_cfg.get("default", 0.7)))
    for region, value in sensitivity_cfg.get("regions", {}).items():
        for i in REGION_INDEX.get(region, []):
            sig[i] = float(value)
    return sig


def region_of(i):
    for name, idx in REGION_INDEX.items():
        if i in idx:
            return name
    return "unknown"


def bed_bounds(cfg):
    b = cfg["bed_bounds"]
    return (tuple(b["x"]), tuple(b["y"]), tuple(b["z"]))


def default_config_dir():
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "config")
