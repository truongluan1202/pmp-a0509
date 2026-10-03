"""Core data types. No ROS imports here."""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

N_KP = 38

# ZED BODY_38 keypoint names, index-ordered.
KP_NAMES = [
    "PELVIS", "SPINE_1", "SPINE_2", "SPINE_3", "NECK", "NOSE",
    "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
    "LEFT_CLAVICLE", "RIGHT_CLAVICLE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_BIG_TOE", "RIGHT_BIG_TOE",
    "LEFT_SMALL_TOE", "RIGHT_SMALL_TOE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_HAND_THUMB_4", "RIGHT_HAND_THUMB_4",
    "LEFT_HAND_INDEX_1", "RIGHT_HAND_INDEX_1",
    "LEFT_HAND_MIDDLE_4", "RIGHT_HAND_MIDDLE_4",
    "LEFT_HAND_PINKY_1", "RIGHT_HAND_PINKY_1",
]
assert len(KP_NAMES) == N_KP

LOWER_LIMB = [i for i, n in enumerate(KP_NAMES) if any(
    t in n for t in ("KNEE", "ANKLE", "TOE", "HEEL"))]
UPPER_LIMB = [i for i, n in enumerate(KP_NAMES) if any(
    t in n for t in ("ELBOW", "WRIST", "HAND"))]
TORSO = [i for i, n in enumerate(KP_NAMES) if any(
    t in n for t in ("PELVIS", "SPINE", "NECK", "CLAVICLE", "SHOULDER", "HIP"))]
HEAD = [i for i, n in enumerate(KP_NAMES) if any(
    t in n for t in ("NOSE", "EYE", "EAR"))]


@dataclass
class Body:
    """One tracked body in one frame."""
    id: int
    keypoints: np.ndarray      # (38,3) world coords, NaN where invalid
    valid: np.ndarray          # (38,) bool
    confidence: np.ndarray     # (38,) float, NaN where unavailable
    tracking_state: int = -1
    label_id: int = -1


@dataclass
class Frame:
    t: float                   # seconds, bag-relative
    bodies: List[Body] = field(default_factory=list)
    t_abs: float = 0.0         # absolute stamp, seconds


@dataclass
class State:
    """Per-keypoint hold state at one instant."""
    t: float
    positions: np.ndarray      # (38,3) last seen position
    ages: np.ndarray           # (38,) seconds since last observation
    d_contact: np.ndarray      # (38,) contact distance, metres
    seen_now: np.ndarray       # (38,) bool
    ever_seen: np.ndarray      # (38,) bool
    subject_id: Optional[int] = None
    id_changed: bool = False


# ---------------------------------------------------------------------------
# Skeleton topology (paper v2). EDGES define the capsule segments of the
# occupancy envelope; CHAINS define the units removed together by structured
# dropout injection. Both are fixed constants of BODY_38, stated here rather
# than inferred from the SDK so the protocol is deterministic and printable.
# ---------------------------------------------------------------------------
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # pelvis .. neck
    (4, 5), (5, 6), (5, 7), (6, 8), (7, 9),    # head
    (4, 10), (4, 11), (10, 12), (11, 13),      # clavicles, shoulders
    (12, 14), (13, 15), (14, 16), (15, 17),    # arms
    (16, 30), (16, 32), (16, 34), (16, 36),    # left hand
    (17, 31), (17, 33), (17, 35), (17, 37),    # right hand
    (0, 18), (0, 19),                          # hips
    (18, 20), (19, 21), (20, 22), (21, 23),    # legs
    (22, 24), (22, 26), (22, 28),              # left foot
    (23, 25), (23, 27), (23, 29),              # right foot
]

# Shoulder-to-hand and hip-to-foot, as stated in the paper.
CHAINS = {
    "left_arm":  [12, 14, 16, 30, 32, 34, 36],
    "right_arm": [13, 15, 17, 31, 33, 35, 37],
    "left_leg":  [18, 20, 22, 24, 26, 28],
    "right_leg": [19, 21, 23, 25, 27, 29],
}
