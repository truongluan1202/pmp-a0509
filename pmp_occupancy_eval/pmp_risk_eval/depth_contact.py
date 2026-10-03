"""Derive d_contact per keypoint from the depth image.

Fixed anthropometric constants do not generalise to real patients, so
d_contact is measured: for each keypoint, project into the depth image, take
the median depth in a small window, and take the difference between the
keypoint's own range and that surface range as the offset from skeleton joint
to skin surface.

Only bags recorded with depth carry this. The 280826 bags do not; the 310826
bags do. Where depth is absent, fall back to tape measurements via
d_contact_from_tape().
"""
import numpy as np
from .types import N_KP


def project(points_cam, fx, fy, cx, cy):
    """(N,3) camera-frame points -> (N,2) pixel coords and (N,) range."""
    p = np.asarray(points_cam, dtype=float)
    z = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * p[:, 0] / z + cx
        v = fy * p[:, 1] / z + cy
    return np.stack([u, v], axis=1), z


def surface_depth(depth_img, uv, win=5):
    """Median finite depth in a (2*win+1) window about each pixel."""
    h, w = depth_img.shape
    out = np.full(len(uv), np.nan)
    for i, (u, v) in enumerate(uv):
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h):
            continue
        patch = depth_img[max(0, vi - win):vi + win + 1,
                          max(0, ui - win):ui + win + 1]
        finite = patch[np.isfinite(patch) & (patch > 0)]
        if finite.size:
            out[i] = float(np.median(finite))
    return out


def accumulate(samples, floor=0.02, ceil=0.25):
    """(F,38) per-frame offsets -> (38,) robust d_contact.

    Median over frames, clipped to a physically plausible band. Keypoints
    never observed fall back to the median of those that were.
    """
    s = np.asarray(samples, dtype=float)
    with np.errstate(all="ignore"):
        d = np.nanmedian(s, axis=0)
    d = np.clip(d, floor, ceil)
    if np.isnan(d).all():
        return np.full(N_KP, 0.08)
    fill = np.nanmedian(d)
    return np.where(np.isfinite(d), d, fill)


def d_contact_from_tape(circumferences):
    """{keypoint_index: circumference_metres} -> (38,) radii.

    Radius = circumference / 2*pi. Unmeasured keypoints take the median of
    those measured. This is the fallback for bags without depth.
    """
    d = np.full(N_KP, np.nan)
    for i, c in circumferences.items():
        d[i] = float(c) / (2.0 * np.pi)
    if np.isnan(d).all():
        return np.full(N_KP, 0.08)
    return np.where(np.isfinite(d), d, np.nanmedian(d))
