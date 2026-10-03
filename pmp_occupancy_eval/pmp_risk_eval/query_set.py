"""Query geometry, independent of where the keypoints happen to be.

The baseline's empty-set branch is structurally unreachable because its
queries are anchored to keypoints. Queries here are anchored to the bed, so
the empty set is reachable and the fallback is actually exercised.
"""
import numpy as np


def grid_bounds(bed_bounds, margin=0.0):
    """Selection box widened by a per-axis margin (x, y, z) in metres, for
    the evaluation grid only. Subject selection keeps the unwidened box."""
    m = margin if isinstance(margin, (list, tuple)) else [margin] * 3
    return tuple((lo - float(mi), hi + float(mi))
                 for (lo, hi), mi in zip(bed_bounds, m))


def bed_grid(bed_bounds, res=0.05):
    (x0, x1), (y0, y1), (z0, z1) = bed_bounds
    xs = np.arange(x0, x1 + 1e-9, res)
    ys = np.arange(y0, y1 + 1e-9, res)
    zs = np.arange(z0, z1 + 1e-9, res)
    g = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    return g.reshape(-1, 3)


def sweep_path(bed_bounds, height=None, n=200, axis=0):
    """Straight line across the bed at fixed height: the approach path."""
    (x0, x1), (y0, y1), (z0, z1) = bed_bounds
    z = z1 if height is None else height
    t = np.linspace(0.0, 1.0, n)
    if axis == 0:
        x = x0 + t * (x1 - x0)
        y = np.full(n, 0.5 * (y0 + y1))
    else:
        y = y0 + t * (y1 - y0)
        x = np.full(n, 0.5 * (x0 + x1))
    return np.stack([x, y, np.full(n, z)], axis=1)
