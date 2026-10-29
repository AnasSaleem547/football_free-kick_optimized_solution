"""Background translation estimate used by extraction and tracking."""

import numpy as np


def camera_translation(previous, current):
    """Robust background translation. No claim of full camera calibration."""
    a, b = np.asarray(previous, dtype=float), np.asarray(current, dtype=float)
    if len(a) < 12 or a.shape != b.shape:
        return np.zeros(2), False
    delta = b - a
    keep = np.isfinite(delta).all(axis=1)
    if keep.sum() < 12:
        return np.zeros(2), False
    median = np.median(delta[keep], axis=0)
    err = np.linalg.norm(delta - median, axis=1)
    inliers = keep & (err < 2.5)
    if inliers.sum() < max(12, 0.55 * len(a)):
        return np.zeros(2), False
    return np.median(delta[inliers], axis=0), True
