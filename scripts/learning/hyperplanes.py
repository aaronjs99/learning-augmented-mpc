"""SVM-based spatial separating hyperplanes for manta pairwise avoidance."""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC


def get_symmetric_hyperplanes_spatial(
    idx_i: int,
    idx_j: int,
    horizon: int,
    traj_i: list[np.ndarray] | np.ndarray,
    traj_j: list[np.ndarray] | np.ndarray,
    safety_margin: float = 0.3,
    safety_margin_i: float | None = None,
    safety_margin_j: float | None = None,
    ignore_distance: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pairwise linear half-spaces for two stored trajectories."""
    path_i = np.asarray(traj_i, dtype=float)
    path_j = np.asarray(traj_j, dtype=float)
    margin_i = safety_margin if safety_margin_i is None else safety_margin_i
    margin_j = safety_margin if safety_margin_j is None else safety_margin_j
    h_i = np.zeros((horizon, 1), dtype=float)
    h_j = np.zeros((horizon, 1), dtype=float)
    H_i = np.zeros((horizon, 2), dtype=float)
    H_j = np.zeros((horizon, 2), dtype=float)

    for k in range(horizon):
        cur_i = min(idx_i + k, len(path_i) - 1)
        cur_j = min(idx_j + k, len(path_j) - 1)
        pos_i = path_i[cur_i, :2].copy()
        pos_j = path_j[cur_j, :2].copy()

        if np.linalg.norm(pos_i - pos_j) > ignore_distance:
            h_i[k, 0] = -100.0
            h_j[k, 0] = -100.0
            continue

        if np.linalg.norm(pos_i - pos_j) < 1e-4:
            pos_j += np.array([1e-3, 1e-3])

        clf = SVC(kernel="linear", C=1e5)
        clf.fit(np.vstack([pos_i, pos_j]), np.array([1, -1]))

        w = clf.coef_[0]
        b = float(clf.intercept_[0])
        norm_w = max(float(np.linalg.norm(w)), 1e-6)
        w_norm = w / norm_w
        b_norm = b / norm_w

        H_i[k, :] = -w_norm
        h_i[k, 0] = -b_norm + margin_i
        H_j[k, :] = w_norm
        h_j[k, 0] = b_norm + margin_j

    return H_i, h_i, H_j, h_j
