"""NumPy UmeTrack hand skinning (port of facebookresearch/hand_tracking_toolkit umetrack_skinning.py).

model: the "umetrack" dict from HOT3D's __hand_shapes.json__ (rest positions in mm).
Outputs are in metres, like the toolkit.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _skin_mats(model, angles, T_wrist, right):
    """angles (N,22), T_wrist (N,4,4) -> (N,17,4,4) skinning transforms."""
    T_wrist = np.array(T_wrist, np.float64)
    if right:
        T_wrist[:, :, 0] *= -1
    N = len(angles)
    axes = np.asarray(model["joint_rotation_axes"], np.float64)[:20]
    rest = np.asarray(model["joint_rest_positions"], np.float64)[:20] * 1e-3
    ang = np.asarray(angles, np.float64)[:, :20]
    R = Rotation.from_rotvec((axes[None] * ang[..., None]).reshape(-1, 3)).as_matrix().reshape(N, 20, 3, 3)
    L = np.zeros((N, 20, 4, 4))
    L[..., :3, :3] = R
    L[..., :3, 3] = rest[None] - np.einsum("nkij,kj->nki", R, rest)
    L[..., 3, 3] = 1
    mats = [T_wrist, T_wrist]
    for f in range(5):
        cur = T_wrist
        for i in range(4):
            cur = cur @ L[:, 4 * f + i]
            if i >= 1:
                mats.append(cur)
    return np.stack(mats, 1)


def _apply(M, W, pts):
    ph = np.c_[pts, np.ones(len(pts))]
    A = np.einsum("vj,njab->nvab", W, M[:, :, :3, :])  # (N,V,3,4)
    return np.einsum("nvab,vb->nva", A, ph)


def skin_vertices(model, angles, T_wrist, right):
    """-> (N,788,3) metres."""
    M = _skin_mats(model, angles, T_wrist, right)
    W = np.asarray(model["dense_bone_weights"], np.float64)
    return _apply(M, W, np.asarray(model["mesh_vertices"], np.float64) * 1e-3)


def skin_landmarks(model, angles, T_wrist, right):
    """-> (N,21,3) metres; canonical 0-4 are fingertips thumb..pinky."""
    M = _skin_mats(model, angles, T_wrist, right)
    pts = np.asarray(model["landmark_rest_positions"], np.float64) * 1e-3
    idx = np.asarray(model["landmark_rest_bone_indices"]).astype(int)
    w = np.asarray(model["landmark_rest_bone_weights"], np.float64)
    W = np.zeros((len(pts), 17))
    for k in range(idx.shape[1]):
        np.add.at(W, (np.arange(len(pts)), idx[:, k]), w[:, k])
    return _apply(M, W, pts)


def faces(model, right):
    F = np.asarray(model["mesh_triangles"]).astype(np.int64)
    return F[:, ::-1].copy() if right else F
