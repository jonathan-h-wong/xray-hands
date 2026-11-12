"""Minimal NumPy MANO (matches smplx.MANO with use_pca=False).

Loads the official MANO_{LEFT,RIGHT}.pkl without chumpy installed.
No left-hand shapedirs "fix" is applied: ARCTIC and SHOW3D both pose with
plain smplx.MANO, so the stock model reproduces their meshes.
"""
import pickle

import numpy as np
import scipy.sparse as sp


class _Ch:
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"_s": state})


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Ch
        return super().find_class(module, name)


def _resolve(x):
    if isinstance(x, _Ch):
        d = x.__dict__
        if "x" in d:
            return np.asarray(d["x"])
        if "a" in d and "idxs" in d:
            return _resolve(d["a"]).ravel()[d["idxs"]].reshape(d["preferred_shape"])
        raise ValueError(f"Unknown chumpy object: {list(d)}")
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def rodrigues(aa):
    """aa: (..., 3) axis-angle -> (..., 3, 3). Same epsilon handling as smplx."""
    angle = np.linalg.norm(aa + 1e-8, axis=-1, keepdims=True)
    axis = aa / angle
    s, c = np.sin(angle)[..., None], np.cos(angle)[..., None]
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = np.zeros_like(x)
    K = np.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], -1).reshape(aa.shape[:-1] + (3, 3))
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + s * K + (1 - c) * (K @ K)


class MANO:
    """Call with a 45-d axis-angle hand_pose; for PCA poses use pca_to_pose(coeffs) first (HOT3D: 15 comps)."""
    def __init__(self, pkl_path, flat_hand_mean=False):
        with open(pkl_path, "rb") as f:
            d = _Unpickler(f, encoding="latin1").load()
        self.v_template = _resolve(d["v_template"]).astype(np.float64)
        self.shapedirs = _resolve(d["shapedirs"]).astype(np.float64)  # (778,3,10)
        self.posedirs = _resolve(d["posedirs"]).reshape(-1, 135).T.astype(np.float64)  # (135, 2334)
        self.J_regressor = _resolve(d["J_regressor"]).astype(np.float64)  # (16,778)
        self.weights = _resolve(d["weights"]).astype(np.float64)  # (778,16)
        self.faces = _resolve(d["f"]).astype(np.int64)
        parents = _resolve(d["kintree_table"])[0].astype(np.int64)
        parents[0] = -1
        self.parents = parents
        self.hands_components = _resolve(d["hands_components"]).astype(np.float64)  # (45,45) PCA basis rows
        hm = _resolve(d["hands_mean"]).astype(np.float64)
        self.pose_mean = np.concatenate([np.zeros(3), np.zeros(45) if flat_hand_mean else hm])

    def pca_to_pose(self, coeffs):
        """(N,k) PCA coefficients -> (N,45) axis-angle, as smplx.MANO(use_pca=True) (mean added later)."""
        coeffs = np.asarray(coeffs, np.float64)
        return coeffs @ self.hands_components[: coeffs.shape[-1]]

    def __call__(self, betas, global_orient, hand_pose, transl=None, chunk=512):
        n = global_orient.shape[0]
        betas = np.broadcast_to(np.asarray(betas, np.float64).reshape(-1, 10), (n, 10))
        out = np.empty((n, 778, 3))
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            out[s:e] = self._forward(betas[s:e], global_orient[s:e], hand_pose[s:e])
        if transl is not None:
            out += np.asarray(transl, np.float64)[:, None, :]
        return out

    def _forward(self, betas, go, hp):
        b = go.shape[0]
        full_pose = np.concatenate([go, hp], 1) + self.pose_mean  # (b,48)
        v_shaped = self.v_template + np.einsum("vck,bk->bvc", self.shapedirs, betas)
        J = np.einsum("jv,bvc->bjc", self.J_regressor, v_shaped)  # (b,16,3)
        R = rodrigues(full_pose.reshape(b, 16, 3))  # (b,16,3,3)
        feat = (R[:, 1:] - np.eye(3)).reshape(b, 135)
        v_posed = v_shaped + (feat @ self.posedirs).reshape(b, 778, 3)

        rel_J = J.copy()
        rel_J[:, 1:] -= J[:, self.parents[1:]]
        T = np.zeros((b, 16, 4, 4))
        T[:, :, :3, :3] = R
        T[:, :, :3, 3] = rel_J
        T[:, :, 3, 3] = 1
        chain = [T[:, 0]]
        for i in range(1, 16):
            chain.append(chain[self.parents[i]] @ T[:, i])
        A = np.stack(chain, 1)  # (b,16,4,4)
        A[:, :, :3, 3] -= np.einsum("bjik,bjk->bji", A[:, :, :3, :3], J)
        W = np.einsum("vj,bjik->bvik", self.weights, A[:, :, :3, :])  # (b,778,3,4)
        return np.einsum("bvik,bvk->bvi", W[..., :3], v_posed) + W[..., 3]
