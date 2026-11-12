"""Hand-object penetration and temporal-smoothness metrics.

Units: millimetres and seconds throughout.

Penetration: signed distance (generalized winding number sign, robust to
non-watertight scans) from a 1x-subdivided MANO surface (~3.1k points,
~2 mm spacing) to the object. Articulated objects are a union of rigid parts
(SDF of a union = min over parts).

Smoothness: the hand is expressed in the object frame (removes camera/rig
motion), then each vertex trajectory is split into a low-pass part (<LP_HZ)
and a residual. The residual is treated as noise; real hand-object motion
during manipulation is far below 6 Hz. Also reports finite-difference
acceleration and world-frame jitter of the hand and object separately to
attribute the noise.
"""
from dataclasses import dataclass, field

import igl
import numpy as np
import scipy.sparse as sp
from scipy.signal import butter, filtfilt

PAD_MM = 30.0          # points farther than this from a part's AABB are not queried
CONTACT_MM = 5.0       # "in contact" if the closest hand point is within this
LP_HZ = 6.0            # low-pass cutoff separating motion from jitter
MIN_RUN = 16           # min contiguous frames to evaluate smoothness
SDF_TYPE = igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER


# ----------------------------------------------------------------------------- geometry
def subdivision(faces, nv):
    """Midpoint subdivision. Returns S (nv+ne, nv) sparse and the new faces."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1)
    edges, inv = np.unique(e, axis=0, return_inverse=True)
    ne = len(edges)
    rows = np.concatenate([np.arange(nv), nv + np.arange(ne), nv + np.arange(ne)])
    cols = np.concatenate([np.arange(nv), edges[:, 0], edges[:, 1]])
    vals = np.concatenate([np.ones(nv), np.full(2 * ne, 0.5)])
    S = sp.csr_matrix((vals, (rows, cols)), shape=(nv + ne, nv))
    nf = len(faces)
    m01, m12, m20 = (nv + inv[i * nf:(i + 1) * nf] for i in range(3))
    a, b, c = faces.T
    F = np.concatenate([np.stack([a, m01, m20], 1), np.stack([b, m12, m01], 1),
                        np.stack([c, m20, m12], 1), np.stack([m01, m12, m20], 1)])
    return S, F


def vertex_areas(V, F):
    """V: (N,P,3), F: (nf,3) -> (N,P) one-third of adjacent triangle area."""
    a = 0.5 * np.linalg.norm(np.cross(V[:, F[:, 1]] - V[:, F[:, 0]], V[:, F[:, 2]] - V[:, F[:, 0]]), axis=-1)
    inc = sp.csr_matrix((np.ones(F.size) / 3.0, (F.ravel(), np.repeat(np.arange(len(F)), 3))),
                        shape=(V.shape[1], len(F)))
    return (inc @ a.T).T


def apply_T(T, X):
    """T: (N,4,4), X: (N,P,3)."""
    return np.einsum("nij,npj->npi", T[:, :3, :3], X) + T[:, None, :3, 3]


def inv_T(T):
    Ti = np.zeros_like(T)
    R = np.swapaxes(T[:, :3, :3], 1, 2)
    Ti[:, :3, :3] = R
    Ti[:, :3, 3] = -np.einsum("nij,nj->ni", R, T[:, :3, 3])
    Ti[:, 3, 3] = 1
    return Ti


@dataclass
class Part:
    V: np.ndarray  # (nv,3) mm, part-local frame
    F: np.ndarray

    def __post_init__(self):
        self.V = np.ascontiguousarray(self.V, np.float64)
        self.F = np.ascontiguousarray(self.F, np.int64)
        self.lo = self.V.min(0) - PAD_MM
        self.hi = self.V.max(0) + PAD_MM


def object_sdf(parts, part_T, P_obj):
    """parts: list[Part]; part_T: (N,K,4,4) object-base -> part-local; P_obj: (N,P,3) in object base frame.
    Returns (N,P) signed distance in mm (negative = inside), clipped to +PAD_MM when far."""
    N, Pn, _ = P_obj.shape
    out = np.full((N, Pn), PAD_MM)
    for k, part in enumerate(parts):
        Q = apply_T(part_T[:, k], P_obj) if part_T is not None else P_obj
        near = np.all((Q > part.lo) & (Q < part.hi), axis=-1)
        if not near.any():
            continue
        s, *_ = igl.signed_distance(np.ascontiguousarray(Q[near], np.float64), part.V, part.F, sign_type=SDF_TYPE)
        out[near] = np.minimum(out[near], s)
    return np.minimum(out, PAD_MM)


@dataclass
class Articulated:
    """ARCTIC-style object: one watertight mesh whose `top` vertices rotate about -z by `angle`
    (exactly how ARCTIC's ObjectTensors poses it; hinge-spanning faces stretch with it)."""
    V: np.ndarray       # (nv,3) mm, base frame at angle 0
    F: np.ndarray
    top: np.ndarray     # (nv,) bool
    QUANT = np.deg2rad(0.05)

    def mesh_at(self, angle):
        from mano_np import rodrigues
        R = rodrigues(np.array([0, 0, -angle]))
        V = self.V.copy()
        V[self.top] = V[self.top] @ R.T
        return np.ascontiguousarray(V), np.ascontiguousarray(self.F, np.int64)


def articulated_sdf(art: Articulated, angles, P_obj):
    N, Pn, _ = P_obj.shape
    out = np.full((N, Pn), PAD_MM)
    q = np.round(angles / art.QUANT).astype(np.int64)
    for qa in np.unique(q):
        fi = np.where(q == qa)[0]
        V, F = art.mesh_at(qa * art.QUANT)
        lo, hi = V.min(0) - PAD_MM, V.max(0) + PAD_MM
        Q = P_obj[fi]
        near = np.all((Q > lo) & (Q < hi), axis=-1)
        if not near.any():
            continue
        s, *_ = igl.signed_distance(np.ascontiguousarray(Q[near], np.float64), V, F, sign_type=SDF_TYPE)
        sub = out[fi]
        sub[near] = np.minimum(s, PAD_MM)
        out[fi] = sub
    return out


# ----------------------------------------------------------------------------- per-frame penetration
class HandSurface:
    def __init__(self, faces, nv=None):
        self.faces = faces
        self.nv = int(nv or faces.max() + 1)  # 778 for MANO, 788 for UmeTrack
        self.S, self.F_dense = subdivision(faces, self.nv)

    def dense(self, V):
        """V: (N,nv,3) -> (N,P,3)."""
        N = V.shape[0]
        X = self.S @ V.transpose(1, 0, 2).reshape(self.nv, N * 3)
        return X.reshape(-1, N, 3).transpose(1, 0, 2)


def penetration_metrics(hs: HandSurface, hand_obj, parts, part_T, chunk=400):
    """hand_obj: (N,778,3) hand vertices in object base frame (mm)."""
    N = hand_obj.shape[0]
    res = {k: np.zeros(N) for k in ["min_sd_mm", "pen_max_mm", "pen_vol_mm3", "n_pen", "n_pen_2mm", "n_pen_5mm",
                                     "pen_area_mm2"]}
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        D = hs.dense(hand_obj[s:e])
        if isinstance(parts, Articulated):
            sd = articulated_sdf(parts, part_T[s:e], D)  # part_T carries the per-frame angles
        else:
            sd = object_sdf(parts, None if part_T is None else part_T[s:e], D)
        A = vertex_areas(D, hs.F_dense)
        depth = np.clip(-sd, 0, None)
        res["min_sd_mm"][s:e] = sd.min(1)
        res["pen_max_mm"][s:e] = depth.max(1)
        res["pen_vol_mm3"][s:e] = (depth * A).sum(1)
        res["pen_area_mm2"][s:e] = (A * (sd < 0)).sum(1)
        res["n_pen"][s:e] = (sd < 0).sum(1)
        res["n_pen_2mm"][s:e] = (sd < -2).sum(1)
        res["n_pen_5mm"][s:e] = (sd < -5).sum(1)
    res["contact"] = res["min_sd_mm"] < CONTACT_MM
    return res


# ----------------------------------------------------------------------------- temporal smoothness
def runs(frame_idx, valid, step):
    """Contiguous runs of valid frames whose index gap == step."""
    ok = valid.copy()
    out, start = [], None
    for i in range(len(frame_idx)):
        if ok[i] and start is not None and frame_idx[i] - frame_idx[i - 1] == step:
            continue
        if start is not None:
            out.append((start, i))
        start = i if ok[i] else None
    if start is not None:
        out.append((start, len(frame_idx)))
    return [(a, b) for a, b in out if b - a >= MIN_RUN]


def jitter(X, frame_idx, valid, fps, step):
    """X: (N,P,3) mm. Returns per-frame (hf_rms_mm, acc_m_s2), NaN where not evaluable.
    hf_rms_mm: mean over points of |x - lowpass(x)|.
    acc_m_s2: mean over points of |second difference| / dt^2."""
    N = X.shape[0]
    hf = np.full(N, np.nan)
    acc = np.full(N, np.nan)
    dt = step / fps
    fs = 1.0 / dt
    if fs / 2 <= 1.5 * LP_HZ:  # sampled too sparsely to separate motion from jitter
        return hf, acc
    b, a = butter(4, LP_HZ / (fs / 2))
    for s, e in runs(frame_idx, valid, step):
        seg = X[s:e]
        flat = seg.reshape(e - s, -1)
        low = filtfilt(b, a, flat, axis=0, padlen=min(3 * max(len(a), len(b)), e - s - 1))
        r = (flat - low).reshape(seg.shape)
        hf[s:e] = np.linalg.norm(r, axis=-1).mean(1)
        d2 = seg[2:] - 2 * seg[1:-1] + seg[:-2]
        acc[s + 1:e - 1] = np.linalg.norm(d2, axis=-1).mean(1) / dt ** 2 / 1000.0
    return hf, acc


@dataclass
class Sequence:
    dataset: str
    name: str
    fps: float                       # rate of frame_idx (Hz)
    step: int                        # expected frame_idx gap between consecutive samples
    frame_idx: np.ndarray            # (N,)
    T_wo: np.ndarray                 # (N,4,4) world <- object base (mm)
    parts: list                      # list[Part]
    part_T: np.ndarray | None        # (N,K,4,4) object base -> part local, None if rigid single part
    hands: dict                      # side -> (verts_world (N,778,3) mm, valid (N,) bool)
    extra: dict = field(default_factory=dict)  # per-frame columns, side-specific keys "<side>:<name>"


def evaluate(seq: Sequence, hs: HandSurface):
    import pandas as pd
    rows = []
    T_ow = inv_T(seq.T_wo)
    # object world jitter: 8 corners of the (base) object's bbox
    Vb = seq.parts.V if isinstance(seq.parts, Articulated) else seq.parts[0].V
    lo, hi = Vb.min(0), Vb.max(0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    obj_world = apply_T(seq.T_wo, np.broadcast_to(corners, (len(seq.frame_idx), 8, 3)))
    for side, (Vw, valid) in seq.hands.items():
        valid = valid & np.isfinite(Vw).all((1, 2)) & np.isfinite(seq.T_wo).all((1, 2))
        if not valid.any():
            continue
        idx = np.where(valid)[0]
        Vo = apply_T(T_ow[idx], Vw[idx])
        pm = penetration_metrics(hs, Vo, seq.parts, None if seq.part_T is None else seq.part_T[idx])
        Vo_full = np.full_like(Vw, np.nan)
        Vo_full[idx] = Vo
        rel_hf, rel_acc = jitter(Vo_full, seq.frame_idx, valid, seq.fps, seq.step)
        hand_hf, hand_acc = jitter(Vw, seq.frame_idx, valid, seq.fps, seq.step)
        obj_hf, obj_acc = jitter(obj_world, seq.frame_idx, valid, seq.fps, seq.step)
        df = pd.DataFrame({
            "dataset": seq.dataset, "seq": seq.name, "hand": side,
            "frame": seq.frame_idx[idx], **{k: v for k, v in pm.items()},
            "rel_hf_mm": rel_hf[idx], "rel_acc_ms2": rel_acc[idx],
            "hand_world_hf_mm": hand_hf[idx], "obj_world_hf_mm": obj_hf[idx],
            "hand_world_acc_ms2": hand_acc[idx], "obj_world_acc_ms2": obj_acc[idx],
        })
        for k, v in seq.extra.items():
            if ":" in k:
                s, name = k.split(":", 1)
                if s == side:
                    df[name] = np.asarray(v)[idx]
            else:
                df[k] = np.asarray(v)[idx]
        # frame-to-frame change in penetration depth between consecutive samples
        f = df["frame"].to_numpy()
        dpen = np.full(len(df), np.nan)
        cons = np.diff(f) == seq.step
        dpen[1:][cons] = np.abs(np.diff(df["pen_max_mm"].to_numpy()))[cons]
        df["dpen_mm"] = dpen
        rows.append(df)
    return pd.concat(rows) if rows else None
