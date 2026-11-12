"""Shared ARCTIC plumbing for the baseline evaluation (HaMeR, WiLoR, HOPformer).

Numpy/OpenCV only, so it imports cleanly inside each model's own Python environment.

Conventions
-----------
- Ground truth comes from ARCTIC raw_seqs (MANO in metres, object translation in mm) and is moved into the
  camera of the requested view: view 0 = egocentric (per-frame extrinsics + distortion), 1..8 = static rigs.
- All camera-space 3D in this package is in METRES; penetration is computed in MILLIMETRES with the same
  code as analysis/run_partwise.py (subdivided MANO surface, winding-number signed distance, 12 regions).
- Images: the egocentric view is read from ARCTIC `cropped_images` (the full ego frame downscaled by 0.3, no
  crop) by default; static views need full `images`. The intrinsics of the image actually fed to a model are
  K_full scaled by the image scale.
- Ego frames are radially distorted. 2D methods are fed an undistorted copy (so the pinhole lift is valid);
  HOPformer is fed the raw distorted frame, exactly as it was trained. ARCTIC GT lives in undistorted space.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

ARCTIC = Path(os.environ.get("ARCTIC_DATA", ROOT / "data/arctic/unpack/arctic_data/data"))
MANO_DIR = Path(os.environ.get("MANO_DIR", ROOT / "data/arctic/unpack/body_models/mano_v1_2/models"))
EGO_SCALE = 0.3          # ARCTIC cropped_images: egocentric frames are the full frame resized by 0.3
TIPS = [744, 320, 443, 554, 671]   # thumb, index, middle, ring, pinky tip vertices (smplx / ARCTIC convention)
SIDES = ("left", "right")


# ----------------------------------------------------------------------------- eval set
def load_eval_set(path):
    """-> dict of arrays (one row per image) + meta dict. Written by make_eval_set.py."""
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    meta = json.loads(str(z["meta"]))
    return d, meta


def image_relpath(sid, seq, view, frame, ioi_offset, kind):
    """kind: 'cropped' (cropped_images) or 'full' (images)."""
    sub = "cropped_images" if kind == "cropped" else "images"
    return f"{sub}/{sid}/{seq}/{view}/{int(frame) + int(ioi_offset):05d}.jpg"


def image_scale(view, kind):
    if kind == "cropped":
        if int(view) != 0:
            raise ValueError("cropped_images for static views are object-centred crops; use --image-kind full")
        return EGO_SCALE
    return 1.0


# ----------------------------------------------------------------------------- cameras
_MISC = None


def misc():
    global _MISC
    if _MISC is None:
        _MISC = json.load(open(ARCTIC / "meta/misc.json"))
    return _MISC


def camera(sid, seq, view):
    """Full-resolution camera of a view.
    -> dict(K (3,3), dist (8,) or None, world2cam (T,4,4) metres, size (W,H), ioi_offset)."""
    m = misc()[sid]
    if int(view) == 0:
        e = np.load(ARCTIC / f"raw_seqs/{sid}/{seq}.egocam.dist.npy", allow_pickle=True).item()
        R = np.asarray(e["R_k_cam_np"], np.float64)
        t = np.asarray(e["T_k_cam_np"], np.float64).reshape(-1, 3)
        W2C = np.zeros((len(R), 4, 4))
        W2C[:, :3, :3], W2C[:, :3, 3], W2C[:, 3, 3] = R, t, 1
        K, dist = np.asarray(e["intrinsics"], np.float64), np.asarray(e["dist8"], np.float64)
    else:
        W2C = np.asarray(m["world2cam"][int(view) - 1], np.float64)[None]
        K, dist = np.asarray(m["intris_mat"][int(view) - 1], np.float64), None
    return {"K": K, "dist": dist, "world2cam": W2C, "size": tuple(m["image_size"][int(view)]),
            "ioi_offset": int(m["ioi_offset"])}


def scale_K(K, s):
    K = np.array(K, np.float64)
    K[:2] *= s
    return K


def project(K, X):
    """X (...,3) camera metres -> (...,2) pixels."""
    uvw = X @ K.T
    return uvw[..., :2] / uvw[..., 2:3]


def undistort(img, K, dist):
    """OpenCV rational model; ARCTIC's dist8 is in OpenCV order (k1,k2,p1,p2,k3,k4,k5,k6)."""
    import cv2
    return cv2.undistort(img, K, dist, None, K)


# ----------------------------------------------------------------------------- MANO / GT
_MANO = {}


def mano(side):
    from mano_np import MANO
    if side not in _MANO:
        _MANO[side] = MANO(MANO_DIR / f"MANO_{side.upper()}.pkl", flat_hand_mean=False)
    return _MANO[side]


def joints21(V, side):
    """(N,778,3) -> (N,21,3): 16 MANO joints regressed from the posed mesh + 5 fingertips.
    Applied identically to GT and to every method's mesh, so MPJPE does not depend on each repo's joint order."""
    J = np.einsum("jv,nvc->njc", mano(side).J_regressor, V)
    return np.concatenate([J, V[:, TIPS]], 1)


def gt_sequence(sid, seq, view):
    """Ground truth for one sequence in the camera of `view`.
    -> dict(verts{side}: (T,778,3) m, joints{side}: (T,21,3) m, T_co (T,4,4) mm, angle (T,), obj name, cam)."""
    from mano_np import rodrigues
    mp = np.load(ARCTIC / f"raw_seqs/{sid}/{seq}.mano.npy", allow_pickle=True).item()
    o = np.load(ARCTIC / f"raw_seqs/{sid}/{seq}.object.npy", allow_pickle=True)
    cam = camera(sid, seq, view)
    T = len(o)
    W2C = cam["world2cam"] if len(cam["world2cam"]) == T else np.repeat(cam["world2cam"], T, 0)
    out = {"verts": {}, "joints": {}, "cam": cam, "obj": seq.split("_")[0], "angle": o[:, 0].astype(np.float64)}
    for side in SIDES:
        h = mp[side]
        Vw = mano(side)(h["shape"], h["rot"], h["pose"], h["trans"])          # ARCTIC uses the stock left MANO
        Vc = np.einsum("nij,nvj->nvi", W2C[:, :3, :3], Vw) + W2C[:, None, :3, 3]
        out["verts"][side] = Vc
        out["joints"][side] = joints21(Vc, side)
    T_wo = np.zeros((T, 4, 4))
    T_wo[:, :3, :3], T_wo[:, :3, 3], T_wo[:, 3, 3] = rodrigues(o[:, 1:4]), o[:, 4:7], 1   # mm
    W2C_mm = W2C.copy()
    W2C_mm[:, :3, 3] *= 1000.0
    out["T_co"] = W2C_mm @ T_wo
    return out


# ----------------------------------------------------------------------------- lifting (2D methods)
def lift_weak_perspective(pred_cam, box_center, box_size, K, crop_focal, crop_res):
    """HaMeR/WiLoR crop camera (s, tx, ty) -> full-image camera translation with real intrinsics.

    Same derivation as HaMeR's cam_crop_to_full, but with the true focal and principal point instead of
    focal = 5000/256 * max(W,H) and the image centre. (crop_focal/crop_res cancel; kept for clarity.)
    """
    f = 0.5 * (K[..., 0, 0] + K[..., 1, 1])
    cx, cy = K[..., 0, 2], K[..., 1, 2]
    bs = box_size * pred_cam[..., 0] + 1e-9
    tz = 2.0 * f / bs
    tx = pred_cam[..., 1] + 2.0 * (box_center[..., 0] - cx) / bs
    ty = pred_cam[..., 2] + 2.0 * (box_center[..., 1] - cy) / bs
    return np.stack([tx, ty, tz], -1)


def lift_fit2d(J_model, kp2d_full, K, t0=None):
    """Translation t minimising the full-perspective reprojection error of model-frame joints J (21,3) against
    the model's own 2D keypoints (21,2, full-image px). Linear least squares (exact for a pinhole camera)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = kp2d_full[:, 0], kp2d_full[:, 1]
    X, Y, Z = J_model.T
    n = len(u)
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    A[:n, 0], A[:n, 2], b[:n] = fx, cx - u, u * Z - fx * X - cx * Z
    A[n:, 1], A[n:, 2], b[n:] = fy, cy - v, v * Z - fy * Y - cy * Z
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    if t[2] + Z.mean() <= 0 and t0 is not None:   # degenerate (behind camera): keep the weak-perspective lift
        return t0
    return t


# ----------------------------------------------------------------------------- penetration per part
class PartPenetration:
    """Per-frame x hand x region penetration, identical in definition to analysis/run_partwise.py."""

    def __init__(self):
        import hoi_metrics as hm
        self.hm = hm
        hm.SDF_TYPE = hm.igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER  # as run_partwise.py
        lab = json.load(open(Path(__file__).parent / "assets/hand_parts.json"))
        self.parts = lab["parts"]
        self.regions = [f"{p}_{s}" for p in self.parts for s in ("palmar", "dorsal")]
        self.hs = hm.HandSurface(mano("right").faces)
        parent = np.asarray(self.hs.S.argmax(axis=1)).ravel()
        self.region = {}
        for side in SIDES:
            part = np.asarray(lab[side]["part"])
            dors = np.asarray([s == "dorsal" for s in lab[side]["side"]])
            self.region[side] = (part * 2 + dors)[parent]
        self._objs = {}

    def obj(self, name):
        if name not in self._objs:
            import trimesh
            d = ARCTIC / "meta/object_vtemplates" / name
            m = trimesh.load(d / "mesh.obj", process=False, force="mesh")
            top = np.array(json.load(open(d / "parts.json")), bool)
            self._objs[name] = self.hm.Articulated(np.asarray(m.vertices, np.float64),
                                                   np.asarray(m.faces, np.int64), top)
        return self._objs[name]

    def __call__(self, V_cam_m, side, T_co_mm, angle, obj_name, chunk=400):
        """V_cam_m (N,778,3) camera metres; T_co_mm (N,4,4); angle (N,). -> dict of (N,) arrays."""
        hm = self.hm
        N = len(V_cam_m)
        Vo = hm.apply_T(hm.inv_T(T_co_mm), V_cam_m * 1000.0)
        art = self.obj(obj_name)
        reg = self.region[side]
        res = {k: np.zeros(N, np.float32) for k in ("min_sd_mm", "pen_max_mm", "pen_vol_mm3", "n_pen")}
        for r in self.regions:
            for m in ("pen", "vol", "npen", "minsd"):
                res[f"{r}_{m}"] = np.zeros(N, np.float32)
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            D = self.hs.dense(Vo[s:e])
            sd = hm.articulated_sdf(art, angle[s:e], D)
            A = hm.vertex_areas(D, self.hs.F_dense)
            depth = np.clip(-sd, 0, None)
            res["min_sd_mm"][s:e] = sd.min(1)
            res["pen_max_mm"][s:e] = depth.max(1)
            res["pen_vol_mm3"][s:e] = (depth * A).sum(1)
            res["n_pen"][s:e] = (sd < 0).sum(1)
            for ri, r in enumerate(self.regions):
                m = reg == ri
                res[f"{r}_pen"][s:e] = depth[:, m].max(1)
                res[f"{r}_vol"][s:e] = (depth[:, m] * A[:, m]).sum(1)
                res[f"{r}_npen"][s:e] = (sd[:, m] < 0).sum(1)
                res[f"{r}_minsd"][s:e] = sd[:, m].min(1)
        res["contact"] = res["min_sd_mm"] < hm.CONTACT_MM
        return res


# ----------------------------------------------------------------------------- prediction shards
def shard_name(sid, seq, view):
    return f"{sid}__{seq}__v{int(view)}.npz"


def group_rows(es):
    """Eval-set rows grouped by (sid, seq, view) -> list of (sid, seq, view, row indices)."""
    keys = np.char.add(np.char.add(es["sid"], "/"), np.char.add(es["seq"], np.char.add("/", es["view"].astype(str))))
    out = []
    for k in dict.fromkeys(keys.tolist()):
        sid, seq, view = k.split("/")
        out.append((sid, seq, int(view), np.where(keys == k)[0]))
    return out
