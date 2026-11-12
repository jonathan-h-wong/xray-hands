"""Run hand-object penetration + smoothness metrics on ARCTIC, SHOW3D, EPIC-Contact and HOT3D.

HOT3D is evaluated twice: "hot3d" with its MANO fits (comparable to the other datasets, per-part capable)
and "hot3d_ume" with its native UmeTrack meshes (the hand model its mocap tracks).

Usage (from project root):
    .venv\\Scripts\\python analysis\\run_penetration.py --dataset arctic --workers 12
    .venv\\Scripts\\python analysis\\run_penetration.py --dataset show3d --limit 50
    .venv\\Scripts\\python analysis\\run_penetration.py --dataset epic
Writes results/<dataset>_frames.parquet (one row per frame x hand).
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hoi_metrics import Articulated, HandSurface, Part, Sequence, evaluate  # noqa: E402
from mano_np import MANO, rodrigues  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANO_DIR = DATA / "arctic/unpack/body_models/mano_v1_2/models"
ARCTIC = DATA / "arctic/unpack/arctic_data/data"
SHOW3D = DATA / "show3d"
HOT3D_MODELS = DATA / "hot3d_models/unzipped/models"
EPIC = DATA / "epic-contact"

SHOW3D_TO_BOP = {
    "dumbbell": 22, "mouse": 31, "keyboard": 28, "mug": 9, "mug2": 8, "balandabowl": 2, "vase": 16,
    "brushholder": 25, "birdhousetoy": 26, "dinotoy": 27, "whiteboardmarker": 32, "milk": 17,
    "orangejuice": 18, "mustard": 13, "ranch": 15, "bbq": 14, "cansoup": 10, "canparmesan": 11,
    "cantomatosauce": 12, "waffles": 20, "vegetables": 21, "aria": 23,
}

_G = {}


def _init():
    import trimesh  # noqa: F401  (warm import in worker)
    _G["mano"] = {
        (side, flat): MANO(MANO_DIR / f"MANO_{side.upper()}.pkl", flat_hand_mean=flat)
        for side in ("left", "right") for flat in (False, True)
    }
    # SHOW3D fits its left hands with the commonly patched MANO_LEFT (shapedirs[:, 0] negated).
    # Verified against SHOW3D's native UmeTrack fingertips: 17.5 mm -> 3.9 mm on a beta0 = -1.15 subject.
    fixed = MANO(MANO_DIR / "MANO_LEFT.pkl", flat_hand_mean=False)
    fixed.shapedirs[:, 0, :] *= -1
    _G["mano"][("left_fixed", False)] = fixed
    _G["hs"] = HandSurface(_G["mano"][("right", False)].faces)
    _G["parts"] = {}


def _T(R, t):
    T = np.zeros((len(R), 4, 4))
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1
    return T


def _load_mesh(path):
    import trimesh
    m = trimesh.load(path, process=False, force="mesh")
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


# ----------------------------------------------------------------------------- ARCTIC
def arctic_list():
    return sorted(glob.glob(str(ARCTIC / "raw_seqs/*/*.mano.npy")))


def arctic_parts(obj):
    if obj not in _G["parts"]:
        d = ARCTIC / "meta/object_vtemplates" / obj
        V, F = _load_mesh(d / "mesh.obj")
        top = np.array(json.load(open(d / "parts.json")), bool)
        _G["parts"][obj] = Articulated(V, F, top)
    return _G["parts"][obj]


def arctic_load(mano_p):
    sid = Path(mano_p).parent.name
    seq = Path(mano_p).name.replace(".mano.npy", "")
    obj = seq.split("_")[0]
    mano = np.load(mano_p, allow_pickle=True).item()
    o = np.load(mano_p.replace(".mano.npy", ".object.npy"), allow_pickle=True)
    N = len(o)
    T_wo = _T(rodrigues(o[:, 1:4]), o[:, 4:7])
    part_T = o[:, 0].astype(np.float64)  # hinge angle per frame (Articulated)
    hands, extra = {}, {}
    for side in ("left", "right"):
        h = mano[side]
        V = _G["mano"][(side, False)](h["shape"], h["rot"], h["pose"], h["trans"]) * 1000.0
        hands[side] = (V, np.isfinite(V).all((1, 2)))
        extra[f"{side}:fit_err"] = h["fitting_err"]
    extra["obj_angle_rad"] = o[:, 0]
    return Sequence("arctic", f"{sid}/{seq}", 30.0, 1, np.arange(N), T_wo, arctic_parts(obj), part_T, hands, extra)


# ----------------------------------------------------------------------------- SHOW3D
def show3d_list():
    import pandas as pd
    out = []
    for split, f in (("train", "index.parquet"), ("test", "test_index.parquet")):
        p = SHOW3D / "object_pose/v2" / f
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        sub = "scenes" if split == "train" else "test_scenes"
        for r in df.itertuples():
            op = SHOW3D / "object_pose/v2" / sub / r.subject_id / r.scene_id / "object_pose.json"
            hp = SHOW3D / "hand_pose/v3" / sub / r.subject_id / r.scene_id / "hand_pose_mano.json"
            if r.object_alias in SHOW3D_TO_BOP and op.exists() and hp.exists():
                out.append((split, r.subject_id, r.scene_id, r.object_alias, str(op), str(hp)))
    return out


def show3d_load(task):
    import orjson
    split, subj, scene, alias, op, hp = task
    if alias not in _G["parts"]:
        _G["parts"][alias] = [Part(*_load_mesh(HOT3D_MODELS / f"obj_{SHOW3D_TO_BOP[alias]:06d}.ply"))]
    objp = orjson.loads(open(op, "rb").read())
    handp = orjson.loads(open(hp, "rb").read())
    prof = json.load(open(SHOW3D / "hand_pose/hand_profiles" / subj / "profile_mano.json"))
    betas = np.array(prof["betas"], np.float64)
    fr = sorted(int(k) for k, v in objp.items() if v["confidence"] > 0)
    fr = np.array(fr)
    N = len(fr)
    R = np.array([objp[str(f)]["R"] for f in fr], np.float64)
    t = np.array([objp[str(f)]["t"] for f in fr], np.float64).reshape(N, 3)
    T_wo = _T(R, t)
    step = int(np.bincount(np.diff(fr)).argmax()) if N > 1 else 1
    hands = {}
    extra = {"obj_conf": np.array([objp[str(f)]["confidence"] for f in fr]), "split": np.full(N, split),
             "object": np.full(N, alias)}
    frames = handp["frames"]
    for slot, side in (("0", "left"), ("1", "right")):
        gt = np.full((N, 6), np.nan)
        pose = np.zeros((N, 45))
        conf = np.zeros(N)
        ferr = np.full(N, np.nan)
        for i, f in enumerate(fr):
            h = frames.get(str(f), {}).get("hand_poses", {}).get(slot)
            if h is None or h.get("pose_parameters") is None:
                continue
            gt[i] = h["global_transform"]
            pose[i] = h["pose_parameters"]
            conf[i] = h["confidence"]
            ferr[i] = h.get("mean_landmark_error_mm", np.nan)
        valid = np.isfinite(gt).all(1) & (conf > 0.1)
        V = np.full((N, 778, 3), np.nan)
        if valid.any():
            model = _G["mano"][("left_fixed" if side == "left" else side, False)]
            V[valid] = model(betas, gt[valid, :3], pose[valid], gt[valid, 3:]) * 1000.0
        hands[side] = (V, valid)
        extra[f"{side}:hand_conf"] = conf
        extra[f"{side}:mano_fit_err_mm"] = ferr
    return Sequence("show3d", f"{subj}/{scene}", 60.0, step, fr, T_wo, _G["parts"][alias], None, hands, extra)


# ----------------------------------------------------------------------------- EPIC-Contact
def epic_list():
    c = sorted(glob.glob(str(EPIC / "central_frames/*")))
    p = sorted(glob.glob(str(EPIC / "per_frame_projected_npz/*/*.npz")))
    return [("central", x) for x in c] + [("perframe", x) for x in p]


def kabsch(A, B):
    """Rigid T (4x4) with B ~= R A + t, A,B: (P,3)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return _T(R[None], (cb - R @ ca)[None])[0]


def epic_load(task):
    kind, p = task
    if kind == "central":
        clip = Path(p).name
        side = "left" if "_left_" in clip else "right"
        Hv, _ = _load_mesh(Path(p) / "hand_mesh.obj")
        Ov, Of = _load_mesh(Path(p) / "obj_mesh.obj")
        hands = {side: (Hv[None] * 1000.0, np.ones(1, bool))}
        extra = {"object": np.array([clip.split("_")[-1]])}
        return Sequence("epic_central", clip, 60.0, 1, np.zeros(1, int), np.eye(4)[None],
                        [Part(Ov * 1000.0, Of)], None, hands, extra)
    z = np.load(p, allow_pickle=True)
    clip = str(z["_clip"])
    side = "left" if "_left_" in clip else "right"
    s = side[0]
    fr = z["_frame_num"].astype(int)
    Ov = z["object.v.cam"].astype(np.float64) * 1000.0
    nv = int(z["object.v_len"][0])
    Ov = Ov[:, :nv]
    Of = z["object.f"][0].astype(np.int64)
    Of = Of[(Of < nv).all(1)]
    T_wo = np.stack([kabsch(Ov[0], Ov[i]) for i in range(len(fr))])
    H = z[f"mano.v3d.cam.{s}"].astype(np.float64) * 1000.0
    valid = (z[f"{side}_valid"] > 0) & (z["is_valid"] > 0)
    step = int(np.bincount(np.diff(fr)).argmax()) if len(fr) > 1 else 1
    extra = {"object": np.full(len(fr), clip.split("_")[-1]),
             "clip_verified": np.full(len(fr), bool(z["clip_verified"]))}
    return Sequence("epic_perframe", clip, 60.0, step, fr, T_wo, [Part(Ov[0], Of)], None,
                    {side: (H, valid)}, extra)


# ----------------------------------------------------------------------------- HOT3D (clips)
HOT3D = DATA / "hot3d/clips"
HOT3D_REACH_M = 0.15  # keep (clip, object) if a wrist comes within object radius + 15 cm


def hot3d_list():
    info = json.load(open(HOT3D_MODELS / "models_info.json"))
    out = []
    for p in sorted(q for q in HOT3D.glob("*/*.npz") if ".tmp" not in q.name):
        z = np.load(p)
        wr = [z[f"{s}_ume_T"][:, :3, 3] for s in ("left", "right")]
        for k, oid in enumerate(z["obj_ids"]):
            ot = z["obj_T"][k, :, :3, 3]
            r = info[str(int(oid))]["diameter"] / 2000.0  # mm -> m radius
            d = np.nanmin([np.nanmin(np.linalg.norm(w - ot, axis=1), initial=np.inf) for w in wr])
            if d < r + HOT3D_REACH_M:
                out.append((str(p), int(oid)))
    return out


def _hot3d_seq(task, hand_model):
    import umetrack_np as U
    p, oid = task
    z = np.load(p)
    if "hot3d_info" not in _G:
        info = json.load(open(HOT3D_MODELS / "models_info.json"))
        _G["hot3d_info"] = {int(k): v["name"] for k, v in info.items()}
    if ("hot3d", oid) not in _G["parts"]:
        _G["parts"][("hot3d", oid)] = [Part(*_load_mesh(HOT3D_MODELS / f"obj_{oid:06d}.ply"))]
    k = list(z["obj_ids"]).index(oid)
    T_wo = z["obj_T"][k].copy()
    T_wo[:, :3, 3] *= 1000.0
    fr = z["frames"].astype(int)
    N = len(fr)
    name = _G["hot3d_info"][oid]
    hands = {}
    ume = json.loads(str(z["umetrack_shape"])) if hand_model == "umetrack" else None
    for side in ("left", "right"):
        if hand_model == "mano":
            valid = z[f"{side}_valid"].copy()
            V = np.full((N, 778, 3), np.nan)
            if valid.any():
                m = _G["mano"][("left_fixed" if side == "left" else side, False)]
                th, wx = z[f"{side}_thetas"][valid], z[f"{side}_wrist"][valid]
                V[valid] = m(z["betas"], wx[:, :3], m.pca_to_pose(th), wx[:, 3:]) * 1000.0
        else:
            valid = z[f"{side}_ume_valid"].copy()
            V = np.full((N, len(ume["mesh_vertices"]), 3), np.nan)
            if valid.any():
                V[valid] = U.skin_vertices(ume, z[f"{side}_ume_angles"][valid], z[f"{side}_ume_T"][valid],
                                           side == "right") * 1000.0
        hands[side] = (V, valid)
    hs = None
    if hand_model == "umetrack":  # topology differs by participant (P0014: 878 verts), so cache per topology
        F = U.faces(ume, False)
        key = (len(ume["mesh_vertices"]), F.tobytes())
        cache = _G.setdefault("hs_ume", {})
        if key not in cache:
            cache[key] = HandSurface(F, nv=len(ume["mesh_vertices"]))
        hs = cache[key]
    split = Path(p).parent.name
    extra = {"object": np.full(N, name), "device": np.full(N, str(z["device"])),
             "participant": np.full(N, str(z["participant_id"])), "split": np.full(N, split)}
    ds = "hot3d" if hand_model == "mano" else "hot3d_ume"
    seq = Sequence(ds, f"{split}/{Path(p).stem}:{name}", 30.0, 1, fr, T_wo, _G["parts"][("hot3d", oid)], None,
                   hands, extra)
    seq.hs = hs
    return seq


def hot3d_load(task):
    return _hot3d_seq(task, "mano")


def hot3d_ume_load(task):
    return _hot3d_seq(task, "umetrack")


# ----------------------------------------------------------------------------- driver
LOADERS = {"arctic": (arctic_list, arctic_load), "show3d": (show3d_list, show3d_load),
           "epic": (epic_list, epic_load), "hot3d": (hot3d_list, hot3d_load),
           "hot3d_ume": (hot3d_list, hot3d_ume_load)}


def _work(args):
    ds, task, hands = args
    try:
        seq = LOADERS[ds][1](task)
        if hands:
            seq.hands = {k: v for k, v in seq.hands.items() if k in hands}
        return evaluate(seq, getattr(seq, "hs", None) or _G["hs"]), None
    except Exception as e:  # keep going; report at the end
        import traceback
        return None, f"{task}: {e!r}\n{traceback.format_exc(limit=3)}"


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(LOADERS), required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N sequences (0 = all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--hands", nargs="*", choices=["left", "right"], default=None,
                    help="only evaluate these hands (output gets a _<hands> suffix)")
    a = ap.parse_args()
    tasks = LOADERS[a.dataset][0]()
    if a.limit:
        # spread the sample over the whole list instead of taking the first subject only
        tasks = [tasks[i] for i in np.linspace(0, len(tasks) - 1, min(a.limit, len(tasks))).astype(int)]
    print(f"{a.dataset}: {len(tasks)} sequences, {a.workers} workers", flush=True)
    t0 = time.time()
    dfs, errs = [], []
    with Pool(a.workers, initializer=_init) as pool:
        for i, (df, err) in enumerate(pool.imap_unordered(_work, [(a.dataset, t, a.hands) for t in tasks], chunksize=1)):
            if df is not None:
                dfs.append(df)
            if err:
                errs.append(err)
            if (i + 1) % 25 == 0 or i + 1 == len(tasks):
                print(f"  {i + 1}/{len(tasks)}  {time.time() - t0:.0f}s  errors={len(errs)}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    out = pd.concat(dfs, ignore_index=True)
    suffix = (f"_limit{a.limit}" if a.limit else "") + ("_" + "_".join(a.hands) if a.hands else "")
    for name, g in out.groupby("dataset"):
        path = Path(a.out) / f"{name}_frames{suffix}.parquet"
        g.reset_index(drop=True).to_parquet(path)
        print("wrote", path, len(g), "rows")
    for e in errs[:10]:
        print("ERROR", e)


if __name__ == "__main__":
    main()
