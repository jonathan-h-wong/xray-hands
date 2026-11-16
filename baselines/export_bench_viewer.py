"""Export one benchmark run for the browser viewer (viewer/bench/index.html).

Inputs are what eval_arctic.py writes:
  --frames  <tag>_frames.parquet          (per hand-frame metrics, one row per method/lift + the GT rows)
  --hands   <dir with <shard>.npz>        (optional; from eval_arctic.py --export. Enables the 3D frame view)

Output: viewer/bench/data/<run>/{meta.json, rows.json, objects.json, hands/<shard>.json|.bin}
and the run is (re)registered in viewer/bench/data/runs.json. viewer/bench/data/ is git-ignored.

Usage (from the repo root):
  python baselines/export_bench_viewer.py --run 20261113_ego_s5 \
      --frames results/baselines/20261113_ego_s5_compare_fine_frames.parquet \
      --hands results/baselines/hands_20261113_ego_s5/native --note "calibrated intrinsics"
  python -m http.server -d viewer 8000      # then open http://localhost:8000/bench/

Your own model: write predictions in the same format as the baselines (see baselines/README.md),
run eval_arctic.py with --export, then export a run that contains your method's rows (e.g. concatenate
your frames parquet with the baseline compare parquet so they are shown side by side).
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "analysis"))
import common as C  # noqa: E402

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
PARTS = ["palm", "thumb", "index", "middle", "ring", "pinky"]
# key, label, unit, lower-is-better, group
METRICS = [
    ("pa_mpjpe", "PA-MPJPE", "mm", True, "pose"),
    ("pa_mpvpe", "PA-MPVPE", "mm", True, "pose"),
    ("mpjpe_ra", "MPJPE (root-aligned)", "mm", True, "pose"),
    ("mpvpe_ra", "MPVPE (root-aligned)", "mm", True, "pose"),
    *[(f"tip_{f}_err_ra", f"{f} tip err (wrist-aligned)", "mm", True, "pose") for f in FINGERS],
    ("mpjpe_cam", "MPJPE (camera frame)", "mm", True, "placement"),
    ("root_err", "Wrist error (camera)", "mm", True, "placement"),
    ("depth_err", "Wrist depth error (signed)", "mm", None, "placement"),
    ("mrrpe_rl_mm", "MRRPE right-left", "mm", True, "placement"),
    *[(f"tip_{f}_err", f"{f} tip err (camera)", "mm", True, "placement") for f in FINGERS],
    ("kp2d_err_px", "2D keypoint error", "px", True, "placement"),
    ("min_sd_mm", "Min signed dist to object", "mm", None, "contact"),
    ("pen_max_mm", "Max penetration", "mm", True, "contact"),
    ("pen_vol_mm3", "Penetration volume", "mm³", True, "contact"),
    ("ra_pen_max_mm", "Max penetration (wrist-aligned)", "mm", True, "contact"),
    ("sd_err_mm", "Surface distance err vs GT (pred − GT)", "mm", None, "contact"),
    ("obj_t_err_mm", "Object translation err", "mm", True, "object"),
    ("obj_rot_err_deg", "Object rotation err", "°", True, "object"),
    ("obj_angle_err_deg", "Object articulation err", "°", True, "object"),
    ("mrrpe_ro_mm", "MRRPE right-object", "mm", True, "object"),
    ("rel_obj_t_mm", "Hand rel. to own object (translation)", "mm", True, "hand-object (own object)"),
    ("rel_obj_frame_mm", "Hand in own object's frame", "mm", True, "hand-object (own object)"),
    ("scene_scaled_mpjpe", "MPJPE after one scene scale", "mm", True, "hand-object (own object)"),
    ("scene_depth_err_pct", "Scene depth error (signed)", "%", None, "hand-object (own object)"),
    ("self_min_sd_mm", "Min signed dist to own object", "mm", None, "hand-object (own object)"),
    ("self_sd_err_mm", "Surface distance err, own object", "mm", None, "hand-object (own object)"),
    ("self_pen_max_mm", "Max penetration into own object", "mm", True, "hand-object (own object)"),
    ("score", "Detector score", "", None, "detection"),
]
BOOLS = ["detected", "contact", "ra_contact", "self_contact", "obj_cls_ok"]
Q = 10.0   # verts stored as int16 in units of 1/Q mm  (range ±3.27 m)


def col(s, nd=1):
    a = s.to_numpy(dtype=np.float64)
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


def part_sd(d, prefix=""):
    out = {}
    for p in PARTS:
        a, b = f"{prefix}{p}_palmar_minsd", f"{prefix}{p}_dorsal_minsd"
        if a in d and b in d:
            out[p] = np.minimum(d[a], d[b])
    return out


def write_hands(npz, out_dir, keys):
    """npz -> <shard>.json (index + object poses) and <shard>.bin (int16 verts/joints, 1/Q mm)."""
    z = np.load(npz, allow_pickle=True)
    meta = {"K_full": z["K_full"].round(3).tolist(), "obj": str(z["obj"]), "sid": str(z["sid"]),
            "seq": str(z["seq"]), "view": int(z["view"]), "q": Q, "sides": {}}
    chunks, off = [], 0

    def add(a):
        nonlocal off
        a = np.asarray(a, np.float64) * 1000.0 * Q
        bad = ~np.isfinite(a)
        a = np.clip(np.where(bad, 0, a), -32767, 32767).astype(np.int16)
        a[bad] = -32768
        chunks.append(a.ravel())
        o = off
        off += a.size
        return o

    for side in C.SIDES:
        if f"{side}.frame" not in z.files:
            continue
        n = len(z[f"{side}.frame"])
        s = {"n": n, "frame": z[f"{side}.frame"].tolist(), "eval_row": z[f"{side}.eval_rows"].tolist(),
             "obj_T_mm": z[f"gt.obj_T_cam_mm.{side}"].round(3).tolist(),
             "obj_rad": z[f"gt.obj_radian.{side}"].round(5).tolist(), "src": {}}
        for k in ["gt"] + keys:
            if f"{k}.{side}.verts" not in z.files:
                continue
            e = {"v": add(z[f"{k}.{side}.verts"]), "j": add(z[f"{k}.{side}.joints"])}
            if f"{k}.{side}.score" in z.files:
                e["score"] = [None if not np.isfinite(v) else round(float(v), 3) for v in z[f"{k}.{side}.score"]]
            if f"{k}.obj_rot.{side}" in z.files:
                R, t, r = z[f"{k}.obj_rot.{side}"], z[f"{k}.obj_cam_t_m.{side}"] * 1000.0, z[f"{k}.obj_radian.{side}"]
                T = np.tile(np.eye(4), (n, 1, 1))
                T[:, :3, :3], T[:, :3, 3] = R, t
                e["obj_T_mm"] = np.where(np.isfinite(T), T, 0).round(3).tolist()
                e["obj_rad"] = np.nan_to_num(r).round(5).tolist()
            s["src"][k] = e
        meta["sides"][side] = s
    np.concatenate(chunks).astype("<i2").tofile(out_dir / f"{npz.stem}.bin")
    (out_dir / f"{npz.stem}.json").write_text(json.dumps(meta, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run id shown in the viewer (folder name)")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--hands", help="dir of <shard>.npz from eval_arctic.py --export")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=str(ROOT / "viewer/bench/data"))
    a = ap.parse_args()

    out = Path(a.out) / a.run
    (out / "hands").mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(a.frames)
    if "scene_scale" in d:   # pred depth / GT depth - 1, from the scale that maps the predicted scene onto GT
        d["scene_depth_err_pct"] = (1.0 / d["scene_scale"] - 1.0) * 100.0
    d["shard"] = [C.shard_name(s, q.split("/")[-1], v).removesuffix(".npz") for s, q, v in zip(d.sid, d.seq, d.view)]
    d = d.sort_values(["shard", "hand", "frame"], kind="stable").reset_index(drop=True)

    meths = d[d.method != "gt"][["method", "lift"]].drop_duplicates().to_records(index=False).tolist()
    order = {"hamer": 0, "wilor": 1, "hopformer": 2}
    meths.sort(key=lambda m: (order.get(m[0].split("_")[0], 9), m[0], m[1]))
    keys = [f"{m}.{l}" for m, l in meths]
    mkey = (d.method + "." + d.lift).where(d.method != "gt", "gt")
    src = ["gt"] + keys
    shards = sorted(d.shard.unique())

    # GT per-part distance + GT contact flag broadcast onto every row of the same hand-frame
    g = d[d.method == "gt"].set_index(["shard", "hand", "frame"])
    gsd = part_sd(g)
    idx = pd.MultiIndex.from_frame(d[["shard", "hand", "frame"]])
    rows = {"shard": [shards.index(s) for s in d.shard], "hand": (d.hand == "right").astype(int).tolist(),
            "frame": d.frame.astype(int).tolist(), "src": [src.index(k) for k in mkey],
            "gt_contact": g["contact"].reindex(idx).fillna(False).astype(int).tolist()}
    for p, s in gsd.items():
        rows[f"gt_sd_{p}"] = col(s.reindex(idx))
    for p, s in part_sd(d).items():
        rows[f"sd_{p}"] = col(s)
    for p, s in part_sd(d, "ra_").items():
        rows[f"ra_sd_{p}"] = col(s)
    metrics = []
    for k, lab, unit, low, grp in METRICS:
        if k in d and d[k].notna().any():
            rows[k] = col(d[k], 3 if k == "score" else 2)
            metrics.append({"key": k, "label": lab, "unit": unit, "lower": low, "group": grp})
    for k in BOOLS:
        if k in d:
            v = d[k]
            rows[k] = [None if pd.isna(x) else int(bool(x)) for x in v]
    (out / "rows.json").write_text(json.dumps(rows, separators=(",", ":")))

    seqs = []
    for s in shards:
        r = d[(d.shard == s) & (d.method == "gt")]
        seqs.append({"shard": s, "sid": r.sid.iloc[0], "seq": r.seq.iloc[0], "view": int(r.view.iloc[0]),
                     "obj": r.obj.iloc[0], "n_left": int((r.hand == "left").sum()),
                     "n_right": int((r.hand == "right").sum()), "has3d": False})

    has_hands = False
    if a.hands:
        hd = Path(a.hands)
        for i, s in enumerate(seqs):
            p = hd / f"{s['shard']}.npz"
            if p.exists():
                write_hands(p, out / "hands", keys)
                s["has3d"] = has_hands = True
        print(f"hands: {sum(s['has3d'] for s in seqs)}/{len(seqs)} sequences")

    objs = {}
    if has_hands:
        import trimesh
        for o in sorted(d.obj.unique()):
            od = C.ARCTIC / "meta/object_vtemplates" / o
            m = trimesh.load(od / "mesh.obj", process=False, force="mesh")
            objs[o] = {"v": np.asarray(m.vertices).round(2).ravel().tolist(),
                       "f": np.asarray(m.faces).astype(int).ravel().tolist(),
                       "top": [int(x) for x in json.load(open(od / "parts.json"))]}
        (out / "objects.json").write_text(json.dumps(objs, separators=(",", ":")))

    meta = {
        "run": a.run, "note": a.note, "frames_file": Path(a.frames).name, "created": dt.date.today().isoformat(),
        "methods": [{"key": k, "method": m, "lift": l} for k, (m, l) in zip(keys, meths)],
        "src": src, "seqs": seqs, "metrics": metrics, "parts": PARTS, "has3d": has_hands,
        "mano_faces": {sd: C.mano(sd).faces.astype(int).ravel().tolist() for sd in C.SIDES},
        "joint_parents": [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9],
        "n_rows": len(d),
    }
    (out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))

    idx_p = Path(a.out) / "runs.json"
    runs = json.loads(idx_p.read_text()) if idx_p.exists() else []
    runs = [r for r in runs if r["run"] != a.run] + [{"run": a.run, "note": a.note, "created": meta["created"],
                                                      "methods": keys, "has3d": has_hands}]
    idx_p.write_text(json.dumps(runs, indent=1))
    mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out} ({mb:.0f} MB): {len(d)} rows, methods {keys}")


if __name__ == "__main__":
    main()
