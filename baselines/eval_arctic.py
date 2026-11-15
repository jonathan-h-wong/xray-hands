"""Evaluate baseline predictions on an ARCTIC eval set (CPU only).

For every selected hand (GT grounded + visible, see make_eval_set.py) and every method x lift:

Hand pose (GT = ARCTIC MANO in the view's camera, metres -> reported in mm):
  mpjpe_ra / mpvpe_ra     joints / vertices after aligning the wrist (root-relative)
  pa_mpjpe / pa_mpvpe     after similarity Procrustes alignment
  root_err, depth_err     absolute wrist error and signed wrist depth error (pred - GT): what lifting gets wrong
  mpjpe_cam               absolute joint error in the camera frame
  kp2d_err_px             reprojection error with the true K, in full-resolution pixels
Penetration (same definitions as analysis/run_partwise.py; mm, mm^3; 12 regions = 6 parts x palmar/dorsal):
  pen_* / <region>_*      lifted predicted hand vs the GT object
  ra_pen_*                predicted hand moved onto the GT wrist position (isolates pose/shape from lifting)
  self_pen_*              HOPformer only: predicted hand vs HOPformer's own predicted object
Joint/hand-object relations: mrrpe_rl_mm (right wrist relative to left), HOPformer object errors
(obj_t_err_mm, obj_rot_err_deg, obj_angle_err_deg, obj_cls_ok, mrrpe_ro_mm).
A "gt" method row holds the GT hand's own penetration on the same frames (reference; must reproduce the
pen_gt stored in the eval set - checked and printed).

Lifts for 2D methods (HaMeR / WiLoR), both computed from the same saved crop outputs:
  wp     HaMeR's cam_crop_to_full with the real focal length and principal point
  fit2d  translation solved by least squares so the model's 3D joints reproject onto its own 2D keypoints
Intrinsics: native (ARCTIC calibration) or moge:<json> (estimate_intrinsics_moge.py --for hand2d).

    python baselines/eval_arctic.py --eval-set baselines/eval_sets/arctic_p2_val_v0_s5.npz \
        --methods hamer wilor hopformer --workers 8
Writes results/baselines/<tag>_frames.parquet and <tag>_summary.md
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

_W = {}
HAND2D = {"hamer", "wilor"}
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]   # order of C.TIPS = joints21[16:21]
# class index order of HOPformer's object.class_logits (HOPformer common/object_tensors.py OBJECTS), not alphabetical
ARCTIC_OBJECTS = ["capsulemachine", "box", "ketchup", "laptop", "microwave", "mixer", "notebook", "espressomachine",
                  "waffleiron", "scissors", "phone"]


def procrustes(X, Y):
    """Similarity-align X (N,P,3) to Y (N,P,3) (Umeyama)."""
    mx, my = X.mean(1, keepdims=True), Y.mean(1, keepdims=True)
    X0, Y0 = X - mx, Y - my
    U, S, Vt = np.linalg.svd(np.einsum("npi,npj->nij", X0, Y0))
    V = np.swapaxes(Vt, 1, 2)
    D = np.tile(np.eye(3), (len(X), 1, 1))
    D[:, 2, 2] = np.sign(np.linalg.det(V @ np.swapaxes(U, 1, 2)))
    R = V @ D @ np.swapaxes(U, 1, 2)
    s = (S * np.diagonal(D, axis1=1, axis2=2)).sum(1) / (X0 ** 2).sum((1, 2))
    return s[:, None, None] * np.einsum("nij,npj->npi", R, X0) + my


def rot_err_deg(Ra, Rb):
    c = (np.einsum("nij,nij->n", Ra, Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def pose_metrics(Vp, Vg, side, K, scale):
    Jp, Jg = C.joints21(Vp, side), C.joints21(Vg, side)
    rp, rg = Jp[:, :1], Jg[:, :1]
    m = {}
    m["mpjpe_ra"] = np.linalg.norm((Jp - rp) - (Jg - rg), axis=-1).mean(1)
    m["mpvpe_ra"] = np.linalg.norm((Vp - rp) - (Vg - rg), axis=-1).mean(1)
    m["pa_mpjpe"] = np.linalg.norm(procrustes(Jp, Jg) - Jg, axis=-1).mean(1)
    m["pa_mpvpe"] = np.linalg.norm(procrustes(Vp, Vg) - Vg, axis=-1).mean(1)
    m["root_err"] = np.linalg.norm(rp[:, 0] - rg[:, 0], axis=-1)
    m["depth_err"] = rp[:, 0, 2] - rg[:, 0, 2]
    m["mpjpe_cam"] = np.linalg.norm(Jp - Jg, axis=-1).mean(1)
    for i, f in enumerate(FINGERS):   # fingertip position error, camera frame and wrist-aligned
        m[f"tip_{f}_err"] = np.linalg.norm(Jp[:, 16 + i] - Jg[:, 16 + i], axis=-1)
        m[f"tip_{f}_err_ra"] = np.linalg.norm((Jp[:, 16 + i] - rp[:, 0]) - (Jg[:, 16 + i] - rg[:, 0]), axis=-1)
    m = {k: v * 1000.0 for k, v in m.items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        m["kp2d_err_px"] = np.linalg.norm(C.project(K, Jp) - C.project(K, Jg), axis=-1).mean(1) / scale
    return m, Jp, Jg


def pen_cols(pp, V, side, T_co, ang, obj, prefix="", regions=True):
    r = pp(V, side, T_co, ang, obj)
    keep = r if regions else {k: r[k] for k in ("min_sd_mm", "pen_max_mm", "pen_vol_mm3", "n_pen", "contact")}
    return {prefix + k: v for k, v in keep.items()}


def _init(intr, sdf_pad=None, export=None):
    _W["pp"] = C.PartPenetration()
    _W["export"] = export
    if export:
        Path(export).mkdir(parents=True, exist_ok=True)
    if sdf_pad:   # signed distances are clipped at +PAD_MM (30 by default); raise it to measure far-from-object hands
        _W["pp"].hm.PAD_MM = float(sdf_pad)
    _W["moge"] = None if intr == "native" else json.load(open(intr.split(":", 1)[1]))


def evaluate_group(task):
    import pandas as pd
    (sid, seq, view, idx), es, methods, preds_dir, lifts = task
    pp = _W["pp"]
    gt = C.gt_sequence(sid, seq, view)
    fr = es["frame"][idx]
    obj = gt["obj"]
    T_co, ang = gt["T_co"][fr], gt["angle"][fr]
    base = {"sid": sid, "seq": f"{sid}/{seq}", "view": view, "obj": obj, "frame": fr}
    dfs, checks = [], []
    roots = {}
    exp = {} if _W.get("export") else None
    for side in C.SIDES:
        sel = es[f"sel_{side}"][idx]
        rows = np.where(sel)[0]
        if not len(rows):
            continue
        Vg = gt["verts"][side][fr[rows]]
        common = {k: (v[rows] if isinstance(v, np.ndarray) else v) for k, v in base.items()}
        common.update({"hand": side, "contact_es": es[f"contact_{side}"][idx][rows],
                       "pen_gt_es": es[f"pen_gt_{side}"][idx][rows]})
        if exp is not None:
            exp[f"{side}.eval_rows"], exp[f"{side}.frame"] = idx[rows], fr[rows]
            exp[f"gt.{side}.verts"] = Vg.astype(np.float32)
            exp[f"gt.{side}.joints"] = C.joints21(Vg, side).astype(np.float32)
            exp[f"gt.obj_T_cam_mm.{side}"] = T_co[rows].astype(np.float32)
            exp[f"gt.obj_radian.{side}"] = ang[rows].astype(np.float32)
        g = pen_cols(pp, Vg, side, T_co[rows], ang[rows], obj)
        checks.append(np.abs(g["pen_max_mm"] - common["pen_gt_es"]).max())
        dfs.append(pd.DataFrame({**common, "method": "gt", "lift": "-", "detected": True, **g}))
        gt_contact = g["contact"]
        for method in methods:
            p = Path(preds_dir) / method / C.shard_name(sid, seq, view)
            if not p.exists():
                continue
            z = np.load(p, allow_pickle=False)
            pos = {r: i for i, r in enumerate(z["rows"])}
            k = np.array([pos[r] for r in idx[rows]])
            found = z["found"][k]
            det = z[f"{side}_det"][k] & found
            hop = "obj_rot" in z.files
            variants = []
            if hop:
                Vp = z[f"{side}_verts"][k].astype(np.float64)
                K_fed = C.scale_K(gt["cam"]["K"], 1.0)
                variants.append(("native", Vp, K_fed, 1.0))
            else:
                scale = float(z["img_hw"][k][found][0][1] / gt["cam"]["size"][0]) if found.any() else 1.0
                K_fed = z["K_native"] if _W["moge"] is None else np.asarray(_W["moge"][f"{sid}/{seq}/{view}"]["K_fed"])
                Vm = z[f"{side}_verts"][k].astype(np.float64)
                for lift in lifts:
                    if lift == "wp":
                        t = C.lift_weak_perspective(z[f"{side}_cam"][k].astype(np.float64), z[f"{side}_center"][k],
                                                    z[f"{side}_size"][k], K_fed, float(z["crop_focal"]),
                                                    int(z["crop_res"]))
                    else:
                        t = np.full((len(k), 3), np.nan)
                        t0 = C.lift_weak_perspective(z[f"{side}_cam"][k].astype(np.float64), z[f"{side}_center"][k],
                                                     z[f"{side}_size"][k], K_fed, 0, 0)
                        for i in np.where(det)[0]:
                            t[i] = C.lift_fit2d(z[f"{side}_kp3d"][k][i].astype(np.float64),
                                                z[f"{side}_kp2d"][k][i].astype(np.float64), K_fed, t0[i])
                    variants.append((lift, Vm + t[:, None, :], C.scale_K(gt["cam"]["K"], scale), scale))
            for lift, Vp, K_eval, scale in variants:
                ok = det & np.isfinite(Vp).all((1, 2))
                if exp is not None:   # exactly the camera-frame hand every metric below is computed on
                    Ve = np.where(ok[:, None, None], Vp, np.nan).astype(np.float32)
                    exp[f"{method}.{lift}.{side}.verts"] = Ve
                    exp[f"{method}.{lift}.{side}.joints"] = C.joints21(np.nan_to_num(Ve), side).astype(np.float32) \
                        * np.where(ok, 1, np.nan)[:, None, None].astype(np.float32)
                    exp[f"{method}.{lift}.{side}.score"] = z[f"{side}_score"][k].astype(np.float32)
                    if hop:
                        exp[f"{method}.{lift}.obj_rot.{side}"] = z["obj_rot"][k].astype(np.float32)
                        exp[f"{method}.{lift}.obj_cam_t_m.{side}"] = z["obj_cam_t"][k].astype(np.float32)
                        exp[f"{method}.{lift}.obj_radian.{side}"] = z["obj_radian"][k].astype(np.float32)
                cols = {"method": method, "lift": lift, "detected": det, "score": z[f"{side}_score"][k]}
                if ok.any():
                    pm, Jp, Jg = pose_metrics(Vp[ok], Vg[ok], side, K_eval, scale)
                    full = lambda a, fill=np.nan: _scatter(a, ok, fill)  # noqa: E731
                    cols.update({k2: full(v) for k2, v in pm.items()})
                    pc = pen_cols(pp, Vp[ok], side, T_co[rows][ok], ang[rows][ok], obj)
                    cols.update({k2: full(v, np.nan if v.dtype != bool else False) for k2, v in pc.items()})
                    Vra = Vp[ok] - Jp[:, :1] + Jg[:, :1]
                    pr = pen_cols(pp, Vra, side, T_co[rows][ok], ang[rows][ok], obj, "ra_")
                    cols.update({k2: full(v, np.nan if v.dtype != bool else False) for k2, v in pr.items()})
                    cols["contact_pred_gt"] = full(gt_contact[ok], False)
                    roots[(method, lift, side)] = (idx[rows][ok], Jp[:, 0], Jg[:, 0])
                    if hop:
                        R, t, rad = z["obj_rot"][k][ok].astype(np.float64), z["obj_cam_t"][k][ok] * 1000.0, \
                            z["obj_radian"][k][ok].astype(np.float64)
                        T_pred = np.tile(np.eye(4), (ok.sum(), 1, 1))
                        T_pred[:, :3, :3], T_pred[:, :3, 3] = R, t
                        ps = pen_cols(pp, Vp[ok], side, T_pred, rad, obj, "self_", regions=False)
                        cols.update({k2: full(v, np.nan if v.dtype != bool else False) for k2, v in ps.items()})
                        Tg = T_co[rows][ok]
                        cols["obj_t_err_mm"] = full(np.linalg.norm(t - Tg[:, :3, 3], axis=-1))
                        cols["obj_rot_err_deg"] = full(rot_err_deg(R, Tg[:, :3, :3]))
                        cols["obj_angle_err_deg"] = full(np.degrees(np.abs(rad - ang[rows][ok])))
                        cols["obj_cls_ok"] = full((np.argmax(z["obj_logits"][k][ok], 1) ==
                                                   ARCTIC_OBJECTS.index(obj)).astype(float))
                        cols["mrrpe_ro_mm"] = full(np.linalg.norm((Jp[:, 0] * 1000 - t) -
                                                                  (Jg[:, 0] * 1000 - Tg[:, :3, 3]), axis=-1))
                dfs.append(pd.DataFrame({**common, **cols}))
    df = pd.concat(dfs, ignore_index=True) if dfs else None
    if df is not None:   # right-relative-to-left wrist error where both hands are evaluated and detected
        df["mrrpe_rl_mm"] = np.nan
        for (method, lift, side), (r_idx, jp, jg) in roots.items():
            if side != "right" or (method, lift, "left") not in roots:
                continue
            l_idx, lp, lg = roots[(method, lift, "left")]
            both, ir, il = np.intersect1d(r_idx, l_idx, return_indices=True)
            if not len(both):
                continue
            err = np.linalg.norm((jp[ir] - lp[il]) - (jg[ir] - lg[il]), axis=-1) * 1000
            m = (df.method == method) & (df.lift == lift) & (df.hand == "right")
            fr_map = dict(zip(es["frame"][both], err))
            df.loc[m, "mrrpe_rl_mm"] = df.loc[m, "frame"].map(fr_map)
    if exp:
        exp.update(K_full=gt["cam"]["K"].astype(np.float32), obj=np.array(obj), sid=np.array(sid), seq=np.array(seq),
                   view=np.array(int(view)), units=np.array("verts/joints: metres, camera frame of the view "
                                                            "(undistorted); joints = 16 MANO + 5 tips; NaN = not detected"))
        np.savez_compressed(Path(_W["export"]) / C.shard_name(sid, seq, view), **exp)
    return df, (max(checks) if checks else 0.0)


def _scatter(v, ok, fill):
    out = np.full(len(ok), fill, dtype=np.asarray(v).dtype if fill is not np.nan else np.float64)
    out[ok] = v
    return out


def _work(task):
    try:
        return evaluate_group(task), None
    except Exception as e:
        import traceback
        return (None, 0.0), f"{task[0][:3]}: {e!r}\n{traceback.format_exc(limit=4)}"


# ----------------------------------------------------------------------------- summary
def summarize(df, regions):
    import pandas as pd
    lines = []
    q = lambda s, p: np.nanpercentile(s, p) if len(s) else np.nan  # noqa: E731
    keys = ["method", "lift"]
    rows = []
    for (m, l), g in df.groupby(keys, sort=False):
        d = g[g.detected]
        r = {"method": m, "lift": l, "hands": len(g), "detected %": 100 * g.detected.mean()}
        if m != "gt":
            for c in ("mpjpe_ra", "pa_mpjpe", "pa_mpvpe", "root_err", "mpjpe_cam", "kp2d_err_px"):
                r[f"{c} p50"] = d[c].median()
            r["depth_err p50"] = d.depth_err.median()
            r["|depth_err| p50"] = d.depth_err.abs().median()
            r["mrrpe_rl p50"] = d.mrrpe_rl_mm.median() if "mrrpe_rl_mm" in d else np.nan
        r["pen p50"] = d.pen_max_mm.median()
        r["pen p90"] = q(d.pen_max_mm, 90)
        r["pen>2mm %"] = 100 * (d.pen_max_mm > 2).mean()
        r["pen>5mm %"] = 100 * (d.pen_max_mm > 5).mean()
        r["contact %"] = 100 * d.contact.mean()
        if m != "gt":
            gc = d.contact_pred_gt.astype(bool)
            r["contact recall %"] = 100 * (d.contact & gc).sum() / max(gc.sum(), 1)
            r["contact precision %"] = 100 * (d.contact & gc).sum() / max(d.contact.sum(), 1)
            r["ra pen p50"] = d.ra_pen_max_mm.median()
            r["ra pen>5mm %"] = 100 * (d.ra_pen_max_mm > 5).mean()
        if "self_pen_max_mm" in d and d.self_pen_max_mm.notna().any():
            r["self pen p50"] = d.self_pen_max_mm.median()
            r["obj t err p50"] = d.obj_t_err_mm.median()
            r["obj rot err p50"] = d.obj_rot_err_deg.median()
            r["mrrpe_ro p50"] = d.mrrpe_ro_mm.median()
            r["obj cls %"] = 100 * d.obj_cls_ok.mean()
        rows.append(r)
    t = pd.DataFrame(rows).set_index(keys).T
    lines += ["## All selected hands (metrics over detected hands)\n", t.round(2).to_markdown(), ""]

    # fair subset: hands every method detected
    meth = [k for k in df.groupby(keys).groups if k[0] != "gt"]
    if len(meth) > 1:
        key = ["seq", "hand", "frame"]
        sets = [set(map(tuple, df[(df.method == m) & (df.lift == l) & df.detected][key].to_numpy().tolist()))
                for m, l in meth]
        common = set.intersection(*sets)
        sub = df[[tuple(x) in common for x in df[key].to_numpy().tolist()]]
        rows = []
        for (m, l), g in sub.groupby(keys, sort=False):
            r = {"method": m, "lift": l, "hands": len(g)}
            if m != "gt":
                for c in ("mpjpe_ra", "pa_mpjpe", "root_err", "kp2d_err_px"):
                    r[f"{c} p50"] = g[c].median()
                r["ra pen p50"] = g.ra_pen_max_mm.median()
            r["pen p50"] = g.pen_max_mm.median()
            r["pen>5mm %"] = 100 * (g.pen_max_mm > 5).mean()
            rows.append(r)
        lines += [f"## Common subset: {len(common)} hands detected by every method\n",
                  pd.DataFrame(rows).set_index(keys).T.round(2).to_markdown(), ""]

    # per part, over frames where the evaluated hand touches the object (pred contact)
    lines.append("## Per part (contact frames of each method; penetration depth mm, share of penetrating volume)\n")
    parts = sorted({r.rsplit("_", 1)[0] for r in regions}, key=lambda p: [r.rsplit("_", 1)[0] for r in regions].index(p))
    for (m, l), g in df.groupby(keys, sort=False):
        d = g[g.detected & g.contact.astype(bool)]
        if not len(d):
            continue
        tot = sum(d[f"{r}_vol"].sum() for r in regions) or 1.0
        rows = []
        for p in parts:
            pen = np.maximum(d[f"{p}_palmar_pen"], d[f"{p}_dorsal_pen"])
            vol = d[f"{p}_palmar_vol"].sum() + d[f"{p}_dorsal_vol"].sum()
            rows.append({"part": p, ">2mm %": 100 * (pen > 2).mean(), ">5mm %": 100 * (pen > 5).mean(),
                         "p50": pen.median(), "p90": q(pen, 90), "vol share %": 100 * vol / tot})
        for s in ("palmar", "dorsal"):
            pen = np.max([d[f"{p}_{s}_pen"] for p in parts], 0)
            vol = sum(d[f"{p}_{s}_vol"].sum() for p in parts)
            rows.append({"part": s, ">2mm %": 100 * (pen > 2).mean(), ">5mm %": 100 * (pen > 5).mean(),
                         "p50": np.median(pen), "p90": q(pen, 90), "vol share %": 100 * vol / tot})
        lines += [f"### {m} ({l}), {len(d)} contact hand-frames\n",
                  pd.DataFrame(rows).set_index("part").round(2).to_markdown(), ""]
    return "\n".join(lines)


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--methods", nargs="+", default=["hamer", "wilor", "hopformer"],
                    help="sub-directories of --preds (hand2d shards or HOPformer shards)")
    ap.add_argument("--preds", default=str(C.ROOT / "baselines/preds"))
    ap.add_argument("--lifts", nargs="+", default=["wp", "fit2d"], choices=["wp", "fit2d"])
    ap.add_argument("--intrinsics", default="native", help="native | moge:<json> (for the 2D methods' lift)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=str(C.ROOT / "results/baselines"))
    ap.add_argument("--limit-seqs", type=int, default=0)
    ap.add_argument("--sdf-pad", type=float, default=None,
                    help="clip signed distances at this many mm instead of 30 (penetration/contact unaffected)")
    ap.add_argument("--export", default=None,
                    help="dir: also write per-sequence npz of the camera-frame GT and predicted hands (every lift)")
    a = ap.parse_args()

    es, meta = C.load_eval_set(a.eval_set)
    groups = C.group_rows(es)[: a.limit_seqs or None]
    tasks = [(g, es, a.methods, a.preds, a.lifts) for g in groups]
    tag = a.tag or (Path(a.eval_set).stem + ("" if a.intrinsics == "native" else "_moge"))
    print(f"{len(groups)} sequence/view groups, methods {a.methods}, intrinsics {a.intrinsics}", flush=True)
    t0, dfs, errs, worst = time.time(), [], [], 0.0
    with Pool(a.workers, initializer=_init, initargs=(a.intrinsics, a.sdf_pad, a.export)) as pool:
        for i, ((df, chk), err) in enumerate(pool.imap_unordered(_work, tasks)):
            if df is not None:
                dfs.append(df)
            worst = max(worst, chk)
            if err:
                errs.append(err)
            print(f"  {i + 1}/{len(tasks)} {time.time() - t0:.0f}s errors={len(errs)}", flush=True)
    for e in errs[:10]:
        print("ERROR", e)
    if not dfs:
        sys.exit("no results")
    df = pd.concat(dfs, ignore_index=True)
    check = (f"GT penetration re-computation vs eval set: max |diff| = {worst:.4f} mm "
             f"({'OK' if worst < 0.05 else 'MISMATCH - camera/object transform is off'})")
    print(check)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{tag}_frames.parquet")
    pp = C.PartPenetration()
    md = [f"# ARCTIC baselines: {tag}\n", f"eval set: {json.dumps(meta)}\n", f"intrinsics: {a.intrinsics}\n", check + "\n",
          summarize(df, pp.regions)]
    (out / f"{tag}_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print("wrote", out / f"{tag}_frames.parquet", "and", out / f"{tag}_summary.md")


if __name__ == "__main__":
    main()
