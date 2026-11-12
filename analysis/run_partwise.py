"""Per-hand-part penetration metrics (separate pass; does not modify run_penetration.py).

Regions: 6 parts (palm, thumb, index, middle, ring, pinky) x 2 sides (palmar, dorsal),
labelled by analysis/hand_parts.py. Subdivided surface points inherit the label of their
first parent vertex.

Per frame x hand x region: max depth (mm), penetrating volume proxy (mm^3), # points
inside, closest signed distance (mm). Also per-vertex penetration counts over contact
frames (for the hand heat map).

Usage:
    .venv\\Scripts\\python analysis\\run_partwise.py --dataset arctic --workers 14
Writes results/parts/<dataset>_parts.parquet and results/parts/<dataset>_vertex.npz
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import igl
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hoi_metrics as hm  # noqa: E402
import run_penetration as rp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "parts"
LAB = json.load(open(ROOT / "results" / "hand_parts.json"))
PARTS = LAB["parts"]
SIDES = ["palmar", "dorsal"]
REGIONS = [f"{p}_{s}" for p in PARTS for s in SIDES]
_W = {}


def _init():
    rp._init()
    hs = rp._G["hs"]
    parent = np.asarray(hs.S.argmax(axis=1)).ravel()  # dense point -> first parent vertex
    _W["region"] = {}
    for side in ("left", "right"):
        part = np.asarray(LAB[side]["part"])
        dors = np.asarray([s == "dorsal" for s in LAB[side]["side"]])
        _W["region"][side] = (part * 2 + dors)[parent]  # region index per dense point


def evaluate_parts(seq):
    import pandas as pd
    hs = rp._G["hs"]
    # Winding-number sign everywhere: pseudonormal was tried for ARCTIC but disagreed with the
    # original run on ~24% of frames (articulated meshes stretch/fold at the hinge).
    hm.SDF_TYPE = igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    T_ow = hm.inv_T(seq.T_wo)
    dfs = []
    acc = {"n_contact": 0, "pen0": np.zeros(778), "pen2": np.zeros(778), "pen5": np.zeros(778),
           "depth_sum": np.zeros(778), "near": np.zeros(778)}
    for side, (Vw, valid) in seq.hands.items():
        valid = valid & np.isfinite(Vw).all((1, 2)) & np.isfinite(seq.T_wo).all((1, 2))
        idx = np.where(valid)[0]
        if not len(idx):
            continue
        Vo = hm.apply_T(T_ow[idx], Vw[idx])
        reg = _W["region"][side]
        N = len(idx)
        cols = {f"{r}_{m}": np.zeros(N, np.float32) for r in REGIONS for m in ("pen", "vol", "npen", "minsd")}
        overall_min = np.zeros(N, np.float32)
        overall_pen = np.zeros(N, np.float32)
        for s in range(0, N, 400):
            e = min(N, s + 400)
            D = hs.dense(Vo[s:e])
            if isinstance(seq.parts, hm.Articulated):
                sd = hm.articulated_sdf(seq.parts, seq.part_T[idx[s:e]], D)
            else:
                sd = hm.object_sdf(seq.parts, None, D)
            A = hm.vertex_areas(D, hs.F_dense)
            depth = np.clip(-sd, 0, None)
            overall_min[s:e] = sd.min(1)
            overall_pen[s:e] = depth.max(1)
            for ri, r in enumerate(REGIONS):
                m = reg == ri
                cols[f"{r}_pen"][s:e] = depth[:, m].max(1)
                cols[f"{r}_vol"][s:e] = (depth[:, m] * A[:, m]).sum(1)
                cols[f"{r}_npen"][s:e] = (sd[:, m] < 0).sum(1)
                cols[f"{r}_minsd"][s:e] = sd[:, m].min(1)
            c = sd.min(1) < hm.CONTACT_MM
            v = sd[c, :778]
            acc["n_contact"] += int(c.sum())
            acc["pen0"] += (v < 0).sum(0)
            acc["pen2"] += (v < -2).sum(0)
            acc["pen5"] += (v < -5).sum(0)
            acc["depth_sum"] += np.clip(-v, 0, None).sum(0)
            acc["near"] += (v < hm.CONTACT_MM).sum(0)
        df = pd.DataFrame({"dataset": seq.dataset, "seq": seq.name, "hand": side,
                           "frame": seq.frame_idx[idx], "min_sd_mm": overall_min, "pen_max_mm": overall_pen,
                           "contact": overall_min < hm.CONTACT_MM, **cols})
        if "object" in seq.extra:
            df["object"] = np.asarray(seq.extra["object"])[idx]
        dfs.append(df)
    return (pd.concat(dfs) if dfs else None), seq.dataset, acc


def _work(args):
    ds, task = args
    try:
        seq = rp.LOADERS[ds][1](task)
        return evaluate_parts(seq), None
    except Exception as e:
        import traceback
        return None, f"{task}: {e!r}\n{traceback.format_exc(limit=3)}"


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(rp.LOADERS), required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    tasks = rp.LOADERS[a.dataset][0]()
    if a.limit:
        tasks = [tasks[i] for i in np.linspace(0, len(tasks) - 1, min(a.limit, len(tasks))).astype(int)]
    print(f"{a.dataset} (parts): {len(tasks)} sequences, {a.workers} workers", flush=True)
    t0 = time.time()
    dfs, errs, accs = [], [], {}
    with Pool(a.workers, initializer=_init) as pool:
        for i, (res, err) in enumerate(pool.imap_unordered(_work, [(a.dataset, t) for t in tasks], chunksize=1)):
            if res is not None:
                df, ds, acc = res
                if df is not None:
                    dfs.append(df)
                tot = accs.setdefault(ds, {k: (0 if k == "n_contact" else np.zeros(778)) for k in acc})
                for k, v in acc.items():
                    tot[k] = tot[k] + v
            if err:
                errs.append(err)
            if (i + 1) % 25 == 0 or i + 1 == len(tasks):
                print(f"  {i + 1}/{len(tasks)}  {time.time() - t0:.0f}s  errors={len(errs)}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    out = pd.concat(dfs, ignore_index=True)
    suffix = f"_limit{a.limit}" if a.limit else ""
    for name, g in out.groupby("dataset"):
        p = Path(a.out) / f"{name}_parts{suffix}.parquet"
        g.reset_index(drop=True).to_parquet(p)
        np.savez(Path(a.out) / f"{name}_vertex{suffix}.npz", **accs[name])
        print("wrote", p, len(g), "rows")
    for e in errs[:10]:
        print("ERROR", e)


if __name__ == "__main__":
    main()
