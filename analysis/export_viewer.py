"""Export compact JSON for the web viewer (viewer/data/*.json).

summary.json  - overall stats, per-object tables, CDFs of penetration & jitter (contact frames)
series.json   - per sequence/hand decimated time series (pen max-pooled, jitter mean)
frames3d.json - curated frames with hand + object meshes in the object frame, hand vertex SDF
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_penetration as rp  # noqa: E402
from hoi_metrics import Articulated, apply_T, articulated_sdf, inv_T, object_sdf  # noqa: E402
from summarize import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "viewer" / "data"
DATASETS = ["arctic", "show3d", "hot3d", "hot3d_ume", "epic_central", "epic_perframe"]
SERIES_PTS = 160


def r1(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 2)


def load():
    D = {}
    for k in DATASETS:
        p = RES / f"{k}_frames.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            if k == "arctic":
                d["object"] = d.seq.str.split("/").str[1].str.split("_").str[0]
            D[k] = d
    return D


def summary(D):
    out = {"datasets": [], "overall": {}, "objects": {}, "cdf": {}}
    qs = np.linspace(0, 100, 201)
    for k, d in D.items():
        out["datasets"].append(k)
        out["overall"][k] = {m: r1(v) for m, v in stats(d).items()}
        c = d[d.contact]
        out["cdf"][k] = {
            "pen": [r1(v) for v in np.percentile(c.pen_max_mm, qs)],
            "jit": [r1(v) for v in np.percentile(c.rel_hf_mm.dropna(), qs)] if c.rel_hf_mm.notna().any() else None,
            "n_pen": int(len(c)), "n_jit": int(c.rel_hf_mm.notna().sum()),
        }
        rows = []
        for o, g in d.groupby("object"):
            s = stats(g)
            gc = g[g.contact]
            rows.append({"object": o, "frames": int(len(g)), "contact": r1(s["contact %"]),
                         "pen50": r1(s["pen p50 mm"]), "pen90": r1(s["pen p90 mm"]), "gt5": r1(s["pen>5mm %"]),
                         "jit50": r1(s["rel jitter p50 mm"]), "jit95": r1(s["rel jitter p95 mm"]),
                         "clean": r1(s["sim-clean % (pen<=1mm & jitter<=0.5mm)"]),
                         "seqs": int(gc.seq.nunique())})
        out["objects"][k] = sorted(rows, key=lambda r: -(r["pen90"] or 0))
    return out


def series(D):
    out = {}
    for k in ("arctic", "show3d", "epic_perframe"):
        if k not in D:
            continue
        d = D[k].sort_values(["seq", "hand", "frame"])
        seqs = {}
        for (s, h), g in d.groupby(["seq", "hand"], sort=False):
            n = len(g)
            b = max(1, int(np.ceil(n / SERIES_PTS)))
            idx = np.arange(n) // b
            gg = pd.DataFrame({"i": idx, "f": g.frame.to_numpy(), "p": g.pen_max_mm.to_numpy(),
                               "j": g.rel_hf_mm.to_numpy(), "c": g.contact.to_numpy()})
            a = gg.groupby("i").agg(f=("f", "first"), p=("p", "max"), j=("j", "mean"), c=("c", "mean"))
            e = seqs.setdefault(s, {"obj": str(g.object.iloc[0]), "hands": {}})
            e["hands"][h] = {
                "f": a.f.astype(int).tolist(),
                "p": [round(float(x), 1) for x in a.p],
                "j": [None if not np.isfinite(x) else round(float(x), 2) for x in a.j],
                "c": [round(float(x), 2) for x in a.c],
                "pen50": r1(g[g.contact].pen_max_mm.median()) if g.contact.any() else None,
                "jit50": r1(g[g.contact].rel_hf_mm.median()) if g.contact.any() else None,
                "n": int(n),
            }
        out[k] = seqs
    return out


def pick_frames(d):
    c = d[d.contact].sort_values("pen_max_mm").reset_index(drop=True)
    picks = {}
    if len(c) == 0:
        return picks
    clean = c[(c.pen_max_mm == 0) & (c.min_sd_mm > -1)]
    if len(clean):
        picks["cleanest contact"] = clean.iloc[len(clean) // 2]
    for name, q in (("median", 0.5), ("p90", 0.9), ("p99", 0.99)):
        picks[name] = c.iloc[min(len(c) - 1, int(q * len(c)))]
    picks["worst"] = c.iloc[-1]
    return picks


def mesh_payload(V, F):
    return {"v": np.round(V, 1).ravel().tolist(), "f": F.astype(int).ravel().tolist()}


def frames3d(D):
    rp._init()
    hs = rp._G["hs"]
    out = {"mano_faces": hs.faces.astype(int).ravel().tolist(), "frames": [], "objects": {}}
    hot_tasks = {}
    if "hot3d" in D or "hot3d_ume" in D:
        info = json.load(open(rp.HOT3D_MODELS / "models_info.json"))
        for p, oid in rp.hot3d_list():
            hot_tasks[f"{Path(p).parent.name}/{Path(p).stem}:{info[str(oid)]['name']}"] = (p, oid)
    epic_tasks = {(f"epic_{k}", Path(p).name if k == "central" else Path(p).stem): (k, p) for k, p in rp.epic_list()}
    show_tasks = {f"{t[1]}/{t[2]}": t for t in rp.show3d_list()} if "show3d" in D else {}
    for k, d in D.items():
        for label, row in pick_frames(d).items():
            if k == "arctic":
                seq = rp.arctic_load(str(rp.ARCTIC / "raw_seqs" / f"{row.seq}.mano.npy"))
            elif k == "show3d":
                seq = rp.show3d_load(show_tasks[row.seq])
            elif k in ("hot3d", "hot3d_ume"):
                seq = (rp.hot3d_load if k == "hot3d" else rp.hot3d_ume_load)(hot_tasks[row.seq])
            else:
                seq = rp.epic_load(epic_tasks[(k, row.seq)])
            i = int(np.where(seq.frame_idx == row.frame)[0][0])
            Vw = seq.hands[row.hand][0][i:i + 1]
            Vo = apply_T(inv_T(seq.T_wo[i:i + 1]), Vw)
            if isinstance(seq.parts, Articulated):
                V, F = seq.parts.mesh_at(float(seq.part_T[i]))
                sd = articulated_sdf(seq.parts, seq.part_T[i:i + 1], Vo)[0]
                okey = f"{k}:{row.seq}:{row.frame}"
            else:
                V, F = seq.parts[0].V, seq.parts[0].F
                sd = object_sdf(seq.parts, None, Vo)[0]
                okey = (f"{k}:{row.object}" if k == "show3d" else f"hot3d:{row.object}" if k.startswith("hot3d")
                        else f"{k}:{row.seq}")
            if okey not in out["objects"]:
                out["objects"][okey] = mesh_payload(V, F)
            out["frames"].append({
                "dataset": k, "label": label, "seq": row.seq, "hand": row.hand, "frame": int(row.frame),
                "object": str(row.object), "obj_key": okey,
                "pen_max": r1(row.pen_max_mm), "pen_vol": r1(row.pen_vol_mm3), "n_pen": int(row.n_pen),
                "jit": r1(row.rel_hf_mm),
                "hand_v": np.round(Vo[0], 1).ravel().tolist(),
                "sd": np.round(sd, 1).tolist(),
                **({"faces": f"ume{seq.hs.nv}"} if k == "hot3d_ume" else {}),
            })
            if k == "hot3d_ume":  # UmeTrack topology varies by participant
                out.setdefault("hand_faces", {})[f"ume{seq.hs.nv}"] = seq.hs.faces.astype(int).ravel().tolist()
            print(k, label, row.seq, row.frame, f"{row.pen_max_mm:.1f} mm", flush=True)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = load()
    for name, fn in (("summary", summary), ("series", series), ("frames3d", frames3d)):
        data = fn(D)
        if name == "frames3d":  # one file per object mesh, fetched on demand by the viewer
            import hashlib
            (OUT / "obj").mkdir(exist_ok=True)
            for old in (OUT / "obj").glob("*.json"):
                old.unlink()
            for old in OUT.glob("objects_*.json"):
                old.unlink()
            files = {}
            for k, m in data.pop("objects").items():
                fn = f"obj/{hashlib.md5(k.encode()).hexdigest()[:12]}.json"
                (OUT / fn).write_text(json.dumps(m, separators=(",", ":")), encoding="utf-8")
                files[k] = fn
            data["obj_files"] = files
            print(f"wrote {len(files)} object meshes, {sum((OUT / f).stat().st_size for f in files.values()) / 1e6:.1f} MB")
        p = OUT / f"{name}.json"
        p.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        print(f"wrote {p} {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
