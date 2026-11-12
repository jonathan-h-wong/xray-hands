"""Summarize results/parts/* into results/parts_summary.md and viewer/data/parts.json.

All rates are over *hand* contact frames (closest hand point within 5 mm of the object).
A part "penetrates > X mm" when its deepest point (either side) is more than X mm inside.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mano_np import MANO  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PD = ROOT / "results" / "parts"
LAB = json.load(open(ROOT / "results" / "hand_parts.json"))
PARTS = LAB["parts"]
SIDES = ["palmar", "dorsal"]
DATASETS = ["arctic", "show3d", "hot3d", "epic_central", "epic_perframe"]


def r2(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 2)


def part_cols(d, p):
    pen = np.maximum(d[f"{p}_palmar_pen"], d[f"{p}_dorsal_pen"])
    vol = d[f"{p}_palmar_vol"] + d[f"{p}_dorsal_vol"]
    minsd = np.minimum(d[f"{p}_palmar_minsd"], d[f"{p}_dorsal_minsd"])
    return pen, vol, minsd


def group_stats(pen, vol, minsd, total_vol):
    near = minsd < 5
    pn = pen[near]
    return {"contact": r2(100 * near.mean()), "gt2": r2(100 * (pen > 2).mean()), "gt5": r2(100 * (pen > 5).mean()),
            "p50": r2(np.median(pn)) if len(pn) else None, "p90": r2(np.percentile(pn, 90)) if len(pn) else None,
            "vol_share": r2(100 * vol.sum() / total_vol) if total_vol > 0 else None}


def main():
    out = {"parts": PARTS, "sides": SIDES, "datasets": [], "by_part": {}, "by_side": {}, "by_region": {},
           "object_part": {}, "vertex": {}}
    md = ["# Per-part penetration summary", "",
          "Rates over hand contact frames. Volume share = part's share of all penetrating volume.", ""]
    for ds in DATASETS:
        p = PD / f"{ds}_parts.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        if ds == "arctic":
            d["object"] = d.seq.str.split("/").str[1].str.split("_").str[0]
        d = d[d.contact]
        out["datasets"].append(ds)
        tot = sum(d[f"{p}_{s}_vol"].sum() for p in PARTS for s in SIDES)
        bp = {p: group_stats(*part_cols(d, p), tot) for p in PARTS}
        bs = {}
        for s in SIDES:
            pen = np.max(np.stack([d[f"{p}_{s}_pen"] for p in PARTS]), 0)
            vol = sum(d[f"{p}_{s}_vol"] for p in PARTS)
            minsd = np.min(np.stack([d[f"{p}_{s}_minsd"] for p in PARTS]), 0)
            bs[s] = group_stats(pd.Series(pen), vol, pd.Series(minsd), tot)
        br = {f"{p}_{s}": group_stats(d[f"{p}_{s}_pen"], d[f"{p}_{s}_vol"], d[f"{p}_{s}_minsd"], tot)
              for p in PARTS for s in SIDES}
        out["by_part"][ds], out["by_side"][ds], out["by_region"][ds] = bp, bs, br
        if "object" in d.columns:
            op = {}
            for o, g in d.groupby("object"):
                op[o] = {p: r2(100 * (part_cols(g, p)[0] > 2).mean()) for p in PARTS}
                op[o]["_n"] = int(len(g))
            out["object_part"][ds] = op
        v = np.load(PD / f"{ds}_vertex.npz")
        n = max(1, int(v["n_contact"]))
        out["vertex"][ds] = {"n": n, "pen2": np.round(100 * v["pen2"] / n, 2).tolist(),
                             "pen0": np.round(100 * v["pen0"] / n, 2).tolist(),
                             "depth": np.round(v["depth_sum"] / n, 3).tolist()}
        md += [f"## {ds}", "", pd.DataFrame(bp).T.to_markdown(floatfmt=".2f"), "",
               pd.DataFrame(bs).T.to_markdown(floatfmt=".2f"), ""]
    m = MANO(ROOT / "data/arctic/unpack/body_models/mano_v1_2/models/MANO_RIGHT.pkl")
    out["template"] = {"v": np.round(m.v_template * 1000, 2).ravel().tolist(), "f": m.faces.ravel().tolist(),
                       "part": LAB["right"]["part"], "side": LAB["right"]["side"],
                       "palmar_dir": LAB["right"]["palmar_dir"]}
    (ROOT / "results" / "parts_summary.md").write_text("\n".join(md), encoding="utf-8")
    p = ROOT / "viewer" / "data" / "parts.json"
    p.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print("\n".join(md))
    print(f"wrote {p} {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
