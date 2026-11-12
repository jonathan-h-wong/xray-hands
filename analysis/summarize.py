"""Summarize results/*_frames.parquet into results/summary.md (+ per-object CSVs).

All statistics over *contact* frames (closest hand point within 5 mm of the object)
unless stated. Smoothness metrics are in the object frame (hand relative to object).
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PEN_OK_MM = 1.0      # strict "sim-clean" penetration tolerance
JIT_OK_MM = 0.5      # strict "sim-clean" high-frequency jitter tolerance


def q(s, p):
    s = s.dropna()
    return np.nan if s.empty else float(np.percentile(s, p))


def stats(d):
    c = d[d.contact]
    j = c.rel_hf_mm.dropna()
    out = {
        "hand-frames": len(d),
        "contact %": 100 * d.contact.mean(),
        "pen>0 %": 100 * (c.pen_max_mm > 0).mean(),
        "pen>2mm %": 100 * (c.pen_max_mm > 2).mean(),
        "pen>5mm %": 100 * (c.pen_max_mm > 5).mean(),
        "pen>10mm %": 100 * (c.pen_max_mm > 10).mean(),
        "pen p50 mm": q(c.pen_max_mm, 50),
        "pen p90 mm": q(c.pen_max_mm, 90),
        "pen p99 mm": q(c.pen_max_mm, 99),
        "pen vol p50 mm3": q(c.pen_vol_mm3, 50),
        "pen vol p90 mm3": q(c.pen_vol_mm3, 90),
        "jitter-evaluable %": 100 * len(j) / max(len(c), 1),
        "rel jitter p50 mm": q(j, 50),
        "rel jitter p95 mm": q(j, 95),
        "rel jitter>0.5mm %": 100 * (j > JIT_OK_MM).mean() if len(j) else np.nan,
        "rel jitter>1mm %": 100 * (j > 1).mean() if len(j) else np.nan,
        "rel acc p50 m/s2": q(c.rel_acc_ms2, 50),
        "rel acc p95 m/s2": q(c.rel_acc_ms2, 95),
        "hand world jitter p50 mm": q(c.hand_world_hf_mm, 50),
        "obj world jitter p50 mm": q(c.obj_world_hf_mm, 50),
        "|dpen| p50 mm": q(c.dpen_mm, 50),
        "|dpen| p95 mm": q(c.dpen_mm, 95),
        "|dpen|>2mm %": 100 * (c.dpen_mm.dropna() > 2).mean() if c.dpen_mm.notna().any() else np.nan,
    }
    ev = c[c.rel_hf_mm.notna()]
    out["sim-clean % (pen<=1mm & jitter<=0.5mm)"] = (
        100 * ((ev.pen_max_mm <= PEN_OK_MM) & (ev.rel_hf_mm <= JIT_OK_MM)).mean() if len(ev) else np.nan)
    out["pen<=1mm % (all contact)"] = 100 * (c.pen_max_mm <= PEN_OK_MM).mean()
    return out


def fmt(df):
    return df.to_markdown(floatfmt=".2f")


def main():
    files = sorted(glob.glob(str(RES / "*_frames.parquet")))
    if not files:
        sys.exit("no results/*_frames.parquet yet")
    D = {Path(f).name.replace("_frames.parquet", ""): pd.read_parquet(f) for f in files}
    lines = ["# Hand-object penetration & smoothness summary", "",
             "Contact frame = closest hand surface point within 5 mm of the object. "
             "Penetration = max depth of the (subdivided) hand surface inside the object. "
             "Jitter = RMS-like mean residual of hand vertices *in the object frame* after a 6 Hz low-pass. "
             f"Sim-clean = pen <= {PEN_OK_MM} mm and jitter <= {JIT_OK_MM} mm.", ""]
    overall = pd.DataFrame({k: stats(v) for k, v in D.items()})
    lines += ["## Overall (contact frames)", "", fmt(overall), ""]
    for k, v in D.items():
        if "object" not in v.columns and k == "arctic":
            v["object"] = v.seq.str.split("/").str[1].str.split("_").str[0]
        if "object" in v.columns:
            per = pd.DataFrame({o: stats(g) for o, g in v.groupby("object")}).T
            cols = ["hand-frames", "contact %", "pen>2mm %", "pen>5mm %", "pen p50 mm", "pen p90 mm",
                    "rel jitter p50 mm", "rel jitter p95 mm", "sim-clean % (pen<=1mm & jitter<=0.5mm)"]
            per = per[cols].sort_values("pen p90 mm", ascending=False)
            per.to_csv(RES / f"{k}_per_object.csv")
            lines += [f"## {k}: per object", "", fmt(per), ""]
        if k in ("arctic", "show3d", "hot3d", "hot3d_ume"):
            c = v[v.contact]
            seq = c.groupby("seq").agg(pen_p50=("pen_max_mm", "median"),
                                       pen_gt5_pct=("pen_max_mm", lambda s: 100 * (s > 5).mean()),
                                       jitter_p50=("rel_hf_mm", "median"), n=("frame", "size"))
            seq = seq[seq.n >= 30]
            lines += [f"## {k}: distribution over sequences (>=30 contact frames)", "",
                      fmt(seq.describe(percentiles=[.1, .5, .9]).T), ""]
            seq.sort_values("pen_p50", ascending=False).to_csv(RES / f"{k}_per_seq.csv")
    (RES / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
