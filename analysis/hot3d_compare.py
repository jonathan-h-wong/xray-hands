"""HOT3D: MANO vs native UmeTrack hands on matched (seq, hand, frame) contact frames,
split by device / hand / participant. Writes results/hot3d_compare.md.

Contact is taken from the UmeTrack mesh (the mocap-tracked GT hand) so both models are scored on the same frames.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def stats(g):
    m, u = g.pen_max_mm_m, g.pen_max_mm_u
    return pd.Series({
        "hand-frames": len(g),
        "MANO pen p50": m.median(), "Ume pen p50": u.median(),
        "MANO >5mm %": (m > 5).mean() * 100, "Ume >5mm %": (u > 5).mean() * 100,
        "MANO p90": m.quantile(.9), "Ume p90": u.quantile(.9),
        "Ume jitter p50": g.rel_hf_mm_u.median(),
        "Ume sim-clean %": ((u <= 1) & (g.rel_hf_mm_u <= .5)).mean() * 100,
    })


def main():
    cols = ["seq", "hand", "frame", "pen_max_mm", "rel_hf_mm", "contact", "device", "participant", "object"]
    m = pd.read_parquet(R / "hot3d_frames.parquet", columns=cols)
    u = pd.read_parquet(R / "hot3d_ume_frames.parquet", columns=cols)
    d = u.merge(m[["seq", "hand", "frame", "pen_max_mm", "rel_hf_mm", "contact"]], on=["seq", "hand", "frame"],
                suffixes=("_u", "_m"))
    d = d[d.contact_u]
    out = ["# HOT3D: MANO vs UmeTrack (matched contact frames, contact judged on UmeTrack)\n",
           f"matched contact hand-frames: {len(d)}; MANO also in contact: {d.contact_m.mean() * 100:.1f}%\n",
           f"per-frame MANO - Ume penetration: median {np.nanmedian(d.pen_max_mm_m - d.pen_max_mm_u):.2f} mm, "
           f"p10 {np.nanpercentile(d.pen_max_mm_m - d.pen_max_mm_u, 10):.2f}, "
           f"p90 {np.nanpercentile(d.pen_max_mm_m - d.pen_max_mm_u, 90):.2f}\n"]
    for key in ("device", "hand", "participant"):
        t = d.groupby(key).apply(stats).round(2)
        out += [f"\n## by {key}\n", t.to_markdown(), "\n"]
    (R / "hot3d_compare.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
