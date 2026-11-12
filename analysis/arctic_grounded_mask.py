"""Write per-frame "physically grounded" masks for ARCTIC, mirroring the raw_seqs layout.

A hand-frame is grounded if its max penetration into the object is below --thresh mm (default 2.0).
Hands that are not in contact (penetration 0) are grounded: free-floating hands are physically valid.

The dataset itself is never modified. For every raw_seqs/sXX/<seq>.mano.npy this writes
    <out>/sXX/<seq>.npz with
        left, right : bool (T,)     grounded mask per hand
        both        : bool (T,)     left & right
        pen_left_mm, pen_right_mm : float32 (T,)  max penetration depth (re-threshold without rerunning)
        thresh_mm   : float scalar
and <out>/index.json with the parameters and per-sequence keep counts.
Row t of every mask matches row t of the sequence's raw_seqs arrays (mano/object/smplx/egocam).

Usage (from project root):
    .venv\\Scripts\\python analysis\\arctic_grounded_mask.py
    .venv\\Scripts\\python analysis\\arctic_grounded_mask.py --thresh 2.0 --out data/arctic/unpack/arctic_data/data/grounded_mask
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCTIC = ROOT / "data/arctic/unpack/arctic_data/data"
FRAMES = ROOT / "results/arctic_frames.parquet"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thresh", type=float, default=2.0, help="max penetration (mm), strict <")
    ap.add_argument("--frames", type=Path, default=FRAMES)
    ap.add_argument("--arctic", type=Path, default=ARCTIC, help="arctic_data/data root")
    ap.add_argument("--out", type=Path, default=ARCTIC / "grounded_mask")
    args = ap.parse_args()

    df = pd.read_parquet(args.frames, columns=["seq", "hand", "frame", "pen_max_mm"])
    if df.pen_max_mm.isna().any():
        sys.exit("pen_max_mm contains NaN; refusing to guess a mask for those frames")
    by_seq = {k: g for k, g in df.groupby("seq")}

    mano_files = sorted((args.arctic / "raw_seqs").glob("s*/*.mano.npy"))
    if not mano_files:
        sys.exit(f"no raw_seqs found under {args.arctic}")

    index = {"thresh_mm": args.thresh, "source": str(args.frames.name), "sequences": {}}
    tot = {"frames": 0, "left": 0, "right": 0, "both": 0}
    missing = []
    for mf in mano_files:
        seq = f"{mf.parent.name}/{mf.name[: -len('.mano.npy')]}"
        mano = np.load(mf, allow_pickle=True).item()
        T = len(mano["right"]["trans"])
        g = by_seq.get(seq)
        if g is None:
            missing.append(seq)
            continue

        out = {}
        for hand in ("left", "right"):
            h = g[g.hand == hand].sort_values("frame")
            frames = h.frame.to_numpy()
            if len(h) != T or not np.array_equal(frames, np.arange(T)):
                sys.exit(f"{seq}/{hand}: parquet has {len(h)} frames, raw_seqs has {T}; metrics out of sync")
            pen = h.pen_max_mm.to_numpy(np.float32)
            out[f"pen_{hand}_mm"] = pen
            out[hand] = pen < args.thresh
        out["both"] = out["left"] & out["right"]

        dst = args.out / mf.parent.name / (mf.name[: -len(".mano.npy")] + ".npz")
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, thresh_mm=np.float32(args.thresh), **out)

        counts = {"frames": T, **{k: int(out[k].sum()) for k in ("left", "right", "both")}}
        index["sequences"][seq] = counts
        for k in tot:
            tot[k] += counts[k]

    index["totals"] = tot
    index["missing_metrics"] = missing
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.json").write_text(json.dumps(index, indent=1))

    n = len(index["sequences"])
    print(f"wrote {n} masks to {args.out}  (thresh < {args.thresh} mm)")
    for k in ("left", "right", "both"):
        print(f"  {k:5s} grounded: {tot[k]:7d}/{tot['frames']} ({100 * tot[k] / tot['frames']:.1f}%)")
    if missing:
        print(f"  WARNING: {len(missing)} raw_seqs without metrics, skipped: {missing[:5]}...")


if __name__ == "__main__":
    main()
