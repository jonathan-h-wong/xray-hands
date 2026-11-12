"""Build the ARCTIC evaluation subset (CPU, no images needed).

A hand-frame is selected when
  1. its GT is physically grounded: max penetration < thresh (analysis/arctic_grounded_mask.py), and
  2. the GT hand is visible in the view: wrist + >= 3 joints project inside the image, in front of the camera
     (same rule as ARCTIC's own is_valid / right_valid / left_valid).
An image is kept when at least one hand is selected; models run on the whole image, metrics only use selected
hands. Contact (GT min signed distance < 5 mm) is stored so results can also be cut to in-contact frames.

Run locally (needs raw_seqs + grounded_mask), commit the output, evaluate remotely:
    .venv\\Scripts\\python baselines\\make_eval_set.py --protocol p2 --split val --views 0 --stride 5
Writes baselines/eval_sets/arctic_<protocol>_<split>_v<views>_s<stride>.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

OUT = Path(__file__).parent / "eval_sets"


def visible(J, K, size):
    """J (T,21,3) camera metres -> (T,) bool."""
    uv = C.project(K, J)
    W, H = size
    inside = (J[..., 2] > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
    return inside[:, 0] & (inside.sum(1) >= 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default="p2", choices=["p1", "p2"])
    ap.add_argument("--split", default="val", help="val (s05) has public GT; test (s03) does not")
    ap.add_argument("--views", type=int, nargs="+", default=[0], help="0 = egocentric, 1..8 = static")
    ap.add_argument("--stride", type=int, default=5, help="take every Nth frame (ARCTIC is 30 fps)")
    ap.add_argument("--mask", default=str(C.ARCTIC / "grounded_mask"))
    ap.add_argument("--frames", default=str(C.ROOT / "results/arctic_frames.parquet"),
                    help="for the GT contact flag; optional")
    ap.add_argument("--require-both", action="store_true", help="keep an image only if both hands are selected")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    seqs = json.load(open(C.ARCTIC / f"splits_json/protocol_{a.protocol}.json"))[a.split]
    mask_dir = Path(a.mask)
    contact = None
    if Path(a.frames).exists():
        import pandas as pd
        df = pd.read_parquet(a.frames, columns=["seq", "hand", "frame", "contact"])
        contact = {(s, h): g.sort_values("frame").contact.to_numpy() for (s, h), g in df.groupby(["seq", "hand"])}

    rows = {k: [] for k in ("sid", "seq", "view", "frame", "image", "obj")}
    for side in C.SIDES:
        for k in ("sel", "grounded", "visible", "contact", "pen_gt"):
            rows[f"{k}_{side}"] = []
    thresh = None
    for full in seqs:
        sid, seq = full.split("/")
        mz = np.load(mask_dir / sid / f"{seq}.npz")
        thresh = float(mz["thresh_mm"])
        gt = None
        for view in a.views:
            gt = C.gt_sequence(sid, seq, view)
            cam = gt["cam"]
            T = len(gt["angle"])
            vis = {s: visible(gt["joints"][s], cam["K"], cam["size"]) for s in C.SIDES}
            sel = {s: mz[s] & vis[s] for s in C.SIDES}
            keep = np.zeros(T, bool)
            keep[::a.stride] = True
            keep &= (sel["left"] & sel["right"]) if a.require_both else (sel["left"] | sel["right"])
            idx = np.where(keep)[0]
            rows["sid"] += [sid] * len(idx)
            rows["seq"] += [seq] * len(idx)
            rows["view"] += [view] * len(idx)
            rows["frame"] += idx.tolist()
            rows["obj"] += [gt["obj"]] * len(idx)
            rows["image"] += [f"{sid}/{seq}/{view}/{t + cam['ioi_offset']:05d}.jpg" for t in idx]
            for s in C.SIDES:
                rows[f"sel_{s}"] += sel[s][idx].tolist()
                rows[f"grounded_{s}"] += mz[s][idx].tolist()
                rows[f"visible_{s}"] += vis[s][idx].tolist()
                rows[f"pen_gt_{s}"] += mz[f"pen_{s}_mm"][idx].tolist()
                c = contact.get((full, s)) if contact is not None else None
                rows[f"contact_{s}"] += (c[idx].tolist() if c is not None and len(c) == T else [False] * len(idx))
        print(f"{full}: {sum(1 for x in rows['seq'] if x == seq)} images", flush=True)

    out = {k: np.asarray(v) for k, v in rows.items()}
    meta = {"protocol": a.protocol, "split": a.split, "views": a.views, "stride": a.stride,
            "grounded_thresh_mm": thresh, "require_both": a.require_both, "contact_available": contact is not None,
            "n_images": int(len(out["frame"])),
            "n_hands": {s: int(out[f"sel_{s}"].sum()) for s in C.SIDES}}
    path = Path(a.out) if a.out else OUT / (f"arctic_{a.protocol}_{a.split}_v{''.join(map(str, a.views))}"
                                            f"_s{a.stride}.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta), **out)
    print(json.dumps(meta, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
