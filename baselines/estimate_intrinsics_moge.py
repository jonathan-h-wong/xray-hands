"""Estimate camera intrinsics with MoGe-2 (github.com/microsoft/MoGe) for the images of an eval set.

ARCTIC ships calibrated intrinsics for every view, so this is the fallback / ablation path: it answers "how
much do the baselines lose when K must be estimated from the image?". One K per (sequence, view) - the
median over --frames sampled images, since ARCTIC cameras are fixed per sequence - plus the per-frame values.

Images are preprocessed exactly as the consumer sees them:
  --for hand2d     undistorted (matches infer_hand2d.py default)
  --for hopformer  raw distorted ego frame (matches infer_hopformer.py)

    pip install git+https://github.com/microsoft/MoGe.git
    python baselines/estimate_intrinsics_moge.py --eval-set baselines/eval_sets/arctic_p2_val_v0_s5.npz --for hand2d
Writes baselines/preds/moge_<for>.json: {"sid/seq/view": {"K_fed": 3x3 px of the fed image, "K_full": 3x3 px of
the full-resolution frame, "scale": fed/full, "per_frame_f": [...]}}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--for", dest="consumer", required=True, choices=["hand2d", "hopformer"])
    ap.add_argument("--image-kind", default="cropped", choices=["cropped", "full"])
    ap.add_argument("--model", default="Ruicheng/moge-2-vitl-normal")
    ap.add_argument("--frames", type=int, default=12, help="images sampled per sequence/view")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import cv2
    import torch
    from moge.model.v2 import MoGeModel
    device = "cuda"
    model = MoGeModel.from_pretrained(a.model).to(device).eval()
    es, meta = C.load_eval_set(a.eval_set)
    out = {}
    for sid, seq, view, idx in C.group_rows(es):
        cam = C.camera(sid, seq, view)
        s = C.image_scale(view, a.image_kind)
        K_nat = C.scale_K(cam["K"], s)
        pick = idx[np.linspace(0, len(idx) - 1, min(a.frames, len(idx))).astype(int)]
        Ks = []
        for r in pick:
            sub = "cropped_images" if a.image_kind == "cropped" else "images"
            bgr = cv2.imread(str(C.ARCTIC / sub / es["image"][r]))
            if bgr is None:
                continue
            if a.consumer == "hand2d" and cam["dist"] is not None:
                bgr = C.undistort(bgr, K_nat, cam["dist"])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            x = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
            with torch.no_grad():
                K = model.infer(x)["intrinsics"].float().cpu().numpy().reshape(3, 3)
            if K[0, 2] < 2.0:          # MoGe returns intrinsics normalised by image width/height
                K[0] *= W
                K[1] *= H
            Ks.append(K)
        if not Ks:
            continue
        Ks = np.stack(Ks)
        K_fed = np.median(Ks, 0)
        key = f"{sid}/{seq}/{view}"
        out[key] = {"K_fed": K_fed.tolist(), "K_full": C.scale_K(K_fed, 1.0 / s).tolist(), "scale": s,
                    "per_frame_f": Ks[:, 0, 0].tolist(), "native_f": float(K_nat[0, 0])}
        print(f"{key}: f {K_fed[0, 0]:.1f} px (native {K_nat[0, 0]:.1f}), "
              f"c ({K_fed[0, 2]:.1f}, {K_fed[1, 2]:.1f}) vs ({K_nat[0, 2]:.1f}, {K_nat[1, 2]:.1f})", flush=True)
    path = Path(a.out) if a.out else C.ROOT / f"baselines/preds/moge_{a.consumer}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
