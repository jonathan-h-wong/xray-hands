"""Run HOPformer (ECCV 2026; github.com/Sid2697/HOPformer) on an ARCTIC egocentric eval set.

Inputs are rebuilt the way HOPformer's own ArcticDataset builds them for the egocentric view (setup p2,
speedup=True), but without needing ARCTIC's processed splits:
  scene image  ARCTIC cropped_images ego frame (full frame x0.3, still distorted, as in training), centre-
               square crop of side max(W,H) resized to 224 (data_utils.rgb_processing, no augmentation),
               ImageNet normalisation.
  intrinsics   the GT ego K rescaled to that 224 patch (data_utils.get_aug_intrix with use_gt_k, which the
               ego setup forces), so predictions come out in metric ego-camera coordinates (undistorted space).
               --intrinsics moge:<json> swaps in MoGe-2 estimates.
  hand crops   ASSUMPTION: the repo reads pre-made 256px crops (cropped_right/, cropped_left/) but does not
               ship or document how ARCTIC crops were made. We detect hands with WiLoR's YOLO detector (HOPformer
               uses WiLoR features) and take a square box of side --crop-rescale x max(w,h) around the detection,
               then apply HOPformer's own loader steps (letterbox to 256, INTER_AREA, flip the left crop, /255,
               ImageNet normalisation). A missing hand gets an all-zero crop, as HOPformer's loader does.
  object name  the GT object name is passed as query_names (HOPformer's ARCTIC protocol does the same).

Setup (on the GPU box): clone HOPformer, follow its README (env, smplx 21-joint patch, MANO in
data/body_models/mano, WiLoR weights), link this repo's ARCTIC data to <hopformer>/data/arctic_data/data, and
download release_models/p2_arctic_ego_epoch8.ckpt. Then
    python baselines/infer_hopformer.py --repo ~/HOPformer --eval-set baselines/eval_sets/arctic_p2_val_v0_s5.npz \
        --detector ~/WiLoR/pretrained_models/detector.pt
Writes <out>/hopformer/<sid>__<seq>__v0.npz
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

IMG_RES = 224
HAND_RES = 256
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def patch_K(K_full, W, H, img_res=IMG_RES):
    """GT K (full-res) -> K of the 224 centre-square patch (get_aug_intrix, use_gt_k=True, sc=1)."""
    dim = float(max(W, H))
    k = img_res / dim
    Kp = np.array(K_full, np.float64)
    Kp[0, 0] *= k
    Kp[1, 1] *= k
    Kp[0, 2] = (Kp[0, 2] - (W / 2.0 - dim / 2.0)) * k
    Kp[1, 2] = (Kp[1, 2] - (H / 2.0 - dim / 2.0)) * k
    return Kp


def square_crop(img_rgb, box, rescale):
    """Square crop (zero padded outside the image) around a detection box."""
    import cv2
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0) * rescale
    M = np.array([[1, 0, s / 2 - cx], [0, 1, s / 2 - cy]], np.float32)
    n = max(int(round(s)), 2)
    return cv2.warpAffine(img_rgb, M, (n, n), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def hand_tensor(crop, flip):
    """HOPformer read_img_reshape + convert_hand_crop_to_tensor + normalize."""
    import cv2
    if crop is None:
        x = np.zeros((HAND_RES, HAND_RES, 3), np.float32)
    else:
        h, w = crop.shape[:2]
        sc = HAND_RES / max(h, w)
        nw, nh = int(w * sc), int(h * sc)
        x = cv2.resize(crop.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
        dh, dw = HAND_RES - nh, HAND_RES - nw
        x = cv2.copyMakeBorder(x, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if flip:
            x = x[:, ::-1, :]
    x = (x / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1), np.float32)


def load_intrinsics(spec):
    if spec == "native":
        return None
    kind, path = spec.split(":", 1)
    assert kind == "moge"
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="cloned HOPformer repo")
    ap.add_argument("--ckpt", default="release_models/p2_arctic_ego_epoch8.ckpt", help="relative to --repo")
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--detector", required=True, help="WiLoR YOLO hand detector (detector.pt)")
    ap.add_argument("--det-conf", type=float, default=0.3)
    ap.add_argument("--crop-rescale", type=float, default=2.0)
    ap.add_argument("--intrinsics", default="native", help="native | moge:<json from estimate_intrinsics_moge.py>")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=str(C.ROOT / "baselines/preds"))
    ap.add_argument("--tag", default="hopformer")
    ap.add_argument("--limit-seqs", type=int, default=0)
    a = ap.parse_args()

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")   # HOPformer builds a pyrender renderer at init (headless)
    import cv2
    es, meta = C.load_eval_set(Path(a.eval_set).resolve())
    out_dir = Path(a.out).resolve() / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    moge = load_intrinsics(a.intrinsics)
    from ultralytics import YOLO
    detector = YOLO(str(Path(a.detector).expanduser().resolve()))

    repo = Path(a.repo).expanduser().resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.argv = ["infer", "--method", "transformer_sf", "--setup", "p2", "--dataset", "arctic",
                "--backbone", "vit-g", "--decoder_depth", "12", "--precision", "32", "--load_ckpt", a.ckpt]
    import torch
    import src.factory as factory
    from common.data_utils import rgb_processing
    from src.parsers.parser import construct_args
    args = construct_args()
    device = "cuda"
    wrapper = factory.fetch_model(args).to(device)
    wrapper.load_from_ckpt(a.ckpt)
    wrapper.eval()
    wrapper.model.arti_head.object_tensors.to(device)
    model = wrapper.model
    model.mode = "val"
    detector.to(device)

    groups = C.group_rows(es)[: a.limit_seqs or None]
    t_start, done = time.time(), 0
    for gi, (sid, seq, view, idx) in enumerate(groups):
        if view != 0:
            raise SystemExit("HOPformer p2 is egocentric; build the eval set with --views 0")
        dest = out_dir / C.shard_name(sid, seq, view)
        if dest.exists():
            continue
        cam = C.camera(sid, seq, view)
        W, H = cam["size"]
        K_full = cam["K"] if moge is None else np.asarray(moge[f"{sid}/{seq}/{view}"]["K_full"], np.float64)
        Kp = patch_K(K_full, W, H)
        n = len(idx)
        res = {"rows": idx, "found": np.zeros(n, bool), "K_full": K_full, "K_patch": Kp}
        for side in C.SIDES:
            res[f"{side}_det"] = np.zeros(n, bool)
            res[f"{side}_score"] = np.zeros(n, np.float32)
            res[f"{side}_box"] = np.full((n, 4), np.nan, np.float32)
            res[f"{side}_verts"] = np.full((n, 778, 3), np.nan, np.float32)
            res[f"{side}_cam_t"] = np.full((n, 3), np.nan, np.float32)
            res[f"{side}_pose"] = np.full((n, 16, 3, 3), np.nan, np.float32)
            res[f"{side}_betas"] = np.full((n, 10), np.nan, np.float32)
        res.update({"obj_rot": np.full((n, 3, 3), np.nan, np.float32), "obj_cam_t": np.full((n, 3), np.nan, np.float32),
                    "obj_radian": np.full(n, np.nan, np.float32), "obj_logits": np.full((n, 11), np.nan, np.float32)})
        obj = seq.split("_")[0]
        for b0 in range(0, n, a.batch):
            bidx = list(range(b0, min(n, b0 + a.batch)))
            imgs, lh, rh, keep = [], [], [], []
            for k in bidx:
                bgr = cv2.imread(str(C.ARCTIC / "cropped_images" / es["image"][idx[k]]))
                if bgr is None:   # no cropped_images: rebuild it (ARCTIC's ego crop is the full frame x0.3)
                    full = cv2.imread(str(C.ARCTIC / "images" / es["image"][idx[k]]))
                    if full is None:
                        continue
                    bgr = cv2.resize(full, (int(full.shape[1] * C.EGO_SCALE), int(full.shape[0] * C.EGO_SCALE)),
                                     interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
                s = rgb.shape[1] / W                         # 0.3 for cropped_images
                aug = {"rot": 0, "sc": 1.0, "pn": np.ones(3), "flip": 0}
                x = rgb_processing(False, rgb, [W / 2 * s, H / 2 * s], max(W, H) / 200.0 * s, aug, img_res=IMG_RES)
                imgs.append((x.transpose(1, 2, 0) - MEAN).transpose(2, 0, 1) / STD[:, None, None])
                best = {}
                for d in detector(bgr, conf=a.det_conf, verbose=False)[0]:
                    side = "right" if int(d.boxes.cls.item()) == 1 else "left"
                    sc = float(d.boxes.conf.item())
                    if side not in best or sc > best[side][1]:
                        best[side] = (d.boxes.xyxy.cpu().numpy().reshape(4), sc)
                for side, lst in (("left", lh), ("right", rh)):
                    crop = square_crop(rgb, best[side][0], a.crop_rescale) if side in best else None
                    lst.append(hand_tensor(crop, flip=side == "left"))
                    if side in best:
                        res[f"{side}_det"][k] = True
                        res[f"{side}_score"][k] = best[side][1]
                        res[f"{side}_box"][k] = best[side][0]
                keep.append(k)
                res["found"][k] = True
            if not keep:
                continue
            B = len(keep)
            inputs = {"img": torch.from_numpy(np.stack(imgs).astype(np.float32)).to(device),
                      "left_hand": torch.from_numpy(np.stack(lh)).to(device),
                      "right_hand": torch.from_numpy(np.stack(rh)).to(device)}
            meta_info = {"intrinsics": torch.from_numpy(np.repeat(Kp[None], B, 0).astype(np.float32)).to(device),
                         "query_names": [obj] * B}
            with torch.no_grad():
                out = model(inputs, meta_info)
            g = lambda key: out[key].float().cpu().numpy()  # noqa: E731
            for side, sfx in (("left", "l"), ("right", "r")):
                res[f"{side}_verts"][keep] = g(f"mano.v3d.cam.{sfx}")
                res[f"{side}_cam_t"][keep] = g(f"mano.cam_t.{sfx}")
                res[f"{side}_pose"][keep] = g(f"mano.pose.{sfx}").reshape(B, 16, 3, 3)
                res[f"{side}_betas"][keep] = g(f"mano.beta.{sfx}")
            res["obj_rot"][keep] = g("object.rot").reshape(B, 3, 3)
            res["obj_cam_t"][keep] = g("object.cam_t")
            res["obj_radian"][keep] = g("object.radian").reshape(B)
            res["obj_logits"][keep] = g("object.class_logits").reshape(B, -1)[:, :11]
        tmp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, info=json.dumps({"method": "hopformer", "ckpt": a.ckpt, "intrinsics": a.intrinsics,
                                                  "crop_rescale": a.crop_rescale, "det_conf": a.det_conf}), **res)
        tmp.replace(dest)
        done += n
        print(f"[{gi + 1}/{len(groups)}] {sid}/{seq}: {res['found'].sum()}/{n} images, det L "
              f"{res['left_det'].mean():.2f} R {res['right_det'].mean():.2f} ({done / (time.time() - t_start):.1f} img/s)",
              flush=True)
    print("done")


if __name__ == "__main__":
    main()
