"""Run HaMeR or WiLoR, each with its own hand detector, on the full images of an ARCTIC eval set.

Detection follows each repo's demo exactly:
  HaMeR: ViTDet (or RegNetY) person boxes -> ViTPose whole-body keypoints -> hand box from >3 confident hand
         keypoints (conf > 0.5); left/right from the keypoint set. Score = mean confidence of those keypoints.
  WiLoR: its YOLO hand detector (conf 0.3), left/right from the detector class. Score = detector confidence.
The best-scoring detection per side is kept (the number of candidates is saved too). Crops use each repo's
ViTDetDataset with rescale_factor 2.0, as in the demos.

Nothing is lifted here. Per hand the raw crop-level outputs are saved (crop camera, box, model-frame mesh and
keypoints, the model's own 2D keypoints in full-image pixels), and eval_arctic.py lifts them with the native
ARCTIC intrinsics or MoGe-2 estimates. Left hands are mirrored back exactly as the demos do.

Images: egocentric frames are undistorted with ARCTIC's calibration before inference (--no-undistort to
disable), so a pinhole lift is valid.

Run from anywhere; --repo points at the model repo (its code uses repo-relative paths for weights):
    python baselines/infer_hand2d.py --method wilor --repo ~/WiLoR --eval-set baselines/eval_sets/arctic_p2_val_v0_s5.npz
    python baselines/infer_hand2d.py --method hamer --repo ~/hamer --eval-set ...
Writes <out>/<method>/<sid>__<seq>__v<view>.npz (one shard per sequence/view; finished shards are skipped).
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

RESCALE = 2.0


class HaMeR:
    name = "hamer"

    def __init__(self, device, body_detector="vitdet"):
        import torch
        from hamer.configs import CACHE_DIR_HAMER
        from hamer.models import DEFAULT_CHECKPOINT, download_models, load_hamer
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        from vitpose_model import ViTPoseModel
        download_models(CACHE_DIR_HAMER)
        self.model, self.cfg = load_hamer(DEFAULT_CHECKPOINT)
        self.model = self.model.to(device).eval()
        if body_detector == "vitdet":
            import hamer
            from detectron2.config import LazyConfig
            cfg = LazyConfig.load(str(Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"))
            cfg.train.init_checkpoint = ("https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_"
                                         "vitdet_h/f328730692/model_final_f05665.pkl")
            for i in range(3):
                cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        else:
            from detectron2 import model_zoo
            cfg = model_zoo.get_config("new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True)
            cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        self.detector = DefaultPredictor_Lazy(cfg)
        self.cpm = ViTPoseModel(device)
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.utils import recursive_to
        self.Dataset, self.recursive_to, self.torch = ViTDetDataset, recursive_to, torch

    def detect(self, img_bgr):
        """-> list of (box xyxy, is_right, score)."""
        det = self.detector(img_bgr)["instances"]
        keep = (det.pred_classes == 0) & (det.scores > 0.5)
        boxes = det.pred_boxes.tensor[keep].cpu().numpy()
        scores = det.scores[keep].cpu().numpy()
        if not len(boxes):
            return []
        poses = self.cpm.predict_pose(img_bgr[:, :, ::-1].copy(), [np.concatenate([boxes, scores[:, None]], 1)])
        out = []
        for p in poses:
            for right, kp in ((0, p["keypoints"][-42:-21]), (1, p["keypoints"][-21:])):
                valid = kp[:, 2] > 0.5
                if valid.sum() > 3:
                    box = [kp[valid, 0].min(), kp[valid, 1].min(), kp[valid, 0].max(), kp[valid, 1].max()]
                    out.append((np.array(box, np.float32), right, float(kp[valid, 2].mean())))
        return out


class WiLoR:
    name = "wilor"

    def __init__(self, device, conf=0.3):
        import torch
        from ultralytics import YOLO
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.models import load_wilor
        from wilor.utils import recursive_to
        self.model, self.cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt",
                                          cfg_path="./pretrained_models/model_config.yaml")
        self.model = self.model.to(device).eval()
        self.detector = YOLO("./pretrained_models/detector.pt").to(device)
        self.conf = conf
        self.Dataset, self.recursive_to, self.torch = ViTDetDataset, recursive_to, torch

    def detect(self, img_bgr):
        det = self.detector(img_bgr, conf=self.conf, verbose=False)[0]
        out = []
        for d in det:
            b = d.boxes.data.cpu().numpy().reshape(-1)
            out.append((b[:4].astype(np.float32), int(d.boxes.cls.cpu().item()), float(d.boxes.conf.cpu().item())))
        return out


def regress(backend, img_bgr, dets, device):
    """dets: list of (box, is_right, score), at most one per side -> list of per-hand dicts."""
    torch = backend.torch
    boxes = np.stack([d[0] for d in dets])
    right = np.array([d[1] for d in dets], np.float32)
    ds = backend.Dataset(backend.cfg, img_bgr, boxes, right, rescale_factor=RESCALE)
    batch = torch.utils.data.default_collate([ds[i] for i in range(len(ds))])
    batch = backend.recursive_to(batch, device)
    with torch.no_grad():
        out = backend.model(batch)
    res = []
    for n in range(len(dets)):
        r = float(batch["right"][n])
        mul = 2 * r - 1
        cam = out["pred_cam"][n].float().cpu().numpy().copy()
        cam[1] *= mul
        V = out["pred_vertices"][n].float().cpu().numpy().copy()
        J = out["pred_keypoints_3d"][n].float().cpu().numpy().copy()
        V[:, 0] *= mul
        J[:, 0] *= mul
        kp = out["pred_keypoints_2d"][n].float().cpu().numpy().copy()    # crop-normalised, origin at centre
        kp[:, 0] *= mul
        c = batch["box_center"][n].float().cpu().numpy()
        s = float(batch["box_size"][n])
        mp = out["pred_mano_params"]
        res.append({"cam": cam, "verts": V, "kp3d": J, "kp2d": c + kp * s, "center": c, "size": s,
                    "go": mp["global_orient"][n].float().cpu().numpy().reshape(3, 3),
                    "pose": mp["hand_pose"][n].float().cpu().numpy().reshape(15, 3, 3),
                    "betas": mp["betas"][n].float().cpu().numpy().reshape(10)})
    return res


def empty(n):
    d = {}
    for side in C.SIDES:
        d.update({f"{side}_det": np.zeros(n, bool), f"{side}_score": np.zeros(n, np.float32),
                  f"{side}_ndet": np.zeros(n, np.int16), f"{side}_box": np.full((n, 4), np.nan, np.float32),
                  f"{side}_center": np.full((n, 2), np.nan, np.float32), f"{side}_size": np.full(n, np.nan, np.float32),
                  f"{side}_cam": np.full((n, 3), np.nan, np.float32),
                  f"{side}_verts": np.full((n, 778, 3), np.nan, np.float32),
                  f"{side}_kp3d": np.full((n, 21, 3), np.nan, np.float32),
                  f"{side}_kp2d": np.full((n, 21, 2), np.nan, np.float32),
                  f"{side}_go": np.full((n, 3, 3), np.nan, np.float32),
                  f"{side}_pose": np.full((n, 15, 3, 3), np.nan, np.float32),
                  f"{side}_betas": np.full((n, 10), np.nan, np.float32)})
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=["hamer", "wilor"])
    ap.add_argument("--repo", required=True, help="path to the cloned hamer / WiLoR repo (weights inside)")
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--image-kind", default="cropped", choices=["cropped", "full"])
    ap.add_argument("--no-undistort", action="store_true")
    ap.add_argument("--out", default=str(C.ROOT / "baselines/preds"))
    ap.add_argument("--hamer-body-detector", default="vitdet", choices=["vitdet", "regnety"])
    ap.add_argument("--wilor-conf", type=float, default=0.3)
    ap.add_argument("--limit-seqs", type=int, default=0, help="debug: only the first N sequence/view groups")
    a = ap.parse_args()

    import cv2
    es, meta = C.load_eval_set(Path(a.eval_set).resolve())
    out_dir = Path(a.out).resolve() / a.method
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(a.repo).expanduser().resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    device = "cuda"
    backend = HaMeR(device, a.hamer_body_detector) if a.method == "hamer" else WiLoR(device, a.wilor_conf)
    focal, res = float(backend.cfg.EXTRA.FOCAL_LENGTH), int(backend.cfg.MODEL.IMAGE_SIZE)

    groups = C.group_rows(es)[: a.limit_seqs or None]
    t_start, done = time.time(), 0
    for gi, (sid, seq, view, idx) in enumerate(groups):
        dest = out_dir / C.shard_name(sid, seq, view)
        if dest.exists():
            continue
        cam = C.camera(sid, seq, view)
        scale = C.image_scale(view, a.image_kind)
        K = C.scale_K(cam["K"], scale)
        undist = (cam["dist"] is not None) and not a.no_undistort
        n = len(idx)
        d = empty(n)
        found = np.zeros(n, bool)
        hw = np.zeros((n, 2), np.int32)
        for k, r in enumerate(idx):
            p = C.ARCTIC / ("cropped_images" if a.image_kind == "cropped" else "images") / es["image"][r]
            img = cv2.imread(str(p))
            if img is None:
                continue
            if undist:
                img = C.undistort(img, K, cam["dist"])
            found[k] = True
            hw[k] = img.shape[:2]
            cands = backend.detect(img)
            best = {}
            for box, right, score in cands:
                side = "right" if right else "left"
                d[f"{side}_ndet"][k] += 1
                if side not in best or score > best[side][2]:
                    best[side] = (box, right, score)
            if not best:
                continue
            sides = list(best)
            for side, h in zip(sides, regress(backend, img, [best[s] for s in sides], device)):
                d[f"{side}_det"][k] = True
                d[f"{side}_score"][k] = best[side][2]
                d[f"{side}_box"][k] = best[side][0]
                for key in ("cam", "verts", "kp3d", "kp2d", "center", "size", "go", "pose", "betas"):
                    d[f"{side}_{key}"][k] = h[key]
        tmp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, rows=idx, found=found, img_hw=hw, K_native=K, undistorted=undist,
                            crop_focal=focal, crop_res=res, image_kind=a.image_kind,
                            info=json.dumps({"method": a.method, "rescale": RESCALE,
                                             "detector": a.hamer_body_detector if a.method == "hamer" else
                                             f"yolo conf {a.wilor_conf}"}), **d)
        tmp.replace(dest)
        done += n
        rate = done / (time.time() - t_start)
        print(f"[{gi + 1}/{len(groups)}] {sid}/{seq} v{view}: {found.sum()}/{n} images, "
              f"det L {d['left_det'].mean():.2f} R {d['right_det'].mean():.2f}  ({rate:.1f} img/s)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
