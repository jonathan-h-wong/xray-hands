"""End-to-end check of the eval pipeline without a GPU or images.

Writes "oracle" prediction shards derived from ARCTIC GT in both formats:
  oracle_hand2d  HaMeR/WiLoR format: model-frame mesh + a crop camera / box / 2D keypoints that are consistent
                 with the true camera, so both lifts must recover the GT wrist (wp exactly up to fx!=fy
                 averaging; fit2d exactly)
  oracle_hop     HOPformer format: GT hands and GT object pose in the camera frame
then runs eval_arctic.py on them. Expected: pose errors ~0 (or ~--noise-mm), pen == the GT row, object errors 0.

    .venv\\Scripts\\python baselines\\selftest.py --eval-set baselines/eval_sets/arctic_p2_val_v0_s5.npz --seqs 3
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from eval_arctic import ARCTIC_OBJECTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--seqs", type=int, default=3)
    ap.add_argument("--noise-mm", type=float, default=0.0, help="iid vertex noise added to the oracle meshes")
    ap.add_argument("--preds", default=str(C.ROOT / "baselines/preds_selftest"))
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    es, meta = C.load_eval_set(a.eval_set)
    groups = C.group_rows(es)[: a.seqs]
    for sid, seq, view, idx in groups:
        gt = C.gt_sequence(sid, seq, view)
        cam = gt["cam"]
        s = C.image_scale(view, "cropped")
        K = C.scale_K(cam["K"], s)
        W, H = (int(cam["size"][0] * s), int(cam["size"][1] * s))
        fr = es["frame"][idx]
        n = len(idx)
        h2 = {"rows": idx, "found": np.ones(n, bool), "img_hw": np.tile([H, W], (n, 1)), "K_native": K,
              "undistorted": True, "crop_focal": 5000.0, "crop_res": 256, "image_kind": "cropped",
              "info": json.dumps({"method": "oracle"})}
        hop = {"rows": idx, "found": np.ones(n, bool), "K_full": cam["K"], "K_patch": np.eye(3),
               "info": json.dumps({"method": "oracle"})}
        f = 0.5 * (K[0, 0] + K[1, 1])
        for side in C.SIDES:
            Vg = gt["verts"][side][fr] + rng.normal(0, a.noise_mm / 1000.0, (n, 778, 3))
            Jg = C.joints21(Vg, side)
            T = Jg[:, 0]
            Vm, Jm = Vg - T[:, None], Jg - T[:, None]
            uv = C.project(K, Jg)
            lo, hi = uv.min(1), uv.max(1)
            c = (lo + hi) / 2
            b = (hi - lo).max(1) * 2.0
            sc = 2 * f / (b * T[:, 2])
            cam_crop = np.stack([sc, T[:, 0] - 2 * (c[:, 0] - K[0, 2]) / (b * sc),
                                 T[:, 1] - 2 * (c[:, 1] - K[1, 2]) / (b * sc)], 1)
            h2.update({f"{side}_det": np.ones(n, bool), f"{side}_score": np.ones(n, np.float32),
                       f"{side}_ndet": np.ones(n, np.int16), f"{side}_box": np.c_[lo, hi].astype(np.float32),
                       f"{side}_center": c.astype(np.float32), f"{side}_size": b.astype(np.float32),
                       f"{side}_cam": cam_crop.astype(np.float32), f"{side}_verts": Vm.astype(np.float32),
                       f"{side}_kp3d": Jm.astype(np.float32), f"{side}_kp2d": uv.astype(np.float32)})
            hop.update({f"{side}_det": np.ones(n, bool), f"{side}_score": np.ones(n, np.float32),
                        f"{side}_verts": Vg.astype(np.float32), f"{side}_cam_t": T.astype(np.float32)})
        Tco = gt["T_co"][fr]
        logits = np.full((n, 11), -10.0)
        logits[:, ARCTIC_OBJECTS.index(gt["obj"])] = 10.0
        hop.update({"obj_rot": Tco[:, :3, :3], "obj_cam_t": Tco[:, :3, 3] / 1000.0, "obj_radian": gt["angle"][fr],
                    "obj_logits": logits})
        for name, d in (("oracle_hand2d", h2), ("oracle_hop", hop)):
            p = Path(a.preds) / name / C.shard_name(sid, seq, view)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, **d)
        print(f"wrote oracle shards for {sid}/{seq} v{view} ({n} images)", flush=True)
    # hand2d detection by HaMeR/WiLoR is keyed on 'obj_rot' being absent; run the evaluator
    cmd = [sys.executable, "-W", "ignore", str(HERE / "eval_arctic.py"), "--eval-set", a.eval_set,
           "--methods", "oracle_hand2d", "oracle_hop", "--preds", a.preds, "--limit-seqs", str(a.seqs),
           "--workers", str(min(a.seqs, 4)), "--tag", f"selftest_noise{a.noise_mm:g}"]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
