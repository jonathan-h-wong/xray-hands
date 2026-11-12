"""Stream HOT3D-Clips (train splits, which carry ground truth) from the BOP Hugging Face repo and keep
only the annotations. Images are read off the stream and discarded, so local storage stays ~MBs.

Per clip -> data/hot3d/clips/<device>/clip-NNNNNN.npz with
    ts (150,) ref timestamps ns; device; sequence_id; participant_id
    betas (10,) MANO shape (shared by both hands, from __hand_shapes.json__)
    {left,right}_valid (150,), {left,right}_thetas (150,15) PCA pose, {left,right}_wrist (150,6) [aa, t(m)]
    {left,right}_ume_valid/_ume_angles (150,22)/_ume_T (150,4,4) UmeTrack pose; umetrack_shape (json str)
    obj_ids (K,) BOP ids; obj_T (K,150,4,4) T_world_from_object (m), NaN where absent

Usage: .venv\\Scripts\\python analysis\\hot3d_fetch.py --split train_aria train_quest3 --workers 8 [--limit N]
"""
import argparse
import json
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import HfApi, hf_hub_url

REPO = "bop-benchmark/hot3d"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "hot3d" / "clips"


def quat_T(q, t):
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def fetch(fname, dest):
    if dest.exists():
        return fname, 0, "skip"
    url = hf_hub_url(REPO, fname, repo_type="dataset")
    hands, objs, info, shape = {}, {}, {}, None
    nbytes = 0
    for attempt in range(5):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                r.raw.decode_content = True
                with tarfile.open(fileobj=r.raw, mode="r|") as tar:
                    for m in tar:
                        nbytes += m.size
                        n = m.name
                        if not n.endswith(".json") and n != "__hand_shapes.json__":
                            continue
                        d = json.load(tar.extractfile(m))
                        if n == "__hand_shapes.json__":
                            shape = d
                            continue
                        k, kind = n.split(".")[0], n.split(".")[1]
                        if kind == "hands":
                            hands[int(k)] = d
                        elif kind == "objects":
                            objs[int(k)] = d
                        elif kind == "info":
                            info[int(k)] = d
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))
            hands, objs, info, shape, nbytes = {}, {}, {}, None, 0
    frames = sorted(info)
    N = len(frames)
    out = {"frames": np.array(frames), "ts": np.array([info[f]["ref_timestamp_ns"] for f in frames], np.int64),
           "device": info[frames[0]]["device"], "sequence_id": info[frames[0]]["sequence_id"],
           "participant_id": info[frames[0]]["participant_id"],
           "betas": np.array(shape["mano"] if shape is not None else np.full(10, np.nan), np.float64),
           "umetrack_shape": json.dumps(shape["umetrack"]) if shape is not None else ""}
    for side in ("left", "right"):
        valid = np.zeros(N, bool)
        th = np.full((N, 15), np.nan)
        wr = np.full((N, 6), np.nan)
        uvalid = np.zeros(N, bool)
        uja = np.full((N, 22), np.nan)
        uT = np.full((N, 4, 4), np.nan)
        for i, f in enumerate(frames):
            h = hands.get(f, {}).get(side)
            if h and h.get("mano_pose"):
                th[i] = h["mano_pose"]["thetas"]
                wr[i] = h["mano_pose"]["wrist_xform"]
                valid[i] = True
            if h and h.get("umetrack_pose"):
                up = h["umetrack_pose"]
                uja[i] = up["joint_angles"]
                uT[i] = quat_T(up["T_world_from_wrist"]["quaternion_wxyz"], up["T_world_from_wrist"]["translation_xyz"])
                uvalid[i] = True
        out[f"{side}_valid"], out[f"{side}_thetas"], out[f"{side}_wrist"] = valid, th, wr
        out[f"{side}_ume_valid"], out[f"{side}_ume_angles"], out[f"{side}_ume_T"] = uvalid, uja, uT
    ids = sorted({int(b) for f in frames for b in objs.get(f, {})})
    T = np.full((len(ids), N, 4, 4), np.nan)
    for i, f in enumerate(frames):
        for b, lst in objs.get(f, {}).items():
            p = lst[0]["T_world_from_object"]
            T[ids.index(int(b)), i] = quat_T(p["quaternion_wxyz"], p["translation_xyz"])
    out["obj_ids"], out["obj_T"] = np.array(ids, np.int64), T
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(dest)
    return fname, nbytes, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["train_aria", "train_quest3"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sib = HfApi().dataset_info(REPO).siblings
    files = [s.rfilename for s in sib if s.rfilename.split("/")[0] in a.split and s.rfilename.endswith(".tar")]
    if a.limit:
        files = [files[i] for i in np.linspace(0, len(files) - 1, min(a.limit, len(files))).astype(int)]
    print(f"{len(files)} clips", flush=True)
    t0, tot, errs = time.time(), 0, []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch, f, OUT / f.replace(".tar", ".npz")): f for f in files}
        for i, fu in enumerate(as_completed(futs)):
            try:
                _, nb, _ = fu.result()
                tot += nb
            except Exception as e:
                errs.append(f"{futs[fu]}: {e!r}")
            if (i + 1) % 25 == 0 or i + 1 == len(files):
                dt = time.time() - t0
                print(f"  {i + 1}/{len(files)}  {dt:.0f}s  {tot / 1e6 / dt:.1f} MB/s  errors={len(errs)}", flush=True)
    for e in errs[:20]:
        print("ERROR", e)
    print("done")


if __name__ == "__main__":
    main()
