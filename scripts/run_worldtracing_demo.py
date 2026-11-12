"""Send trial images through the World-Tracing HF Space ("single object photo" tab, /run_object)
and unpack the returned point cloud.

For each input image <name> this writes, under trial_images/worldtracing_out/<name>/:
    cutout.png   background-removed image the model saw (from the Space)
    viewer.html  the Space's own self-contained viewer page (iframe srcdoc, unescaped)
    points.npz   pos [N,3] float32 (Three.js Y-up), col [N,3] uint8, layer [N] uint8, K [3,3]
    meta.json

Usage: python scripts/run_worldtracing_demo.py [image ...]   (default: every image in trial_images/)
"""
import base64
import gzip
import html
import json
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
from gradio_client import Client, handle_file

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "trial_images"
OUT = IMAGES / "worldtracing_out"
SPACE = "haoz19/world-tracing-demo"
STEPS, SEED = 20, 42
# viz_glb._layer_colors: layer 0 keeps input RGB, layer i>=1 is grey 210 - 22*(i-1)
GREYS = [max(60, 210 - 22 * i) for i in range(9)]


def parse_wtd1(raw):
    assert raw[:4] == b"WTD1", raw[:4]
    T, fx, fy, cx, cy, w, h = struct.unpack_from("<I6f", raw, 4)
    o = 32
    N = struct.unpack_from("<I", raw, o)[0]; o += 4
    pos = np.frombuffer(raw, np.float32, N * 3, o).reshape(N, 3); o += N * 12
    col = np.frombuffer(raw, np.uint8, N * 3, o).reshape(N, 3)
    return pos.copy(), col.copy(), dict(fx=fx, fy=fy, cx=cx, cy=cy, img_w=w, img_h=h)


def label_layers(col):
    """Layers are concatenated in order; walk back from the end peeling off grey runs."""
    lay = np.zeros(len(col), np.uint8)
    grey_idx = np.full(len(col), -1)
    is_grey = (col[:, 0] == col[:, 1]) & (col[:, 1] == col[:, 2])
    for i, g in enumerate(GREYS):
        grey_idx[is_grey & (col[:, 0] == g)] = i + 1
    end = len(col)
    expected = grey_idx[-1] if len(col) else 0
    while expected >= 1 and end > 0:
        start = end
        while start > 0 and grey_idx[start - 1] == expected:
            start -= 1
        lay[start:end] = expected
        end = start
        expected -= 1
    return lay


def run_one(client, img):
    name = img.stem
    d = OUT / name
    if (d / "points.npz").exists():
        print("skip", name)
        return
    d.mkdir(parents=True, exist_ok=True)
    dt = None
    if not (d / "viewer.html").exists():
        t0 = time.time()
        cutout, viewer = client.predict(handle_file(str(img)), STEPS, SEED, True, api_name="/run_object")
        dt = round(time.time() - t0, 1)
        shutil.copy(cutout["path"] if isinstance(cutout, dict) else cutout, d / "cutout.png")
        inner = html.unescape(re.search(r'srcdoc="(.*?)"\s+sandbox', viewer, re.S).group(1))
        (d / "viewer.html").write_text(inner, encoding="utf-8")
    inner = (d / "viewer.html").read_text(encoding="utf-8")
    b64 = re.search(r'BLOB_DATA_URI = "data:application/octet-stream;base64,([^"]+)"', inner).group(1)
    manifest = json.loads(re.search(r"const MANIFEST = (\{.*?\});", inner, re.S).group(1))
    pos, col, intr = parse_wtd1(gzip.decompress(base64.b64decode(b64)))
    lay = label_layers(col)
    K = np.array(manifest["intrinsics"], np.float32)
    np.savez_compressed(d / "points.npz", pos=pos, col=col, layer=lay, K=K)
    counts = np.bincount(lay, minlength=6).tolist()
    meta = dict(name=name, input=img.name, space=SPACE, endpoint="/run_object", num_steps=STEPS, seed=SEED,
                remove_bg=True, seconds=dt, points=int(len(pos)), points_per_layer=counts,
                intrinsics=intr, manifest=manifest)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{name}: {len(pos)} pts, layers {counts}, {dt}s", flush=True)


if __name__ == "__main__":
    imgs = [Path(p) for p in sys.argv[1:]] or sorted(
        p for p in IMAGES.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    client = Client(SPACE, token=(Path.home() / ".cache/huggingface/token").read_text().strip())
    for img in imgs:
        for attempt in range(3):
            try:
                run_one(client, img)
                break
            except Exception as e:
                print(f"{img.stem}: attempt {attempt + 1} failed: {e!r}", flush=True)
                time.sleep(20 * (attempt + 1))
