"""Download only the ARCTIC images an eval set needs.

ARCTIC ships one zip per sequence (all 9 views). Each zip is streamed to disk, the needed view/frame jpgs are
extracted to <arctic>/<cropped_images|images>/<sid>/<seq>/<view>/<nnnnn>.jpg (the layout ARCTIC's own unzip
script produces), and the zip is deleted. Re-running skips sequences whose images are all present.

Credentials come from the environment (e.g. Runpod secrets), never from the command line:
    ARCTIC_USERNAME, ARCTIC_PASSWORD

    python baselines/fetch_arctic_images.py --eval-set baselines/eval_sets/arctic_p2_val_v0_s5.npz
Sizes: cropped_images is ~116 GB for all of ARCTIC (~0.35 GB per sequence zip); full images ~649 GB.
Use --all-frames to keep every frame of the needed views instead of only the eval-set frames.
"""
import argparse
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

BASE = ("https://download.is.tue.mpg.de/download.php?domain=arctic&resume=1&sfile=arctic_release/"
        "c7216c3b205186106a1f8326ed7b948f838e4907e69b21c8b3c87bb69d87206e/v1_0/data/{kind}_zips/{sid}/{seq}.zip")


def download(url, dest, verify):
    data = {"username": os.environ["ARCTIC_USERNAME"], "password": os.environ["ARCTIC_PASSWORD"]}
    for attempt in range(5):
        try:
            with requests.post(url, data=data, stream=True, verify=verify, allow_redirects=True, timeout=120) as r:
                if r.status_code == 401:
                    sys.exit("ARCTIC authentication failed (check ARCTIC_USERNAME / ARCTIC_PASSWORD)")
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 22):
                        f.write(chunk)
            if not zipfile.is_zipfile(dest):
                head = open(dest, "rb").read(200)
                sys.exit(f"not a zip (license not accepted or bad credentials?): {head!r}")
            return
        except (requests.RequestException, OSError) as e:
            if attempt == 4:
                raise
            print(f"  retry {attempt + 1}: {e!r}", flush=True)
            time.sleep(15 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--image-kind", default="cropped", choices=["cropped", "full"],
                    help="cropped = ARCTIC cropped_images (ego view only); full = 2K images")
    ap.add_argument("--all-frames", action="store_true")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification (upstream script does this)")
    a = ap.parse_args()
    if not (os.environ.get("ARCTIC_USERNAME") and os.environ.get("ARCTIC_PASSWORD")):
        sys.exit("set ARCTIC_USERNAME and ARCTIC_PASSWORD in the environment")

    es, meta = C.load_eval_set(a.eval_set)
    sub = "cropped_images" if a.image_kind == "cropped" else "images"
    zkind = "cropped_images" if a.image_kind == "cropped" else "images"
    for sid, seq, view, idx in C.group_rows(es):
        C.image_scale(view, a.image_kind)  # validates kind/view
    seqs = {}
    for sid, seq, view, idx in C.group_rows(es):
        want = seqs.setdefault((sid, seq), set())
        for name in es["image"][idx]:
            _, _, v, img = name.split("/")
            want.add((v, img))
    print(f"{len(seqs)} sequences, {sum(map(len, seqs.values()))} images", flush=True)
    for i, ((sid, seq), want) in enumerate(seqs.items()):
        out_dir = C.ARCTIC / sub / sid / seq
        missing = [w for w in want if not (out_dir / w[0] / w[1]).exists()]
        if not missing and not a.all_frames:
            continue
        views = {v for v, _ in want}
        t0 = time.time()
        with tempfile.TemporaryDirectory(dir=C.ARCTIC) as tmp:
            zp = Path(tmp) / f"{seq}.zip"
            download(BASE.format(kind=zkind, sid=sid, seq=seq), zp, verify=not a.insecure)
            n = 0
            with zipfile.ZipFile(zp) as z:
                for m in z.infolist():
                    parts = Path(m.filename).parts
                    if m.is_dir() or len(parts) < 2 or not parts[-1].endswith(".jpg"):
                        continue
                    v, img = parts[-2], parts[-1]
                    if v not in views or (not a.all_frames and (v, img) not in want):
                        continue
                    dst = out_dir / v / img
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(m) as src, open(dst, "wb") as f:
                        f.write(src.read())
                    n += 1
        print(f"[{i + 1}/{len(seqs)}] {sid}/{seq}: {n} images in {time.time() - t0:.0f}s", flush=True)
    still = sum(1 for (sid, seq), want in seqs.items() for v, img in want
                if not (C.ARCTIC / sub / sid / seq / v / img).exists())
    print(f"done; {still} eval-set images missing (frames ARCTIC did not ship are skipped at inference)")


if __name__ == "__main__":
    main()
