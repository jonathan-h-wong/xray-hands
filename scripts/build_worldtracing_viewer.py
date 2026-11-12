"""Pack trial_images/worldtracing_out/* into the static viewer at trial_images/worldtracing_viewer/.

Each sample becomes data/<name>.js (classic script, so the page works from file://):
    WT.register("<name>", <meta>, "<base64 payload>")
payload = uint16 xyz quantised to the sample bbox [N*3] | uint8 rgb [N*3] | uint8 layer [N]
"""
import base64
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "trial_images"
SRC = IMAGES / "worldtracing_out"
DST = IMAGES / "worldtracing_viewer"
OBJECTS = ["box", "capsulemachine", "espressomachine", "ketchup", "laptop", "microwave",
           "mixer", "notebook", "phone", "scissors", "waffleiron"]


def describe(name):
    if name.startswith("hand_"):
        return "hands", "hand", "photo", name[5:].replace("_", " ")
    obj = name.split("_")[0]
    if "_render_" in name:
        return "arctic", obj, "render", "ARCTIC mesh render, az " + name.rsplit("az", 1)[1] + "°"
    return "arctic", obj, "photo", "real photo (Wikimedia Commons)"


def main():
    (DST / "data").mkdir(parents=True, exist_ok=True)
    (DST / "img").mkdir(exist_ok=True)
    sources = {s["file"]: s for s in json.loads((IMAGES / "sources.json").read_text(encoding="utf-8"))}
    samples = []
    for d in sorted(p for p in SRC.iterdir() if (p / "points.npz").exists()):
        name = d.name
        z = np.load(d / "points.npz")
        meta = json.loads((d / "meta.json").read_text())
        pos, col, lay = z["pos"], z["col"], z["layer"]
        lo, hi = pos.min(0), pos.max(0)
        q = np.round((pos - lo) / np.maximum(hi - lo, 1e-9) * 65535).astype("<u2")
        payload = q.tobytes() + col.astype(np.uint8).tobytes() + lay.astype(np.uint8).tobytes()
        group, obj, kind, label = describe(name)
        info = dict(name=name, group=group, object=obj, kind=kind, label=label, n=int(len(pos)),
                    lo=lo.tolist(), hi=hi.tolist(), per_layer=meta["points_per_layer"],
                    seconds=meta["seconds"], steps=meta["num_steps"], seed=meta["seed"],
                    source=sources.get(meta["input"]))
        (DST / "data" / f"{name}.js").write_text(
            f"WT.register({json.dumps(name)},{json.dumps(info)},\"{base64.b64encode(payload).decode()}\");\n")
        im = Image.open(IMAGES / meta["input"]).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
        im.thumbnail((512, 512))
        im.save(DST / "img" / f"{name}_input.jpg", quality=88)
        shutil.copy(d / "cutout.png", DST / "img" / f"{name}_cutout.png")
        samples.append(info)
    order = {o: i for i, o in enumerate(OBJECTS + ["hand"])}
    samples.sort(key=lambda s: (s["group"] == "hands", order.get(s["object"], 99), s["kind"] != "photo", s["name"]))
    (DST / "data" / "manifest.js").write_text("window.WT_MANIFEST = " + json.dumps(samples, indent=1) + ";\n")
    print(f"{len(samples)} samples -> {DST}")


if __name__ == "__main__":
    main()
