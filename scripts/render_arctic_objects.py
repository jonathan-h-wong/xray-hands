"""Render ARCTIC textured object templates to RGBA PNGs (numpy z-buffer rasterizer).

Usage:
    python scripts/render_arctic_objects.py sheet            # contact sheet of candidate views
    python scripts/render_arctic_objects.py render obj:az:el [obj:az:el ...]
"""
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TPL = ROOT / "data/arctic/unpack/arctic_data/data/meta/object_vtemplates"
OUT = ROOT / "trial_images"
OBJECTS = ["box", "capsulemachine", "espressomachine", "ketchup", "laptop", "microwave",
           "mixer", "notebook", "phone", "scissors", "waffleiron"]


def load(obj):
    m = trimesh.load(TPL / obj / "mesh_tex.obj", process=False)
    tex = np.asarray(Image.open(TPL / obj / "material.jpg").convert("RGB"), dtype=np.float32) / 255
    v = m.vertices - m.bounds.mean(0)
    return v / np.linalg.norm(m.extents) * 2, np.asarray(m.faces), np.asarray(m.visual.uv), tex


def look_at(az, el, up_axis):
    a, e = np.radians(az), np.radians(el)
    up = np.eye(3)[up_axis]
    # build a horizontal basis orthogonal to up
    h1 = np.eye(3)[(up_axis + 1) % 3]
    h2 = np.cross(up, h1)
    d = np.cos(e) * (np.cos(a) * h1 + np.sin(a) * h2) + np.sin(e) * up  # camera position direction
    fwd = -d
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    cup = np.cross(right, fwd)
    return d, np.stack([right, cup, fwd])


def render(obj, az, el, up_axis=1, size=640, dist=3.2, f=1.9):
    v, F, uv, tex = load(obj)
    eye, R = look_at(az, el, up_axis)
    pc = (v - eye * dist) @ R.T                      # camera coords: x right, y up, z forward
    z = pc[:, 2]
    px = np.stack([pc[:, 0] / z, -pc[:, 1] / z], 1) * f * size / 2 + size / 2
    # face normals for Lambert shading (two-sided)
    n = np.cross(v[F[:, 1]] - v[F[:, 0]], v[F[:, 2]] - v[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = -R[2] * 0.7 + R[1] * 0.6 - R[0] * 0.3
    light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.abs(n @ light)

    zbuf = np.full((size, size), np.inf)
    rgb = np.zeros((size, size, 3), np.float32)
    th, tw = tex.shape[:2]
    for fi in range(len(F)):
        i0, i1, i2 = F[fi]
        p = px[[i0, i1, i2]]
        x0, y0 = np.floor(p.min(0)).astype(int); x1, y1 = np.ceil(p.max(0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, size - 1), min(y1, size - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = p
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-9:
            continue
        w0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
        w1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        zz = w0 * z[i0] + w1 * z[i1] + w2 * z[i2]
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        upd = inside & (zz < sub)
        if not upd.any():
            continue
        sub[upd] = zz[upd]
        # perspective-correct UV
        iz = np.stack([w0 / z[i0], w1 / z[i1], w2 / z[i2]], -1)[upd]
        t = (iz[:, :, None] * uv[[i0, i1, i2]][None]).sum(1) / iz.sum(1, keepdims=True)
        tx = np.clip((t[:, 0] % 1) * (tw - 1), 0, tw - 1).astype(int)
        ty = np.clip((1 - t[:, 1] % 1) * (th - 1), 0, th - 1).astype(int)
        rgb[y0:y1 + 1, x0:x1 + 1][upd] = tex[ty, tx] * shade[fi]
    alpha = np.isfinite(zbuf)
    out = np.dstack([np.clip(rgb, 0, 1) * 255, alpha * 255]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def on_white(im):
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB")


if __name__ == "__main__":
    if sys.argv[1] == "sheet":
        # candidate views: each up-axis, two azimuths
        views = [(u, az, 25) for u in (0, 1, 2) for az in (35, 215)]
        tile = 192
        for obj in OBJECTS:
            sheet = Image.new("RGB", (tile * len(views), tile), "white")
            for k, (u, az, el) in enumerate(views):
                sheet.paste(on_white(render(obj, az, el, u, size=tile)), (k * tile, 0))
            sheet.save(Path(sys.argv[2]) / f"sheet_{obj}.png")
            print("sheet", obj, flush=True)
    else:
        OUT.mkdir(exist_ok=True)
        for spec in sys.argv[2:]:
            obj, up, az, el = spec.split(":")
            name = f"{obj}_render_az{az}"
            render(obj, float(az), float(el), int(up)).save(OUT / f"{name}.png")
            print("wrote", name, flush=True)
