"""Label MANO vertices by hand part and by palmar/dorsal side.

Part: argmax of MANO skinning weights -> joint -> part
  (joint 0 -> palm, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb).
Side: sign of the rest-pose vertex normal against the palmar direction. The palmar
direction is the mean rest-pose normal of the vertices that EPIC-Contact annotators
painted as "palm" contact (human-labelled, so it cannot be flipped by a convention error).

Writes results/hand_parts.json: {"left": {"part": [...778], "side": [...778]}, "right": ...}
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).parent))
from mano_np import MANO  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANO_DIR = ROOT / "data/arctic/unpack/body_models/mano_v1_2/models"
EPIC = ROOT / "data/epic-contact/central_frames"
PARTS = ["palm", "thumb", "index", "middle", "ring", "pinky"]
JOINT_PART = [0, 2, 2, 2, 3, 3, 3, 5, 5, 5, 4, 4, 4, 1, 1, 1]  # MANO joint -> PARTS index


def main():
    counts = {"left": np.zeros(778), "right": np.zeros(778)}
    for clip in glob.glob(str(EPIC / "*")):
        side = "left" if "_left_" in Path(clip).name else "right"
        c = json.load(open(Path(clip) / "contact_vertices_sparse.json"))
        counts[side][np.asarray(c.get("palm", []), int)] += 1
    out = {}
    for side in ("left", "right"):
        m = MANO(MANO_DIR / f"MANO_{side.upper()}.pkl")
        part = np.asarray(JOINT_PART)[m.weights.argmax(1)]
        mesh = trimesh.Trimesh(m.v_template, m.faces, process=False)
        n = mesh.vertex_normals
        palm_annot = counts[side] >= np.percentile(counts[side][counts[side] > 0], 50)
        palmar_dir = n[palm_annot].mean(0)
        palmar_dir /= np.linalg.norm(palmar_dir)
        palmar = (n @ palmar_dir) > 0
        # sanity: annotated palm-contact vertices must be palm part and palmar side
        print(f"{side}: annotated palm-contact verts {palm_annot.sum()}, "
              f"{100 * (part[palm_annot] == 0).mean():.0f}% labelled palm, "
              f"{100 * palmar[palm_annot].mean():.0f}% labelled palmar")
        print("  verts per part:", {p: int((part == i).sum()) for i, p in enumerate(PARTS)},
              "palmar:", int(palmar.sum()), "dorsal:", int((~palmar).sum()))
        out[side] = {"part": part.tolist(), "side": ["palmar" if x else "dorsal" for x in palmar],
                     "palmar_dir": palmar_dir.tolist()}
    out["parts"] = PARTS
    p = ROOT / "results/hand_parts.json"
    p.write_text(json.dumps(out))
    print("wrote", p)


if __name__ == "__main__":
    main()
