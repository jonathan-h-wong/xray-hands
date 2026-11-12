"""Check the NumPy MANO path for HOT3D against HOT3D's independent UmeTrack hand fits.

UmeTrack landmark skinning is re-implemented here in NumPy (ported from
facebookresearch/hand_tracking_toolkit, umetrack_skinning.py / umetrack_hand_model.py).
Canonical landmarks 0-4 are fingertips (thumb..pinky); MANO fingertips are vertices 744,320,443,554,671
(toolkit's MANO convention). Compares the left hand with and without the shapedirs fix.

Usage: .venv\\Scripts\\python analysis\\hot3d_validate.py <extracted_clip_dir> [...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from mano_np import MANO  # noqa: E402

MDIR = ROOT / "data/arctic/unpack/body_models/mano_v1_2/models"
TIPS = [744, 320, 443, 554, 671]


def aa_to_R(aa):
    return Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix()


def ume_landmarks(model, joint_angles, T_wrist, right):
    """UmeTrack skin_landmarks for one hand (metres)."""
    T_wrist = T_wrist.copy()
    if right:
        T_wrist[:, 0] *= -1
    axes = np.asarray(model["joint_rotation_axes"])[:20]
    rest = np.asarray(model["joint_rest_positions"])[:20] * 1e-3
    ang = np.asarray(joint_angles)[:20]
    R = aa_to_R(axes * ang[:, None])
    L = np.tile(np.eye(4), (20, 1, 1))
    L[:, :3, :3] = R
    L[:, :3, 3] = rest - np.einsum("nij,nj->ni", R, rest)
    mats = [T_wrist, T_wrist]
    for f in range(5):
        chain = [T_wrist]
        for i in range(4):
            chain.append(chain[-1] @ L[4 * f + i])
        mats += chain[2:]
    M = np.stack(mats)  # (17,4,4)
    pts = np.asarray(model["landmark_rest_positions"]) * 1e-3
    idx = np.asarray(model["landmark_rest_bone_indices"]).astype(int)
    w = np.asarray(model["landmark_rest_bone_weights"])
    W = np.zeros((len(pts), 17))
    for k in range(idx.shape[1]):
        np.add.at(W, (np.arange(len(pts)), idx[:, k]), w[:, k])
    ph = np.c_[pts, np.ones(len(pts))]
    out = np.einsum("vj,jab,vb->va", W, M, ph)
    return out[:, :3]


def quatT(q, t):
    w, x, y, z = q
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = t
    return T


def main():
    m_r = MANO(MDIR / "MANO_RIGHT.pkl")
    m_l = MANO(MDIR / "MANO_LEFT.pkl")
    m_lf = MANO(MDIR / "MANO_LEFT.pkl")
    m_lf.shapedirs[:, 0, :] *= -1
    rows = []
    for d in map(Path, sys.argv[1:]):
        sh = json.load(open(d / "__hand_shapes.json__"))
        beta = np.array(sh["mano"])
        ume = sh["umetrack"]
        for hf in sorted(d.glob("*.hands.json")):
            j = json.load(open(hf))
            for side in ("left", "right"):
                h = j.get(side)
                if not h or "mano_pose" not in h or "umetrack_pose" not in h:
                    continue
                th = np.array(h["mano_pose"]["thetas"])[None]
                wx = np.array(h["mano_pose"]["wrist_xform"])[None]
                up = h["umetrack_pose"]
                Lu = ume_landmarks(ume, up["joint_angles"], quatT(up["T_world_from_wrist"]["quaternion_wxyz"],
                                                                 up["T_world_from_wrist"]["translation_xyz"]),
                                   side == "right")
                for name, m in ([("right", m_r)] if side == "right" else [("left_fix", m_lf), ("left_nofix", m_l)]):
                    V = m(beta, wx[:, :3], m.pca_to_pose(th), wx[:, 3:])[0]
                    rows.append((d.name, beta[0], name, np.linalg.norm(V[TIPS] - Lu[:5], axis=-1).mean() * 1000))
    import pandas as pd
    df = pd.DataFrame(rows, columns=["clip", "beta0", "variant", "tip_err_mm"])
    print(df.groupby(["clip", "variant"]).agg(beta0=("beta0", "first"), n=("tip_err_mm", "size"),
                                              p50=("tip_err_mm", "median"),
                                              p90=("tip_err_mm", lambda x: np.percentile(x, 90))).round(2))


if __name__ == "__main__":
    main()
