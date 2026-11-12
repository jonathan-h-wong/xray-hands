#!/usr/bin/env bash
# One-time setup on a Linux GPU box (e.g. a Runpod pod with a network volume at /workspace).
# Creates: .venv-eval (CPU eval), conda envs hamer / wilor / hopformer, model repos under $REPOS,
# ARCTIC meta + raw_seqs + MANO, and the eval-set images.
#
# Secrets expected in the environment (Runpod: map secrets to these env vars):
#   ARCTIC_USERNAME ARCTIC_PASSWORD MANO_USERNAME MANO_PASSWORD   (MPI accounts)
#   HF_TOKEN   (Hugging Face; HOPformer weights are gated - accept the terms on huggingface.co/Sid2697/HOPformer)
#
# Third-party install steps follow each repo's README as of 2026-11; if a step fails, check that README first.
# Usage: bash baselines/setup_remote.sh [eval_set.npz]
set -euo pipefail
cd "$(dirname "$0")/.."
PROJ=$(pwd)
REPOS=${REPOS:-/workspace/baseline_repos}
EVAL_SET=${1:-baselines/eval_sets/arctic_p2_val_v0_s5.npz}
export ARCTIC_DATA=${ARCTIC_DATA:-$PROJ/data/arctic/unpack/arctic_data/data}
export MANO_DIR=${MANO_DIR:-$PROJ/data/arctic/unpack/body_models/mano_v1_2/models}   # folder with MANO_{LEFT,RIGHT}.pkl
MANO_MODELS=$MANO_DIR
mkdir -p "$REPOS"
NEED="HF_TOKEN"
[ -d "$ARCTIC_DATA/raw_seqs" ] && [ -f "$MANO_MODELS/MANO_RIGHT.pkl" ] || NEED="$NEED ARCTIC_USERNAME ARCTIC_PASSWORD MANO_USERNAME MANO_PASSWORD"
for v in $NEED; do
  [ -n "${!v:-}" ] || { echo "missing env var $v"; exit 1; }
done
eval "$(conda shell.bash hook)"

echo "== CPU eval env + ARCTIC GT (meta, raw_seqs, MANO)"
[ -d .venv-eval ] || python3 -m venv .venv-eval
.venv-eval/bin/pip install -q -r baselines/requirements-eval.txt
[ -d tools/arctic ] || git clone -q https://github.com/zc-alexfan/arctic tools/arctic
if [ ! -d "$ARCTIC_DATA/raw_seqs" ] || [ ! -f "$MANO_MODELS/MANO_RIGHT.pkl" ]; then
  .venv-eval/bin/python scripts/download_arctic.py
else
  echo "ARCTIC GT found at $ARCTIC_DATA, MANO at $MANO_MODELS - skipping download"
fi
[ -d "$ARCTIC_DATA/grounded_mask" ] || echo "NOTE: grounded_mask not found; only needed to rebuild eval sets (they are committed)"

echo "== eval-set images (skips images already on the volume)"
.venv-eval/bin/python baselines/fetch_arctic_images.py --eval-set "$EVAL_SET" --image-kind "${IMAGE_KIND:-cropped}"

echo "== HaMeR (github.com/geopavlakos/hamer)"
if [ ! -d "$REPOS/hamer" ]; then
  git clone -q --recursive https://github.com/geopavlakos/hamer "$REPOS/hamer"
  conda create -y -q -n hamer python=3.10
  conda activate hamer
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
  (cd "$REPOS/hamer" && pip install -q -e .[all] && pip install -q -v -e third-party/ViTPose && bash fetch_demo_data.sh)
  mkdir -p "$REPOS/hamer/_DATA/data/mano" && cp "$MANO_MODELS/MANO_RIGHT.pkl" "$REPOS/hamer/_DATA/data/mano/"
  conda deactivate
fi

echo "== WiLoR (github.com/rolpotamias/WiLoR)"
if [ ! -d "$REPOS/WiLoR" ]; then
  git clone -q https://github.com/rolpotamias/WiLoR "$REPOS/WiLoR"
  conda create -y -q -n wilor python=3.10
  conda activate wilor
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
  (cd "$REPOS/WiLoR" && pip install -q -r requirements.txt)
  W=https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models
  wget -q -P "$REPOS/WiLoR/pretrained_models" "$W/detector.pt" "$W/wilor_final.ckpt"
  mkdir -p "$REPOS/WiLoR/mano_data" && cp "$MANO_MODELS/MANO_RIGHT.pkl" "$REPOS/WiLoR/mano_data/"
  conda deactivate
fi

echo "== HOPformer (github.com/Sid2697/HOPformer @ fdff116)"
H=$REPOS/HOPformer
if [ ! -d "$H" ]; then
  git clone -q https://github.com/Sid2697/HOPformer "$H"
  (cd "$H" && git checkout -q fdff116b856e3ee4d05e15db64291a3606475494)
  conda env create -q -f "$H/environment.yml"
  conda activate hopformer
  # smplx must return 21 hand joints (HOPformer README step 2): uncomment the vertex_joint_selector line
  python - <<'PY'
import os, re, smplx
p = os.path.join(os.path.dirname(smplx.__file__), "body_models.py")
L = open(p).read().split("\n")
i = 1680  # README: line 1681
assert re.match(r"\s*#\s*joints = self\.vertex_joint_selector\(vertices, joints\)", L[i]), L[i]
L[i] = L[i].replace("# ", "", 1)
open(p, "w").write("\n".join(L))
print("patched", p)
PY
  mkdir -p "$H/data/body_models/mano" "$H/pretrained_models/wilor" "$H/data/arctic_data" "$H/release_models"
  cp "$MANO_MODELS"/MANO_{LEFT,RIGHT}.pkl "$H/data/body_models/mano/"
  wget -q -P "$H/data/body_models/mano" https://raw.githubusercontent.com/rolpotamias/WiLoR/main/mano_data/mano_mean_params.npz
  cp "$REPOS/WiLoR/pretrained_models/model_config.yaml" "$REPOS/WiLoR/pretrained_models/wilor_final.ckpt" "$H/pretrained_models/wilor/"
  ln -sfn "$ARCTIC_DATA" "$H/data/arctic_data/data"
  hf download Sid2697/HOPformer p2_arctic_ego_epoch8.ckpt --local-dir "$H/release_models" --token "$HF_TOKEN"
  pip install -q git+https://github.com/microsoft/MoGe.git      # MoGe-2, for the estimated-intrinsics ablation
  conda deactivate
fi
echo "setup done"
