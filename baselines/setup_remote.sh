#!/usr/bin/env bash
# One-time setup, run on ONE pod with the network volume at /workspace (needs a GPU only for the last step).
# Installs everything onto the volume so every later pod can run any method:
#   $CONDA_ROOT (Miniforge), $ENVS/{hamer,wilor,hopformer}, $REPOS/{hamer,WiLoR,HOPformer} + weights,
#   .venv-eval (CPU eval + awscli), eval-set images, MoGe-2 intrinsics for the HaMeR/WiLoR ablation.
# Each component writes a marker ($VOL/.setup/<name>.ok); re-running skips finished components, so a failed
# step can be fixed and the script simply re-run.
#
# Env: HF_TOKEN (HOPformer weights are gated). ARCTIC_*/MANO_* only if ARCTIC GT or images are missing.
# Third-party steps follow each repo's README as of 2026-11.
# Usage: bash baselines/setup_remote.sh
set -euo pipefail
source "$(dirname "$0")/pod_env.sh"
cd "$PROJ"
MARK=$VOL/.setup
mkdir -p "$MARK" "$REPOS" "$ENVS" logs
done_() { [ -f "$MARK/$1.ok" ]; }
ok() { touch "$MARK/$1.ok"; echo "== $1 ok"; }
[ -n "$HF_TOKEN" ] || { echo "missing HF_TOKEN"; exit 1; }

if ! done_ eval; then
  echo "== CPU eval env + ARCTIC GT"
  [ -d .venv-eval ] || python3 -m venv .venv-eval
  .venv-eval/bin/pip install -q -r baselines/requirements-eval.txt
  if [ ! -d "$ARCTIC_DATA/raw_seqs" ] || [ ! -f "$MANO_DIR/MANO_RIGHT.pkl" ]; then
    [ -d tools/arctic ] || git clone -q https://github.com/zc-alexfan/arctic tools/arctic
    .venv-eval/bin/python scripts/download_arctic.py
  fi
  ok eval
fi

if ! done_ images; then
  .venv-eval/bin/python baselines/fetch_arctic_images.py --eval-set "$EVAL_SET" --image-kind "$IMAGE_KIND"
  ok images
fi

if ! done_ hamer; then
  echo "== HaMeR (github.com/geopavlakos/hamer)"
  [ -d "$REPOS/hamer" ] || git clone -q --recursive https://github.com/geopavlakos/hamer "$REPOS/hamer"
  [ -d "$ENVS/hamer" ] || conda create -y -q -p "$ENVS/hamer" python=3.10
  conda activate "$ENVS/hamer"
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
  (cd "$REPOS/hamer" && pip install -q -e .[all] && pip install -q -v -e third-party/ViTPose && bash fetch_demo_data.sh)
  mkdir -p "$REPOS/hamer/_DATA/data/mano" && cp "$MANO_DIR/MANO_RIGHT.pkl" "$REPOS/hamer/_DATA/data/mano/"
  conda deactivate
  ok hamer
fi

if ! done_ wilor; then
  echo "== WiLoR (github.com/rolpotamias/WiLoR)"
  [ -d "$REPOS/WiLoR" ] || git clone -q https://github.com/rolpotamias/WiLoR "$REPOS/WiLoR"
  [ -d "$ENVS/wilor" ] || conda create -y -q -p "$ENVS/wilor" python=3.10
  conda activate "$ENVS/wilor"
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
  (cd "$REPOS/WiLoR" && pip install -q -r requirements.txt)
  W=https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models
  for f in detector.pt wilor_final.ckpt; do
    [ -f "$REPOS/WiLoR/pretrained_models/$f" ] || wget -q -P "$REPOS/WiLoR/pretrained_models" "$W/$f"
  done
  mkdir -p "$REPOS/WiLoR/mano_data" && cp "$MANO_DIR/MANO_RIGHT.pkl" "$REPOS/WiLoR/mano_data/"
  conda deactivate
  ok wilor
fi

if ! done_ hopformer; then
  echo "== HOPformer (github.com/Sid2697/HOPformer @ fdff116)"
  H=$REPOS/HOPformer
  if [ ! -d "$H" ]; then
    git clone -q https://github.com/Sid2697/HOPformer "$H"
    (cd "$H" && git checkout -q fdff116b856e3ee4d05e15db64291a3606475494)
  fi
  [ -d "$ENVS/hopformer" ] || conda env create -q -p "$ENVS/hopformer" -f "$H/environment.yml"
  conda activate "$ENVS/hopformer"
  python - <<'PY'
# HOPformer README step 2: smplx must return 21 hand joints (uncomment the vertex_joint_selector line)
import os, re, smplx
p = os.path.join(os.path.dirname(smplx.__file__), "body_models.py")
L = open(p).read().split("\n")
i = 1680  # README: line 1681
if re.match(r"\s*joints = self\.vertex_joint_selector\(vertices, joints\)", L[i]):
    print("smplx already patched")
else:
    assert re.match(r"\s*#\s*joints = self\.vertex_joint_selector\(vertices, joints\)", L[i]), L[i]
    L[i] = L[i].replace("# ", "", 1)
    open(p, "w").write("\n".join(L))
    print("patched", p)
PY
  mkdir -p "$H/data/body_models/mano" "$H/pretrained_models/wilor" "$H/data/arctic_data" "$H/release_models"
  cp "$MANO_DIR"/MANO_{LEFT,RIGHT}.pkl "$H/data/body_models/mano/"
  [ -f "$H/data/body_models/mano/mano_mean_params.npz" ] || wget -q -P "$H/data/body_models/mano" \
    https://raw.githubusercontent.com/rolpotamias/WiLoR/main/mano_data/mano_mean_params.npz
  cp "$REPOS/WiLoR/pretrained_models/model_config.yaml" "$REPOS/WiLoR/pretrained_models/wilor_final.ckpt" \
    "$H/pretrained_models/wilor/"
  ln -sfn "$ARCTIC_DATA" "$H/data/arctic_data/data"
  [ -f "$H/release_models/p2_arctic_ego_epoch8.ckpt" ] || python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Sid2697/HOPformer', 'p2_arctic_ego_epoch8.ckpt', local_dir='$H/release_models')"
  pip install -q git+https://github.com/microsoft/MoGe.git      # MoGe-2, estimated-intrinsics ablation
  conda deactivate
  ok hopformer
fi

if ! done_ moge_hand2d && nvidia-smi >/dev/null 2>&1; then
  echo "== MoGe-2 intrinsics for HaMeR/WiLoR (shared by both jobs)"
  conda activate "$ENVS/hopformer"
  python baselines/estimate_intrinsics_moge.py --eval-set "$EVAL_SET" --for hand2d --image-kind "$IMAGE_KIND"
  conda deactivate
  ok moge_hand2d
fi
echo "setup done"
