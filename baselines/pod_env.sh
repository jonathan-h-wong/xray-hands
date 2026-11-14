# Source me: shared environment for every pod that mounts the network volume at /workspace.
# Everything persistent (conda, envs, model repos, weights, ARCTIC, outputs) lives on the volume, so all pods
# see the same installs; only OS packages (installed below, idempotent) live on each pod's container disk.
export VOL=${VOL:-/workspace}
export CONDA_ROOT=${CONDA_ROOT:-$VOL/miniforge3}
export ENVS=${ENVS:-$VOL/envs}
export REPOS=${REPOS:-$VOL/baseline_repos}
export PROJ=${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export ARCTIC_DATA=${ARCTIC_DATA:-$PROJ/data/arctic/unpack/arctic_data/data}
export MANO_DIR=${MANO_DIR:-$PROJ/data/arctic/unpack/body_models/mano_v1_2/models}
export IMAGE_KIND=${IMAGE_KIND:-cropped}
export EVAL_SET=${EVAL_SET:-baselines/eval_sets/arctic_p2_val_v0_s5.npz}
export S3_PREFIX=${S3_PREFIX:-s3://xray-hands/eval/arctic}
export PYOPENGL_PLATFORM=egl
export TAR_OPTIONS=--no-same-owner   # the network volume refuses chown
export PIP_CACHE_DIR=$VOL/.cache/pip HF_HOME=$VOL/.cache/huggingface TORCH_HOME=$VOL/.cache/torch
export HF_TOKEN=${HF_TOKEN:-}

if ! dpkg -s libgl1 libegl1 libglu1-mesa libxrender1 >/dev/null 2>&1; then
  (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgl1 libegl1 libgles2 libglu1-mesa libglib2.0-0 libxrender1 libxext6 libx11-6 \
     build-essential git wget >/dev/null) || echo "WARN: apt install failed"
fi
if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
  wget -q -O /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$CONDA_ROOT"
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
