#!/usr/bin/env bash
# Inference for all three baselines, optional MoGe-2 intrinsics, then CPU evaluation.
# Each inference script skips finished shards, so this can be re-run after an interruption.
# Usage: bash baselines/run_all.sh [eval_set.npz]
set -euo pipefail
cd "$(dirname "$0")/.."
PROJ=$(pwd)
REPOS=${REPOS:-/workspace/baseline_repos}
EVAL_SET=${1:-baselines/eval_sets/arctic_p2_val_v0_s5.npz}
export ARCTIC_DATA=${ARCTIC_DATA:-$PROJ/data/arctic/unpack/arctic_data/data}
export PYOPENGL_PLATFORM=egl
export MANO_DIR=${MANO_DIR:-$PROJ/data/arctic/unpack/body_models/mano_v1_2/models}
KIND=${IMAGE_KIND:-cropped}   # cropped = ARCTIC cropped_images, full = 2K images
eval "$(conda shell.bash hook)"
mkdir -p logs

conda activate wilor
python baselines/infer_hand2d.py --method wilor --repo "$REPOS/WiLoR" --eval-set "$EVAL_SET" --image-kind "$KIND" 2>&1 | tee logs/infer_wilor.log
conda deactivate

conda activate hamer
python baselines/infer_hand2d.py --method hamer --repo "$REPOS/hamer" --eval-set "$EVAL_SET" --image-kind "$KIND" 2>&1 | tee logs/infer_hamer.log
conda deactivate

conda activate hopformer
python baselines/infer_hopformer.py --repo "$REPOS/HOPformer" --eval-set "$EVAL_SET" \
  --detector "$REPOS/WiLoR/pretrained_models/detector.pt" 2>&1 | tee logs/infer_hopformer.log
# MoGe-2 ablation: estimated instead of calibrated intrinsics
python baselines/estimate_intrinsics_moge.py --eval-set "$EVAL_SET" --for hand2d --image-kind "$KIND" 2>&1 | tee logs/moge_hand2d.log
python baselines/estimate_intrinsics_moge.py --eval-set "$EVAL_SET" --for hopformer --image-kind "$KIND" 2>&1 | tee logs/moge_hop.log
python baselines/infer_hopformer.py --repo "$REPOS/HOPformer" --eval-set "$EVAL_SET" --tag hopformer_moge \
  --intrinsics "moge:$PROJ/baselines/preds/moge_hopformer.json" \
  --detector "$REPOS/WiLoR/pretrained_models/detector.pt" 2>&1 | tee logs/infer_hopformer_moge.log
conda deactivate

.venv-eval/bin/python -W ignore baselines/eval_arctic.py --eval-set "$EVAL_SET" \
  --methods hamer wilor hopformer 2>&1 | tee logs/eval_baselines.log
.venv-eval/bin/python -W ignore baselines/eval_arctic.py --eval-set "$EVAL_SET" \
  --methods hamer wilor hopformer_moge --intrinsics "moge:$PROJ/baselines/preds/moge_hand2d.json" \
  2>&1 | tee logs/eval_baselines_moge.log
echo "results in results/baselines/"
