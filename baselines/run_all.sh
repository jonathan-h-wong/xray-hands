#!/usr/bin/env bash
# Sequential fallback (one GPU): the three jobs one after another, then the joint comparison.
# For parallel runs start `RUN_ID=<id> bash baselines/run_method.sh <method>` on separate pods/GPUs instead,
# then `RUN_ID=<id> bash baselines/compare.sh` once all three have finished.
set -euo pipefail
cd "$(dirname "$0")/.."
export RUN_ID=${RUN_ID:-$(date -u +%Y%m%d)_$(git rev-parse --short HEAD)_$(basename "${EVAL_SET:-arctic_p2_val_v0_s5}" .npz)}
for m in wilor hamer hopformer; do bash baselines/run_method.sh "$m"; done
bash baselines/compare.sh
