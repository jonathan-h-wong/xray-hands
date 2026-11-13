#!/usr/bin/env bash
# After all run_method.sh jobs of a run finish: joint evaluation of every method on the same hands (adds the
# common-subset table that per-method evals cannot produce), then upload as <run>/compare/. CPU only.
# Usage: RUN_ID=<id> bash baselines/compare.sh
set -euo pipefail
source "$(dirname "$0")/pod_env.sh"
cd "$PROJ"
RUN_ID=${RUN_ID:?set RUN_ID}
PREDS=baselines/preds/$RUN_ID
mkdir -p "logs/$RUN_ID"
exec > >(tee -a "logs/$RUN_ID/compare.log") 2>&1
M=""; for m in hamer wilor hopformer; do [ -d "$PREDS/$m" ] && M="$M $m"; done
echo "== compare $RUN_ID:$M"
EV=".venv-eval/bin/python -W ignore baselines/eval_arctic.py --eval-set $EVAL_SET --preds $PREDS"
$EV --methods $M --tag "${RUN_ID}_compare"
MM=""; for m in hamer wilor hopformer_moge; do [ -d "$PREDS/$m" ] && MM="$MM $m"; done
[ -f "$PREDS/moge_hand2d.json" ] && $EV --methods $MM --intrinsics "moge:$PREDS/moge_hand2d.json" --tag "${RUN_ID}_compare_moge"
bash baselines/upload_run.sh "$RUN_ID" compare
