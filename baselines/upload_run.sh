#!/usr/bin/env bash
# Upload one method's (or the comparison's) outputs of a run to $S3_PREFIX/$RUN_ID/<name>/.
# Usage: bash baselines/upload_run.sh <run_id> <hamer|wilor|hopformer|compare>
set -uo pipefail
RUN_ID=${1:?run id}; NAME=${2:?name}
source "$(dirname "$0")/pod_env.sh"
cd "$PROJ"
[ -n "${AWS_ACCESS_KEY_ID:-}" ] || { echo "no AWS credentials in env; skipped S3 upload"; exit 0; }
AWS=.venv-eval/bin/aws
DST=$S3_PREFIX/$RUN_ID/$NAME
PREDS=baselines/preds/$RUN_ID
set -e
if [ "$NAME" != compare ]; then
  $AWS s3 sync "$PREDS" "$DST/preds/" --exclude "*" --include "$NAME/*" --include "${NAME}_moge/*" \
    --include "moge_*.json" --only-show-errors
fi
for f in results/baselines/${RUN_ID}_${NAME}*; do [ -e "$f" ] && $AWS s3 cp "$f" "$DST/" --only-show-errors; done
[ -f "logs/$RUN_ID/$NAME.log" ] && $AWS s3 cp "logs/$RUN_ID/$NAME.log" "$DST/logs/$NAME.log" --only-show-errors
echo "uploaded to $DST"
