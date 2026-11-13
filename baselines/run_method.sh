#!/usr/bin/env bash
# One benchmark job = one model, so the three models can run in parallel on separate pods / GPUs that share the
# network volume. Inference -> per-method eval (native + MoGe-2 intrinsics) -> manifest -> S3 upload.
# Restartable: inference skips finished shards.
#
# Usage: RUN_ID=<shared id> bash baselines/run_method.sh hamer|wilor|hopformer
# Outputs (on the volume): baselines/preds/$RUN_ID/<method>/, results/baselines/${RUN_ID}_<method>*,
#   logs/$RUN_ID/<method>.log; uploaded to $S3_PREFIX/$RUN_ID/<method>/ when AWS credentials are set.
set -euo pipefail
METHOD=${1:?usage: run_method.sh hamer|wilor|hopformer}
source "$(dirname "$0")/pod_env.sh"
cd "$PROJ"
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d)_$(git rev-parse --short HEAD)_$(basename "$EVAL_SET" .npz)}
PREDS=baselines/preds/$RUN_ID
mkdir -p "logs/$RUN_ID" "$PREDS" results/baselines
LOG=logs/$RUN_ID/$METHOD.log
exec > >(tee -a "$LOG") 2>&1
echo "== $METHOD run $RUN_ID on $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
[ -f "$VOL/.setup/$METHOD.ok" ] && [ -f "$VOL/.setup/eval.ok" ] || { echo "run setup_remote.sh first"; exit 1; }

EV=".venv-eval/bin/python -W ignore baselines/eval_arctic.py --eval-set $EVAL_SET --preds $PREDS"
case "$METHOD" in
  hamer|wilor)
    conda activate "$ENVS/$METHOD"
    REPO=$REPOS/$([ "$METHOD" = wilor ] && echo WiLoR || echo hamer)
    python baselines/infer_hand2d.py --method "$METHOD" --repo "$REPO" --eval-set "$EVAL_SET" \
      --image-kind "$IMAGE_KIND" --out "$PREDS"
    conda deactivate
    $EV --methods "$METHOD" --tag "${RUN_ID}_${METHOD}"
    if [ -f baselines/preds/moge_hand2d.json ]; then
      cp baselines/preds/moge_hand2d.json "$PREDS/"
      $EV --methods "$METHOD" --intrinsics "moge:$PREDS/moge_hand2d.json" --tag "${RUN_ID}_${METHOD}_moge"
    fi
    ;;
  hopformer)
    conda activate "$ENVS/hopformer"
    DET=$REPOS/WiLoR/pretrained_models/detector.pt
    python baselines/infer_hopformer.py --repo "$REPOS/HOPformer" --eval-set "$EVAL_SET" --detector "$DET" --out "$PREDS"
    python baselines/estimate_intrinsics_moge.py --eval-set "$EVAL_SET" --for hopformer --image-kind "$IMAGE_KIND" \
      --out "$PREDS/moge_hopformer.json"
    python baselines/infer_hopformer.py --repo "$REPOS/HOPformer" --eval-set "$EVAL_SET" --detector "$DET" --out "$PREDS" \
      --tag hopformer_moge --intrinsics "moge:$PREDS/moge_hopformer.json"
    conda deactivate
    $EV --methods hopformer --tag "${RUN_ID}_hopformer"
    $EV --methods hopformer_moge --tag "${RUN_ID}_hopformer_moge"
    ;;
  *) echo "unknown method $METHOD"; exit 1 ;;
esac

.venv-eval/bin/python - "$METHOD" "$RUN_ID" <<'PY'
import hashlib, json, os, subprocess, sys, datetime
m, run = sys.argv[1:3]
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
repos = os.environ["REPOS"]
es = os.environ["EVAL_SET"]
man = {"run_id": run, "method": m, "finished_utc": datetime.datetime.utcnow().isoformat() + "Z",
       "git_commit": sh("git rev-parse HEAD"), "git_dirty": bool(sh("git status --porcelain")),
       "eval_set": es, "eval_set_sha256": hashlib.sha256(open(es, "rb").read()).hexdigest(),
       "image_kind": os.environ["IMAGE_KIND"], "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader | head -1"),
       "model_repo_commit": sh(f"git -C {repos}/{dict(hamer='hamer', wilor='WiLoR', hopformer='HOPformer')[m]} rev-parse HEAD"),
       "checkpoint": {"hopformer": "Sid2697/HOPformer p2_arctic_ego_epoch8.ckpt", "wilor": "wilor_final.ckpt + detector.pt",
                      "hamer": "hamer.ckpt (fetch_demo_data.sh) + ViTDet-H + ViTPose"}[m]}
os.makedirs(f"results/baselines", exist_ok=True)
json.dump(man, open(f"results/baselines/{run}_{m}_manifest.json", "w"), indent=1)
print(json.dumps(man, indent=1))
PY

bash baselines/upload_run.sh "$RUN_ID" "$METHOD" || true
echo "== $METHOD done"
