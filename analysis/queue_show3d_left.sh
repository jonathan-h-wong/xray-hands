#!/bin/bash
# After the per-part queue finishes: re-run SHOW3D whole-hand metrics for LEFT hands only
# (left-hand MANO shapedirs fix), merge into show3d_frames.parquet, rebuild summary + viewer data.
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python
while ! grep -q "wrote" logs/export_parts.log 2>/dev/null; do sleep 60; done
$PY -W ignore analysis/run_penetration.py --dataset show3d --workers 12 --hands left > logs/run_show3d_left.log 2>&1
grep -E "wrote|ERROR" logs/run_show3d_left.log | head -5
$PY -I - <<'EOF'
import pandas as pd, shutil
old = "results/show3d_frames.parquet"
shutil.copy(old, "results/show3d_frames_leftbug_backup.parquet")
a = pd.read_parquet(old)
b = pd.read_parquet("results/show3d_frames_left.parquet")
m = pd.concat([a[a.hand != "left"], b], ignore_index=True)
m.to_parquet(old)
print(f"merged: {len(a)} -> {len(m)} rows ({(a.hand=='left').sum()} old left rows replaced by {len(b)})")
EOF
$PY -W ignore analysis/summarize.py > /dev/null 2>&1
$PY -W ignore analysis/export_viewer.py > logs/export_viewer.log 2>&1
$PY -W ignore analysis/export_parts.py > logs/export_parts_final.log 2>&1
tail -n 1 logs/export_viewer.log logs/export_parts_final.log
