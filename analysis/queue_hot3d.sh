#!/bin/bash
# After the HOT3D annotation fetch finishes: whole-hand metrics (MANO + UmeTrack), per-part (MANO),
# then rebuild summary + viewer data.
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python
while ! grep -q "^done" logs/hot3d_fetch.log 2>/dev/null; do sleep 60; done
grep -E "ERROR" logs/hot3d_fetch.log | head -5
$PY -W ignore analysis/run_penetration.py --dataset hot3d --workers 12 > logs/run_hot3d.log 2>&1
grep -E "wrote|ERROR|sequences" logs/run_hot3d.log | head -5
$PY -W ignore analysis/run_penetration.py --dataset hot3d_ume --workers 12 > logs/run_hot3d_ume.log 2>&1
grep -E "wrote|ERROR|sequences" logs/run_hot3d_ume.log | head -5
$PY -W ignore analysis/run_partwise.py --dataset hot3d --workers 10 > logs/parts_hot3d.log 2>&1
grep -E "wrote|ERROR" logs/parts_hot3d.log | head -5
$PY -W ignore analysis/summarize.py > /dev/null 2>&1
$PY -W ignore analysis/export_viewer.py > logs/export_viewer_hot3d.log 2>&1
$PY -W ignore analysis/export_parts.py > logs/export_parts_hot3d.log 2>&1
tail -n 2 logs/export_viewer_hot3d.log logs/export_parts_hot3d.log
