#!/bin/bash
# Run the per-part pass after the SHOW3D whole-hand run and the per-part validation finish.
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python
while ! grep -q "wrote" logs/run_show3d.log 2>/dev/null; do sleep 30; done
sleep 60  # let the post-SHOW3D export finish its heavy part
rm -f results/smoke/arctic_parts_limit3.parquet results/smoke/arctic_vertex_limit3.npz
$PY -W ignore analysis/run_partwise.py --dataset arctic --limit 3 --workers 3 --out results/smoke > logs/parts_validate.log 2>&1

$PY -I - <<'EOF' || { echo "VALIDATION FAILED - per-part run not started"; exit 1; }
import pandas as pd
a = pd.read_parquet("results/smoke/arctic_frames_limit3.parquet")
b = pd.read_parquet("results/smoke/arctic_parts_limit3.parquet")
m = a.merge(b, on=["seq", "hand", "frame"], suffixes=("_o", "_p"))
d = (m.pen_max_mm_o - m.pen_max_mm_p).abs()
bad = (d > 0.05).mean()
print(f"validation: {len(m)} rows, max |diff| {d.max():.4f} mm, {100*bad:.3f}% rows > 0.05 mm")
assert len(m) == len(a) and bad < 0.001
EOF

for ds in epic arctic show3d; do
  $PY -W ignore analysis/run_partwise.py --dataset $ds --workers 14 > logs/parts_$ds.log 2>&1
  grep -E "wrote|ERROR" logs/parts_$ds.log | head -5
done
$PY -W ignore analysis/export_parts.py > logs/export_parts.log 2>&1
tail -n 1 logs/export_parts.log
