#!/bin/bash
# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

# GPU queue after E1: re-time the 10 contended rows, then E3-A and E3-B.
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
while tmux has-session -t e1 2>/dev/null; do sleep 60; done
echo "[queue] E1 finished at $(date)"
# 1. drop the rows whose runtime was measured under contention, then re-run them
$PY - <<'PYEOF'
import csv, os
from pathlib import Path
ids = set(Path("experiments/paper01_qubo_baseline/queue/e1_rerun_ids.txt").read_text().split())
p = Path("experiments/paper01_qubo_baseline/results/paper01_e1_cross.csv")
rows = list(csv.DictReader(open(p)))
keep = [r for r in rows if r["run_id"] not in ids]
with open(p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(keep)
for i in ids:
    a = Path(f"experiments/paper01_qubo_baseline/results/e1_runs/{i}.npz")
    if a.exists(): a.unlink()
print(f"[queue] dropped {len(rows)-len(keep)} contended rows")
PYEOF
$PY -m experiments.paper01_qubo_baseline.run_e1_cross --backend lightning_gpu --precision single --resume 2>&1 | grep -v Warning >> logs/e1_full.log
echo "[queue] E1 re-run done at $(date)"
# 2. E3 controls
$PY -m experiments.paper01_qubo_baseline.run_e1_cross --backend lightning_gpu --precision single --resume \
   --instances 0 1 2 3 4 5 6 7 8 9 --penalties Aheur Amargin --stepsize 0.03 --steps 100 --results-tag e3_adam003 2>&1 | grep -v Warning >> logs/e3_adam003.log
echo "[queue] E3 adam003 done at $(date)"
$PY -m experiments.paper01_qubo_baseline.run_e1_cross --backend lightning_gpu --precision single --resume \
   --instances 0 1 2 3 4 5 6 7 8 9 --penalties Aheur Amargin --optimizer lbfgs --steps 100 --results-tag e3_lbfgs 2>&1 | grep -v Warning >> logs/e3_lbfgs.log
echo "[queue] E3 lbfgs done at $(date)"
# 3. (removed 2026-09-11) the seed extension 47-51 is deferred; the command
#    and the reason are in the project's deferred-experiments list (not distributed).
echo "[queue] all done at $(date)"
