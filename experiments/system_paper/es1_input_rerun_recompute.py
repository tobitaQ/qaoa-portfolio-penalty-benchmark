# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Re-simulate every re-run row of the input-perturbation ladders, not a sample.

``state-recompute`` samples five runs per size and always includes the reference
instance's A_heur seed-42 run, so on the spliced records of the ladder it fires
on the pair that holds instance 0 and is silent on the pair that does not,
whatever the perturbation did. This driver restricts the spliced record to the
re-run rows and re-simulates all of them, so the ladder reports, per rung and
pair, how many of the re-run states the re-simulation finds off and by how much.

    python -m experiments.system_paper.es1_input_rerun_recompute
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.system_paper import es1_fault_injection as es1f
from experiments.system_paper import es1_input_rerun_fault as irf

RESULTS = es1f.RESULTS_DIR
LADDERS = (("", "0 1"), ("_pair23", "2 3"))


def main() -> int:
    clean = es1f.load_record((12,))
    rows = []
    for ladder, instances in LADDERS:
        frame = pd.read_csv(RESULTS / f"paper02_es1_input_rerun{ladder}.csv")
        for _, q in frame.iterrows():
            tag = str(q.tag)
            arm_tag = "d6all" if tag == "d1e6" else tag    # the 1e-6 arm of the first pair
            arm = es1f.load_record((12,), suffix=f"_{arm_tag}", label="arm")
            faulted, _, _ = irf.splice(clean, arm, "input_rerun")
            sub = faulted.copy("rerun-rows")
            sub.rows = sub.rows[sub.rows.run_id.isin(set(arm.rows.run_id))].reset_index(drop=True)
            t0 = time.perf_counter()
            found = es1f.check_state_recompute(sub, per_size=10_000)
            secs = time.perf_counter() - t0
            gaps = [float(m.group(1)) for m in (re.search(r"is ([0-9.e+-]+) from", f) for f in found) if m]
            rows.append({"ladder": ladder or "_pair01", "instances": instances, "tag": tag,
                         "delta": float(q.delta), "rows_resimulated": len(sub.rows),
                         "rows_off": len(found), "max_gap": max(gaps) if gaps else 0.0,
                         "tolerance": 1e-5, "seconds": round(secs, 1)})
            print(f"{ladder or '_pair01':<8} {tag:<8} delta={q.delta:g}: {len(found)}/{len(sub.rows)} rows off, "
                  f"max gap {max(gaps) if gaps else 0:.2e}, {secs:.0f} s", flush=True)
    out = RESULTS / "paper02_es1_input_rerun_recompute.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
