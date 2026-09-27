# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT
"""Does greedy repair still find the optimum at sizes the QAOA record does not cover?

`probe_condition_changes` repairs a *stored* portfolio at N = 12, 16 and 20, where the
feasible set is small. This script asks the same question where the feasible set is
large and the start carries no information: 100 random feasible portfolios per size at
N = 24 (C(24,4) = 10,626) and N = 30 (C(30,6) = 593,775), repaired by the same
single-asset-swap greedy search, against the enumerated optimum.

Deterministic: the starts come from a seeded generator, the instance is the published
one (instance 0) on the pinned price snapshot, and nothing is downloaded.

Run:
    python -m experiments.a1_reuse.probe_repair_scaling        # ~2 min CPU
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from experiments.a1_reuse.probe_condition_changes import W1, Instance, bits_to_index
from experiments.paper01_qubo_baseline.run_experiment import write_csv
from src.finance.data_loader import FinanceDataLoader

OUT = Path(__file__).parent / "results"
SIZES = (24, 30)
STARTS = 100
SEED = 42


def random_feasible(rng: np.random.Generator, n: int, k: int) -> int:
    bits = np.zeros(n, dtype=int)
    bits[rng.choice(n, size=k, replace=False)] = 1
    return bits_to_index(bits)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    loader = FinanceDataLoader()
    rows = []
    for n in SIZES:
        t0 = time.perf_counter()
        inst = Instance(n, 0, W1, loader=loader)
        rng = np.random.default_rng(SEED)
        reached, moves, missed_gaps, gaps = 0, [], [], []
        for _ in range(STARTS):
            start = random_feasible(rng, n, inst.k)
            idx, m = inst.repair(start)
            r = inst.regret(idx)
            reached += r["optimal"]
            moves.append(m)
            gaps.append(r["g_F"])
            if not r["optimal"]:
                missed_gaps.append(r["g_F"])
        rows.append({"n": n, "k": inst.k, "feasible_set_size": int(inst.feas.indices.size),
                     "starts": STARTS, "reached_optimum": reached,
                     "moves_median": float(np.median(moves)),
                     "g_F_median_when_missed": (float(np.median(missed_gaps)) if missed_gaps else 0.0),
                     "g_F_max": float(np.max(gaps)),
                     "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  N={n} K={inst.k}: {reached}/{STARTS} reached the optimum, "
              f"median {rows[-1]['moves_median']} moves, worst g_F {rows[-1]['g_F_max']:.4f}")
    write_csv(OUT / "a1_repair_scaling.csv", rows)
    print(f"wrote {OUT / 'a1_repair_scaling.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
