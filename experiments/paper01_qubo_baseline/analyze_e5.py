# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Paper ① / TQE plan E5: the E1 block on a second, non-overlapping price window.

E5 re-runs the pre-fixed block of E1 — instances 0-9 × seeds 42-46 ×
{Aheur, Amargin} at N ∈ {12, 16, 20}, the baseline solver — on the three-year
window that immediately precedes the paper's (2020-08-05 – 2023-08-05 against
2023-08-06 – 2026-08-05), same fifty-ticker universe, same instance-drawing
rule. Every instance is therefore the same ticker set as its E1 namesake with
different returns and covariance.

Reads ``paper01_e1_cross.csv`` (restricted to the block) and
``paper01_e1_cross_e5.csv`` with their reference files, aggregates both with
the E1 functions (same statistics, same pairing), and writes

    paper01_e5_period_compare.csv   one row per (N, quantity): the E1-window
                                    value beside the E5-window value

The question is plan §9.1's: does the direction of the main contrast, the
deflation, the overshoot and the E2 conclusion depend on the market period the
instances were drawn from? It is a check of the benchmark's behaviour under a
second input condition, not a backtest.

Run:
    python -m experiments.paper01_qubo_baseline.analyze_e5
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from experiments.paper01_qubo_baseline.analyze_e1 import (
    _f,
    _q,
    contrasts,
    decomposition,
    load,
    shot_budget_table,
    summarise,
)
from experiments.paper01_qubo_baseline.run_e1_cross import RESULTS_DIR
from experiments.paper01_qubo_baseline.run_experiment import write_csv

BLOCK_INSTANCES = set(range(10))
BLOCK_PENALTIES = {"Aheur", "Amargin"}
WINDOWS = {"E1": "", "E5": "_e5"}


def block(rows: list[dict]) -> list[dict]:
    return [r for r in rows if int(r["instance"]) in BLOCK_INSTANCES
            and r["penalty_label"] in BLOCK_PENALTIES]


def _ref_stats(refs: list[dict], n: int) -> dict:
    rr = [r for r in refs if int(r["n_assets"]) == n and int(r["instance"]) in BLOCK_INSTANCES]
    over = [_f(r["a_heur_over_crit"]) for r in rr if r["a_crit_is_zero"] != "True"]
    return {
        "instances": len(rr),
        "a_crit_zero": sum(r["a_crit_is_zero"] == "True" for r in rr),
        "overshoot_median": _q(over, 50) if over else np.nan,
        "delta_F_median": _q([_f(r["delta_F"]) for r in rr], 50),
        "f_star_median": _q([_f(r["f_star"]) for r in rr], 50),
        "P_tau_uniform_F_median": _q([_f(r["P_tau_main_uniform_F"]) for r in rr], 50),
        "data_start": rr[0]["data_start"] if rr else "",
        "data_end": rr[0]["data_end"] if rr else "",
    }


def compare(per_window: dict[str, dict]) -> list[dict]:
    """Side-by-side rows: one per (N, quantity)."""
    out = []
    ns = sorted({int(r["n_assets"]) for w in per_window.values() for r in w["summary"]})
    for n in ns:
        rows: dict[str, dict] = {}

        def put(name, key, value):
            rows.setdefault(name, {"n_assets": n, "quantity": name})[key] = value

        for w, d in per_window.items():
            rs = _ref_stats(d["refs"], n)
            put("window", w, f"{rs['data_start']} – {rs['data_end']}")
            put("instances with A_crit = 0", w, rs["a_crit_zero"])
            put("A_heur / A_crit, median", w, rs["overshoot_median"])
            put("Δ_F, median", w, rs["delta_F_median"])
            put("P_τ under uniform-over-F, median", w, rs["P_tau_uniform_F_median"])
            for s in d["summary"]:
                if int(s["n_assets"]) != n:
                    continue
                lab = s["penalty_label"]
                put(f"{lab}: P_F median", w, s["final_P_F_median"])
                put(f"{lab}: P_τ median", w, s["final_P_tau_main_median"])
                put(f"{lab}: Q_τ(1000) median", w, s["final_Q_tau_main_1000_median"])
                put(f"{lab}: g_off median (%)", w, s["g_off_median"] * 100)
                put(f"{lab}: g_F median (%)", w, s["g_F_median"] * 100)
                put(f"{lab}: optimal rate", w, s["optimal_rate"])
                put(f"{lab}: runs with no feasible shot", w, s["no_feasible_shot_runs"])
                put(f"{lab}: pass g_off ≤ 0.1 % but fail g_F ≤ τ", w, s["g_off_success_but_not_tau_rate"])
                put(f"{lab}: deflation D median", w, s["deflation_D_median"])
            for c in d["contrasts"]:
                if int(c["n_assets"]) != n or c["treatment"] != "Amargin":
                    continue
                fld = c["field"]
                scale = 100 if fld in ("g_off", "g_F") else 1
                put(f"Amargin − Aheur: {fld} median diff", w, c["instance_median_diff"] * scale)
                put(f"Amargin − Aheur: {fld} CI low", w, c["ci95_low"] * scale)
                put(f"Amargin − Aheur: {fld} CI high", w, c["ci95_high"] * scale)
                put(f"Amargin − Aheur: {fld} instances better / tied / of", w,
                    f"{c['instances_treatment_better']} / {c['instances_tied']} / {c['instances']}")
            for sh in d["shots"]:
                if int(sh["n_assets"]) != n or sh["shots"] != 1000:
                    continue
                lab = sh["penalty_label"]
                put(f"{lab}: E2 optimised beats uniform-F at S = 1000 (rate)", w, sh["final_beats_uniform_F_rate"])
                put(f"{lab}: E2 Q_τ(1000) optimised / unoptimised / uniform-F", w,
                    f"{sh['Q_tau_final_median']:.3f} / {sh['Q_tau_initial_median']:.3f} / {sh['Q_tau_uniform_F_median']:.3f}")
            for dc in d["decomposition"]:
                if int(dc["n_assets"]) != n or dc["treatment"] != "Amargin":
                    continue
                put("Amargin / Aheur g_off ratio: pairs", w, dc["pairs"])
                # Medians are levels, and three of them do not multiply; the
                # geometric means over the same pairs do (analyze_e1.decomposition).
                put("Amargin / Aheur g_off ratio: median total / solution / denominator", w,
                    f"{dc['g_off_ratio_median']:.1f} / {dc['solution_part_median']:.2f} / {dc['denominator_part_median']:.1f}")
                put("Amargin / Aheur g_off ratio: geometric total = solution × denominator", w,
                    f"{dc['g_off_ratio_geomean']:.1f} = {dc['solution_part_geomean']:.2f} × {dc['denominator_part_geomean']:.1f}")
        out.extend(rows.values())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    per_window = {}
    for w, suffix in WINDOWS.items():
        cross = RESULTS_DIR / f"paper01_e1_cross{suffix}.csv"
        refs = RESULTS_DIR / f"paper01_e1_reference{suffix}.csv"
        if not cross.exists():
            print(f"missing {cross}", file=sys.stderr)
            return 1
        rows = block(load(cross))
        ref_rows = load(refs)
        per_window[w] = {
            "rows": rows, "refs": ref_rows,
            "summary": summarise(rows), "contrasts": contrasts(rows),
            "shots": shot_budget_table(rows, ref_rows),
            "decomposition": decomposition(rows),
        }
        print(f"{w}: {len(rows)} runs in the block, "
              f"{sum(r['status'] != 'ok' for r in rows)} failed, window "
              f"{ref_rows[0]['data_start']} – {ref_rows[0]['data_end']}")
    table = compare(per_window)
    write_csv(RESULTS_DIR / "paper01_e5_period_compare.csv", table)
    for r in table:
        e1, e5 = r.get("E1", ""), r.get("E5", "")
        fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
        print(f"N={r['n_assets']:2d} {r['quantity']:<70s} {fmt(e1):>28s} | {fmt(e5):>28s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
