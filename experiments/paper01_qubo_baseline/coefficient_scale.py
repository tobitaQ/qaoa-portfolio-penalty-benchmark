# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""How small the objective is next to the penalty, in the units that matter.

Sec. V-B and Sec. VI-C said that at A_heur "the objective is 0.5 % of the
largest coefficient". Review round 3 (M6) asked which coefficient: the
largest entry of the QUBO matrix Q and the scale s_H = max(max|h_i|,
max|J_ij|) the circuit is actually normalised by are different numbers,
because each h_i sums several Q entries. This computes both, per instance
and penalty setting, together with the quantity the argument needs:

    ε_F(A) = Δ_F / s_H(A)

the width of the objective over the feasible set in units of the normalised
Hamiltonian -- on F the penalty vanishes and the constant cancels in
differences, so ε_F is the energy range the optimiser can still resolve
between feasible solutions once H_C / s_H is what it descends.

    python -m experiments.paper01_qubo_baseline.coefficient_scale

No circuit runs; the instances are the E1 references. Writes
``results/paper01_coefficient_scale.csv`` (one row per N × instance × A) and
prints the per-(N, A) medians and ranges.
"""

from __future__ import annotations

import argparse

import numpy as np

from experiments.paper01_qubo_baseline.run_e1_cross import (
    DATA_END_DATE,
    DEFAULT_INSTANCES,
    InstanceReference,
    RESULTS_DIR,
    RISK_AVERSION,
)
from experiments.paper01_qubo_baseline.run_experiment import write_csv
from src.finance.data_loader import FinanceDataLoader
from src.qubo.portfolio import PortfolioQUBO


def _ising_scale(formulator: PortfolioQUBO, problem) -> float:
    h, J, _ = formulator.to_ising(problem)
    return float(max(np.max(np.abs(h)), np.max(np.abs(J))))


def scales_for(ref: InstanceReference, formulator: PortfolioQUBO, label: str, a_value: float) -> dict:
    kw = dict(num_select=ref.k, risk_aversion=RISK_AVERSION)
    full = formulator.formulate(ref.data.returns, ref.data.covariance, penalty_strength=a_value, **kw)
    objective_only = formulator.formulate(ref.data.returns, ref.data.covariance, penalty_strength=0.0, **kw)
    q_full, q_obj = full.Q, objective_only.Q
    q_pen = q_full - q_obj
    s_h = _ising_scale(formulator, full)
    s_obj_ising = _ising_scale(formulator, objective_only)
    # The penalty part alone, as an Ising problem: h_i = A(K − N/2), J_ij = A/2.
    pen_problem = type(full)(Q=q_pen, offset=0.0, n_variables=full.n_variables,
                             asset_names=list(full.asset_names), metadata={})
    s_pen_ising = _ising_scale(formulator, pen_problem)
    return {
        "n_assets": ref.n, "instance": ref.instance, "penalty_label": label, "penalty": a_value,
        "n_select": ref.k, "delta_F": ref.feasible.delta,
        "max_abs_Q": float(np.abs(q_full).max()),
        "max_abs_Q_objective": float(np.abs(q_obj).max()),
        "max_abs_Q_penalty": float(np.abs(q_pen).max()),
        "qubo_ratio_objective_over_max": float(np.abs(q_obj).max() / np.abs(q_full).max()),
        "s_H": s_h,
        "s_H_objective_only": s_obj_ising,
        "s_H_penalty_only": s_pen_ising,
        "ising_ratio_objective_over_s_H": s_obj_ising / s_h,
        "epsilon_F": ref.feasible.delta / s_h,
        "delta_F_over_A": ref.feasible.delta / a_value if a_value > 0 else np.nan,
    }


def run(sizes: list[int], instances: list[int]) -> list[dict]:
    loader, formulator = FinanceDataLoader(), PortfolioQUBO()
    out = []
    for n in sizes:
        for inst in instances:
            ref = InstanceReference(n, inst, loader, formulator, DATA_END_DATE)
            settings = [("Aheur", ref.a_heur), ("Amargin", ref.a_margin)]
            if ref.a_crit > 0:
                settings.append(("1.1xAcrit", 1.1 * ref.a_crit))
            for label, a in settings:
                out.append(scales_for(ref, formulator, label, a))
    return out


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for n in sorted({r["n_assets"] for r in rows}):
        for label in ("Aheur", "Amargin", "1.1xAcrit"):
            g = [r for r in rows if r["n_assets"] == n and r["penalty_label"] == label]
            if not g:
                continue
            row = {"n_assets": n, "penalty_label": label, "instances": len(g)}
            for f in ("qubo_ratio_objective_over_max", "ising_ratio_objective_over_s_H", "epsilon_F"):
                v = np.asarray([r[f] for r in g], dtype=float)
                row[f"{f}_median"] = float(np.median(v))
                row[f"{f}_min"] = float(v.min()); row[f"{f}_max"] = float(v.max())
            out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--sizes", type=int, nargs="+", default=[12, 16, 20])
    ap.add_argument("--instances", type=int, nargs="+", default=list(DEFAULT_INSTANCES))
    args = ap.parse_args(argv)
    rows = run(args.sizes, args.instances)
    write_csv(RESULTS_DIR / "paper01_coefficient_scale.csv", rows)
    summary = summarise(rows)
    write_csv(RESULTS_DIR / "paper01_coefficient_scale_summary.csv", summary)
    print(f"{'N':>2} {'A':>9} {'inst':>4} {'obj/max|Q|':>22} {'obj/s_H (Ising)':>22} {'ε_F = Δ_F/s_H':>22}")
    for r in summary:
        def rng(f):
            return f"{r[f + '_median']:.4f} [{r[f + '_min']:.4f}, {r[f + '_max']:.4f}]"
        print(f"{r['n_assets']:>2} {r['penalty_label']:>9} {r['instances']:>4} "
              f"{rng('qubo_ratio_objective_over_max'):>22} {rng('ising_ratio_objective_over_s_H'):>22} "
              f"{rng('epsilon_F'):>22}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
