# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT
"""The verification relations a run record satisfies on its own two CSVs.

The platform's thirteen checks live in the E-S1 harness
(``experiments/system_paper/es1_fault_injection.py``), and most of them need the
per-run ``.npz`` artifacts or the pinned statistics. Six do not: they read the
manifest and the reference table and nothing else. Those six are here, so that
the aggregation can run them before it reads a number -- a record that fails one
of them should not reach a table, and until this module existed the only thing
standing between an edited CSV and a published aggregate was the job's exit
status.

The harness imports these rather than keeping its own copy: one expression per
fact, in one place, or the two drift and the paper describes neither.

What is *not* here, and why:

* ``row-npz``, ``npz-internal``, ``identities``, ``seed-distinct`` and
  ``state-recompute`` need the artifacts, which a clone does not carry;
* ``input-digest`` and ``reference-input`` need the pinned statistics;
* ``manifest`` asks whether a block holds every cell its reference prescribes,
  which a deliberate partial run (a reproduction of one block) does not.

Run those with the harness. ``check_record`` is the subset that costs a second
and applies to any run of the driver, whole or partial.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.qubo import scoring

#: Binomial tail below which a batch is not a draw from its state.
#: The threshold is applied to the smaller *one-sided* tail, so a correct row
#: fails with probability up to 2 x TAIL_P and a record of ~5,000 tests has a
#: union bound of ~1e-5 rather than ~5e-6.
TAIL_P = 1e-9

#: Columns every ``ok`` row must carry filled. ``g_F``/``g_off`` may be empty
#: when a batch had no feasible shot, so they are not in this list.
REQUIRED = [
    "run_id", "n_assets", "n_select", "instance", "instance_digest", "input_digest",
    "penalty_label", "penalty", "a_crit", "a_heur", "a_margin", "qaoa_seed", "backend",
    "precision", "p_layers", "steps", "optimizer", "f_star", "delta_F",
    "feasible_set_size", "data_start", "data_end", "status", "shots", "feasible_shots",
    "distinct_shots", "best_all_index", "best_all_E", "best_all_f",
    "energy_no_offset_solver", "best_shot_scorer_agrees", "feasible_shot_fraction",
    "final_P_F", "final_P_opt", "initial_P_F", "initial_P_opt", "hamiltonian_scale",
    "final_expectation", "expectation_at_final_angles", "initial_angles", "final_angles",
]

#: The penalty labels that are a multiple of A_crit, as the cross driver writes them.
MULTIPLES = [(1.1, "1.1xAcrit"), (2.0, "2xAcrit"), (5.0, "5xAcrit"), (10.0, "10xAcrit")]


def check_schema(rows: pd.DataFrame, reference: pd.DataFrame) -> list[str]:
    """Every required column is present and filled on every completed run."""
    out = []
    missing = [c for c in REQUIRED if c not in rows.columns]
    if missing:
        out.append(f"missing columns: {missing}")
    ok = rows[rows.status == "ok"] if "status" in rows else rows
    for c in REQUIRED:
        if c in ok.columns and ok[c].isna().any():
            out.append(f"{c}: {int(ok[c].isna().sum())} empty cells on ok rows")
    return out


def check_penalty(rows: pd.DataFrame, reference: pd.DataFrame) -> list[str]:
    """The penalty value is the one its label names."""
    out = []
    mult = {lab: m for m, lab in MULTIPLES}
    for _, r in rows.iterrows():
        lab = r.penalty_label
        if lab == "Aheur":
            want = r.a_heur
        elif lab == "Amargin":
            want = r.a_margin
        elif lab == "A0":
            want = 0.0
        elif lab in mult:
            want = mult[lab] * r.a_crit
        else:
            out.append(f"{r.run_id}: unknown label {lab}")
            continue
        if not np.isclose(r.penalty, want, rtol=1e-12, atol=1e-15):
            out.append(f"{r.run_id}: penalty {r.penalty!r} is not {lab} = {want!r}")
        if r.a_crit > 0 and not np.isclose(r.a_over_crit, r.penalty / r.a_crit, rtol=1e-9):
            out.append(f"{r.run_id}: a_over_crit inconsistent")
    return out


def check_reference_row(rows: pd.DataFrame, reference: pd.DataFrame) -> list[str]:
    """The instance quantities copied onto a run row equal the reference row's."""
    out = []
    ref = reference.set_index(["n_assets", "instance"])
    for _, r in rows.iterrows():
        key = (r.n_assets, r.instance)
        if key not in ref.index:
            out.append(f"{r.run_id}: no reference row")
            continue
        q = ref.loc[key]
        for col in ("f_star", "delta_F", "feasible_set_size", "a_crit", "a_heur",
                    "a_margin", "instance_digest", "input_digest", "n_select"):
            a, b = r[col], q[col]
            same = (a == b) if isinstance(a, str) else np.isclose(a, b, rtol=0, atol=0)
            if not same:
                out.append(f"{r.run_id}: {col} {a!r} != reference {b!r}")
    return out


def check_scorer(rows: pd.DataFrame, reference: pd.DataFrame) -> list[str]:
    """Solver and scorer agree, and the two gaps follow from f, f*, Delta_F and A."""
    out = []
    for _, r in rows.iterrows():
        if r.best_shot_scorer_agrees is not True and str(r.best_shot_scorer_agrees) != "True":
            out.append(f"{r.run_id}: solver and scorer disagree on the best shot")
        card = r.best_all_cardinality
        E = r.best_all_f + r.penalty * (card - r.n_select) ** 2 - r.penalty * r.n_select ** 2
        if not np.isclose(E, r.best_all_E, rtol=1e-9, atol=1e-9):
            out.append(f"{r.run_id}: best_all_E {r.best_all_E!r} != f + A(m-k)^2 - Ak^2 = {E!r}")
        if not np.isclose(r.energy_no_offset_solver, r.best_all_E, rtol=1e-6, atol=1e-6):
            out.append(f"{r.run_id}: solver energy {r.energy_no_offset_solver!r} "
                       f"!= scorer {r.best_all_E!r}")
        if pd.notna(r.get("f_best_feasible", np.nan)):
            gF = (r.f_best_feasible - r.f_star) / r.delta_F
            if not np.isclose(gF, r.g_F, rtol=1e-9, atol=1e-12) or gF < -1e-12 or gF > 1 + 1e-12:
                out.append(f"{r.run_id}: g_F {r.g_F!r} != (f_bf - f*)/Delta_F = {gF!r}")
            goff = scoring.gap_offset(r.f_best_feasible, r.f_star, r.penalty, int(r.n_select))
            if not np.isclose(goff, r.g_off, rtol=1e-9, atol=1e-12):
                out.append(f"{r.run_id}: g_off {r.g_off!r} != formula {goff!r}")
    return out


def check_state_batch(rows: pd.DataFrame, reference: pd.DataFrame) -> list[str]:
    """The shot batch is a draw from the *final* state, not from another one."""
    out = []
    S = rows.shots.astype(float)
    from scipy.stats import binom
    for prefix, obs in (("P_F", rows.feasible_shots), ("P_opt", rows.shots_optimal)):
        p = rows[f"final_{prefix}"].clip(0, 1).to_numpy()
        x = obs.to_numpy(dtype=float)
        n = S.to_numpy()
        # Exact binomial tail. A Gaussian band misfires at the
        # rates the optimum has (S·P_opt ≈ 0.1: two hits are a 0.5 % event,
        # not a five-sigma one); the exact tail flags only what no draw from
        # the state explains. TAIL_P is compared with the smaller one-sided
        # tail, so the per-row false-alarm probability is up to 2 x TAIL_P.
        tail = np.minimum(binom.cdf(x, n, p), binom.sf(x - 1, n, p))
        bad = rows[tail < TAIL_P]
        for _, r in bad.iterrows():
            out.append(f"{r.run_id}: {obs.name} {int(r[obs.name])} has binomial tail "
                       f"{tail[r.name]:.1e} under S x final_{prefix} = {n[r.name] * p[r.name]:.1f}")
    for (n, lab), block in rows.groupby(["n_assets", "penalty_label"]):
        if len(block) < 10 or block.final_P_F.std() == 0:
            continue
        r_final = np.corrcoef(block.feasible_shot_fraction, block.final_P_F)[0, 1]
        r_init = (np.corrcoef(block.feasible_shot_fraction, block.initial_P_F)[0, 1]
                  if block.initial_P_F.std() > 0 else np.nan)
        if not (r_final > 0.9) or (np.isfinite(r_init) and r_init > r_final):
            out.append(f"N={n} {lab}: feasible fraction tracks final P_F at r={r_final:.2f}"
                       f" (initial P_F r={r_init:.2f})")
    return out


def check_ranges(rows: pd.DataFrame, reference: pd.DataFrame) -> list[str]:
    """Probabilities lie in [0, 1] and the nested counts and masses are ordered."""
    out = []
    for col in ("final_P_F", "final_P_opt", "initial_P_F", "initial_P_opt",
                "feasible_shot_fraction"):
        if col not in rows:
            continue
        bad = rows[(rows[col] < -1e-12) | (rows[col] > 1 + 1e-12)]
        for _, r in bad.iterrows():
            out.append(f"{r.run_id}: {col} {r[col]!r} is not a probability")
    tau = "shots_within_tau_main"
    for _, r in rows.iterrows():
        if r.feasible_shots > r.shots:
            out.append(f"{r.run_id}: feasible_shots {r.feasible_shots} > shots {r.shots}")
        if r.shots_optimal > r.feasible_shots:
            out.append(f"{r.run_id}: shots_optimal {r.shots_optimal} > feasible_shots "
                       f"{r.feasible_shots}")
        if tau in rows.columns and pd.notna(r[tau]):
            if not (r.shots_optimal <= r[tau] <= r.feasible_shots):
                out.append(f"{r.run_id}: optimum {r.shots_optimal} <= within-tau {r[tau]} "
                           f"<= feasible {r.feasible_shots} does not hold")
        if r.final_P_opt > r.final_P_F + 1e-12:
            out.append(f"{r.run_id}: final_P_opt {r.final_P_opt!r} > final_P_F "
                       f"{r.final_P_F!r}")
        if pd.notna(r.get("f_best_feasible", np.nan)) and r.f_best_feasible < r.f_star - 1e-12:
            out.append(f"{r.run_id}: f_best_feasible {r.f_best_feasible!r} is below the "
                       f"instance optimum f* {r.f_star!r}")
    return out


#: The relations of this module, in the order the harness runs them.
RECORD_CHECKS = (
    ("schema", check_schema),
    ("penalty", check_penalty),
    ("reference-row", check_reference_row),
    ("scorer", check_scorer),
    ("ranges", check_ranges),
    ("state-batch", check_state_batch),
)


def check_record(rows: pd.DataFrame, reference: pd.DataFrame,
                 skip: tuple[str, ...] = ()) -> dict[str, list[str]]:
    """Run every relation of this module and return the findings by name.

    Args:
        rows: The manifest, as the driver wrote it.
        reference: The reference table for the instances the manifest names.
        skip: Names to leave out.

    Returns:
        Check name to its findings; a name is present only if it found something.
    """
    found = {}
    for name, fn in RECORD_CHECKS:
        if name in skip:
            continue
        try:
            messages = fn(rows, reference)
        except Exception as exc:                      # a check that cannot run is a finding
            messages = [f"check raised {type(exc).__name__}: {exc}"]
        if messages:
            found[name] = messages
    return found
