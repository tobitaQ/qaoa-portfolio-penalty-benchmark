# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Tests for the E1 cross-experiment runner.

The runner's own logic is small: which penalty settings an instance gets
(and that A_crit = 0 collapses four of them into one), how a run is scored
by the Q-free evaluator, and that a resumed manifest refuses a different spec.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from experiments.paper01_qubo_baseline import run_e1_cross as e1
from experiments.paper01_qubo_baseline.run_degeneracy import RISK_FREE_RATE
from experiments.paper01_qubo_baseline.run_experiment import RISK_AVERSION
from src.finance.data_loader import FinanceDataLoader
from src.finance.metrics import PortfolioMetrics
from src.qubo import scoring
from src.qubo.portfolio import PortfolioQUBO
from src.solvers.braket_solver import BraketSolver


@pytest.fixture(scope="module")
def ref():
    return e1.InstanceReference(12, 0, FinanceDataLoader(), PortfolioQUBO(), e1.DATA_END_DATE)


class TestConditions:
    def test_published_instance_gets_six_distinct_penalties(self, ref):
        conds = ref.conditions()
        assert [c[0] for c in conds] == ["1.1xAcrit", "2xAcrit", "5xAcrit", "10xAcrit",
                                         "Aheur", "Amargin"]
        values = [c[1] for c in conds]
        assert len(set(values)) == 6
        assert values[0] == pytest.approx(1.1 * ref.a_crit)
        assert ref.a_heur == pytest.approx(13.0, abs=1e-6)
        assert ref.a_margin > ref.a_crit

    def test_acrit_zero_collapses_the_multiples(self, ref, monkeypatch):
        monkeypatch.setattr(ref, "a_crit", 0.0)
        conds = ref.conditions()
        assert [c[0] for c in conds] == ["A0", "Aheur", "Amargin"]
        assert conds[0][1] == 0.0

    def test_reference_row_matches_the_published_census(self, ref):
        """Table VI's N=12 numbers: |F| = 66, deflation ~80x, A_heur/A_crit ~1213x."""
        row = ref.row()
        assert row["feasible_set_size"] == 66
        assert float(row["a_heur_over_crit"]) == pytest.approx(1213.07, rel=1e-3)
        assert row["a_crit_binding_m"] == 1
        assert row["P_F_uniform_all"] == pytest.approx(66 / 4096)
        assert row["n_optimal_ties"] == 1
        assert scoring.deflation_factor(row["f_star"], ref.a_heur, 2, row["delta_F"]) == \
            pytest.approx(80.51, rel=1e-3)


class TestScoring:
    def test_a_run_is_scored_consistently(self, ref):
        problem = PortfolioQUBO().formulate(
            ref.data.returns, ref.data.covariance, num_select=ref.k,
            risk_aversion=RISK_AVERSION, penalty_strength=ref.a_heur)
        res = BraketSolver("lightning_cpu", p_layers=1, n_shots=300,
                           n_optimizer_steps=3, record_state=True).solve(problem, seed=42)
        row, artifact = e1.score_run(res, problem, ref, ref.a_heur,
                                     PortfolioMetrics(risk_free_rate=RISK_FREE_RATE))
        assert row["shots"] == 300
        assert row["best_shot_scorer_agrees"]
        assert row["feasible_shot_fraction"] == pytest.approx(row["feasible_shots"] / 300)
        assert 0.0 <= row["final_P_F"] <= 1.0
        assert row["final_P_opt"] <= row["final_P_tau_main"] <= row["final_P_tau_alt"] <= row["final_P_F"]
        assert row["final_probs_sum"] == pytest.approx(1.0, abs=1e-6)
        assert row["final_Q_tau_main_1000"] == pytest.approx(
            1 - (1 - row["final_P_tau_main"]) ** 1000)
        if row["g_F"] != "":
            assert row["g_F"] == pytest.approx(row["deflation_D"] * row["g_off"], rel=1e-9)
            assert row["rank_best_feasible"] >= 1
            assert row["is_optimal"] == (row["rank_best_feasible"] == 1)
        assert artifact["final_probs_feasible"].shape == (66,)
        assert artifact["shot_counts"].sum() == 300
        assert artifact["angles_trajectory"].shape == (3, 2)
        # The expectation of the violation must be zero mass on F and positive
        # elsewhere: E[V] = sum over cardinalities of mass * (m-K)^2.
        card = artifact["final_cardinality_mass"]
        ev = sum(card[m] * (m - ref.k) ** 2 for m in range(13))
        assert row["final_expected_violation"] == pytest.approx(ev, rel=1e-9)

    def test_no_feasible_shot_is_recorded_as_such(self, ref):
        problem = PortfolioQUBO().formulate(
            ref.data.returns, ref.data.covariance, num_select=ref.k,
            risk_aversion=RISK_AVERSION, penalty_strength=ref.a_heur)
        res = BraketSolver("lightning_cpu", p_layers=1, n_shots=20,
                           n_optimizer_steps=2, record_state=True).solve(problem, seed=1)
        # Force every shot infeasible: all-ones bitstrings.
        res.metadata["last_samples"] = np.ones((20, 12))
        res.bitstring = np.ones(12)
        res.energy_no_offset = float(np.ones(12) @ problem.Q @ np.ones(12))
        row, _ = e1.score_run(res, problem, ref, ref.a_heur,
                              PortfolioMetrics(risk_free_rate=RISK_FREE_RATE))
        assert row["feasible_shots"] == 0
        assert row["best_all_feasible"] is False
        assert row["best_all_cardinality"] == 12
        assert row["g_F"] == "" and row["rank_best_feasible"] == ""


class TestResume:
    def test_a_foreign_spec_version_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(e1, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(e1, "ARTIFACT_DIR", tmp_path / "e1_runs")
        manifest = tmp_path / "paper01_e1_cross_x.csv"
        with open(manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["spec_version", *e1.RUN_KEY])
            w.writeheader()
            w.writerow({"spec_version": "e1-v0", "n_assets": "12", "instance": "0",
                        "penalty_label": "Aheur", "qaoa_seed": "42"})
        assert e1.main(["--results-tag", "x", "--resume", "--backend", "lightning_cpu",
                        "--n", "12", "--instances", "0", "--seeds", "42"]) == 1


    def test_an_untagged_run_over_the_committed_cross_is_refused(self, tmp_path, monkeypatch, capsys):
        """The 2,565-run manifest and its artifacts are the baseline; a bare run must not replace them."""
        monkeypatch.setattr(e1, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(e1, "ARTIFACT_DIR", tmp_path / "e1_runs")
        manifest = tmp_path / "paper01_e1_cross.csv"
        manifest.write_text("spec_version,n_assets\n")
        assert e1.main(["--backend", "lightning_cpu", "--n", "12",
                        "--instances", "0", "--seeds", "42"]) == 2
        assert "--write-canonical" in capsys.readouterr().err
        assert manifest.read_text() == "spec_version,n_assets\n"
        assert not (tmp_path / "e1_runs").exists()

    def test_resume_is_the_sanctioned_way_to_touch_the_canonical_files(self, tmp_path, monkeypatch):
        """--resume appends the missing rows only, so it passes the guard (and then meets the spec check)."""
        monkeypatch.setattr(e1, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(e1, "ARTIFACT_DIR", tmp_path / "e1_runs")
        manifest = tmp_path / "paper01_e1_cross.csv"
        with open(manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["spec_version", *e1.RUN_KEY])
            w.writeheader()
            w.writerow({"spec_version": "e1-v0", "n_assets": "12", "instance": "0",
                        "penalty_label": "Aheur", "qaoa_seed": "42"})
        # Past the overwrite guard (exit 2), stopped by the spec check (exit 1).
        assert e1.main(["--resume", "--backend", "lightning_cpu", "--n", "12",
                        "--instances", "0", "--seeds", "42"]) == 1


class TestBestFeasibleRule:
    """The output rule of Sec. III-C, case by case (review round 3, M1 / §5.1).

    The rule returns the lowest-f shot among the *feasible* shots, and a batch
    with no feasible shot is a counted outcome. Because E_A and f induce the
    same order on F, "the batch contains a qualifying feasible solution" and
    "the rule returns one" are the same event, so Q_τ(S) = 1 − (1 − P_τ)^S is
    also the probability of *returning* a qualifying solution -- which it would
    not be for a rule that takes the lowest-E_A shot without checking
    feasibility.
    """

    @staticmethod
    def _batch(ref, indices: list[int]) -> np.ndarray:
        return scoring.bitstrings_from_indices(np.asarray(indices), ref.n)

    @staticmethod
    def _score(ref, samples, a_value):
        return e1.score_shots(samples, ref, a_value, PortfolioMetrics(risk_free_rate=RISK_FREE_RATE))

    @pytest.fixture(scope="class")
    def picks(self, ref):
        feas = ref.feasible
        order = np.argsort(feas.f, kind="stable")
        g = feas.gap_range(feas.f)
        outside = [int(feas.indices[i]) for i in order if g[i] > e1.TAU_ALT]
        inside = [int(feas.indices[i]) for i in order if g[i] <= e1.TAU_MAIN]
        # The infeasible state with the lowest f: at A = 0 its E_A is below
        # every feasible shot's, so a feasibility-blind rule would return it.
        infeasible = np.flatnonzero(ref.states.cardinality != ref.k)
        cheapest_infeasible = int(infeasible[np.argmin(ref.states.f[infeasible])])
        assert ref.states.f[cheapest_infeasible] < feas.f_star
        return {"optimum": int(feas.indices[order[0]]), "second": int(feas.indices[order[1]]),
                "inside": inside, "outside": outside, "cheapest_infeasible": cheapest_infeasible}

    def test_only_infeasible_shots_is_a_counted_failure(self, ref, picks):
        row = self._score(ref, self._batch(ref, [picks["cheapest_infeasible"]] * 5), ref.a_heur)
        assert row["feasible_shots"] == 0
        assert row["best_feasible_index"] == "" and row["g_F"] == ""
        assert row["within_tau_main"] == "" and row["is_optimal"] == ""
        assert row["shots_within_tau_main"] == 0

    def test_a_cheaper_infeasible_shot_cannot_displace_a_qualifying_feasible_one(self, ref, picks):
        batch = self._batch(ref, [picks["cheapest_infeasible"], picks["inside"][-1]])
        row = self._score(ref, batch, a_value=0.0)
        # At A = 0 the infeasible shot is the minimum-E_A shot of the batch...
        assert row["best_all_index"] == picks["cheapest_infeasible"]
        assert row["best_all_feasible"] is False
        # ...and the rule still returns the feasible one, which qualifies.
        assert row["best_feasible_index"] == picks["inside"][-1]
        assert row["within_tau_main"] is True

    def test_feasible_but_outside_tau_is_returned_and_does_not_qualify(self, ref, picks):
        row = self._score(ref, self._batch(ref, picks["outside"][-2:]), ref.a_heur)
        assert row["best_feasible_index"] == picks["outside"][-2]
        assert row["within_tau_main"] is False and row["within_tau_alt"] is False
        assert row["is_optimal"] is False

    def test_several_qualifying_shots_return_the_lowest_f(self, ref, picks):
        worst_inside = picks["inside"][-1]
        batch = self._batch(ref, [worst_inside, picks["optimum"], worst_inside])
        row = self._score(ref, batch, ref.a_heur)
        assert row["best_feasible_index"] == picks["optimum"]
        assert row["is_optimal"] is True and row["rank_best_feasible"] == 1
        assert row["shots_within_tau_main"] == 3

    def test_exact_ties_break_to_the_lowest_basis_index(self, ref, picks):
        """Two feasible states with identical f: the rule is deterministic."""
        import copy
        feas = ref.feasible
        lo, hi = sorted((picks["optimum"], picks["second"]))
        f = feas.f.copy()
        f[feas.position(hi)] = f[feas.position(lo)]       # make them tie exactly
        tied = scoring.FeasibleSet(indices=feas.indices, f=f, f_sorted=np.sort(f))
        states_f = ref.states.f.copy(); states_f[hi] = states_f[lo]
        fake = copy.copy(ref)
        fake.feasible = tied
        fake.states = scoring.AllStates(f=states_f, cardinality=ref.states.cardinality)
        row = self._score(fake, self._batch(ref, [hi, lo, hi]), ref.a_heur)
        assert row["best_feasible_index"] == lo
        assert row["rank_ties"] == 2 and row["rank_best_feasible"] == 1
        assert row["is_optimal"] is True

    def test_tolerance_boundaries_are_inclusive(self, ref, picks):
        """f − f* ≤ OPT_TOL·Δ_F counts as optimal; g_F ≤ τ counts as within."""
        feas = ref.feasible
        assert bool(feas.is_optimal(e1.OPT_TOL)[feas.position(picks["optimum"])])
        # A state exactly on the τ line qualifies (≤, not <).
        g = feas.gap_range(feas.f)
        assert feas.within(e1.TAU_MAIN).sum() == int((g <= e1.TAU_MAIN).sum())
        assert not bool(feas.within(e1.TAU_MAIN)[np.argmax(g)])

    def test_qualifying_output_is_the_same_event_as_a_qualifying_shot(self, ref):
        """{g_F(x_out) ≤ τ} ⇔ {batch ∩ F_τ ≠ ∅}, batch by batch, under i.i.d. shots.

        So Q_τ(S) = 1 − (1 − P_τ)^S is the probability of *returning* a
        qualifying solution under this rule, not only of drawing one.
        """
        rng = np.random.default_rng(3)
        # A synthetic sampler: uniform over all 2^N bitstrings, so P_τ is |F_τ| / 2^N.
        n_states = ref.states.f.shape[0]
        tau_set = set(int(i) for i in ref.feasible.indices[ref.within_main])
        p_tau = len(tau_set) / n_states
        S, batches, hits = 40, 300, 0
        for _ in range(batches):
            idx = rng.integers(0, n_states, size=S)
            row = self._score(ref, scoring.bitstrings_from_indices(idx, ref.n), ref.a_heur)
            returned_qualifies = row["within_tau_main"] is True
            batch_contains = bool(tau_set & set(int(i) for i in idx))
            assert returned_qualifies == batch_contains
            hits += returned_qualifies
        q = 1 - (1 - p_tau) ** S
        assert abs(hits / batches - q) < 4 * np.sqrt(q * (1 - q) / batches) + 1e-12
