# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""The solver's angle/state recording (plan §5.5) must not alter the result.

Two things are pinned here: the default path returns exactly what it did
before the recording was added, and the recorded probability vector uses the
same bit order as the shots it is meant to explain.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qubo import scoring
from src.qubo.portfolio import PortfolioQUBO
from src.solvers.braket_solver import BraketSolver


def _problem(n: int = 6, k: int = 2, penalty=None):
    rng = np.random.default_rng(3)
    mu = rng.normal(size=n)
    a = rng.normal(size=(n, n))
    cov = a @ a.T / n
    return PortfolioQUBO().formulate(mu, cov, num_select=k,
                                     penalty_strength=penalty), mu, cov


class TestRecording:
    def test_angles_are_recorded_on_the_default_path(self):
        problem, _, _ = _problem()
        res = BraketSolver("lightning_cpu", p_layers=2, n_shots=50,
                           n_optimizer_steps=5).solve(problem, seed=1)
        md = res.metadata
        assert md["angles_trajectory"].shape == (5, 4)
        assert np.array_equal(md["initial_angles"], md["angles_trajectory"][0])
        assert md["final_angles"].shape == (4,)
        assert not np.array_equal(md["final_angles"], md["initial_angles"])
        assert md["last_samples"].shape == (50, 6)
        assert md["optimisation_seconds"] > 0 and md["sampling_seconds"] > 0
        assert "final_probabilities" not in md

    def test_record_state_leaves_the_bitstring_and_samples_unchanged(self):
        problem, _, _ = _problem()
        kw = dict(p_layers=2, n_shots=200, n_optimizer_steps=8)
        plain = BraketSolver("lightning_cpu", **kw).solve(problem, seed=4)
        rec = BraketSolver("lightning_cpu", record_state=True, **kw).solve(problem, seed=4)
        assert np.array_equal(plain.bitstring, rec.bitstring)
        assert plain.energy_no_offset == rec.energy_no_offset
        assert np.array_equal(plain.metadata["last_samples"], rec.metadata["last_samples"])
        assert np.array_equal(plain.metadata["final_angles"], rec.metadata["final_angles"])

    def test_recorded_probabilities_match_the_shot_frequencies(self):
        """Bit order: wire 0 is the MSB of the probability index."""
        # No penalty, so the state is shaped by the (asymmetric) objective and
        # the per-wire marginals differ; with the penalty on, the distribution
        # is close to permutation-symmetric and cannot tell wires apart.
        problem, _, _ = _problem(penalty=0.0)
        res = BraketSolver("lightning_cpu", p_layers=1, n_shots=40_000,
                           n_optimizer_steps=3, record_state=True).solve(problem, seed=2)
        P = res.metadata["final_probabilities"]
        assert P.shape == (64,)
        assert P.sum() == pytest.approx(1.0, abs=1e-9)
        idx, cnt = scoring.shot_counts(res.metadata["last_samples"])
        freq = np.zeros(64)
        freq[idx] = cnt / 40_000
        # 40k shots over 64 states: binomial sd per state <= 0.0025.
        assert np.max(np.abs(freq - P)) < 0.01
        assert np.corrcoef(freq, P)[0, 1] > 0.99
        # Per-wire marginals discriminate the bit order even when the joint
        # distribution is nearly permutation-symmetric (the penalty dominates
        # a random instance): the sampled column i must match the mass of the
        # states whose bit (n-1-i) is set, and the wires must differ.
        X = scoring.bitstrings_from_indices(np.arange(64), 6)
        marg_probs = X.T @ P
        marg_shots = res.metadata["last_samples"].mean(axis=0)
        assert np.max(np.abs(marg_probs - marg_shots)) < 0.01
        assert np.ptp(marg_probs) > 0.02
        assert np.max(np.abs(marg_probs[::-1] - marg_shots)) > 0.01

    def test_initial_state_and_final_expectation(self):
        problem, _, _ = _problem()
        res = BraketSolver("lightning_cpu", p_layers=2, n_shots=10,
                           n_optimizer_steps=6, record_state=True).solve(problem, seed=5)
        md = res.metadata
        assert md["initial_probabilities"].shape == (64,)
        assert md["initial_probabilities"].sum() == pytest.approx(1.0, abs=1e-9)
        assert md["best_expectation"] == min(md["convergence"])
        assert md["convergence"][md["best_expectation_step"]] == md["best_expectation"]
        assert np.array_equal(md["best_expectation_angles"],
                              md["angles_trajectory"][md["best_expectation_step"]])
        # The final angles are one step past the last recorded expectation.
        assert isinstance(md["expectation_at_final_angles"], float)

    def test_record_state_is_refused_off_the_local_simulators(self):
        with pytest.raises(ValueError, match="local simulator"):
            BraketSolver("braket_sv1", s3_bucket="b", record_state=True)


class TestOptimizerOptions:
    def test_default_is_adam_at_the_published_step(self):
        problem, _, _ = _problem()
        res = BraketSolver("lightning_cpu", p_layers=1, n_shots=10,
                           n_optimizer_steps=4).solve(problem, seed=1)
        md = res.metadata
        assert md["optimizer"] == "adam" and md["stepsize"] == 0.1
        assert md["objective_evaluations"] == 4 and md["gradient_evaluations"] == 4

    def test_a_smaller_step_moves_less(self):
        problem, _, _ = _problem()
        big = BraketSolver("lightning_cpu", p_layers=1, n_shots=10,
                           n_optimizer_steps=3).solve(problem, seed=1).metadata
        small = BraketSolver("lightning_cpu", p_layers=1, n_shots=10,
                             n_optimizer_steps=3, stepsize=0.03).solve(problem, seed=1).metadata
        assert np.array_equal(big["initial_angles"], small["initial_angles"])
        d_big = np.abs(big["final_angles"] - big["initial_angles"]).sum()
        d_small = np.abs(small["final_angles"] - small["initial_angles"]).sum()
        assert d_small < d_big

    def test_lbfgs_respects_the_gradient_budget_and_records_evaluations(self):
        problem, _, _ = _problem()
        res = BraketSolver("lightning_cpu", p_layers=2, n_shots=10,
                           n_optimizer_steps=6, optimizer="lbfgs",
                           record_state=True).solve(problem, seed=3)
        md = res.metadata
        assert md["optimizer"] == "lbfgs" and md["stepsize"] is None
        assert md["gradient_evaluations"] <= 6
        assert md["objective_evaluations"] >= md["gradient_evaluations"]
        assert len(md["convergence"]) == md["objective_evaluations"]
        assert md["angles_trajectory"].shape == (md["objective_evaluations"], 4)
        # The first evaluation is at the same initial angles ADAM would use.
        adam = BraketSolver("lightning_cpu", p_layers=2, n_shots=10,
                            n_optimizer_steps=1).solve(problem, seed=3).metadata
        assert np.allclose(md["initial_angles"], adam["initial_angles"])
        # It descends: the best evaluation is strictly below the first, and
        # the angles moved (a zero gradient would leave both untouched).
        assert md["gradient_evaluations"] >= 2
        assert md["best_expectation"] < md["convergence"][0]
        assert not np.allclose(md["final_angles"], md["initial_angles"])
        assert md["final_probabilities"].shape == (64,)

    def test_unknown_optimizer_is_refused(self):
        with pytest.raises(ValueError, match="optimizer"):
            BraketSolver("lightning_cpu", optimizer="sgd")


class TestSamplingAngles:
    """The batch must be drawn at the angles the optimiser ended on.

    The ADAM loop rebinds the parameter variable to the array each step
    returns; the L-BFGS-B wrapper receives a copy and returns its result in a
    separate variable. Both paths must hand the *final* angles to the sampler,
    and the test inspects the argument the sampler receives, not the shots.
    Until 2026-09-13 the sampling call still read the original variable, so
    the L-BFGS-B path handed over the initial angles (the E3 L-BFGS-B batches'
    feasible fraction tracked ``initial_P_F`` with r = 0.99 and ``final_P_F``
    with r = -0.30) and every batch-derived column of that block described an
    unoptimised state.
    """

    @pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
    def test_the_sampler_gets_the_final_angles(self, optimizer, monkeypatch):
        problem, _, _ = _problem()
        seen = {}
        original = BraketSolver._sample_best

        def spy(self, dev, params, *args, **kwargs):
            seen["angles"] = np.asarray(params, dtype=float).copy()
            return original(self, dev, params, *args, **kwargs)

        monkeypatch.setattr(BraketSolver, "_sample_best", spy)
        md = BraketSolver("lightning_cpu", p_layers=2, n_shots=10,
                          n_optimizer_steps=6, optimizer=optimizer,
                          record_state=True).solve(problem, seed=3).metadata
        assert np.array_equal(seen["angles"], md["final_angles"])
        assert not np.allclose(seen["angles"], md["initial_angles"])

    def test_lbfgs_batch_frequencies_follow_the_recorded_final_state(self):
        problem, _, _ = _problem(penalty=0.0)
        md = BraketSolver("lightning_cpu", p_layers=1, n_shots=40_000,
                          n_optimizer_steps=6, optimizer="lbfgs",
                          record_state=True).solve(problem, seed=2).metadata
        idx, cnt = scoring.shot_counts(md["last_samples"])
        emp = np.zeros(64)
        emp[idx] = cnt / cnt.sum()
        tv_final = 0.5 * np.abs(emp - md["final_probabilities"]).sum()
        tv_initial = 0.5 * np.abs(emp - md["initial_probabilities"]).sum()
        assert tv_final < 0.02
        assert tv_initial > 5 * tv_final
