# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Independent evaluator for the penalty-encoded portfolio QUBO (plan E0).

Every number the paper has reported so far went through ``x @ Q @ x`` on the
matrix that ``PortfolioQUBO.formulate`` builds. That is one code path, and the
2026-09-11 revision plan (§4.1) asks for a second one that never touches ``Q``:
the objective, the constraint violation and the penalised energy are computed
straight from the normalised inputs, so an error in the matrix assembly, the
offset bookkeeping or the bit order would show up as a disagreement between the
two paths rather than be scored consistently by both.

Notation follows plan §3.1 and is kept apart on purpose:

    f(x)    = λ xᵀΣ̂x − (1−λ) μ̂ᵀx          the portfolio objective
    V(x)    = (Σᵢ xᵢ − K)²                   the cardinality violation
    H_A(x)  = f(x) + A·V(x)                  the penalised objective
    E_A(x)  = H_A(x) − A·K²                  what ``x @ Q @ x`` returns

On the feasible set F = {x : Σx = K} the penalty vanishes, so H_A = f and
E_A = f − A·K². The −A·K² is a constant that the expanded QUBO drops into
``offset``; it is *not* a penalty on feasible solutions.

Basis-state indices use PennyLane's convention throughout: wire 0 is the most
significant bit, so ``index = Σᵢ xᵢ · 2^(n−1−i)``. ``qml.probs(wires=range(n))``
and ``qml.sample(qml.PauliZ(i))`` (with z = +1 ↦ x = 0) both agree with it.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from math import comb

import numpy as np

#: Largest N ``enumerate_all`` will attempt (2^22 states, ~1.4 GB of f values
#: in chunks). The plan's cross experiment stops at N=20.
MAX_ENUMERATED_N = 22

#: Chunk size for the all-states sweep: 2^16 rows of N floats at a time.
_CHUNK = 1 << 16


# ----------------------------------------------------------------------
# Objective
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Objective:
    """The portfolio objective f, built from the normalised inputs directly.

    Args:
        mu_hat: Normalised expected returns μ̂ = μ / mu_scale, shape (N,).
        cov_hat: Normalised covariance Σ̂ = Σ / cov_scale, shape (N, N).
        risk_aversion: λ ∈ [0, 1].
        num_select: Cardinality K.
    """

    mu_hat: np.ndarray
    cov_hat: np.ndarray
    risk_aversion: float
    num_select: int

    @classmethod
    def from_problem(cls, problem, returns: np.ndarray, covariance: np.ndarray) -> "Objective":
        """Rebuild the objective from a formulated problem's recorded scales.

        Only the scalars in ``problem.metadata`` are used; ``problem.Q`` is not
        read, which is the whole point of this path.
        """
        md = problem.metadata
        return cls(
            mu_hat=np.asarray(returns, dtype=float) / float(md["mu_scale"]),
            cov_hat=np.asarray(covariance, dtype=float) / float(md["cov_scale"]),
            risk_aversion=float(md["risk_aversion"]),
            num_select=int(md["num_select"]),
        )

    @property
    def n(self) -> int:
        return int(self.mu_hat.shape[0])

    def objective(self, X: np.ndarray) -> np.ndarray:
        """f(x) for one bitstring or a batch, shape (m, N) → (m,)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        lam = self.risk_aversion
        quad = np.einsum("ij,ij->i", X @ self.cov_hat, X)
        return lam * quad - (1.0 - lam) * (X @ self.mu_hat)

    def violation(self, X: np.ndarray) -> np.ndarray:
        """V(x) = (Σx − K)² for a batch."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return (X.sum(axis=1) - self.num_select) ** 2

    def penalised(self, X: np.ndarray, penalty: float) -> np.ndarray:
        """H_A(x) = f(x) + A·V(x)."""
        return self.objective(X) + penalty * self.violation(X)

    def offset_energy(self, X: np.ndarray, penalty: float) -> np.ndarray:
        """E_A(x) = H_A(x) − A·K², the quantity ``x @ Q @ x`` reports."""
        return self.penalised(X, penalty) - penalty * self.num_select ** 2


# ----------------------------------------------------------------------
# Bit order
# ----------------------------------------------------------------------


def bitstrings_from_indices(indices: np.ndarray, n: int) -> np.ndarray:
    """Basis-state indices → (m, N) float bitstrings, wire 0 as the MSB."""
    idx = np.asarray(indices, dtype=np.int64).reshape(-1, 1)
    shifts = (n - 1 - np.arange(n, dtype=np.int64)).reshape(1, -1)
    return ((idx >> shifts) & 1).astype(float)


def indices_from_bitstrings(X: np.ndarray) -> np.ndarray:
    """(m, N) bitstrings → basis-state indices, wire 0 as the MSB."""
    X = np.atleast_2d(np.asarray(X))
    n = X.shape[1]
    weights = (1 << (n - 1 - np.arange(n, dtype=np.int64)))
    return (np.rint(X).astype(np.int64) @ weights).astype(np.int64)


def cardinality_of_indices(n: int) -> np.ndarray:
    """Popcount of every basis-state index 0 … 2^N − 1, as int8."""
    if n > MAX_ENUMERATED_N:
        raise ValueError(f"N={n} exceeds MAX_ENUMERATED_N={MAX_ENUMERATED_N}")
    card = np.zeros(1, dtype=np.int8)
    for _ in range(n):
        card = np.concatenate([card, card + 1])
    return card


# ----------------------------------------------------------------------
# Enumeration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class AllStates:
    """f over every one of the 2^N assignments, indexed by basis state.

    Attributes:
        f: Objective per state, shape (2^N,).
        cardinality: Σx per state, shape (2^N,), int8.
    """

    f: np.ndarray
    cardinality: np.ndarray

    @property
    def n(self) -> int:
        return int(round(np.log2(self.f.shape[0])))

    def cardinality_minima(self) -> np.ndarray:
        """a_m = min over states with Σx = m, for m = 0 … N (plan §3.4)."""
        n = self.n
        return np.array([self.f[self.cardinality == m].min() for m in range(n + 1)])

    def cardinality_maxima(self) -> np.ndarray:
        """max over states with Σx = m, for m = 0 … N."""
        n = self.n
        return np.array([self.f[self.cardinality == m].max() for m in range(n + 1)])


def enumerate_all(objective: Objective) -> AllStates:
    """Evaluate f on all 2^N assignments, in basis-state order."""
    n = objective.n
    if n > MAX_ENUMERATED_N:
        raise ValueError(f"N={n} exceeds MAX_ENUMERATED_N={MAX_ENUMERATED_N}")
    total = 1 << n
    f = np.empty(total, dtype=float)
    for start in range(0, total, _CHUNK):
        stop = min(start + _CHUNK, total)
        X = bitstrings_from_indices(np.arange(start, stop), n)
        f[start:stop] = objective.objective(X)
    return AllStates(f=f, cardinality=cardinality_of_indices(n))


def max_penalised_energy(states: AllStates, objective: Objective, penalty: float) -> tuple[float, int]:
    """max_x H_A(x) over all 2^N states, and the cardinality where it occurs.

    Plan §3.5 asks that the all-states maximum behind the Abbas-style range
    not be *assumed* to sit at the most violating bitstring; this takes the
    maximum over every state explicitly.
    """
    k = objective.num_select
    m = states.cardinality.astype(float)
    h = states.f + penalty * (m - k) ** 2
    i = int(h.argmax())
    return float(h[i]), int(states.cardinality[i])


@dataclass(frozen=True)
class FeasibleSet:
    """Every K-subset, its objective, and the reference quantities of plan §4.2.

    Attributes:
        indices: Basis-state index of each feasible x, in ``combinations`` order.
        f: Objective per feasible x, same order.
        f_sorted: ``f`` sorted ascending, for rank queries.
    """

    indices: np.ndarray
    f: np.ndarray
    f_sorted: np.ndarray

    @property
    def size(self) -> int:
        return int(self.f.shape[0])

    def position(self, index: int) -> int:
        """Position in ``indices``/``f`` of the feasible state with this basis index.

        Ranks must be computed from the feasible set's *own* value of a
        solution: the same bitstring evaluated in another batch can differ by
        one ulp (2.2e-16 measured at N=20), which is enough to count the
        solution as strictly below itself and shift its rank by one.

        Raises:
            KeyError: If the index is not a feasible state.
        """
        order = self._order()
        pos = int(np.searchsorted(self.indices[order], index))
        if pos >= self.size or self.indices[order[pos]] != index:
            raise KeyError(f"basis index {index} is not in the feasible set")
        return int(order[pos])

    def _order(self) -> np.ndarray:
        cached = getattr(self, "_index_order", None)
        if cached is None:
            cached = np.argsort(self.indices, kind="stable")
            object.__setattr__(self, "_index_order", cached)
        return cached

    @property
    def f_star(self) -> float:
        return float(self.f_sorted[0])

    @property
    def f_max(self) -> float:
        return float(self.f_sorted[-1])

    @property
    def delta(self) -> float:
        """Δ_F = f_F^max − f*, the feasible-range denominator."""
        return self.f_max - self.f_star

    @property
    def f_bar(self) -> float:
        """Mean objective over the feasible set (used by A_margin)."""
        return float(self.f.mean())

    @property
    def optimum_index(self) -> int:
        """Basis-state index of the first minimiser (ties: lowest index)."""
        minimisers = self.indices[self.f == self.f_star]
        return int(minimisers.min())

    def gap_range(self, f_values: np.ndarray | float) -> np.ndarray:
        """g_F = (f − f*) / Δ_F. Undefined (nan) on a degenerate set."""
        f_values = np.asarray(f_values, dtype=float)
        if self.delta <= 0.0:
            return np.full(f_values.shape, np.nan)
        return (f_values - self.f_star) / self.delta

    def rank(self, f_value: float) -> tuple[int, int]:
        """(rank, ties): 1 + #{feasible f strictly below}, and #{equal}."""
        lo = int(np.searchsorted(self.f_sorted, f_value, side="left"))
        hi = int(np.searchsorted(self.f_sorted, f_value, side="right"))
        return lo + 1, hi - lo

    def within(self, tau: float) -> np.ndarray:
        """Mask over the feasible set: g_F ≤ τ."""
        g = self.gap_range(self.f)
        return np.nan_to_num(g, nan=0.0) <= tau

    def is_optimal(self, tol: float) -> np.ndarray:
        """Mask over the feasible set: f − f* ≤ tol·Δ_F (ties included)."""
        return (self.f - self.f_star) <= tol * max(self.delta, 0.0)


def enumerate_feasible(objective: Objective) -> FeasibleSet:
    """Evaluate f on every K-subset of N (C(N, K) states)."""
    n, k = objective.n, objective.num_select
    combos = np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(range(n), k)),
        dtype=np.int64,
        count=comb(n, k) * k,
    ).reshape(-1, k)
    weights = 1 << (n - 1 - combos)
    indices = weights.sum(axis=1).astype(np.int64)
    f = np.empty(indices.shape[0], dtype=float)
    for start in range(0, indices.shape[0], _CHUNK):
        stop = min(start + _CHUNK, indices.shape[0])
        f[start:stop] = objective.objective(bitstrings_from_indices(indices[start:stop], n))
    return FeasibleSet(indices=indices, f=f, f_sorted=np.sort(f, kind="stable"))


# ----------------------------------------------------------------------
# Penalty thresholds (plan §3.4, §5.3)
# ----------------------------------------------------------------------


def critical_penalty(a_m: np.ndarray, k: int) -> tuple[float, int]:
    """A_crit = max(0, max_{m≠K} (f* − a_m)/(m−K)²), and the binding m.

    At A = A_crit an infeasible solution ties the constrained optimum; any
    weight strictly above it makes every infeasible x cost more than x*.
    A_crit = 0 means the constrained optimum already beats every other
    cardinality without help, and no multiple of 0 is a distinct condition.

    Args:
        a_m: Exact per-cardinality minima, length N + 1, from ``AllStates``.
        k: Cardinality K.

    Returns:
        ``(A_crit, binding cardinality)``. When the clamp at 0 is active, the
        binding m is the cardinality whose (negative) ratio was largest.
    """
    f_star = float(a_m[k])
    m = np.arange(a_m.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = (f_star - a_m) / (m - k) ** 2
    ratios[k] = -np.inf
    binding = int(ratios.argmax())
    return max(float(ratios[binding]), 0.0), binding


def margin_penalty(a_m: np.ndarray, k: int, f_bar: float, eps: float) -> tuple[float, float]:
    """A_margin per plan §5.3: separate x* from every infeasible x by a margin.

    T = (f* + f̄_F)/2 is the target level the best infeasible solution must
    lie above, following Brandhofer et al.'s feasible-mean criterion. The
    weight that achieves exactly that is
        max(0, max_{m≠K} (T − a_m)/(m−K)²),
    and ``eps`` is the recorded safety margin added on top.

    Args:
        a_m: Exact per-cardinality minima.
        k: Cardinality K.
        f_bar: Mean objective over the feasible set.
        eps: ε_A added to the threshold (recorded with the run).

    Returns:
        ``(A_margin, T)``.
    """
    f_star = float(a_m[k])
    target = 0.5 * (f_star + f_bar)
    m = np.arange(a_m.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = (target - a_m) / (m - k) ** 2
    ratios[k] = -np.inf
    return max(float(ratios.max()), 0.0) + eps, target


# ----------------------------------------------------------------------
# Gaps and acquisition probabilities (plan §4.2)
# ----------------------------------------------------------------------


def gap_offset(f_value: float, f_star: float, penalty: float, k: int) -> float:
    """g_off = (f − f*) / |f* − A·K²|, the conventional reported gap.

    Returns nan when the denominator is zero.
    """
    denom = abs(f_star - penalty * k * k)
    if denom == 0.0:
        return float("nan")
    return (f_value - f_star) / denom


def deflation_factor(f_star: float, penalty: float, k: int, delta: float) -> float:
    """D(A) = |f* − A·K²| / Δ_F, so that g_F = D(A) · g_off (plan §4.3)."""
    if delta <= 0.0:
        return float("nan")
    return abs(f_star - penalty * k * k) / delta


def success_probability(p_single: float, shots: int) -> float:
    """Q_τ(S) = 1 − (1 − P_τ)^S for i.i.d. shots.

    Evaluated as −expm1(S·log1p(−P)) rather than as the formula reads. Written
    directly, a small P makes (1 − P)^S a number just under one and the
    subtraction cancels the leading digits away: at P = 10⁻⁶ the two agree to
    only eleven significant figures, and the surviving digits are whatever the
    platform's ``pow`` rounded to. That was enough for this analysis to produce
    different CSVs on the two machines it runs on, which is precisely the kind
    of drift this paper reports. expm1 and log1p are accurate where the naive
    form is not, and the identity is exact. What remains after the rewrite is
    the libm difference itself: macOS and glibc round expm1/log1p one ulp
    apart in 26 of the aggregate cells (2026-09-12), none of which reaches a
    reported digit. The committed CSVs are the GPU host's.
    """
    p = min(max(float(p_single), 0.0), 1.0)
    if p >= 1.0:
        return 1.0
    return -math.expm1(shots * math.log1p(-p))


def selected_output_tv_bound(d: float, shots: int) -> float:
    """Upper bound on TV between the *selected-output* distributions of two samplers.

    Two single-shot distributions at TV distance d, each drawn S times i.i.d.
    and passed through the same deterministic selection rule T (best feasible
    shot, or ⊥): couple the two batches shot by shot with a maximal coupling,
    so each pair differs with probability d and the batches differ with
    probability 1 − (1 − d)^S; the data-processing inequality then gives
    TV(L(T(X)), L(T(Y))) ≤ 1 − (1 − d)^S ≤ S·d (Sec. V-D, Supplementary S-VII).

    This bounds how far the *distribution* of the returned portfolio can move
    when the arithmetic changes. It does not bound the probability that two
    independently sampled batches return different portfolios: at S = 1 with
    two identical fair coins, d = 0 and the bound is 0, yet independent draws
    disagree half the time. ``tests/test_e4_bounds.py`` keeps that
    distinction explicit.
    """
    return success_probability(d, shots)


def cardinality_mass(probs: np.ndarray, cardinality: np.ndarray, n: int) -> np.ndarray:
    """Probability mass per cardinality m = 0 … N."""
    return np.bincount(cardinality.astype(np.int64), weights=probs, minlength=n + 1)


def shot_counts(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique basis-state indices and their counts from an (S, N) sample array."""
    idx = indices_from_bitstrings(samples)
    return np.unique(idx, return_counts=True)
