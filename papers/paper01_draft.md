# Measuring Metric and Sampling Distortions in Penalty-Encoded QAOA Portfolio Benchmarks

**Draft v3.2 — 2026-09-13.** Target venue: IEEE Transactions on Quantum Engineering. Every number is
from a committed result file. The Supplementary Material holds the single-instance measurements,
the full tables the main text summarises, and the literature census.

---

## Abstract

Penalty-encoded quantum optimization benchmarks can conflate solution quality with normalization and
finite-shot selection. We quantify both for cardinality-constrained portfolio optimization with the
quantum approximate optimization algorithm, against exact optima from feasible-set enumeration. A
penalty-encoded quadratic unconstrained binary optimization energy carries a constant shift, so a gap
taken on it divides by a formulation choice: re-scoring one fixed portfolio as only the penalty weight
varies moves its offset-normalized gap from 33.6% to 0.22% while a feasible-range gap stays at 37.8%,
and the deflation reaches 121–279× at 16–30 assets. A cross of 30 instances × 5 initial angles × 6
penalty settings at 12, 16 and 20 assets (2,565 runs) moves the reported gap and the sampler in
opposite directions: a margin-based weight worsens the gap in 29 of 30 instances at 20 assets while
raising, in all 30, the probability that 1,000 shots hold a feasible solution within 1% of the feasible
range (+0.086, 95% confidence interval 0.071–0.116); a second price window reproduced both directions.
At the heuristic weight the optimised sampler, conditioned on feasibility, is close to uniform (total
variation ≤ 0.04 in 450 runs) and never exceeds uniform feasible sampling in success probability. At
fixed angles, double-precision implementations agreed on the output distribution to 2.6·10⁻¹⁴ in total
variation, while independent resampling of one distribution at 20 assets returned nine or ten distinct
portfolios in ten batches. Reports should separate objective quality, feasibility, acquisition
probability and execution conditions.

**Keywords:** benchmark testing, cloud computing, constrained optimization, portfolio optimization,
quadratic unconstrained binary optimization (QUBO), quantum approximate optimization algorithm (QAOA),
quantum computing, reproducibility of results.

---
## I. Introduction

Portfolio selection under a cardinality constraint — choose exactly K of N candidate assets to
optimize a risk/return trade-off [1] — is NP-hard [2] and a canonical target for quantum optimization
via its quadratic unconstrained binary optimization (QUBO)/Ising encoding [3], [4], surveyed in [5]–[7]. The encoding used here is
the textbook quadratic penalty [3], the algorithm is the quantum approximate optimization algorithm (QAOA) [4], and the runtime results agree with the closest prior benchmark [8] rather than
competing with it. The question is what the numbers such a benchmark publishes measure.

We study a controlled failure mode in evaluating penalty-encoded QAOA: a quality score taken on the
energies the solver returns can improve while the probability of acquiring a useful feasible
solution decreases. A penalty-encoded QUBO's energies carry the constant −A·K² that the cardinality
term contributes to every feasible assignment, so an optimality gap taken on them divides by a number
the formulation chose. We call that the *offset-normalized gap*, g_off. It is the gap this
implementation first reported, and it is measured here on our own runs; the fourteen published
QAOA and annealing portfolio benchmarks we read (Supplementary Sec. S-I) mostly report no gap at
all, and none of the four that do uses this denominator. Two further conventions are examined with
it: the *best-of-shots* rule, which returns the lowest-energy bitstring of a batch (here the
lowest-energy *feasible* one, §III-C) and says nothing about how often a batch contains a usable
solution; and the run's *execution conditions* — initial
angles, backend, precision, sampler — which are usually fixed at one value and not reported as
factors. That an offset-sensitive score is sensitive to the penalty weight is known [9], [10], [11]. We
provide a crossed quantification of it: how large the effect is on our instances, how it interacts
with the acquisition probability, and under which conditions it holds. Rather than proposing a new
normalization, we measure the discrepancy across instances, initializations and penalty settings,
against exact optima from feasible-set enumeration, scored by an evaluator blind to the QUBO
matrix. Three contributions follow; the second is the central one, the first its starting point.

**C1 — The compression caused by the metric, quantified.** Re-scoring *fixed* feasible solutions as
only the penalty weight A changes (E0) separates what the number does from what the solver does: the
same median-rank portfolio at N = 20 reports a 33.6 % offset-normalized gap at 1.1·A_crit and 0.22 % at
the baseline heuristic weight, while a gap normalised by the objective's spread over the feasible
set stays at 37.8 %. The deflation of the offset-normalized gap at the heuristic weight is
121–279× on the reference instances at N = 16–30 (57–279× over N = 8–30), rises with N, and no
instance of thirty drawn per size escapes it (§V-A).

**C2 — Solution quality separated from acquisition probability.** A cross of 30 instances × 5
initial-angle seeds × 6 penalty settings at N ∈ {12, 16, 20} (E1, 2,565 runs), each run scored both
on its exact final state (feasibility and acquisition probabilities) and on the best feasible sample
of its 1,000-shot batch (gaps), shows the offset-normalized gap and the sampler moving in opposite directions:
the margin weight of [9] worsens the reported gap in 29 of 30 instances at N = 20 while raising the
probability that 1,000 shots hold a feasible solution within 1 % of the feasible range in 30 of 30
(+0.086, 95 % CI [+0.071, +0.116]). Re-using the same states (E2), the optimised sampler's
one-shot success probability at the primary threshold (τ = 1 % of the feasible range) exceeds that
of a uniform draw from the feasible set in 0 of 450 heuristic-weight runs, so at no shot budget;
its distribution conditioned on feasibility is close to uniform on the feasible set (median total
variation 0.002, maximum 0.04). Two optimizer settings (E3) and a second price window (E5) bound
where those conclusions hold (§V-B, §V-C). The value of the cross is the judgement it changes, not
its size: one weight change reads as a loss on the reported gap and a gain on acquisition, and the
conditional decomposition separates that gain into feasible mass and preference within the
feasible set, which dominate at different sizes (§V-B).

**C3 — A reproducible evaluation procedure and its record.** Every run is tied to its inputs (μ and Σ
sliced from one price snapshot pinned by value), formulation constants, initial and final angles, evaluation and gradient
counts, shots, exact-state scores, wall-clock and, on metered backends, task counts and dollars; a
manifest states what was planned, what collapsed, what ran and what failed (Table II). With the
angles held fixed (E4-A), the double-precision implementations — local CPU, local GPU and a managed
cloud simulator — agree on the feasible-outcome probabilities to 2.6·10⁻¹⁴ in total variation and
the single-precision arm to 6.1·10⁻⁷, while ten 1,000-shot batches of one state return nine or
ten distinct selected portfolios at N = 20 on every backend (§V-D).

The critique is of the encoding and the reporting conventions in this configuration, not of QAOA: a
classical annealer run on the same penalty-encoded QUBO reproduces the pattern (§V-A). Runtime is
reported as context for the evaluated implementation — simulated QAOA had no speed advantage at
any size we could run — not as evidence about quantum advantage, and the value of the measurement
does not depend on it.

## II. Related Work

**QUBO portfolio benchmarks.** Mean–variance selection [1] with a cardinality constraint is NP-hard
[2]; the standard route to a quantum device is the QUBO/Ising encoding catalogued by Lucas [3] with
the constraint absorbed into a quadratic penalty. Rosenberg et al. [12] applied it to a multi-period
trading trajectory on a quantum annealer; Venturelli and Kondratyev [13] seed reverse annealing with a
greedy solution, and note that enforcing the cardinality penalty needs a large P that "is typically
associated with precision issues" — a hardware-side argument for keeping the penalty scale out of
the number being scored; Mugel et al. [14] compared annealers, gate-based devices and tensor
networks on 52 assets. The closest benchmark to this paper is Stopfer and Wagner [8], who evaluate
annealing and QAOA against mixed-integer programming and classical heuristics on 250 real-market
instances of up to 1,000 assets, restrict the quantum methods to at most 30 assets, and find that
classical solvers reach optimality within seconds; our runtime results agree (§V-A). Uotila et al.
[15] run 100 DJIA instances against an exact eigensolver and report QAOA meeting both quality and
feasibility criteria on 3 of 100, observing QUBO and HUBO spectra that agree to 10⁻⁹–10⁻¹¹ while
"the allocations start differing substantially". Brandhofer et al. [9] define the approximation
ratio over the feasible set alone, "independent of the particular method used to enforce the
constraint", and reject the dominate-the-objective penalty rule as "unnecessarily large"; §V-A is a
measurement of what those two corrections are worth, not a rediscovery of them. Baker and Radha [10]
report, in a QAOA portfolio case study, run-to-run fluctuation exceeding finite-shot noise and reject
the approximation ratio as "sensitive to … the constraint enforcement scheme chosen"; the constant
offset carrying that sensitivity is written as c = AK² in [11]. How to set the penalty weight is a
studied question: Verma and Lewis [16] give exact weights that make the unconstrained optima
coincide with the feasible ones, Ayodele [17] compares static rules, and Montanez-Barrera et al. [18]
introduce an unbalanced penalty formulation for inequality constraints that avoids additional slack
variables; how a constraint is encoded and whether candidates are re-scored on the original
objective are separate choices.

**What benchmarks report, and what is proposed.** Three documents propose what a benchmark in this
field should report; none is a formal standard. Abbas et al. [19] prescribe a range-normalized
approximation ratio (C_max − ⟨C⟩)/(C_max − C_min) and require mean-sample and best-sample ratios
listed separately, since the best sample "may improve with a higher number of shots". The Quantum
Optimization Benchmark Library [20] mandates run, feasible-run and successful-run counts with a
success threshold, warns that "the objective values of two QUBOs representing the same problem using
different penalty factors cannot be directly compared", and prescribes scoring on the original
problem; its smallest portfolio instance has 710 binary variables, beyond any statevector
simulator, which is why we do not benchmark on it. Lozano [21] proposes a four-metric protocol for
hybrid portfolio benchmarks, the only one to require billed device time, with a gap regularized by
max(1, |f*|) on the original constrained objective. We adopt the reporting fields of [19], [20]
where they apply. A census of fourteen QAOA and annealing portfolio benchmarks — [8]–[15],
[22]–[27], selected by a stated protocol (Supplementary Sec. S-I, Table S1) — is background for the measurements: ten of the fourteen
report no optimality gap, the four that do use four different normalizations, none reports task
counts or billed cost, and none of the seven published after [19], [20] cites either. Not citing a
proposal is what the census counts; it is not evidence that any paper uses the denominator studied
here. Barkoutsos et al. [28] aggregate the shot distribution by conditional value-at-risk because
its mean and minimum are unstable summaries; best-of-shots is the extreme case of that family.

**Initialization, sampling and the simulation stack.** QAOA's weak point is the classical outer loop:
Zhou et al. [29] introduced the INTERP and FOURIER initialization heuristics; Sack and Serbyn [30]
showed random initialization prone to local minima; barren plateaus [31] and the NP-hardness of
training [32] make no initialization universally safe. Guerreschi and Matsuura [33] concluded that
hundreds of qubits would be needed before QAOA on Max-Cut could outrun a classical solver, the
precedent for the negative runtime result reported here. Amazon Braket [34] exposes simulators and
hardware behind one API with per-task and per-shot pricing; Oralkhan and Zhaxalykov [23] use it to
compare algorithms across trapped-ion and superconducting backends. Our simulation backends are
PennyLane [35] with the Lightning CPU/GPU statevector devices, the latter on NVIDIA cuQuantum
[36], with gradients by adjoint differentiation [37]. Reproducibility in computer-systems research
depends on recording the exact inputs [38]–[40]; a cloud quantum experiment adds a billable
environment to that list, which §VI treats as part of the method.

**Position of this work.** The formulation, the algorithm and the runtime conclusion are standard;
what is added is the measurement: a crossed design (instance × seed × penalty) evaluated on both
exact final-state probabilities and best-feasible sample quality, with exact references, explicit sampling controls and a recorded execution environment,
reported because it contradicts what a single-seed, single-instance, single-weight run would have
let us claim.

## III. Formulation and Evaluation Quantities

### A. The penalty encoding

Let x ∈ {0,1}^N be a binary selection vector, μ ∈ ℝ^N the vector of annualized expected returns, and
Σ ∈ ℝ^{N×N} the annualized covariance matrix. With risk-aversion λ ∈ [0,1] and target cardinality K,
the cardinality-constrained mean–variance problem

> minimize  λ·xᵀΣx − (1−λ)·μᵀx   subject to  Σᵢ xᵢ = K

is cast as an unconstrained QUBO via a quadratic penalty. We keep four quantities apart, because
the paper's central measurement is about which of them a reported number divides by:

> f(x) = λ·xᵀΣ̂x − (1−λ)·μ̂ᵀx  (the portfolio objective),

> V(x) = (Σᵢ xᵢ − K)²  (the constraint violation),

> H_A(x) = f(x) + A·V(x)  (the penalised objective the QUBO encodes),

> E_A(x) = H_A(x) − A·K²  (what xᵀQx returns once the expansion's constant is dropped).

Let F be the feasible set, the assignments with Σᵢ xᵢ = K. On F the penalty vanishes, so H_A(x) = f(x) and
E_A(x) = f(x) − A·K²: the −A·K² is a constant the expansion sheds into an offset, not a penalty
paid by feasible solutions. Every energy the results report is E_A, and every gap denominated by
|E_A\*| therefore carries that constant. The QAOA circuit sees a fifth quantity, the Ising form of
H_A rescaled by its largest coefficient (below); we write ⟨H_norm⟩ for its expectation and never
interchange it with H_A or E_A, nor compare it between runs at different A.

μ and Σ are normalized by their max-abs value (μ̂, Σ̂ ∈ [−1,1]) so the penalty A has a
dataset-independent meaning. Two consequences are stated now because §V-A reads a financial metric
off the same x: the portfolio evaluated there is the equal-weight one, w = x/K, and because μ and Σ
are scaled separately, the relative weight of return and risk in f is not the one λ = 0.5 would give
on the raw financial units — f is what is optimized, and the Sharpe ratio is measured afterwards on
the raw series. And because w = x/K, the objective on a feasible x reads f = λK²·wᵀΣ̂w − (1−λ)K·μ̂ᵀw,
so a fixed λ weighs risk against return differently at each K: the sizes are not the same financial
risk preference. We use λ = 0.5 throughout. Expanding the penalty yields an upper-triangular QUBO
matrix Q with

- diagonal:  Q_ii = λ·Σ̂_ii − (1−λ)·μ̂_i + A·(1 − 2K),
- off-diagonal:  Q_ij = 2λ·Σ̂_ij + 2A,

and a constant offset A·K². For QAOA the QUBO is mapped to an Ising Hamiltonian by x_i = (1 − z_i)/2,
giving linear (h) and quadratic (J) coefficients plus a constant. This layer is device-independent
and is unchanged whether the solver is classical or quantum.

*The circuit, stated once.* With s_H = max(max_i |h_i|, max_{i<j} |J_ij|) and the normalised cost
Hamiltonian

> H_C = (1/s_H) · ( Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j ),   H_M = −Σ_i X_i,

the state at depth p is

> |ψ(γ, β)⟩ = ∏_{ℓ=p..1} e^{−i β_ℓ H_M} e^{−i γ_ℓ H_C} · H^{⊗N}|0⟩^{⊗N},

layers applied cost-first, and the objective minimised is ⟨ψ(γ, β)| H_C |ψ(γ, β)⟩ = ⟨H_norm⟩. Terms
with |coefficient| ≤ 10⁻¹² are dropped before the circuit is built. Each exponential is one Trotter
step of the exact commuting decomposition (every term of H_C is diagonal, and H_M's terms commute
among themselves), giving p·[C(N,2) + 2N] parameterised rotations. Angles are drawn once per run as
γ_ℓ, β_ℓ ~ U(0, π) from the run's seed and are unbounded during optimisation. We use p = 2 for
E1–E5 (the depth sweep of Supplementary Sec. S-VI is the exception), so there are four variational
parameters. The rescaling by s_H leaves argmin unchanged
and reported energies use the original Q, but it is not neutral to the search: on the circuit it is
the reparameterisation γ → γ/s_H, so the same initial-angle range explores a different region of
the landscape than the unnormalised problem would (§VI-C).

### B. Penalty weights

Three reference weights recur. The heuristic the formulation uses, and the one we call A_heur, is
the "make the penalty dominate the objective" rule

> A_heur = 2·s_obj·N + 1,  s_obj = max(λ·max|Σ̂|, (1−λ)·max|μ̂|);

because the normalization makes max|Σ̂| = max|μ̂| = 1, at λ = 0.5 this is A_heur = N + 1 for every
instance in this paper. It is the baseline penalty of this implementation. The *idea* of making the
penalty dominate the objective is common; this particular formula is not a field standard and we do
not treat it as one. The experiments vary it against reference weights computed from the instance
(A_crit, A_margin), not against competing conventions.

A_crit is the smallest A at which the constrained optimum x\* is a global minimizer of H_A: with a_m
the minimum of f over assignments of cardinality m,

> A_crit = max(0, max_{m≠K} (f\* − a_m)/(m − K)²).

If the inner maximum is positive, an infeasible assignment ties x\* at A = A_crit and any larger A
separates them strictly; if it is zero, some infeasible assignment already ties x\* at A = 0; if it
is negative (the clamp is active), x\* is strictly better than every infeasible assignment at A = 0
and no tie occurs at any A ≥ 0. All nine of the thirty N = 12 instances with A_crit = 0 are the
third case, with inner maxima from −9.0·10⁻² to −5.9·10⁻⁴, so for them the multiples of A_crit
collapse to the single condition A = 0. Ties are decided at OPT_TOL = 10⁻⁹ relative to Δ_F, not by
floating-point equality. For N ≤ 20 we compute a_m exactly over all 2^N assignments; for N > 20 we
bound the cardinalities far from K and call the result A_safe, an upper bound on A_crit.

A_margin follows Brandhofer et al. [9] and places the best infeasible assignment at the midpoint
between f\* and the feasible-set mean f̄_F,

> T = (f\* + f̄_F)/2,  A_margin = max(0, max_{m≠K} (T − a_m)/(m − K)²) + ε_A,

with ε_A = 10⁻⁶ recorded per instance. At A_margin every infeasible assignment costs at least
T − f\* more than x\* does, up to ε_A; the best infeasible assignment sits at T (before ε_A is added)
when the inner maximum is nonnegative; when it is negative every infeasible assignment already
exceeds T at A = 0. Both A_crit and A_margin use exact information about the instance; they
are reference conditions, not a procedure we claim scales. Computing them took 0.06 s, 0.8 s and
10.3 s per thirty instances at N = 12, 16 and 20.

### C. Evaluation quantities

All scores are computed by an evaluator that reads x, μ̂, Σ̂, λ, K and A and never reads Q; on
every one of the 2^N states at N = 8 and N = 12 it agrees with the QUBO path on H_A, E_A, the
Ising coefficients and the bit order (E0, §IV-C). With f\* = min_F f, f_F^max = max_F f and the
feasible-range spread Δ_F = f_F^max − f\*, the quantities are those of Table I.

**Table I. The evaluation quantities.** Every score in §V is one of these, computed by the evaluator from
x, μ̂, Σ̂, λ, K and A.

| Quantity | Definition | Role |
|---|---|---|
| Offset-normalized gap g_off | (f(x) − f\*) / \|f\* − A·K²\|, feasible x | the gap taken on the solver's own energies; analysed, not used for judgement |
| Feasible-range gap g_F | (f(x) − f\*) / Δ_F | primary quality of a feasible solution; free of the constant and of A |
| Feasible probability P_F | Pr[x ∈ F] under the sampler's exact distribution | the chance one shot is usable |
| Quality-hit probability P_τ | Pr[x ∈ F and g_F(x) ≤ τ] | the unconditional one-shot success probability |
| S-shot success Q_τ(S) | 1 − (1 − P_τ)^S for i.i.d. shots | success at a fixed measurement budget |
| Rank | 1 + number of feasible solutions with f strictly below f(x) | position in F; implies neither a small objective difference nor a similar portfolio |
| Deflation D(A) | \|f\* − A·K²\| / Δ_F = g_F / g_off | how many times smaller the offset-normalized number reads |
| Conditional distance D_cond | ½ Σ_{x∈F} \| p(x)/P_F − 1/\|F\| \| | how far the sampler, conditioned on feasibility, is from uniform on F |
| Domain metric | return, volatility and Sharpe ratio of w = x/K on the raw series | auxiliary; not what the QUBO optimizes |

The primary threshold is τ = 0.01 of the feasible range; the exact optimum and τ = 0.05 are
sensitivity settings. τ = 1 % of Δ_F is not the 0.1 % of |E\*| that the earlier tables used as a
success line; the two are compared as two decision rules in §V-B, not treated as equivalent.
Two denominators can vanish. Δ_F = 0 makes g_F undefined by definition (the within-τ test then
counts the whole feasible set as qualifying). |f\* − A·K²| = 0 makes g_off undefined; D(A) is then
0 as a formula, but the evaluator guards both and returns them as undefined together, a choice of
implementation rather than of definition — none of the settings in Table II reaches it. P_F = 0
makes D_cond undefined. The identity D(A) = g_F / g_off holds for non-optimal x; at the optimum both
gaps are 0. Runs in which any of these occurs are counted rather than dropped.

Under i.i.d. shots Q_τ(S) is a formula in P_τ, so shot budgets are compared without re-running
QAOA; on hardware, whose shots may be correlated, it is a model. Two consequences follow. It is
monotone in P_τ at every S, so one sampler exceeding another at several budgets is one comparison
reported at several operating points, decided on P_τ. And Q_τ(S) is the probability that a batch
contains at least one qualifying feasible solution; under the output rule used here it is also the
probability of *returning* one. The rule selects the lowest-E_A shot among the feasible shots
(ties to the lowest basis index), a batch with no feasible shot is a counted outcome rather than a
missing value, and g_F is not defined for an infeasible x. Because E_A and f induce the same
ordering on F, the returned shot has g_F ≤ τ exactly when the batch holds a feasible shot with
g_F ≤ τ; an infeasible shot of lower E_A cannot displace it, so the equivalence survives the weak
penalty weights at which the batch minimum is infeasible. It need not hold for a rule that selects
the minimum-energy shot without first checking feasibility (`tests/test_run_e1_cross.py` pins each
case). Within one instance at one A, g_off and g_F are positive multiples of each other and
preserve the order of solutions; what D(A) changes is every comparison across A, across N and
against a threshold.

Two identities relate the conditional quantities. With u_τ = |F_τ| / |F| the share of the feasible
set within τ and q_τ = P_τ / P_F the sampler's conditional hit rate, |q_τ − u_τ| ≤ D_cond, and the
sampler's one-shot success relative to uniform feasible sampling factors as
P_τ / P_τ^{uniform-F} = P_F · R_τ with R_τ = q_τ / u_τ: the first factor is the chance of landing in
F, the second the preference for good solutions within it. D_cond has no direction — it does not
say whether mass moved toward or away from F_τ — and a small D_cond does not guarantee that R_τ is
close to one when u_τ is small (|R_τ − 1| ≤ D_cond / u_τ), so comparisons against uniform-F are made on P_τ directly and D_cond and R_τ are read as
their explanation.

## IV. Experimental Protocol

### A. Data and problem instances

Nikkei 225 constituents via `yfinance`, a three-year window pinned in code to 2023-08-06 –
2026-08-05 (731 trading days), annualized returns and covariance. For each N the target cardinality
is K = max(2, ⌊N/5⌋). The universe is a fixed list of fifty Nikkei 225 constituents chosen by the author for
liquidity and availability on `yfinance`, not by a numerical ranking rule, written into the source
code in the order listed there before any experiment was run; every one of the
fifty has a complete daily series in both windows (731 and 735 trading days), so no ticker was
excluded for missing data. Instances are subsets of that universe drawn from a single cached price
snapshot: instance 0 is the *reference instance*, the subset the single-instance tables use (the
first N tickers in the listed order); instance i ≥ 1 is an N-subset drawn without replacement from
the fifty by a generator seeded with i — one stream per index, reused at every size, so instances
with the same index at different N are not independent draws (they share 4–7 tickers where
independent draws would share 3.8–6.4 in expectation) while instances with different indices are
independent draws of subsets from the fixed snapshot, not independent market observations; there
is no exclusion between instances and no de-duplication; every instance's sorted ticker list and a
digest of it are recorded with its results (`paper01_e1_reference.csv`). All
E0–E4 instances share one market period and one asset class, so the spread across them measures
subset sensitivity, not robustness across market periods; Supplementary Table S19 repeats the main
contrast without instance 0. E5 repeats a pre-fixed block on the
adjacent, non-overlapping window 2020-08-05 – 2023-08-05 (735 trading days) from a second cached
snapshot of the same fifty tickers.

### B. Solvers and backends

- **Classical reference.** Exact enumeration of the *feasible* set for N ≤ 30 (C(N, K) candidates —
  5.9·10⁵ instead of 1.07·10⁹ at N = 30 — a proven optimum in 5.7 s); the reference solver uses
  simulated annealing [41] only at N = 50 (the annealing arms of §V-A are controls, not the
  reference). Every gap reported for N ≤ 30 is measured against a proven optimum.
- **QAOA baseline.** Depth p = 2, ADAM [42] with step 0.1 for 50 steps from uniform random initial
  angles in [0, π) drawn from a named seed, gradients by adjoint differentiation [37], 1,000 final
  measurement shots, on `lightning.gpu` at single precision. It is the *baseline* of E3 and of the
  single-seed measurements in the Supplementary Material, not a claim about the best QAOA.
- **Simulation host.** RTX A6000 (48 GB VRAM, driver 570.133), Python 3.11, PennyLane 0.45.1 [35],
  pennylane-lightning-gpu 0.45, custatevec-cu12 1.14 [36]. Every number in §V comes from this host
  unless a table says otherwise; numbers from a second machine are never mixed into a table (the same
  deterministic classical solver's Sharpe ratio differs by 3.4·10⁻¹⁶ between the Linux host and an
  Apple-silicon laptop, a BLAS difference).
- **Cloud and hardware backends.** The identical code, selected only by a backend name, on AWS Braket
  SV1 (managed statevector simulator, double precision, us-east-1) and, for the final sampling task
  only, on IonQ Forte-1 (the optimisation loop never runs on hardware).

### C. The experiments

Four factors are varied, and they answer different questions. An *instance* is a ticker subset, so
the spread over instances measures problem dependence within one universe and one period; an
*initial-angle seed* repeats the same problem from a different initialisation; a *penalty setting*
changes the weight A of the same problem, which moves both the reported metric and the
distribution the sampler draws from; the *second price window* (E5) re-runs the same subsets on a
different input period. Thirty instances × five seeds are thus thirty problems each run five
times, not 150 problems, and are aggregated as such (§IV-E).

**E0 — evaluator and re-scoring.** The evaluator of §III-C was checked against the QUBO path on
every one of the 2^N states at N = 8 and N = 12 (H_A = E_A + A·K² on all states, H_A = f on F, the
same per-cardinality minima, ties and worst feasible value, the same bit order; the exact A_crit
never above the bounded A_safe). Then, without re-running QAOA, fixed feasible solutions of each
reference instance — the optimum, the solutions of rank 2, 5, 10, 50, 100 and 1,000, the
median-rank solution, the worst feasible solution and the single-seed QAOA solution of
Supplementary Table S2 — were scored
under g_off, g_F and the all-states approximation ratio of [19] as only A takes the values 1.1, 2,
5 and 10 × A_crit, A_margin and A_heur (354 rows).

**E1 — instance × seed × penalty cross.** N ∈ {12, 16, 20}; 30 instances per size; initial-angle
seeds 42–46; six penalty settings {1.1, 2, 5, 10} × A_crit, A_margin, A_heur; the baseline solver;
1,000 shots. For every run the initial and final angles, the optimisation trajectory, the exact
probabilities of every feasible bitstring under the initial and the final state, the per-cardinality
mass and the shot counts are retained (§IV-D). The nominal design is 2,700 runs; the nine N = 12
instances with A_crit = 0 collapse their four multiples into one condition A = 0, so 2,565 were
expected and 2,565 completed (Table II). Instance 0 at A_heur reproduces an earlier five-seed sweep
and the seed-42 A_heur arm an earlier 90-run instance sweep, to the bitstring (energies within
4·10⁻¹⁶ relative, the ulp of the two evaluators), so those sweeps are not reported separately.

**E2 — what the best shot means.** From the retained states of every E1 run, four samplers are
compared at S ∈ {10, 100, 1000, 10000}: the optimised QAOA state; the *unoptimised* state at the
same initial angles; the uniform distribution over all 2^N bitstrings; and the uniform distribution
over F. P_F, P_τ and D_cond are read off the exact state and Q_τ(S) follows by the formula. The
uniform-over-F sampler draws K distinct assets uniformly without optimising the objective — a
control that uses the constraint directly and is matched to QAOA on the number of candidates
generated, not on computational cost; exact enumeration is used to score its success probability,
not to define it. No optimisation is re-run.

**E3 — optimizer controls.** A block fixed before any E3 result was seen — instances 0–9, seeds
42–46, {A_heur, A_margin}, at each N; 300 runs — was re-run under two other settings: **A**, ADAM
with step 0.03 for 100 steps; **B**, SciPy's L-BFGS-B [43], [44] on the analytic expectation with
adjoint gradients, stopped at SciPy's own criteria or at 100 gradient evaluations, whichever comes
first. Each run pairs with its E1 baseline on (N, instance, A, seed) and the same initial angles, so
⟨H_norm⟩ is comparable within a pair. Neither setting is proposed as a better QAOA; the block asks
whether the baseline's results are an artefact of one weak setting.

**E4-A — the same angles on four backends.** Twelve circuits fixed before any E4-A result was seen —
N ∈ {12, 16, 20} × instances {0, 1} × seeds {42, 43} at A_heur, the final angles of the stored E1
runs — are evaluated without optimisation on `lightning.gpu` at double precision (the reference),
`lightning.gpu` at single precision (the precision of every E1 run), `lightning.qubit` at double
precision on one thread, and SV1 at double precision. Each arm returns the exact probabilities of
the output distribution (the full 2^N vector locally; on SV1, which refuses the `Probability` result
type at zero shots, the feasible-set amplitudes from one analytic task) and ten independent
batches of 1,000 shots, each a separate circuit execution and, on SV1, a separate task. Locally the
sampler is re-seeded per batch through the device's own generator (§VI-C); SV1's sampler cannot be
seeded. The design answers two questions — whether the output distribution at fixed angles is the
same, and what repeated sampling from it returns — and not whether an optimisation run on SV1
reaches the same angles.

**E5 — the E1 block on a second price window.** The E3 block (300 runs) was re-run with the
baseline solver on the window that immediately precedes the paper's, from a second cached snapshot
of the same fifty tickers under the same instance-drawing rule; the feasible set, A_crit, A_margin
and the uniform-sampler baselines are recomputed for the new window, which was fixed before any E5
result was seen. The question is whether the direction of the main contrast, the deflation, the
disagreement of the two success rules and the E2 conclusion depend on the market period the
instances were drawn from. It is a check under a second input condition, not a backtest.

**Earlier single-seed series.** The single-instance benchmark at N = 8–30, the depth sweep and the
steps × p × seed grid, the SV1 cross-backend, matched-precision and repeat runs, the four IonQ
Forte-1 tasks and the classical annealing arms are single-seed or single-instance measurements,
read as such; their tables are in the Supplementary Material and the main text keeps a paragraph on
each.

### D. Reproducibility of the inputs, and the record

All conditions are in code, seeds are named and the window is pinned. Four prerequisites were learned by losing results to them. (i) Prices are pinned
by *value*, not by date: `yfinance` revises adjusted closes between calls, so the raw series are
cached at full precision and read back with round-trip parsing (two fifty-ticker snapshots,
≈1.2 MB). The snapshots are third-party data and are not redistributed; what the released package
carries is the only thing the pipeline reads from them, the annualised mean-return vector and
covariance matrix of each fifty-ticker window (μ, Σ; ≈54 kB per window), and every instance's μ
and Σ are a slice of those, bit for bit, so every number reproduces from a clone with no network
access. (ii) Every instance slices one snapshot, so no two sizes see different revisions of a
price. (iii) Statistics are computed on a C-contiguous float64 copy, because `DataFrame.cov()`
sums in memory-layout order.
(iv) `lightning.qubit` is run at `OMP_NUM_THREADS=1`, without which it drifts by ~10⁻¹³ between
runs; `lightning.gpu` is bitwise reproducible regardless. Figures are regenerated from the committed
CSVs with timestamps suppressed. The failures behind each rule — including the 10⁻¹⁰-level input
changes that single-precision QAOA amplified into different solutions — are in Supplementary
Sec. S-IX.

Every run is recorded as one CSV row (initial and final angles, evaluation counts, shot counts,
exact-state scores, wall-clock and, on metered backends, task identifiers) and every instance as
one row with its reference solution, A_crit, A_margin and the uniform-sampler baselines. What E1
retains per run is not the full state vector — 2,565 dense initial and final vectors at these
sizes would occupy about 16 GB — but the exact probability of every feasible bitstring under the
initial and the final state, the per-cardinality mass, the angles, the trajectory and the shot
counts (≈100 MB compressed), from which every E2 quantity, including D_cond, is recomputed; the
full vector for any run is reconstructed from its recorded angles and seed in seconds.

### E. Statistics

The unit of generalisation is the instance; initial-angle seeds are repeats within an instance;
shots and offline draws are neither. A paired contrast (treatment − control on the same instance,
seed and, for E3, initial angles) is first reduced to one value per instance by the median over
seeds, and the instance-level values are then bootstrapped by resampling instances (2,000 draws,
percentile 95 % interval), with the number of instances on which the treatment was better, tied or
worse. Two conditions on what the intervals estimate are stated rather than left to be inferred.
Because the thirty subsets share one market period, every interval is conditional on that universe,
window and drawing rule. And because every instance uses the same five seeds, the instance-level
value is a contrast *summarised by the median over that fixed set of initial angles*: resampling
instances propagates the uncertainty in which subsets were drawn, not in which angles were. Supplementary
Table S19 shows what that costs — at each of the two larger sizes one of the five seeds reverses the
sign of the main contrast on its own, while dropping any one seed leaves it unchanged — so a claim
about initialisations drawn from a wider distribution would need a design that samples them. The
primary contrast is A_margin against A_heur; the primary acquisition quantity is Q_0.01(1000); P_F,
g_F of the best feasible shot, the rate of exact optima and wall-clock are reported beside it, and
5 × A_crit against A_heur is a secondary contrast on the instances with A_crit > 0. All other cuts
are exploratory. Failed runs, runs without a feasible shot, and runs whose lowest-energy shot is
infeasible are counted, not dropped; a contrast on g_F is formed only where both sides have a
feasible shot, and the number of such pairs is printed next to the estimate. The design was not
sized from a minimum detectable effect: the instance count was fixed by the compute budget of the
cross rather than by a power calculation — thirty subsets per size with five seeds and six penalty
settings is 2,565 runs, 13.9 h on one GPU — and the resolution it affords is shown with every
contrast as "better in k of 30"; five seeds show that a within-instance tail exists, not its
probability.

### F. What ran

**Table II. What ran, and what it cost.** E1 rows: nominal is the full cross; "expected" applies the
A_crit = 0 collapse (nine N = 12 instances run three conditions rather than six); "failed" counts
execution failures (a run that raised or wrote no result), not optimizer stopping reasons — an
L-BFGS-B line-search failure is a recorded termination, not a failed run (Supplementary Sec. S-VI).
"No feasible shot"
counts completed runs whose 1,000 shots held no feasible bitstring; "min-energy shot infeasible"
counts runs whose lowest-energy shot violated the constraint although a feasible shot may exist.
Time is wall-clock on the host of §IV-B (E1 includes 0.06, 0.82 and 10.3 s of reference
enumeration per size); metered charges are AWS list price as billed, and a "—" in that column is an
unmetered local run, not a zero cost. The earlier single-seed series are listed in Supplementary
Table S33.

| Experiment | Runs nominal / expected / done / failed | No feasible shot | Min-energy shot infeasible | Environment (precision) | Time | Metered tasks | Metered charge |
|---|---:|---:|---:|---|---:|---:|---:|
| E1, N = 12 (K = 2, \|F\| = 66; 9 of 30 instances with A_crit = 0) | 900 / 765 / 765 / 0 | 0 | 8 | lightning.gpu (single) | 2.20 h | — | — |
| E1, N = 16 (K = 3, \|F\| = 560) | 900 / 900 / 900 / 0 | 56 | 124 | lightning.gpu (single) | 4.23 h | — | — |
| E1, N = 20 (K = 4, \|F\| = 4,845) | 900 / 900 / 900 / 0 | 1 | 95 | lightning.gpu (single) | 7.43 h | — | — |
| E0 re-scoring (354 rows) and E2 (from E1 states) | 0 new runs | — | — | CPU (double) | minutes | — | — |
| E3, ADAM 0.03 × 100 | 300 / 300 / 300 / 0 | 7 | 8 | lightning.gpu (single) | 3.2 h | — | — |
| E3, L-BFGS-B ≤ 100 gradients | 300 / 300 / 300 / 0 | 10 | 12 | lightning.gpu (single) | 2.3 h | — | — |
| E5, second window | 300 / 300 / 300 / 0 | 11 | 13 | lightning.gpu (single) | 1.6 h | — | — |
| E4-A, local arms (3 arms × 12 circuits × 11) | 396 / 396 / 396 / 0 | 46 batches | — | lightning.gpu / .qubit (double, single) | 3 min | — | — |
| E4-A, SV1 (12 circuits × 11 + 2 pilot) | 134 / 134 / 134 / 0 | 14 batches | — | Braket SV1 (double) | 21 min | 134 | $0.50 |
| IonQ Forte-1, 100 shots per task | 4 / 4 / 4 / 0 | 0 | — | Forte-1 | — | 4 | $33.20 |
| Earlier SV1 series (Supplementary Sec. S-VII) | 30 | — | — | Braket SV1 (double) | — | ≈1,300 | ≈$4.2 |
| Deployment and smoke test | — | — | — | SV1 | — | 1 | $0.004 |

The Braket total is $37.90 against a $50 budget.

## V. Results

The sections follow the question rather than the order the experiments were run: what the metric
measures (§V-A), what the sampler delivers and how the penalty weight moves the two apart (§V-B),
whether the optimizer setting is responsible (§V-C), and what the backend and the hardware add
(§V-D). §V-A is re-scoring and enumeration — no QAOA is re-run for it; §V-B and §V-C are the cross
and its controls; §V-D is fixed-angle evidence from four backends and single-seed evidence from
metered ones, read as such.

### A. The metric: what a reported gap denominates

**Scale and reference.** The single-instance, single-seed benchmark (N = 8–30, seed 42; Supplementary
Table S2) locates the simulator envelope: classical runtime is flat in the seconds range while QAOA
grows from 30 s at N = 20 to 3.14 h at N = 30, N = 30 at single precision uses 41.3 of 49 GB of VRAM,
double precision does not fit, and N = 50 (2^50 × 8 B ≈ 9 PB for a dense statevector) is beyond
any memory this study could reach. Its reported gaps are 0.000 % at N ≤ 16 and 0.024–0.053 % at N = 20–30, every row
feasible. The rest of this section measures what those numbers denominate.

**The same solutions, three denominators (E0; Fig. 1).** Fig. 1 (left) takes one fixed feasible
portfolio of the reference N = 20 instance — the one of median rank, 2,423 of 4,845 — and scores it
as only A changes. Its feasible-range gap g_F is 37.8 % at every A, because neither f(x) nor Δ_F
depends on A. Its offset-normalized gap falls from 33.6 % at 1.1·A_crit to 3.9 % at 10·A_crit and to
0.22 % at A_heur, and the all-states approximation gap 1 − AR of [19] falls from 1.36 % to 0.014 %
over the same range. Nothing about the portfolio changed. Fig. 1 (middle) is the deflation
D(A) = |f\* − A·K²|/Δ_F for every size: below 1 near A_crit at N ≤ 12, linear in A, and 57×, 81×,
121×, 174×, 207×, 240× and 279× at the A_heur of N = 8 to 30 (Supplementary Table S3). The
mechanism is the one §III states: for feasible x the numerator f − f\* is independent of A while the
denominator |f\* − A·K²| is not, so two studies using the same formulation with different penalty
weights report gaps differing by roughly the ratio of their weights for identical portfolios. With
A = N + 1 and K = ⌊N/5⌋ the constant grows as Θ(N³), while the objective's measured spread over F
rises only from 1.27 at N = 16 to 4.00 at N = 30 against A·K² rising from 153 to 1,116: the same
solution quality reads as a smaller number at larger N.

**Over thirty instances (Fig. 1, right; Supplementary Table S9).** The median deflation at A_heur
rises monotonically with N — 68×, 118×, 162×, 184×, 239× and 282× at N = 12 to 30 — and no instance
at any size escapes it: the smallest value observed anywhere is 36× at N = 12 and the largest 411×
at N = 30. The heuristic weight sits a median 103–179× above the threshold at every size — the exact A_crit
at N ≤ 20 (over the 21 N = 12 instances with A_crit > 0) and the upper bound A_safe at N > 20, so
that A_heur/A_safe is a lower bound on the true overshoot there — so the overshoot is a property of the rule
rather than of one covariance matrix; on the reference instance it is 148–1213×, because that
instance's A_crit is unusually small at N ≤ 12 and overshoot is a ratio against a threshold that
can be near zero. The reference instance is near the median in deflation at every size, and below
the median at every size in the Sharpe spread its 0.1 % band contains (next paragraph), so the
single-instance results understate rather than exaggerate the effect.

**Feasible solutions admitted by an offset-normalized threshold.** Exact enumeration makes the set a
reported gap admits countable. On the reference instances a 0.1 % offset-normalized band holds 30
of 560 feasible portfolios at N = 16 (5.4 %), 368 of 4,845 at N = 20 (7.6 %) and 146,484 of 593,775
at N = 30 (24.7 %), and the band is a step function of the threshold with no broad stable plateau over the
evaluated range: at N = 30 it
holds 1.5 % of the feasible set at 0.05 % and 94.7 % at 0.2 % (Supplementary Tables S4–S5). These
solutions are not near-degenerate in any penalty-independent sense: at a deflation of 279× a 0.1 %
offset-normalized band admits solutions up to 28 % of Δ_F from the optimum, and they are admitted
because the threshold was set on a compressed scale. The single-seed QAOA solutions at N = 20–30, reported at
0.024–0.053 %, sit at 5.7–9.2 % of Δ_F and at ranks 43 to 844 in the feasible set; a fixed
universe-wide normalization instead of the per-instance one of §III widens rather than narrows
the band (Supplementary Table S6).

The domain metric is reported as an auxiliary consequence rather than a solver error, since the
Sharpe ratio is not what the QUBO optimizes (§III-A). Similar values of the optimized objective did
not guarantee similar Sharpe ratios: the N = 30 band spans Sharpe ratios from 0.172 to 1.695, the
single-seed N = 24–30 QAOA solutions have Sharpe ratios 15–22 % below the optimum's at gaps of
0.024–0.031 %, and among the 47 sub-optimal solutions of the seed-42 instance sweep at N ≤ 20, 17
have a Sharpe ratio *above* the exact optimum's while the worst deviations reach −61 %
(Supplementary Sec. S-IV, Fig. S7). The deviations are of both signs, and the reported gap does not
carry the sign.

**The prescribed all-states ratio.** Applied literally to this QUBO, the range-normalized ratio of
[19] removes the offset but inherits the scale: C_max over all 2^N assignments is attained by a
violating assignment and grows with A, while the spread over F does not, so every feasible
solution's ratio tends to 1 as A grows. At A_heur the QAOA solution and the *worst* feasible
portfolio score 1.000000 and 0.999560 at N = 16 and 0.999982 and 0.999777 at N = 30
(Supplementary Table S8). [19] calibrates the metric on unconstrained instances and does not say
which range a constrained problem should use; the rank within F, in neither proposal, is what this
paper reports beside the gap.

**The pattern does not need a quantum sampler.** Simulated annealing on the *feasible set*
(cardinality-preserving swaps, five seeds × 30 instances × 6 sizes, 900 runs) reaches the exact
optimum in 895 of 900 runs at a median 3.4 s, so it leaves the metric nothing to hide. The same
annealer on the penalty-encoded Q with single-bit flips, five seeds at N ∈ {12, 16, 20}, reproduces
the QAOA arm's pattern: every run feasible, offset-normalized gaps of 0.00–0.59 %, Sharpe ratios up
to 64 % below the optimum's, ranks as deep as 4,307 of 4,845 (Supplementary Table S11). The
combination of the encoding's offset with an offset-sensitive score is what produces the pattern,
and a classical solver scored the same way inherits it. The two arms are not on matched budgets (50
ADAM steps against 50,000 annealing moves), so this is a check on whether the reporting critique
needs an imperfect solver, not a verdict on which method is better; the seed-42 QAOA arm over the
same thirty instances reaches the exact optimum in 26, 17 and 0 of 30 at N = 12, 16 and 20
(Supplementary Table S12), and the configuration measured here trails the classical baseline on
quality as on runtime.

### B. The sampler: what the penalty weight does to the metric and to the shots (E1, E2, E5)

§V-A fixed the solution and moved A. The cross moves A with the solver running, over thirty
instances and five initial-angle seeds per size, evaluating acquisition probabilities from each
final state and solution quality from its best feasible sample.
Fig. 2 shows the four quantities of §III-C against the penalty setting, and Table III the paired
contrasts of §IV-E.

**Table III. Main effects of the penalty setting.** Paired A_margin − A_heur on the
same instance and seed, reduced to an instance median, then bootstrapped over instances (95 %
percentile interval); "better" counts instances on which A_margin is better, with ties in
parentheses. Gaps are of the best *feasible* shot and are formed only where both sides have one
(pairs shown); P_F, P_τ and Q_τ(1000) are exact-state quantities and exist for every run. Levels
are medians over runs at each setting; contrasts are medians of within-instance median paired
differences, so a difference of levels need not equal the paired contrast.

| N | Quantity | A_heur level | A_margin level | Paired median diff [95 % CI] | Better (tied) of 30 |
|---:|---|---:|---:|---|---:|
| 12 | P_F | 0.179 | 0.328 | +0.006 [+0.005, +0.012] | 28 |
| 12 | P_τ | 2.6·10⁻³ | 7.6·10⁻³ | +2.1·10⁻³ [+1.3·10⁻³, +3.3·10⁻³] | 27 |
| 12 | Q_τ(1000) | 0.928 | 0.9995 | +0.011 [+0.005, +0.034] | 27 |
| 12 | g_off (%) | 0.000 | 0.000 | 0 [0, 0] | 0 (29) |
| 12 | g_F (%) | 0.00 | 0.00 | 0 [0, 0] | 1 (28) |
| 16 | P_F | 0.229 | 0.301 | +0.0007 [+0.0004, +0.0010] | 27 |
| 16 | P_τ | 4.1·10⁻⁴ | 7.7·10⁻⁴ | +8.2·10⁻⁵ [+6.9·10⁻⁵, +1.4·10⁻⁴] | 30 |
| 16 | Q_τ(1000) | 0.336 | 0.536 | +0.040 [+0.031, +0.061] | 30 |
| 16 | g_off (%), 129 pairs | 0.0143 | 0.5437 | **+0.44 [+0.18, +0.75]** (worse) | 4 (3) |
| 16 | g_F (%), 129 pairs | 1.70 | 1.49 | 0.00 [−0.43, 0.00] | 13 (8) |
| 20 | P_F | 0.167 | 0.303 | +0.034 [+0.017, +0.056] | 30 |
| 20 | P_τ | 6.1·10⁻⁵ | 1.1·10⁻⁴ | +9.2·10⁻⁵ [+7.6·10⁻⁵, +1.4·10⁻⁴] | 30 |
| 20 | Q_τ(1000) | 0.059 | 0.108 | **+0.086 [+0.071, +0.116]** | 30 |
| 20 | g_off (%), 149 pairs | 0.039 | 0.964 | **+0.81 [+0.64, +1.11]** (worse) | 1 (0) |
| 20 | g_F (%), 149 pairs | 6.23 | 4.22 | **−1.61 [−3.11, −0.67]** | 24 (1) |
| 20 | Rank of best feasible shot, 149 pairs | 23 | 10.5 | −11.8 [−17.5, −3.0] | 24 (1) |

*The offset-normalized gap and the sampler move in opposite directions.* Replacing A_heur by
A_margin makes g_off **worse** in 29 of 30 instances at N = 20 (median +0.81 percentage points) and
23 of 30 at N = 16, while it makes every acquisition quantity better: P_F, P_τ and Q_τ(1000) in 30
of 30 instances at N = 20 and in 27–30 of 30 at the other sizes, and the feasible-range gap of the
best feasible shot better in 24 of 30 at N = 20 (median −1.6 points of Δ_F) and unchanged in the
median at N = 16 (better in 13, tied in 8, worse in 9). The secondary contrast, 5·A_crit against
A_heur on the instances with A_crit > 0, agrees on Q_τ(1000) (better in 29 of 30 at N = 16,
+0.094 [+0.069, +0.113]; 30 of 30 at N = 20, +0.019 [+0.002, +0.045]) and on the direction of
g_off, and is mixed on the best feasible shot's g_F (worse in 17 of 30 at N = 16, better in 23 of
30 at N = 20): which weight yields the best single shot and which the highest Q_τ(1000) are
different questions.

*The gap's movement is the denominator's.* The paired ratio of offset-normalized gaps factors, pair
by pair, into a solution part (f(x_A₂) − f\*)/(f(x_A₁) − f\*) and a denominator
part |f\* − A₁K²|/|f\* − A₂K²|; the identity does not survive a median, so the decomposition is
taken in the geometric mean over the same pairs. At N = 20, on the 128 pairs with a positive gap on both
sides, the A_margin run reports a gap 27.8× the A_heur run's in the geometric mean (29.0× in the
median), of which 36.6× is the denominator and 0.76× is the solution — which is *better*, on 59 % of
pairs. The reported gap moves because the denominator moves, with the solution pulling the other
way (all sizes: Supplementary Table S13).

*Two success rules disagree, and the disagreement belongs to the weight.* Under the offset-normalized
rule g_off ≤ 0.1 %, the line this implementation's own tables drew, the A_heur runs succeed at 81 %,
85 % and 82 % (N = 12, 16, 20); under the feasible-range rule g_F ≤ 1 % at 78 %, 40 % and 6 %. The
share of runs that pass the first and fail the second is 3 %, 46 % and **76 %** at A_heur, and 0 %
at every other setting at N = 16 and ≤ 5 % at N = 20. Read at four gap thresholds and three quality
thresholds (Supplementary Table S16) the disagreement is monotone in the gap threshold — 0.06,
0.49, 0.76 and 0.89 at N = 20, A_heur, for 0.01 % to 0.2 % — and stays between 0.03 and 0.07 at
A_margin at every one of them; the 0.1 % line is not a threshold at which anything changes character.

*Within the feasible set, the heuristic-weight sampler is close to uniform.* P_τ factors as
P_F · q_τ with q_τ = Pr[g_F ≤ τ | x ∈ F], and at A_heur the second factor is the share u_τ = |F_τ|/|F|
an indifferent draw from F would give: the ratio R_τ = q_τ/u_τ is 0.999, 0.998 and 1.009 at
N = 12, 16 and 20 with interquartile ranges of 0.02 or less (Supplementary Table S17). That ratio,
however, only checks the mass on one subset; a distribution such as (½, 0, ½, 0) on a four-element F
has R_τ = 1 on a two-element F_τ and is not uniform. So the retained probabilities of every feasible
bitstring were used to compute the conditional distance D_cond of Table I directly, per run
(Table IV; Supplementary Table S18). At A_heur it is 0.001–0.002 in the median at every size and
never above 0.038 (N = 12), 0.009 (N = 16) or 0.005 (N = 20) in 450 runs — against a point mass at
0.985–1.000 and against 0.10–0.81 for what a 1,000-shot empirical draw from the uniform distribution
itself would show at these |F|. Small is not zero: the distribution is *near*-uniform, and because
D_cond carries no direction, the departures from uniform it measures are read through R_τ, whose
median stays within 1 % of 1 at every size (interquartile range ≤ 0.02). At A_margin the
conditional distribution departs further from uniform (D_cond 0.17, 0.14 and 0.09 in the median);
enrichment of the primary quality set is clear at N = 12 and 16 (median R_τ 1.40 and 1.55), whereas
at N = 20 the median R_τ stays close to one (1.01) although the distribution is not uniform. The
multiples of A_crit depart more the closer A is to A_crit (D_cond 0.37–0.49 at 1.1·A_crit).

**Table IV. The conditional distribution on the feasible set.** D_cond per run (Table I), summarised
over the 150 runs of each (N, setting) — 30 instances × 5 seeds — as the median and, for A_heur,
the maximum; R_τ is the median over the same runs (Supplementary Tables S17–S18 give quartiles,
the initial state and every setting). Rounded to the digits shown.

| N | \|F\| | D_cond at A_heur, median (max) | R_τ at A_heur | D_cond at A_margin, median | R_τ at A_margin |
|---:|---:|---:|---:|---:|---:|
| 12 | 66 | 0.002 (0.038) | 0.999 | 0.173 | 1.40 |
| 16 | 560 | 0.001 (0.009) | 0.998 | 0.139 | 1.55 |
| 20 | 4,845 | 0.002 (0.005) | 1.009 | 0.090 | 1.01 |

*What optimisation does at a fixed weight, and what the weight does.* The two are separated by
pairing each run's final state with its initial state (Supplementary Table S34). At A_heur,
optimisation raises the feasible mass — P_F from 0.114, 0.035 and 0.087 (median initial) to 0.179,
0.229 and 0.167, a paired median increase of +0.070, +0.128 and +0.134 on 30 of 30 instances at
N = 12, 16 and 20 — and the expected constraint violation E[V] falls from 21.9, 34.8 and 45.7 to
10.6, 4.0 and 14.3, while the conditional distribution stays where it started: D_cond moves from
0.002–0.004 to 0.001–0.002 in the median, and the initial states that were as far as 0.41 from
uniform on F for some seeds end within 0.005 of it in every N = 20 run. Optimisation at A_heur thus
increased the feasible mass while leaving the conditional distribution close to uniform, with
little systematic enrichment of the primary quality set. Separately, changing the weight from
A_heur to A_margin after optimisation raises the median P_F from 0.167 to 0.303 at N = 20
(Table III); at A_margin optimisation raises P_F (+0.18, +0.22 and +0.24) and the final conditional
distribution departs from uniformity, with the enrichment of Table IV at N ≤ 16. As a
complementary multiplicative summary of the same matched runs, P_τ = P_F · u_τ · R_τ on each
(instance, seed) pair, where u_τ cancels, splits the geometric-mean ratio of P_τ into a
feasible-mass factor and a conditional-enrichment factor: over the 150 pairs per size,
P_τ(A_margin)/P_τ(A_heur) ≈ 1.64 ≈ 1.35 × 1.22, 1.59 ≈ 1.10 × 1.45 and 2.29 ≈ 1.79 × 1.28 as
P_F ratio × R_τ ratio at N = 12, 16 and 20 (the identity holds on the unrounded values), so the
feasible mass accounts for 60 %, 21 % and 70 % of the mean log-ratio; the R_τ ratio exceeds one on
113, 125 and 88 of the 150 pairs (Supplementary Table S38). These shares refer to the mean
log-ratio: they complement rather than decompose the median additive contrasts of Table III, the
identity is algebraic and not a mediation analysis, and the geometric-mean R_τ ratio of 1.28 at
N = 20 is a different statistic from the median R_τ of 1.01 in Table IV. The paired increases at A_heur are nearly the same
on every instance (interquartile range ≤ 0.001), which is what a circuit dominated by the
permutation-symmetric penalty term would give: with f removed, the cost, the mixer and the initial
state are all invariant under qubit permutations, so every bitstring of one cardinality has the
same probability at any angles and the conditional distribution on F is exactly uniform. The
measured ε_F(A) = Δ_F / s_H — the objective's whole feasible range in units of the normalised
Hamiltonian the optimiser descends — is 0.015, 0.015 and 0.016 at A_heur in the median over
instances (Δ_F is 1.5–1.6 % of s_H) (the objective's largest Ising coefficient is 0.7–1.2 % of s_H, and its largest QUBO
entry 0.3–1.2 % of the matrix maximum; Supplementary Table S35), against 0.45–0.52 at A_margin.
That scale is consistent with the near-uniformity and is offered as context for it, not as its
demonstration: the symmetry argument is exact only for the penalty alone, and how far a 1.5 %
objective perturbation is resolved by a given optimiser at finite A is what D_cond measures.
Most of the A_heur state's mass remains far from cardinality K after optimisation, and P_F is what
best-of-shots has to rescue.

*The seed axis, within every instance.* The within-instance spread over five seeds of Q_τ(1000) has
a median over instances of 0.86, 0.49 and 0.19 at A_heur (N = 12, 16, 20) — most of the quantity's
range at the two smaller sizes. At N = 16 and A_heur, 14 of 150 runs returned no feasible bitstring
among 1,000 shots (9 %; 7 % at A_margin; under 1 % at N = 12 and N = 20 at either weight) — a
within-instance tail, not a size trend. Across settings the lowest-energy shot is infeasible in 8,
124 and 95 runs at the three sizes (Table II), and in 55 % of N = 20 runs at 1.1·A_crit, where the
best *feasible* shot nevertheless has the smallest g_F of any setting (−3.3 points of Δ_F against
A_heur, 29 of 30 instances): the selection rule of §III-C is not a formality, and under it a
near-threshold weight is judged by P_τ, g_F and cost, not by the batch minimum.

**What the best shot means (E2; Fig. 3).** The same exact states answer the question the
best-of-shots rule begs: how often does a batch of S shots contain a solution within τ, and how
does that compare with not optimising at all? Optimisation *works*, in the sense the literature's
convergence curves show: at A_heur the optimised state raises Q_τ(1000) over the unoptimised state
at the same initial angles from 0.062 to 0.336 at N = 16 and from 0.019 to 0.059 at N = 20, and it
beats the uniform-over-2^N sampler by one to two orders of magnitude. And its one-shot success
probability at the primary threshold τ = 1 % never exceeds a uniform draw's from the feasible set:
P_τ < P_τ^{uniform-F} in 450 of 450 A_heur runs, hence at every S from 10 to 10,000 — one direct
comparison of P_τ, shown at four operating points because Q_τ(S) is monotone in P_τ. This is a
measured inequality, not a consequence of the small D_cond above (§III-C), and it holds at the two
sensitivity thresholds as well, 0 of 450 at the exact optimum and at τ = 5 % (Supplementary Table
S14). At S = 1,000
and N = 20 a uniform feasible draw succeeds with probability 0.34 and the optimised state with 0.06;
a benchmark that reported its best-of-1,000 gap would not show the difference at N = 12, where
both samplers usually contain a good shot (median Q_τ(1000) 0.93 and 1.00), whereas at N = 16 the
gap between them is already wide (0.34 against 0.83) and at N = 20 neither reliably does (levels at
every S: Supplementary Table S14). The comparison depends on the penalty setting: at A_margin the
optimised state beats the feasible-uniform sampler in 7–9 % of runs, and at 1.1·A_crit in 41 % of
N = 20 runs; the 0 of 450 is a result at A_heur. So **at these sizes and budgets a best-of-shots figure alone cannot confirm that the
optimisation contributed**; the exact-state comparison with explicit control samplers can, and it
shows that at A_heur and the evaluated thresholds optimisation improves acquisition relative to
the unoptimised state while the resulting sampler remains far better than an uninformed one and
below uniform feasible sampling. The uniform-over-F sampler is not a
computational peer of QAOA (§IV-C); it is the floor a constraint-respecting method has to clear,
and the classical annealers of §V-A clear it by a wide margin.

**A second market period (E5; Table V).** Everything above is conditional on one three-year
window. E5 re-ran the pre-fixed 300-run block on the adjacent window, where Δ_F is 23–34 % wider at
every size and one N = 12 instance whose A_crit is zero in the paper's window has a positive
threshold. The main contrast, the disagreement of the two success rules and the uniform-F
comparison keep their direction (Table V; the conditional-distribution and initial → final
analyses were not repeated on this window). At N = 20, A_margin against A_heur
raises Q_τ(1000) on 10 of 10 instances in both windows (+0.09 and +0.12), lowers g_F on 9 and on
7, and worsens g_off on 9 and on 10; the g_off ratio decomposes, in the geometric mean, as
22.3 ≈ 0.64 × 34.6 in the paper's window and 12.4 ≈ 0.45 × 27.4 in the earlier one, the denominator
carrying the sign in both. The two success rules disagree at the same rate — 74 % and 66 % of
N = 20 A_heur runs pass g_off ≤ 0.1 % and fail g_F ≤ 1 % — and the optimised state beats the
uniform-over-F sampler in 0 of 50 A_heur runs per size in the earlier window, as in 0 of 150 in the
paper's. The levels move (Q_τ(1000) at A_heur, N = 20 is 0.060 in one window and 0.098 in the
other), which is what a different instance set produces; the signs and orderings do not. This is one
further window from the same universe and asset class, not a sample of market periods (§VII); the
full comparison is Supplementary Table S15.

**Table V. The E1 block on two price windows (E5).** Instances 0–9 × seeds 42–46 × {A_heur,
A_margin}, 300 runs per window; "2023–26" is the paper's window and "2020–23" the adjacent earlier
one. Paired rows are A_margin − A_heur reduced to an instance median and bootstrapped over the ten
instances (95 % interval), with the count of instances on which A_margin is better (ties in
parentheses); g_off and g_F are of the best feasible shot. The E2 row counts A_heur runs whose
optimised state beats the uniform-over-F sampler at S = 1,000.

| Quantity | N = 12, 2023–26 | N = 12, 2020–23 | N = 16, 2023–26 | N = 16, 2020–23 | N = 20, 2023–26 | N = 20, 2020–23 |
|---|---:|---:|---:|---:|---:|---:|
| A_heur: pass g_off ≤ 0.1 %, fail g_F ≤ 1 % | 0.04 | 0.04 | 0.47 | 0.49 | 0.74 | 0.66 |
| Paired Q_τ(1000) [CI]; better of 10 | +0.020 [+0.000, +0.067]; 8 | +0.010 [+0.000, +0.061]; 9 | +0.041 [+0.004, +0.062]; 10 | +0.057 [+0.029, +0.074]; 10 | +0.090 [+0.067, +0.202]; 10 | +0.122 [+0.062, +0.219]; 10 |
| Paired g_off (pt) [CI]; better of 10 | 0 [0, 0]; 0 (9) | 0 [0, 0]; 0 (10) | +0.28 [−0.00, +1.41]; 2 (2) | +0.19 [−0.01, +0.76]; 4 (0) | +0.63 [+0.41, +0.95]; 1 (0) | +0.52 [+0.20, +0.90]; 0 (0) |
| Paired g_F (pt) [CI]; better of 10 | 0 [0, 0]; 0 (9) | 0 [0, 0]; 0 (10) | 0.00 [−2.31, +0.07]; 4 (4) | −0.22 [−2.05, +0.13]; 5 (3) | −1.75 [−4.49, −0.73]; 9 (0) | −1.93 [−3.80, 0.00]; 7 (2) |
| E2: A_heur beats uniform-F at S = 1,000 | 0 / 50 | 0 / 50 | 0 / 50 | 0 / 50 | 0 / 50 | 0 / 50 |

### C. The optimizer: is the result an artefact of one setting? (E3)

The baseline runs ADAM at a fixed step of 0.1 for 50 steps. One objection to §V-B is that a better
optimizer would change it. The single-seed depth sweep and the steps × p × seed grid
(Supplementary Sec. S-VI) had shown that p = 2 is where depth stops helping monotonically, that
⟨H_norm⟩ saturates by 50–100 ADAM steps and does not improve with an 8× larger budget, and that a
run can barely optimise and still report a small best-of-1,000 gap (⟨H_norm⟩ = −0.91 against
−5.01 in the same cell, gaps 0.038 % and 0.022 %). Which of initialisation, local minima and
expressivity produces the non-monotonicity in p is not identified by those runs. E3 asks the
narrower question for p = 2: on a block fixed before it was run, do two other optimizer settings
change the acquisition quantities?

**Table VI. Optimizer controls on the E3 block.** Instances 0–9 × seeds 42–46 at each size and
weight; the paired columns are the instance-level median difference from the ADAM 0.1 × 50 baseline
with its 95 % bootstrap interval, and the count of instances (of 10) on which the control reached a
lower ⟨H_norm⟩. Budget is the median number of gradient evaluations used (every ADAM step is one
objective and one gradient evaluation; for L-BFGS-B the objective and gradient counts were equal in
every run, n_f = n_∇, line-search evaluations included, since each trial point evaluates both) and
the median optimisation wall-clock. "Optimum hit" counts runs whose best feasible shot
is the exact optimum over all fifty runs of the cell, with the number that produced no feasible shot
in parentheses. Levels, ΔP_τ and the dispersion behind each median are in Supplementary Tables
S22–S23.

| N | A | Setting | Budget (∇, s) | Δ⟨H_norm⟩ vs. baseline [CI]; lower in | ΔQ_τ(1000) [CI] | Optimum hit / 50 (no feas.) |
|---:|---|---|---:|---|---|---:|
| 12 | A_heur | ADAM 0.1 × 50 | 50, 10 | — | — | 38/50 (0) |
| 12 | A_heur | ADAM 0.03 × 100 | 100, 20 | −2.4·10⁻³ [−2.5, −2.3]·10⁻³; 10/10 | +0.0010 [−0.0001, +0.0036] | 42/50 (0) |
| 12 | A_heur | L-BFGS-B | 52, 13 | −3.0·10⁻³ [−3.1, −2.4]·10⁻³; 10/10 | +0.0087 [+0.0044, +0.0094] | 40/50 (0) |
| 16 | A_heur | ADAM 0.1 × 50 | 50, 17 | — | — | 14/50 (3) |
| 16 | A_heur | ADAM 0.03 × 100 | 100, 35 | −6.5·10⁻³ [−6.8, −6.2]·10⁻³; 10/10 | +2.0·10⁻⁵ [−0.56·10⁻⁵, +3.1·10⁻⁵] | 14/50 (4) |
| 16 | A_heur | L-BFGS-B | 53, 21 | −3.6·10⁻³ [−3.7, −3.6]·10⁻³; 9/10 | +1.6·10⁻⁴ [+1.3·10⁻⁴, +2.5·10⁻⁴] | 17/50 (8) |
| 20 | A_heur | ADAM 0.1 × 50 | 50, 30 | — | — | 2/50 (0) |
| 20 | A_heur | ADAM 0.03 × 100 | 100, 61 | **+2.9·10⁻³** [+2.8, +3.2]·10⁻³; 0/10 | −0.0060 [−0.0092, −0.0027] | 1/50 (0) |
| 20 | A_heur | L-BFGS-B | 51, 36 | −1.4·10⁻³ [−1.5, −1.2]·10⁻³; 10/10 | −1.5·10⁻⁴ [−2.5·10⁻⁴, −0.7·10⁻⁴] | 1/50 (0) |
| 12 | A_margin | ADAM 0.1 × 50 | 50, 10 | — | — | 38/50 (0) |
| 12 | A_margin | ADAM 0.03 × 100 | 100, 20 | +1.3·10⁻³ [−1·10⁻⁴, +9.6·10⁻²]; 4/10 | −3.8·10⁻⁵ [−3.0·10⁻³, +1.4·10⁻³] | 36/50 (0) |
| 12 | A_margin | L-BFGS-B | 65, 17 | −4.3·10⁻³ [−15.9, −1.5]·10⁻³; 10/10 | +2.5·10⁻⁶ [−2.9·10⁻³, +1.0·10⁻³] | 36/50 (0) |
| 16 | A_margin | ADAM 0.1 × 50 | 50, 17 | — | — | 20/50 (2) |
| 16 | A_margin | ADAM 0.03 × 100 | 100, 35 | −1.0·10⁻³ [−1.2·10⁻³, +5.5·10⁻²]; 7/10 | −0.0281 [−0.0710, −0.0128] | 12/50 (3) |
| 16 | A_margin | L-BFGS-B | 68, 27 | −2.1·10⁻² [−2.4, −1.2]·10⁻²; 10/10 | −0.0160 [−0.0271, +0.0020] | 17/50 (2) |
| 20 | A_margin | ADAM 0.1 × 50 | 50, 30 | — | — | 6/50 (0) |
| 20 | A_margin | ADAM 0.03 × 100 | 100, 62 | **+0.318** [+0.292, +0.331]; 0/10 | **−0.1060 [−0.1866, −0.0604]** | 0/50 (0) |
| 20 | A_margin | L-BFGS-B | 63, 45 | −4.1·10⁻³ [−60.2, −2.0]·10⁻³; 10/10 | +0.0005 [−0.0064, +0.0049] | 3/50 (0) |

The controls did not yield a uniform improvement in acquisition probability. *L-BFGS-B* reduced the
optimised expectation modestly and consistently — lower than the baseline's in 59 of 60
instance-cells, by 1.4·10⁻³ to 2.1·10⁻² in the median, at 1.2–1.6× the wall-clock, stopping before
the 100-gradient cap in 88–96 % of runs — and its effect on the batch probability Q_τ(1000) is of
both signs and not uniformly small: +0.009 [+0.004, +0.009] at N = 12, A_heur, and −0.016 [−0.027,
+0.002] at N = 16, A_margin, where 7 of 10 instances are worse and the 95th percentile of |ΔQ| is
0.079. Its paired median changes to the one-shot P_τ are within ±4·10⁻⁴ in every cell (instance
maxima ≤ 1.0·10⁻³); Q compounds P over a thousand draws
(∂Q_τ(S)/∂P_τ = S(1 − P_τ)^(S−1)), so a difference in P_τ too small to print can be worth a tenth of
a batch, and a paper that reports only P_τ has not reported the quantity its own selection rule
uses. "Stopped" is not "converged": the E3 block ran at single precision, and a diagnostic re-run of
a fixed sub-block at both precisions (Supplementary Sec. S-VI) shows that in single precision every
run that stopped on a tolerance stopped on the *objective* criterion with a final gradient norm of
10⁻⁵–10⁻¹, several failed their line search or exhausted the budget, while in double precision 11 of
24 stopped on the gradient criterion at gradient norms one to three orders smaller. What the two
precisions agree on is narrower than the stopping reasons suggest, and is stated as two facts. The
optimised ⟨H_norm⟩ agrees to 6·10⁻⁴ in 14 of 16 paired conditions (4.7·10⁻⁵ at N = 20), and — since
a similar expectation does not by itself imply similar mass on the success set — the same runs'
exact final states agree on P_F, P_τ and Q_τ(1000) to 1.0·10⁻⁵, 7·10⁻⁸ and 3·10⁻⁵ in the median over
the 24 pairs and to 3.6·10⁻³, 1.7·10⁻⁵ and 4.4·10⁻³ in 22 of them. The remaining two pairs differ
in expectation by 0.04–0.05 and in Q_τ(1000) by 0.043 and 0.227 — different optimisation outcomes
reached from the same initialisation, and material for the acquisition probability — yet in both
pairs P_τ stays below the instance's uniform-F success probability at either precision
(Supplementary Table S36 and Sec. S-VI). The acquisition quantities are thus precision-sensitive in
2 of 24 pairs, while the comparison against uniform feasible sampling kept its direction in those
two; the conclusion of this section rests on the latter and not on the stopping reason. *ADAM at step 0.03 with twice
the budget* did not improve on the baseline within the tested budget at N = 20: its expectation
after 100 steps is higher than the baseline's after 50 in 10 of 10 instances at both weights, by
0.32 at A_margin, and its Q_τ(1000) is lower by 0.006 and by 0.106 [−0.187, −0.060], worse on 10 of
10 instances in both cells — the largest effect any control has on any quantity in this block, and
it is an optimizer setting. Whether those runs were still descending or had settled in a different
local minimum is not identified by the trajectories, which record no stopping reason for ADAM.

*The scale of the seed.* The baseline's own spread of ⟨H_norm⟩ over the five seeds *within an
instance* has a median of 3.4–5.5 at every N and both weights, and moves Q_τ(1000) by 0.2–0.9
(§V-B). At this budget and p = 2, a large within-instance variation across initial angles thus
remained in the tested configuration, alongside optimizer-dependent changes in acquisition
probability; the comparison is of observed ranges — a paired median against a within-instance
max − min — and not a ranking of factor contributions, and ADAM at 0.03 moves ⟨H_norm⟩ by 0.32 at
N = 20. Where a control moves a run by order 1
it has reached a different optimisation outcome, in either direction. Whether a different
initialisation rule [29], [30] or a larger budget changes the picture is not tested.

*Multistart at equal shots.* The five E1 seeds can be re-read as a five-start strategy — optimise
five times, draw 1,000 shots from each state, keep the best feasible candidate of the 5,000 — and
compared with the single-start strategy a user can actually run at the same shot budget: optimise
once from a seed drawn at random, sample that state 5,000 times, with Q computed per seed and then
averaged (the median seed's state is a summary, not a strategy: no one can select it before
optimising). By the AM–GM inequality the pooled strategy is never worse at equal shots, so the
question is the size of the increment against the four extra optimisations. The increment in the
median over instances is +0.095, +0.153 and +0.046 at A_heur and +0.055, +0.336 and +0.060 at
A_margin (N = 12, 16, 20), exceeding 0.01 on 30 of 30 instances in every cell but one (23 of 30 at
N = 12, A_margin), with intervals clear of zero, for 41–118 s of additional optimisation per instance
(Supplementary Table S24). Five initial angles do buy acquisition probability that five thousand
shots from one start do not, most at N = 16 where the single-start Q_τ(5000) is 0.64–0.68; none of
the strategies reaches uniform sampling from the feasible set at the same 5,000 candidates (1.000,
1.000 and 0.873), nor the annealing baseline of §V-A.

### D. The backend and the hardware: fixed angles on four backends, and four hardware tasks

**Numerics or sampling (E4-A; Fig. 4, Table VII).** Single runs of the same configuration on the
local GPU and on SV1 return bitwise-identical portfolios at N ≤ 12 and different feasible
portfolios at N = 16 and N = 20; three submissions of one configuration to SV1 return three
different portfolios at N ≥ 16; and matching the precision does not restore agreement
(Supplementary Sec. S-VII). Those single runs cannot say whether the output distribution differs
between implementations or the same distribution was sampled differently. E4-A fixes the angles
and asks both questions separately. What is compared is the exact probability of each computational
basis state, not the quantum state (equal basis probabilities leave the relative phases
unconstrained, and probabilities are what the selection rule consumes), on F ∪ {⊥}: the feasible
bitstrings together with one symbol for "this shot is infeasible", p_⊥ = 1 − P_F. That p_⊥ is a
one-shot quantity; the probability that a whole batch of 1,000 shots holds no feasible solution is
(1 − P_F)^1000, and the two are not the same event.

*The double-precision implementations agree; single precision differs by 10⁻⁷.* On F ∪ {⊥} the
managed cloud simulator's exact probabilities differ from the GPU double-precision reference by at
most 2.6·10⁻¹⁴ in total variation (5.3·10⁻¹⁶ per state, 2.8·10⁻¹³ relative in P_τ) at every one of
the twelve circuits, and the single-threaded CPU's by at most 4.3·10⁻¹⁵. Single precision — the
precision of every E1 run — is the only arm that moves the distribution: by up to 6.1·10⁻⁷ in total
variation (5.0·10⁻⁹ per state, 5.5·10⁻⁶ relative in P_τ). How much such a distance can change the
distribution of the *selected* output is bounded directly: if two one-shot distributions are at
total variation d, couple the S shots of a batch pairwise by a maximal coupling, so that the two
batches differ with probability at most 1 − (1 − d)^S, and apply the same deterministic selection
rule to both; by data processing the selected-output distributions are then within
1 − (1 − d)^S ≤ S·d in total variation (derivation in Supplementary Sec. S-VII). At S = 1,000 the
selected-output distributions differ by at most 2.6·10⁻¹¹ for SV1 and 6.1·10⁻⁴ for the
single-precision arm, relative to the double-precision GPU reference. These bounds quantify how far
the arithmetic can move the distribution of the returned portfolio at fixed angles; they do not
bound how often two *independently* sampled batches return different portfolios, which is a
different quantity — two identical fair coins are at d = 0 and still disagree on half of
independent draws — and is what the next paragraph measures.

*The draws are not.* Ten batches of 1,000 shots from one state on one backend return **nine or ten
distinct best-feasible portfolios in ten batches** at N = 20, on every arm and in every circuit —
ten in 15 of the 16 arm × circuit cells and nine feasible portfolios in the one SV1 cell where a
batch held no feasible shot (ten selected outcomes there if ⊥ is counted); three to five at N = 16 and one or two at
N = 12 for the seed-42 states, and one to five and five to ten for the seed-43 states. Pooled over
the four arms, the forty batches of one N = 20 state name 33–36 distinct portfolios, and in no
N = 20 batch is the best feasible shot the exact optimum. The spread is what i.i.d. sampling from the
arm's own exact distribution predicts: a Monte Carlo of ten batches from each arm's distribution
puts the observed number of distinct portfolios inside its 95 % interval in 46 of 48 arm × circuit
cells, and the feasible-shot count of 456 of 480 batches lies within two binomial standard
deviations of 1,000·P_F. The seed-43 states have P_F = 0.0005–0.0006 at N = 16, so five to nine of
their ten batches hold no feasible shot, on the GPU as on SV1.

**Table VII. Fixed angles on four backends (E4-A).** Left: distance between each arm's exact output
probabilities on F ∪ {⊥} and the GPU double-precision ones at the same angles, maximum over the
twelve circuits, and the upper bound 1 − (1 − d)^1000 it implies on the total variation between the
two arms' selected-output distributions at S = 1,000 (not a rate of disagreement between
independent batches).
Right: distinct best-feasible portfolios over ten 1,000-shot batches of one state, range over the
four circuits of each size (the Monte Carlo 95 % interval under i.i.d. draws from the arm's own
distribution in the last row); these counts exclude ⊥, so a batch with no feasible shot adds
nothing to them and is counted in the last column (N = 16, seed 43, where they occur), while the
TV columns are on the output space F ∪ {⊥}.
The full comparison, with per-state and F-only distances, is Supplementary Table S30.

| Arm | TV on F ∪ {⊥} | Upper bound on selected-output TV (S = 1,000) | \|ΔP_τ/P_τ\| | Distinct best, N = 12 (seed 42 / 43) | N = 16 (42 / 43) | N = 20 | No-feasible-shot batches, N = 16 seed 43 |
|---|---:|---:|---:|---|---|---|---|
| GPU double (reference) | — | — | — | 1–2 / 5–6 | 3–4 / 1–5 | 10 | 5–9 |
| CPU double, 1 thread | 4.3·10⁻¹⁵ | 4.3·10⁻¹² | 2.1·10⁻¹⁴ | 1–2 / 6–10 | 3–5 / 3 | 10 | 7 |
| GPU single | 6.1·10⁻⁷ | 6.1·10⁻⁴ | 5.5·10⁻⁶ | 1 / 6–7 | 3–4 / 1 | 10 | 9 |
| SV1 double (cloud) | 2.6·10⁻¹⁴ | 2.6·10⁻¹¹ | 2.8·10⁻¹³ | 2 / 7–8 | 3–4 / 3–4 | 9–10 | 6–7 |
| i.i.d. from own distribution (MC 95 %) | | | | 1–3 / 5–9 | 2–6 / 1–7.5 | 9–10 | — |

Repeated sampling from a fixed output distribution was therefore sufficient to produce portfolio
disagreement of the magnitude the single runs showed, while the arithmetic can move the
selected-output distribution by at most the bounds above; the observed disagreements are not
individually attributed to either channel. What E4-A does not test is the third possibility — different final angles reached by
optimisation runs whose analytic trajectories differ by ~10⁻⁶ per step on the two backends — so the
contribution of backend-dependent optimisation trajectories was not isolated, and the paper does
not claim that sampling is the only source. Three things stay apart here, and only the first two are
measured: compression of a wide quality band by the offset denominator (§V-A), finite-shot sampling
variability from one distribution (this section), and different optimisation outcomes in angle
space (not identified). The practical consequence for anyone building on a device-abstraction layer
is that backend-independence of the QUBO does not imply backend-independence of the returned
portfolio: agreement at small sizes does not establish output-level agreement at larger sizes,
where the feasible set is large, and the check that shows it cost $1.47 for four sizes.

**Hardware: four tasks on IonQ Forte-1.** Each task samples 100 shots from a circuit whose angles
were optimised on the local GPU (two seeds at N = 8, one at N = 12, one at N = 16; $33.20 in all).
In the N = 12 task, 89 of 100 recorded shots violated the cardinality constraint and the remaining
11 contained the exact optimum, so a best-of-shots report would record a 0.0000 % gap from a
circuit whose fidelity under a product-of-gate-fidelities model is 26 %; at N = 16, 18 shots were
feasible and the best of them was 0.0438 % from the exact optimum. The two N = 8 tasks, which differ
only in initial angles, returned 25 % and 7 % feasible shots and best-feasible portfolios with
Sharpe ratios of 1.34 and 1.09 (the optimum's is 1.32). These observations illustrate the distinction between the quality of the best feasible
outcome and the frequency of feasible outcomes, which the reporting of §VI keeps separate. They do
not identify a hardware size limit or separate the contributions of noise, instance structure and
initialization: four tasks at one calibration are not a design that could, and the product-fidelity
model that predicted a ceiling near N = 12 was contradicted by the N = 16 task. The per-task
counts, their provenance from the device's nested result schema, and the fidelity model are in
Supplementary Sec. S-VIII.

## VI. Engineering Implications

The measurements identify three engineering requirements: formulation-independent scoring,
feasibility-aware acquisition metrics, and a recorded execution configuration. Runtime comparisons
provide context for the evaluated implementation rather than evidence about quantum advantage: on
a statevector simulator the exact classical baseline dominated on runtime by three to six orders
of magnitude and on solution quality at every size we could run (§V-A). What the measurement
leaves behind is reusable: what to report, what the same experiment cost on metered backends, the
implementation defects that changed results before they were caught, and an artifact that lets a
reader re-run any of it (§IV-D and the availability statement). The infrastructure, its cost
guards and the provider-specific defects met in deploying it are documented in the artifact, not
here.

### A. What to report

Three reporting proposals exist [19], [20], [21] and none is a formal standard; we adopt the
fields of [19], [20] where they apply and add nothing as a standard. Table VIII lists what each
result of §V showed to be necessary, linking each reporting item to the measurement that
motivates it.

**Table VIII. What each measurement showed a report has to carry.**

| Evaluated quantity | Report | Result that showed the need |
|---|---|---|
| Solution quality | g_F against the objective's spread over F, or the rank in F; the penalty weight A alongside any offset-normalized figure | E0: the same portfolio reads 33.6 % or 0.22 % as A alone changes; the all-states ratio of [19] is non-discriminative at A_heur (§V-A) |
| Optimisation | the optimised expectation beside the best-of-shots figure ([19]'s mean/best split) | a run at ⟨H_norm⟩ = −0.91 reports a gap indistinguishable from one at −5.01 (§V-C) |
| Acquisition | P_F, P_τ and Q_τ(S) at the budget used, against a constraint-respecting control sampler | E2: optimisation raises the median Q_τ(1000) from 0.06 to 0.34 at N = 16, yet at A_heur and τ = 1 % beats a uniform feasible draw's P_τ in 0 of 450 runs (§V-B) |
| Execution conditions | the initial-angle seed distribution, precision, backend and sampler seeding, as factors | one seed reverses the main contrast (§IV-E); single runs on two backends disagree at N ≥ 16 (§V-D) |
| Cost | measured task counts and billed time per run | one sampling call was billed as 177 tasks, not one, until its angles were marked non-trainable (§VI-B) |
| Domain metric | the finance quantity as an auxiliary, with its definition | similar objective values do not guarantee similar Sharpe ratios (§V-A) |

The sharpest form of the first three rows is that the penalty weight is a free parameter of the
reported metric (Table III). Re-scoring fixed feasible solutions at a larger penalty weight reduces
the offset-normalized gap without changing the solutions or their sampling distribution (E0); in
the primary re-optimisation contrast, the larger heuristic weight also yielded a lower acquisition
probability than the margin weight under the tested settings (E1) — two effects, one of the
number and one of the run, and neither a law that a larger A always degrades the sampler.
Reporting A is the minimum; reporting
against the penalty-free objective, or a rank, removes the degree of freedom; reporting Q_τ(S)
against the uniform-feasible control says whether the optimiser did anything a constraint-aware
guess would not. The case of Mancilla et al. [24] makes the cost of under-reporting concrete: a
realized Sharpe of 1.81 for QAOA against 1.31 for annealing is read as a property of the energy
landscape without the objective value of a single selected portfolio, and without the exact
optimum, which C(10, 5) = 252 candidates is small enough to enumerate directly.

### B. Compute and billing

Table II reports, for every metered run, the task count and the dollars billed, and the totals of
the study ($37.90 on Braket). Task counts are reported because one of them was wrong by two orders
of magnitude before it was measured: in the N = 8 SV1 pilot the final 1,000-shot sampling call was
handed the optimiser's angles while they were still marked trainable, and the service's task log
shows 227 tasks for the run — 50 analytic tasks for the fifty optimiser steps and 177 for the one
sampling call — against 51 once the angles were marked non-trainable. The 177 is the forward
circuit plus the 176 shifted tapes that the gradient transform returns for this ansatz at N = 8
(2·p·[C(N, 2) + 2N], reproduced locally with no device; 920 at N = 20), a count consistent with the
framework falling back to parameter shift for a shots-based circuit it cannot differentiate by the
adjoint method, and computing a gradient no one reads. The metered path was not re-run under
instrumentation, so the mechanism is inferred from the count and the fix; the per-task log, reduced
to a per-run ledger, is in the artifact. What the episode establishes for reporting is narrower
than its cause: on a metered backend the executed circuit count is not implied by the algorithm's
description, so it has to be measured and reported per run. Operationally, cloud and hardware
paths require an explicit backend name and a results tag, and the optimiser is never dispatched to
hardware; a default invocation runs locally and bills nothing.

### C. Checks on the recorded results

The results of §V rest on four correspondences that were checked rather than assumed, each
because an implementation defect had at some point broken it; the defects, in the order found,
and what each would have reported are in Supplementary Sec. S-IX. (i) *Objective against QUBO.* The
evaluator of §III-C scores from the returns and covariance, not from the QUBO matrix, and agrees
with the QUBO path on every state at N = 8 and 12 (E0). The cost Hamiltonian is divided by its
largest coefficient so that one ADAM step resolves the γ-landscape (§III-A); at A_heur the
objective's whole feasible range is 1.5–1.6 % of that coefficient in the median (§V-B), so the
normalised landscape is almost entirely the constraint, which is the context for the near-uniform
conditional distribution of §V-B. (ii) *State against batch.* The analytic state quantities and the
final sampling call receive the same final parameters, checked by regression tests on both
optimizer paths. The E3 L-BFGS-B arm had violated this — its sampler read the initial parameter
vector — so its batch columns (the "no feasible shot" and "min-energy shot infeasible" counts of
Table II, the optimum hits of Table VI) described unoptimised states while its state columns
(⟨H_norm⟩, P_F, P_τ, Q_τ) did not. The check that caught it is reported with every arm: the
batch's feasible fraction must track the final state's P_F (it tracked the initial state's, r =
0.99, and the final state's at r = −0.30), and the batch counts must fall within what the recorded
states predict under independent draws (Supplementary Table S22). The arm was re-run with the
fix on the same host: every recorded state of the 300 runs reproduced bit for bit, the batch
columns printed are the re-run's (r = 0.998 against the final state's P_F), and its counts sit
within 3.0 hits and 2.7 no-feasible runs of the independent-draw prediction in every cell, as the
ADAM cells do. (iii) *Sampler against seed.* E4-A seeds the device's own generator: the Lightning
devices draw a seed from NumPy's global generator once, at construction, so a control that
re-seeded the global generator before sampling changed nothing, and two earlier sweeps reported as
varying the shots had re-drawn identical ones. A reproducibility claim about a sampler names the
generator it seeds. (iv) *Hardware records against device schema.* The four IonQ task records are read
from the device's nested result schema, which the client library had misread into an energy of
exactly 0.0 (Supplementary Sec. S-VIII); the committed records are the device's own.

## VII. Limitations and Threats to Validity

**Formulation and data.** Every measurement is on the K-of-N cardinality penalty with a single
weight A; slack variables, a Lagrangian update or a constraint-preserving mixer [45], [46] would
place the constant differently or remove it and would change P_F by construction, and an
implementation that restores the constant and scores on the original objective would not report
g_off at all. The cross covers N ≤ 20, where the feasible set is enumerable; the census extends to
N = 30 by enumeration on thirty instances per size (Supplementary Table S9) while the QAOA runs
above N = 20 are single; N = 50 was not enumerated in this study. All E0–E4 instances are subsets of one
fifty-ticker Nikkei universe over one three-year window, so every interval in §V is conditional on
that universe, window and drawing rule (§IV-E); E5 adds one adjacent window of the same universe,
chosen by adjacency rather than sampled, and does not make the intervals unconditional. Thirty
subsets is fewer than [15]'s 100 or [8]'s 250. The Sharpe ratio is auxiliary: it is measured on
the raw series for the equal-weight portfolio, is not what the QUBO optimizes, and its deviations
from the optimum's are reported as domain-level consequences, not predicted.

**Optimisation.** Five seeds per instance are a tail detector, not a rate: the design was not sized
from a minimum detectable effect, and the intervals are conditional on the fixed seed set, one
member of which reverses the main contrast on its own at each of the two larger sizes
(Supplementary Table S19). E3 tests two settings, not the optimizer space, and one of them did not
improve on the baseline within its budget; the other's single-precision terminations include
line-search failures (§V-C); a schedule, a
larger budget, a different initialisation rule [29], [30] or depth beyond 2 with a warm start could
change the relative sizes of the effects §V-C summarises, and the depth non-monotonicity remains an
observation whose cause is not identified. The multistart comparison is at equal shots with the
optimisation cost stated beside it, not at equal total cost. A_crit and A_margin are reference
conditions computed by enumeration, not procedures that scale. The critical penalty is a
ground-state feasibility threshold, not a guarantee of reliable finite-shot acquisition: near it,
infeasible batch minima are common (55 % of N = 20 runs at 1.1·A_crit), which makes the
feasibility-aware selection rule essential; these observations do not by themselves rule out
near-threshold weights under that rule, whose utility is assessed through the acquisition
probability and solution quality at the chosen budget (§V-B). E2's
strongest control is matched on candidates generated, not on cost: that the optimised state at A_heur beat a uniform
draw from F in none of 450 runs at the tested thresholds says the best-of-shots figure cannot
confirm the optimisation's contribution at these sizes, not that a constraint-aware quantum method
would not; Q_τ(S) assumes i.i.d. shots.

**Backend and hardware.** E4-A separates the output distribution from the draws, not the
optimisation: twelve circuits at one weight, evaluated at fixed angles, bound how far the
arithmetic can move the selected-output distribution and show sampling sufficient for the observed
disagreement, while whether an optimisation run on
SV1 reaches the same angles as the GPU's was not tested (E4-B, ≈1,500 tasks), so the backend
attribution is partial. Hardware is four tasks — two seeds at N = 8, one each at N = 12 and 16, 100
shots each — at one calibration; they illustrate the separation of best-outcome quality from
feasible-shot frequency and establish no size limit, no noise ranking and no distribution.
Continuous weights and richer constraints are deferred.

## VIII. Conclusion

Two effects were measured separately for the tested penalty encoding. Re-scoring fixed portfolios
as only the penalty weight changed moved the offset-normalized gap from 33.6 % to 0.22 % while the
feasible-range gap stayed at 37.8 %; re-optimising at a different weight, across 30 instances × 5
initial angles at N = 20, worsened that gap in 29 of 30 instances while raising the probability
that 1,000 shots hold a feasible solution within 1 % of the feasible range in 30 of 30 (+0.086),
and a 300-run subset on a second price window reproduced each direction. At the heuristic weight the optimised
sampler's distribution conditioned on feasibility was close to uniform (median total variation 0.002,
maximum 0.04 over 450 runs), and its one-shot success probability, compared directly, exceeded
uniform feasible sampling's in none of those runs at the primary threshold or at the two
sensitivity thresholds; relative to the unoptimised state at the same initial angles, the median
1,000-shot success probability rose from 0.062 to 0.336 at N = 16 and from 0.019 to 0.059 at
N = 20. The optimizer controls changed that batch probability by −0.106 to +0.009 depending on the
setting, and a five-start strategy at equal shots added 0.05–0.34 to it. With the angles fixed,
double-precision implementations agreed on the output distribution to 2.6·10⁻¹⁴ in total
variation and the single-precision arm to 6.1·10⁻⁷, so the arithmetic can move the distribution
of the selected portfolio by at most 2.6·10⁻¹¹ and 6.1·10⁻⁴ at 1,000 shots, whereas independent
resampling of one distribution returned a different selected portfolio in nearly every batch at
N = 20; the contribution of backend-dependent optimisation trajectories was not separated. Exact
references, feasibility-aware sampling controls and explicitly recorded execution conditions made
those differences visible. These findings characterise the evaluated benchmark configuration —
one encoding, N ≤ 20, two adjacent windows of one universe, p = 2 — rather than the performance
limit of QAOA.

## Acknowledgment

**AI assistance disclosure.** Claude Opus (Anthropic), accessed through the Claude Code
command-line interface [47], was used throughout this work: implementing the QUBO formulation, the
solvers, the evaluator, the experiment drivers and the infrastructure code; drafting and revising
the prose of this manuscript (Sections I–VIII and Supplementary Sections S-I–S-X); generating the figures and tables from the committed result files;
screening the benchmark literature of the census; and checking numerical claims against the
result files. The research questions, the experimental design, the choice of what to measure and
the interpretation of every result are the author's, who reviewed and edited every portion of the
manuscript and verified every reported number against the committed artifact that produced it. The
author is responsible for all content, including any error introduced with the tool's assistance
and not caught in that review.

## References

[1] H. Markowitz, "Portfolio selection," *The Journal of Finance*, vol. 7, no. 1, pp. 77–91, Mar. 1952,
doi: 10.1111/j.1540-6261.1952.tb01525.x.
[2] D. Bienstock, "Computational study of a family of mixed-integer quadratic programming problems,"
*Mathematical Programming*, vol. 74, pp. 121–140, 1996.
[3] A. Lucas, "Ising formulations of many NP problems," *Frontiers in Physics*, vol. 2, art. 5, 2014,
doi: 10.3389/fphy.2014.00005.
[4] E. Farhi, J. Goldstone, and S. Gutmann, "A quantum approximate optimization algorithm,"
arXiv:1411.4028, 2014.
[5] R. Orús, S. Mugel, and E. Lizaso, "Quantum computing for finance: Overview and prospects,"
*Reviews in Physics*, vol. 4, art. 100028, 2019, doi: 10.1016/j.revip.2019.100028.
[6] D. J. Egger *et al.*, "Quantum computing for finance: State-of-the-art and future prospects,"
*IEEE Transactions on Quantum Engineering*, vol. 1, pp. 1–24, 2020.
[7] D. Herman *et al.*, "Quantum computing for finance," *Nature Reviews Physics*, vol. 5, pp. 450–465,
2023, doi: 10.1038/s42254-023-00603-1.
[8] E. Stopfer and F. Wagner, "Quantum portfolio optimization: An extensive benchmark,"
arXiv:2509.17876, 2025 (rev. Jul. 2026).
[9] S. Brandhofer, D. Braun, V. Dehn, G. Hellstern, M. Hüls, Y. Ji, I. Polian, A. S. Bhatia, and
T. Wellens, "Benchmarking the performance of portfolio optimization with QAOA," *Quantum Information
Processing*, vol. 22, no. 1, art. 25, 2023; preprint arXiv:2207.10555, Jul. 2022.
[10] J. S. Baker and S. K. Radha, "Wasserstein solution quality and the quantum approximate
optimization algorithm: A portfolio optimization case study," arXiv:2202.06782, 2022.
[11] L. Lozano, "A penalty-free pipeline for direct quantum-annealer portfolio optimization,"
*arXiv preprint* arXiv:2605.17628, May 2026.
[12] G. Rosenberg, P. Haghnegahdar, P. Goddard, P. Carr, K. Wu, and M. López de Prado, "Solving the
optimal trading trajectory problem using a quantum annealer," *IEEE Journal of Selected Topics in
Signal Processing*, vol. 10, no. 6, pp. 1053–1060, Sep. 2016.
[13] D. Venturelli and A. Kondratyev, "Reverse quantum annealing approach to portfolio optimization
problems," *Quantum Machine Intelligence*, vol. 1, no. 1, pp. 17–30, 2019,
doi: 10.1007/s42484-019-00001-w.
[14] S. Mugel *et al.*, "Dynamic portfolio optimization with real datasets using quantum processors
and quantum-inspired tensor networks," *Physical Review Research*, vol. 4, art. 013006, 2022,
doi: 10.1103/PhysRevResearch.4.013006.
[15] V. Uotila, J. Ripatti, and B. Zhao, "Higher-order portfolio optimization with the quantum
approximate optimization algorithm," in *Proc. IEEE Int. Conf. Quantum Computing and Engineering
(QCE)*, 2025, pp. 2238–2249, doi: 10.1109/QCE65121.2025.00244.
[16] A. Verma and M. Lewis, "Penalty and partitioning techniques to improve performance of QUBO
solvers," *Discrete Optimization*, vol. 44, art. 100594, 2022.
[17] M. Ayodele, "Penalty weights in QUBO formulations: permutation problems," in *Proc. EvoCOP 2022*,
LNCS 13222, pp. 159–174; preprint arXiv:2206.11040.
[18] J. A. Montanez-Barrera, D. Willsch, A. Maldonado-Romo, and K. Michielsen, "Unbalanced
penalization: a new approach to encode inequality constraints of combinatorial problems for quantum
optimization algorithms," *Quantum Science and Technology*, vol. 9, no. 2, art. 025022, 2024;
preprint arXiv:2211.13914.
[19] A. Abbas *et al.*, "Challenges and opportunities in quantum optimization," *Nature Reviews
Physics*, vol. 6, no. 12, pp. 718–735, Dec. 2024, doi: 10.1038/s42254-024-00770-9.
[20] T. Koch *et al.*, "Quantum optimization benchmarking library — The intractable decathlon,"
arXiv:2504.03832, 2025.
[21] L. Lozano, "Where the quantum lives in D-Wave hybrid portfolio optimization: an operational
decomposition audit," *arXiv preprint* arXiv:2605.17623, 2026.
[22] M. Hodson, B. Ruck, H. Ong, D. Garvin, and S. Dulman, "Portfolio rebalancing experiments using
the Quantum Alternating Operator Ansatz," *arXiv preprint* arXiv:1911.05296, Nov. 2019.
[23] A. Oralkhan and T. Zhaxalykov, "Investigation of hardware architecture effects on quantum
algorithm performance: A comparative hardware study," arXiv:2601.05286, 2026.
[24] J. Mancilla, T. D. Bouloumis, and F. Goguikian, "Constrained portfolio optimization via QAOA with
XY-mixers and Trotterized initialization: A hybrid approach for direct indexing," *arXiv preprint*
arXiv:2602.14827, Feb. 2026.
[25] N. N. Hegade, P. Chandarana, K. Paul, X. Chen, F. Albarrán-Arriagada, and E. Solano, "Portfolio
optimization with digitized counterdiabatic quantum algorithms," *arXiv preprint* arXiv:2112.08347,
Dec. 2021.
[26] B. Wu and L. Wang, "A two-step quantum approximate optimization algorithm for portfolio
optimization and risk assessment," *Quantum Reports*, vol. 8, no. 2, art. 45, May 2026,
doi: 10.3390/quantum8020045.
[27] N. Innan, A. Saleem, A. Marchisio, and M. Shafique, "Quantum portfolio optimization with expert
analysis evaluation," *arXiv preprint* arXiv:2507.20532, Jul. 2025.
[28] P. K. Barkoutsos, G. Nannicini, A. Robert, I. Tavernelli, and S. Woerner, "Improving variational
quantum optimization using CVaR," *Quantum*, vol. 4, art. 256, 2020.
[29] L. Zhou, S.-T. Wang, S. Choi, H. Pichler, and M. D. Lukin, "Quantum approximate optimization
algorithm: Performance, mechanism, and implementation on near-term devices," *Physical Review X*,
vol. 10, art. 021067, 2020, doi: 10.1103/PhysRevX.10.021067.
[30] S. H. Sack and M. Serbyn, "Quantum annealing initialization of the quantum approximate
optimization algorithm," *Quantum*, vol. 5, art. 491, Jul. 2021, doi: 10.22331/q-2021-07-01-491.
[31] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, "Barren plateaus in quantum
neural network training landscapes," *Nature Communications*, vol. 9, art. 4812, 2018,
doi: 10.1038/s41467-018-07090-4.
[32] L. Bittel and M. Kliesch, "Training variational quantum algorithms is NP-hard," *Physical Review
Letters*, vol. 127, no. 12, art. 120502, 2021, doi: 10.1103/PhysRevLett.127.120502.
[33] G. G. Guerreschi and A. Y. Matsuura, "QAOA for Max-Cut requires hundreds of qubits for quantum
speed-up," *Scientific Reports*, vol. 9, art. 6903, 2019, doi: 10.1038/s41598-019-43176-9.
[34] Amazon Web Services, "Amazon Braket Developer Guide," AWS documentation. [Online]. Available:
https://docs.aws.amazon.com/braket/ (accessed Aug. 8, 2026).
[35] V. Bergholm *et al.*, "PennyLane: Automatic differentiation of hybrid quantum-classical
computations," arXiv:1811.04968, 2018.
[36] H. Bayraktar *et al.*, "cuQuantum SDK: A high-performance library for accelerating quantum
science," in *Proc. IEEE Int. Conf. Quantum Computing and Engineering (QCE)*, 2023, pp. 1050–1061,
doi: 10.1109/QCE57702.2023.00119.
[37] T. Jones and J. Gacon, "Efficient calculation of gradients in classical simulations of
variational quantum algorithms," arXiv:2009.02823, 2020.
[38] C. Collberg and T. A. Proebsting, "Repeatability in computer systems research,"
*Communications of the ACM*, vol. 59, no. 3, pp. 62–69, Mar. 2016, doi: 10.1145/2812803.
[39] G. K. Sandve, A. Nekrutenko, J. Taylor, and E. Hovig, "Ten simple rules for reproducible
computational research," *PLoS Computational Biology*, vol. 9, no. 10, art. e1003285, 2013,
doi: 10.1371/journal.pcbi.1003285.
[40] Association for Computing Machinery, "Artifact review and badging — version 1.1," ACM policy,
Aug. 2020. [Online]. Available: https://www.acm.org/publications/policies/artifact-review-and-badging-current
(accessed Aug. 8, 2026).
[41] S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi, "Optimization by simulated annealing," *Science*,
vol. 220, no. 4598, pp. 671–680, 1983, doi: 10.1126/science.220.4598.671.
[42] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf.
Learning Representations (ICLR)*, 2015. arXiv:1412.6980.
[43] R. H. Byrd, P. Lu, J. Nocedal, and C. Zhu, "A limited memory algorithm for bound constrained
optimization," *SIAM J. Sci. Comput.*, vol. 16, no. 5, pp. 1190–1208, 1995, doi: 10.1137/0916069.
[44] P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python,"
*Nature Methods*, vol. 17, pp. 261–272, 2020, doi: 10.1038/s41592-019-0686-2.
[45] S. Hadfield, Z. Wang, B. O'Gorman, E. G. Rieffel, D. Venturelli, and R. Biswas, "From the quantum
approximate optimization algorithm to a quantum alternating operator ansatz," *Algorithms*, vol. 12,
no. 2, art. 34, 2019.
[46] Z. Wang, N. C. Rubin, J. M. Dominy, and E. G. Rieffel, "XY mixers: analytical and numerical
results for the quantum alternating operator ansatz," *Physical Review A*, vol. 101, art. 012320,
2020.
[47] Anthropic, "Claude Code," 2026. [Online]. Available: https://claude.com/claude-code

---

### Figure manifest

Main-text figures. Files carry experiment names; the mapping here is the one the text uses.

- **Fig. 1** — `notebooks/figures/fig_e0_rescore.pdf` — E0. (a) The median-rank feasible solution of
  the reference N = 20 instance scored three ways as only A changes: the offset-normalized gap
  g_off = (f − f\*)/|f\* − AK²|, the all-states range gap 1 − AR = (E − E\*)/(C_max − E\*) of [19],
  and the feasible-range gap g_F = (f − f\*)/Δ_F. (b) The deflation D(A) = g_F/g_off per size on
  the reference instances, with A_heur marked by a star; N > 20 uses A_safe and is drawn hollow.
  (c) D(A_heur) over thirty instances per size (box = IQR, whisker = range), with the reference
  instance marked.
- **Fig. 2** — `notebooks/figures/fig_e1_penalty.pdf` — E1. Four panels against the penalty setting
  (A = 0 for the A_crit = 0 instances, then 1.1/2/5/10 × A_crit, A_margin, A_heur), one series per N:
  (a) g_off and (b) g_F of the best feasible shot (symlog, every run as a faint point, exact hits on
  the zero line), (c) P_F of the optimised state, and (d) Q_τ(1000) at τ = 1 %. Markers are
  instance-level medians, bars the IQR over instances; the gap panels draw only runs with a feasible
  shot. The share of runs at exactly zero gap, per setting and size over these E1 runs, is
  Supplementary Table S37.
- **Fig. 3** — `notebooks/figures/fig_e2_shots.pdf` — E2. Q_τ(S) at τ = 1 % against the shot budget
  S ∈ {10, 100, 1,000, 10,000} for four samplers — the optimised QAOA state, the unoptimised state
  at the same initial angles, uniform over F and uniform over all bitstrings — at A_heur, for
  (a) N = 12, (b) N = 16 and (c) N = 20; instance-level medians with IQR bands.
- **Fig. 4** — `notebooks/figures/fig_e4_backends.pdf` — E4-A. (a) Same angles, different backend:
  total variation distance on F ∪ {⊥} between each arm's exact output probabilities and the GPU
  double-precision ones, per circuit (i0/i1 = instance, s42/s43 = seed; log y; the double- and
  single-precision ε lines for scale). (b) Same output distribution, ten batches: distinct
  best-feasible portfolios over ten 1,000-shot batches of one state, per circuit and arm, with the
  95 % interval of the same count under i.i.d. draws from the arm's own exact distribution.

Supplementary Figs. S1–S12 and Tables S1–S38 are in the Supplementary Material, generated by the
same scripts from the same committed files.

### Data/artifact availability

The code, the derived input statistics of the two price windows (μ and Σ of the fifty-ticker
universe, from which every instance is sliced; the price snapshots themselves are third-party data
and are not redistributed), the census and its per-paper cells, every result CSV this paper reads
(one row per run and one per instance, with the aggregates the tables are checked against), the
figure scripts and the table-checking script are at
`https://github.com/tobitaQ/qaoa-portfolio-penalty-benchmark`, release `v1.0`, archived at
`https://doi.org/10.5281/zenodo.22743582`; the README indexes the files by experiment. A clone
reproduces every number without network access to a market-data provider, at three levels: every figure and table from the committed CSVs (seconds, no GPU); one N = 8 or
N = 12 instance end to end (minutes on a CPU); the cross from the manifest (13.9 GPU h). The
per-run E1 artifacts (exact probabilities on F under the initial and final state, angles,
trajectory, shot counts; ≈100 MB compressed, 2,565 files) are deposited in the same archive, and
are needed only to recompute the E2 columns and D_cond, which the committed CSVs already hold; any
full state vector is reconstructed from the recorded angles and seed. The four IonQ task records
are committed in the device's own nested schema. Cloud and hardware paths require an explicit
backend name and a results tag.
