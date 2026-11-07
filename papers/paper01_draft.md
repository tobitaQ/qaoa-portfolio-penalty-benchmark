# Feasibility Without Preference: What the Quantum Approximate Optimization Algorithm Contributes at a Dominant Penalty Weight

**Draft v5.5 — 2026-10-26.** Submitted to IEEE Transactions on Quantum Engineering and declined
without review on 2026-10-18; the revision goes to IEEE Access. The abstract, introduction and
conclusion were rewritten around what the circuit contributes and what sets it, and Table III
carries the penalty ladder over all six weights and the two A = 0 encodings (`analyze_ladder.py`).
In v5.1, ε_F is measured at every setting including A = 0 — it had been asserted as 1 for the E6
rows — and is described as the ratio it is rather than as a share of the Hamiltonian; the title and
the claims about the weight name the rule this implementation uses rather than a textbook. In v5.2
the claims about the circuit are stated as what the measurements show, ε_F is described against the
Ising coefficient taken before the normalization, the introduction factors the one-shot probability
rather than the batch's, and the abstract names the statistic behind each number. The run files are
unchanged throughout; the ladder and the coefficient scales were re-derived from them. In v5.3 the
figure caption names the coefficient as the one taken before normalization, and the conclusion
separates the conditional distance from the preference ratio, both as medians. In v5.4 the
introduction and the conclusion were compressed to what they are for — the problem, what a reported
gap misses, the approach, the findings and the contributions in the first; three paragraphs in the
second — and the detail they had been previewing stays in §V, §VI and §VII. The main text is 20
pages. In v5.5 the conclusion ends on what the measurements mean for a benchmark rather than on
their limits, and §VI-C, §V-E and §V-F hand their per-cell figures and their defect history to the
Supplementary Material. Reference numbering is in order of first appearance.

---

## Abstract

An optimality gap does not show how often a constrained quantum optimizer returns a feasible
solution, or which feasible solutions it prefers. We measure both for the depth-2 quantum
approximate optimization algorithm (QAOA) on a cardinality-constrained portfolio: 2,565 runs over 30
instances × 5 initial angles × 6 penalty weights at 12, 16, and 20 assets, the zero-penalty
conditions of a nominal 2,700 merged. The preference tracks one ratio, ε_F: the objective's spread
over the feasible set against the cost's largest Ising coefficient. Under this implementation's
dominate-the-objective rule, ε_F is 0.015–0.016, where the penalty dominates it and cannot separate
equal-cardinality selections. There the optimized state sampled inside that set much as a uniform
draw would: median preference ratios 0.998–1.009, median conditional distance 0.001–0.002. Its
one-shot success probability stayed below that draw's in all 900 runs at this weight, across the
baseline and three control blocks. Optimization raised the feasible probability P_F instead. At 20
assets, 1.1 times the feasibility threshold yields ε_F 0.98, a median ratio of 3.10, and 41% of runs
above that draw, a share that rises with ε_F at every size. Moving the constraint into the ansatz (a
Dicke state and a weight-preserving mixer, same depth, steps, and shots) puts 420 of 450 runs above
the draw and the exact optimum in 402, against 170 at the dominant weight. Changing the weight to a
margin-based one worsens the offset-normalized gap in 29 of 30 instances at 20 assets and raises the
1,000-shot success probability in all 30.

**Keywords:** benchmark testing, cloud computing, constrained optimization, portfolio optimization,
quadratic unconstrained binary optimization (QUBO), quantum approximate optimization algorithm (QAOA),
quantum computing, reproducibility of results.

---
## I. Introduction

Any benchmark for a quantum optimizer has to answer one question: did the quantum part contribute
anything to the result? Portfolio selection under a cardinality constraint asks for exactly K of N
candidate assets to optimize a risk–return trade-off [1], and it is NP-hard [2]. Its usual route to
a quantum device is the quadratic unconstrained binary optimization (QUBO), or Ising, encoding [3],
[4], surveyed in [5]–[7]. The usual answer is an optimality gap taken on the energies that the
solver returns. That number cannot answer the question. It describes the best sample of one
measurement batch, and under a penalty encoding, it is divided by a constant that the formulation
chose: one fixed portfolio's gap reads 33.6% at one weight and 0.22% at another, while its distance
from the optimum in the objective's own range stays at 37.8% (§V-D). Of 14 published quantum
approximate optimization algorithm (QAOA) and annealing portfolio benchmarks, 10 report no gap, and
the four that do use four different normalizations (Supplementary Sec. S-I): the normalizations
these benchmarks report are not consistent with each other. That an offset-sensitive score depends
on the weight is already known [8]–[10].

What the question needs is the probability that a batch of shots contains a feasible portfolio
within a quality threshold fixed in advance. A batch is built from one shot: under independent and
identically distributed (i.i.d.) shots it holds one with probability 1 − (1 − P_τ)^S. The one-shot
success probability is the one that factors, as P_τ = P_F · q_τ: how often a shot is feasible, times
which solutions the sampler prefers among those that are (§III-C). A uniform draw from the feasible
set is the control that makes the second readable: it is what respecting the constraint alone
already delivers. We measure both against that control for the QAOA at depth p = 2, on a cross of 30
instances × 5 initial-angle seeds × 6 penalty weights at N ∈ {12, 16, 20}, for a total of 2,565
runs, scored against exact optima and exact feasible-set statistics.

One formulation constant moves the second factor: how wide the objective's spread over the feasible
set is next to the largest Ising coefficient of the cost, ε_F = Δ_F / s_H (§III-A). Under the
dominate-the-objective rule that this implementation uses, ε_F is 0.015–0.016, and there is little
systematic enrichment of the quality set. The optimized state's conditional distribution sits where
its initial state already was, and its one-shot success probability remains below that of a uniform
feasible draw in all 900 runs of that weight. What optimization changed is the other factor: the
feasible probability P_F rises on 30 of 30 instance medians at each size. Over the six weights, the
share of runs that exceed the control rises monotonically with ε_F, from 0% at the dominant weight
to 41% at 1.1 times the feasibility threshold at 20 assets (§V-C).

That end is also reached by changing the encoding instead of the weight. Carried by the ansatz
instead — the Dicke state |D^N_K⟩, a mixer commuting with the Hamming weight [11], [12], and the
objective as the whole cost — the sampler exceeds the control in 420 of 450 runs at the same depth,
budget, and shots, and returns the exact optimum in 402 versus 170 at the dominant weight. A second
block of 450 runs removes the penalty from the cost but keeps the baseline transverse-field ansatz;
because its preference ratios are as large but its feasible probability is not, it remains at 0.4 of
the control. Only the product of the two factors rises above the control, and what that control
bounds is this penalty encoding, not what a quantum sampler can do. Neither weakening the weight nor
changing the ansatz is recommended here, for reasons given in §V-C.

The reported gap and the acquisition probability are two readings of the same weight, measured on
the same runs, and they disagree. Lowering the weight from the dominate-the-objective rule to a
margin-based one worsens the gap in 29 of 30 instances at N = 20, and raises, in 30 of 30 instances,
the probability that 1,000 shots hold a feasible portfolio within 1% of the feasible range (+0.086).
The gap moves because its denominator moves; the acquisition probability moves with how much of the
objective the normalized cost carries, under the settings tested. In that comparison, the gap ranked
best the weight at which the optimized state showed no preference.

Scale is a separate question. §V-G reports the resource counts and the envelope that we measured,
and §VI-A marks which reporting items survive where the feasible set cannot be enumerated. We make
no claim about post-classical performance. The question answered here comes before the speedup
question: feasibility and preference are what a runtime comparison has to be read against.

**C1 — What the circuit contributes, measured at six penalty settings and two ansatzes.** The
ingredients are known: scoring on the feasible set, the warning against an oversized penalty, and
the Dicke-plus-XY alternative are all documented in [8], [11], [12]. What is added is the
measurement: both factors, per run, from the exact final state and against a uniform feasible draw,
over the six penalty settings of one crossed design and the two A = 0 control blocks beside it
(§V-A, §V-C).

**C2 — What that disagreement costs a benchmark, and what to report instead.** It is quantified on
the same runs: 76% of the runs at 20 assets and the dominant weight pass the gap rule while failing
the quality rule (§V-D). §VI-A lists what each measurement showed a report has to carry, and which
of those items survive where the feasible set cannot be enumerated.

**C3 — The procedure and its record.** Every run is tied to its inputs, constants, angles, counts,
shots, exact-state scores, and, on metered backends, task counts and dollars, under a manifest of
what was planned, what collapsed, what ran, and what failed (§IV-F). Every aggregation in §V is
re-derivable from the run rows by the script named beside it.

The target of this criticism is the penalty encoding and the reporting conventions, not QAOA. A
classical annealer on the same penalty-encoded QUBO reproduces the metric pattern (§V-D), and the
simulated QAOA had no speed advantage at any size that we could run, which agrees with the closest
prior benchmark [13] (§V-G). The crossed experiment and its controls are statevector simulations,
with four hardware sampling tasks beside them (§V-F), and the arithmetic was measured instead of
assumed [14]. At these sizes, it is the draw, not the arithmetic, that decides which portfolio is
reported (§V-F).

## II. Related Work

**QUBO portfolio benchmarks.** Markowitz's mean–variance selection [1] becomes NP-hard once a
cardinality constraint is added [2]. The standard route to a quantum device is the QUBO/Ising
encoding catalogued by Lucas [3], with the constraint absorbed into a quadratic penalty. Rosenberg
et al. [15] applied it to multi-period trading on an annealer, and Mugel et al. [16] compared
annealers, gate-based devices, and tensor networks on 52 assets. Venturelli and Kondratyev [17] seed
reverse annealing with a greedy solution and note that the cardinality penalty needs a large P that
"is typically associated with precision issues", which is a hardware-side argument for keeping the
penalty scale out of the number being scored. The closest benchmark to this paper is Stopfer and
Wagner [13]. They evaluate annealing and QAOA against mixed-integer programming and classical
heuristics on 250 real-market instances of up to 1,000 assets, restrict the quantum methods to at
most 30 assets, and find that classical solvers reach optimality within seconds; our runtime results
agree (§V-G). Uotila et al. [18] run 100 DJIA instances against an exact eigensolver. They report
QAOA meeting both quality and feasibility criteria on 3 of 100, and they observe spectra agreeing to
10⁻⁹–10⁻¹¹ while "the allocations start differing substantially". Brandhofer et al. [8] define the
approximation ratio over the feasible set alone, "independent of the particular method used to
enforce the constraint", and reject the dominate-the-objective penalty rule as "unnecessarily
large"; §V-D measures what those two corrections are worth rather than rediscovering them. Baker and
Radha [9] report, in a QAOA portfolio case study, run-to-run fluctuation exceeding finite-shot
noise, and they reject the approximation ratio as "sensitive to … the constraint enforcement scheme
chosen". The constant offset that carries that sensitivity is written as c = AK² in [10]. How to set
the weight is a studied question of its own: Verma and Lewis [19] give exact weights that make the
unconstrained optima coincide with the feasible ones, Ayodele [20] compares static rules, and
Montanez-Barrera et al. [21] give an unbalanced penalty for inequality constraints that avoids slack
variables.

**What benchmarks report, and what is proposed.** Three documents propose what a benchmark in this
field should report, and none of them is a formal standard. Abbas et al. [22] prescribe a
range-normalized approximation ratio (C_max − ⟨C⟩)/(C_max − C_min), with mean-sample and best-sample
ratios listed separately, since the best sample "may improve with a higher number of shots". The
Quantum Optimization Benchmark Library [23] mandates run, feasible-run, and successful-run counts
against a success threshold, and prescribes scoring on the original problem. It warns that "the
objective values of two QUBOs representing the same problem using different penalty factors cannot
be directly compared". Its smallest portfolio instance has 710 binary variables, which is beyond any
statevector simulator. Lozano [24] proposes a four-metric protocol, the only one to require billed
device time, with a gap regularized by max(1, |f*|). We adopt the reporting fields of [22], [23]
where they apply. A census of 14 QAOA and annealing portfolio benchmarks — [8]–[10], [13],
[15]–[18], [25]–[30], selected by a stated protocol (Supplementary Sec. S-I, Table S1) — is
background for the measurements: 10 of the 14 report no optimality gap, the four that do use four
different normalizations, none reports task counts or billed cost, and none of the seven published
after [22], [23] cites either. What the census counts is that a proposal was not cited. That is not
evidence that any paper uses the denominator studied here. Barkoutsos et al. [31] aggregate the shot
distribution by conditional value-at-risk, because the mean and the minimum are unstable summaries,
and best-of-shots is that family's extreme case.

**Initialization, sampling and the simulation stack.** The classical outer loop is where QAOA is
most often limited. Zhou et al. [32] introduced the INTERP and FOURIER heuristics, Sack and Serbyn
[33] showed random initialization to be prone to local minima, and barren plateaus [34] together
with the NP-hardness of training [35] leave no initialization universally safe. Guerreschi and
Matsuura [36] concluded that hundreds of qubits would be needed before QAOA on Max-Cut could outrun
a classical solver, which is the precedent for the runtime result here. Amazon Braket [37] exposes
simulators and hardware behind one API with per-task and per-shot pricing, and Oralkhan and
Zhaxalykov [26] use it across trapped-ion and superconducting backends. We run PennyLane [38] with
the Lightning CPU/GPU statevector devices on NVIDIA cuQuantum [39] and adjoint gradients [40].
Reproducibility in computer-systems research depends on recording the exact inputs [41]–[43], and a
cloud quantum experiment adds a billable environment to that list (§VI).

**Position of this work.** The formulation, the algorithm and the runtime conclusion are standard.
What is added is the measurement: a crossed design (instance × seed × penalty), evaluated on both
exact final-state probabilities and best-feasible sample quality, with exact references, explicit
sampling controls, and a recorded execution environment. We report it because it contradicts what a
single-seed, single-instance, single-weight run would have let us claim.

## III. Formulation and Evaluation Quantities

### A. The penalty encoding

Let x ∈ {0,1}^N be a binary selection vector, μ ∈ ℝ^N the vector of annualized expected returns, and
Σ ∈ ℝ^{N×N} the annualized covariance matrix. With risk-aversion λ ∈ [0,1] and target cardinality K,
the cardinality-constrained mean–variance problem

> minimize λ·xᵀΣx − (1−λ)·μᵀx subject to Σᵢ xᵢ = K

is cast as an unconstrained QUBO via a quadratic penalty. We keep four quantities apart, because the
paper's central measurement is about which of them a reported number divides by:

> f(x) = λ·xᵀΣ̂x − (1−λ)·μ̂ᵀx (the portfolio objective),

> V(x) = (Σᵢ xᵢ − K)² (the constraint violation),

> H_A(x) = f(x) + A·V(x) (the penalized objective the QUBO encodes),

> E_A(x) = H_A(x) − A·K² (what xᵀQx returns once the expansion's constant is dropped).

Let F be the feasible set, the assignments with Σᵢ xᵢ = K. On F the penalty vanishes, so H_A(x) =
f(x) and E_A(x) = f(x) − A·K²: the −A·K² is a constant the expansion sheds into an offset, not a
penalty paid by feasible solutions. Every energy the results report is E_A, and every gap
denominated by |E_A\*| therefore carries that constant. The QAOA circuit sees a fifth quantity, the
Ising form of H_A rescaled by its largest coefficient (below); we write ⟨H_norm⟩ for its expectation
and never interchange it with H_A or E_A, nor compare it between runs at different A.

μ and Σ are normalized by their max-abs value (μ̂, Σ̂ ∈ [−1,1]) so the penalty A has a
dataset-independent meaning. Two consequences are stated now because §V-D reads a financial metric
off the same x. The portfolio evaluated there is the equal-weight one, w = x/K. And because μ and Σ
are scaled separately, the relative weight of return and risk in f is not the one λ = 0.5 would give
on the raw financial units: f is what is optimized, and the Sharpe ratio is measured afterwards on
the raw series. And because w = x/K, the objective on a feasible x reads f = λK²·wᵀΣ̂w −
(1−λ)K·μ̂ᵀw, so a fixed λ weighs risk against return differently at each K: the sizes are not the
same financial risk preference. We use λ = 0.5 throughout. Expanding the penalty yields an
upper-triangular QUBO matrix Q with

- diagonal: Q_ii = λ·Σ̂_ii − (1−λ)·μ̂_i + A·(1 − 2K),
- off-diagonal: Q_ij = 2λ·Σ̂_ij + 2A,

and a constant offset A·K². For QAOA the QUBO is mapped to an Ising Hamiltonian by x_i = (1 −
z_i)/2, giving linear (h) and quadratic (J) coefficients plus a constant. This layer is
device-independent and is unchanged whether the solver is classical or quantum.

*The circuit.* With s_H = max(max_i |h_i|, max_{i<j} |J_ij|) and the normalized cost Hamiltonian

> H_C = (1/s_H) · ( Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j ), H_M = −Σ_i X_i,

the state at depth p is

> |ψ(γ, β)⟩ = ∏_{ℓ=p..1} e^{−i β_ℓ H_M} e^{−i γ_ℓ H_C} · H^{⊗N}|0⟩^{⊗N},

layers applied cost-first, and the objective minimized is ⟨ψ(γ, β)| H_C |ψ(γ, β)⟩ = ⟨H_norm⟩. Terms
with |coefficient| ≤ 10⁻¹² are dropped before the circuit is built. Each exponential is one Trotter
step of the exact commuting decomposition (every term of H_C is diagonal, and H_M's terms commute
among themselves), giving p·[C(N,2) + 2N] parameterized rotations. Angles are drawn once per run as
γ_ℓ, β_ℓ ~ U(0, π) from the run's seed and are unbounded during optimization. We use p = 2 for E1–E5
(the depth sweep of Supplementary Sec. S-VI is the exception), so there are four variational
parameters. The rescaling by s_H leaves argmin unchanged and reported energies use the original Q,
but it is not neutral to the search: on the circuit it is the reparameterization γ → γ/s_H, so the
same initial-angle range explores a different region of the landscape than the unnormalized problem
would.

### B. Penalty weights

Three reference weights recur. The heuristic the formulation uses, and the one we call A_heur, is
the "make the penalty dominate the objective" rule

> A_heur = 2·s_obj·N + 1, s_obj = max(λ·max|Σ̂|, (1−λ)·max|μ̂|);

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
and no tie occurs at any A ≥ 0. All 9 of the 30 N = 12 instances with A_crit = 0 are the
third case, with inner maxima from −9.0·10⁻² to −5.9·10⁻⁴, so for them the multiples of A_crit
collapse to the single condition A = 0. Ties are decided at `OPT_TOL` = 10⁻⁹ relative to Δ_F, not by
floating-point equality. For N ≤ 20 we compute a_m exactly over all 2^N assignments; for N > 20 we
bound the cardinalities far from K and call the result A_safe, an upper bound on A_crit.

A_margin follows Brandhofer et al. [8] and places the best infeasible assignment at the midpoint
between f\* and the feasible-set mean f̄_F,

> T = (f\* + f̄_F)/2, A_margin = max(0, max_{m≠K} (T − a_m)/(m − K)²) + ε_A,

with ε_A = 10⁻⁶ recorded per instance. At A_margin every infeasible assignment costs at least T −
f\* more than x\* does, up to ε_A; the best infeasible assignment sits at T (before ε_A is added)
when the inner maximum is nonnegative; when it is negative every infeasible assignment already
exceeds T at A = 0. Both A_crit and A_margin use exact information about the instance; they are
reference conditions, not a procedure we claim scales. Computing them took 0.06 s, 0.8 s and 10.3 s
per 30 instances at N = 12, 16 and 20.

### C. Evaluation quantities

All scores are computed by an evaluator that reads x, μ̂, Σ̂, λ, K and A, and never reads Q. On
every one of the 2^N states at N = 8 and N = 12 it agrees with the QUBO path on H_A, E_A, the Ising
coefficients and the bit order (E0, §IV-C). With f\* = min_F f, f_F^max = max_F f and the
feasible-range spread Δ_F = f_F^max − f\*, the quantities are those of Table I.

**Table I. The evaluation quantities.** Every score in §V is one of these, computed by the evaluator
from x, μ̂, Σ̂, λ, K and A.

| Quantity | Definition | Role |
|---|---|---|
| Offset-normalized gap g_off | (f(x) − f\*) / \|f\* − A·K²\|, feasible x | the gap taken on the solver's own energies; analyzed, not used for judgment |
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
success line. The two are compared as decision rules in §V-D rather than treated as equivalent.
Three denominators can vanish, and the evaluator returns the affected quantity as undefined rather
than as a number: Δ_F = 0 for g_F (the within-τ test then counts the whole feasible set as
qualifying), |f\* − A·K²| = 0 for g_off and D(A) together, and P_F = 0 for D_cond. No setting in
Table II reaches any of them. The identity D(A) = g_F / g_off holds for non-optimal x; at the
optimum both gaps are 0. Runs in which any of these occurs are counted rather than dropped.

Under i.i.d. shots Q_τ(S) is a formula in P_τ, so shot budgets are compared without re-running QAOA;
on hardware, whose shots may be correlated, it is a model. Two consequences follow. It is monotone
in P_τ at every S, so one sampler exceeding another at several budgets is one comparison reported at
several operating points, decided on P_τ. And Q_τ(S) is the probability that a batch contains at
least one qualifying feasible solution; under the output rule used here it is also the probability
of *returning* one. The rule selects the lowest-E_A shot among the feasible shots (ties to the
lowest basis index), a batch with no feasible shot is a counted outcome rather than a missing value,
and g_F is not defined for an infeasible x. Because E_A and f induce the same ordering on F, the
returned shot has g_F ≤ τ exactly when the batch holds a feasible shot with g_F ≤ τ. An infeasible
shot of lower E_A cannot displace it, so the equivalence survives the weak penalty weights at which
the batch minimum is infeasible. It need not hold for a rule that selects the minimum-energy shot
without first checking feasibility (`tests/test_run_e1_cross.py` pins each case). Within one
instance at one A, g_off and g_F are positive multiples of each other and preserve the order of
solutions; what D(A) changes is every comparison across A, across N and against a threshold.

Two identities relate the conditional quantities. Write u_τ = |F_τ| / |F| for the share of the
feasible set within τ, and q_τ = P_τ / P_F for the sampler's conditional hit rate. Then |q_τ − u_τ|
≤ D_cond, and the sampler's one-shot success relative to uniform feasible sampling factors as P_τ /
P_τ^{uniform-F} = P_F · R_τ with R_τ = q_τ / u_τ. The first factor is the chance of landing in F,
the second the preference for good solutions within it. Neither quantity replaces the comparison.
D_cond has no direction: it does not say whether mass moved toward or away from F_τ. A small D_cond
does not guarantee that R_τ is close to one when u_τ is small (|R_τ − 1| ≤ D_cond / u_τ).
Comparisons against uniform-F are therefore made on P_τ directly, with D_cond and R_τ read as their
explanation.

## IV. Experimental Protocol

### A. Data and problem instances

Nikkei 225 constituents via `yfinance`, a three-year window pinned in code to 2023-08-06 –
2026-08-05 (731 trading days), annualized returns and covariance. For each N the target cardinality
is K = max(2, ⌊N/5⌋). The universe is a fixed list of 50 Nikkei 225 constituents, chosen by the
author for liquidity and availability on `yfinance` rather than by a numerical ranking rule, and
written into the source code in the order listed there before any experiment was run. All 50 have a
complete daily series in both windows (731 and 735 trading days), so no ticker was excluded for
missing data. Instances are subsets of that universe, drawn from a single cached price snapshot.
Instance 0 is the *reference instance*, the subset that the single-instance tables use: the first N
tickers in the listed order. Instance i ≥ 1 is an N-subset drawn without replacement from the 50 by
a generator seeded with i, one stream per index, reused at every size. Instances with the same index
at different N are therefore not independent draws, since they share 4–7 tickers where independent
draws would share 3.8–6.4 in expectation. Instances with different indices are independent draws of
subsets from the fixed snapshot, which is not the same as independent market observations. There is
no exclusion between instances and no de-duplication. Every instance's sorted ticker list, and a
digest of it, are recorded with its results (`paper01_e1_reference.csv`). All E0–E4 instances share
one market period and one asset class, so the spread across them measures subset sensitivity rather
than robustness across market periods; Supplementary Table S19 repeats the main contrast without
instance 0. E5 repeats a pre-fixed block on the adjacent, non-overlapping window 2020-08-05 –
2023-08-05 (735 trading days), from a second cached snapshot of the same 50 tickers.

### B. Solvers and backends

- **Classical reference.** Exact enumeration of the *feasible* set for N ≤ 30 (C(N, K) candidates,
  5.9·10⁵ instead of 1.07·10⁹ at N = 30, so a proven optimum in 5.7 s); the reference solver uses
  simulated annealing [44] only at N = 50 (the annealing arms of §V-D are controls, not the
  reference). Every gap reported for N ≤ 30 is measured against a proven optimum.
- **QAOA baseline.** Depth p = 2, ADAM [45] with step 0.1 for 50 steps from uniform random initial
  angles in [0, π) drawn from a named seed, gradients by adjoint differentiation [40], 1,000 final
  measurement shots, on `lightning.gpu` at single precision. It is the *baseline* of E3 and of the
  single-seed measurements in the Supplementary Material, not a claim about the best QAOA.
- **Simulation host.** RTX A6000 (48 GB VRAM, driver 570.133), Python 3.11, PennyLane 0.45.1 [38],
  pennylane-lightning-gpu 0.45, custatevec-cu12 1.14 [39]. Every number in §V comes from this host
  unless a table says otherwise, and numbers from a second machine are never mixed into a table. The
  same deterministic classical solver's Sharpe ratio differs by 3.4·10⁻¹⁶ between the Linux host and
  an Apple-silicon laptop, a BLAS difference.
- **Cloud and hardware backends.** The identical code, selected only by a backend name, on AWS
  Braket SV1 (managed statevector simulator, double precision, us-east-1) and, for the final
  sampling task only, on IonQ Forte-1 (the optimization loop never runs on hardware).

### C. The experiments

Four factors are varied, and they answer different questions. An *instance* is a ticker subset, so
the spread over instances measures problem dependence within one universe and one period; an
*initial-angle seed* repeats the same problem from a different initialization; a *penalty setting*
changes the weight A of the same problem, which moves both the reported metric and the distribution
the sampler draws from; the *second price window* (E5) re-runs the same subsets on a different input
period. Thirty instances × five seeds are thus 30 problems each run five times, not 150
problems, and are aggregated as such (§IV-E).

**E0 — evaluator and re-scoring.** The evaluator of §III-C was checked against the QUBO path on
every one of the 2^N states at N = 8 and N = 12: H_A = E_A + A·K² on all states, H_A = f on F, the
same per-cardinality minima, ties and worst feasible value, the same bit order, and the exact A_crit
never above the bounded A_safe. Fixed feasible solutions of each reference instance were then
re-scored without re-running QAOA. Those solutions are the optimum, the solutions of rank 2, 5, 10,
50, 100 and 1,000, the median-rank solution, the worst feasible solution, and the single-seed QAOA
solution of Supplementary Table S2. Each was scored under g_off, g_F and the all-states
approximation ratio of [22], as only A takes the values 1.1, 2, 5 and 10 × A_crit, A_margin and
A_heur (354 rows).

**E1 — instance × seed × penalty cross.** N ∈ {12, 16, 20}; 30 instances per size; initial-angle
seeds 42–46; six penalty settings {1.1, 2, 5, 10} × A_crit, A_margin, A_heur; the baseline solver;
1,000 shots. For every run the initial and final angles, the optimization trajectory, the exact
probabilities of every feasible bitstring under the initial and the final state, the per-cardinality
mass and the shot counts are retained (§IV-D). The nominal design is 2,700 runs; the nine N = 12
instances with A_crit = 0 collapse their four multiples into one condition A = 0, so 2,565 were
expected and 2,565 completed (Table II). Instance 0 at A_heur reproduces an earlier five-seed sweep
and the seed-42 A_heur arm an earlier 90-run instance sweep, to the bitstring (energies within
4·10⁻¹⁶ relative, the ulp of the two evaluators), so those sweeps are not reported separately.

**E2 — what the best shot means.** From the retained states of every E1 run, four samplers are
compared at S ∈ {10, 100, 1000, 10000}: the optimized QAOA state; the *unoptimized* state at the
same initial angles; the uniform distribution over all 2^N bitstrings; and the uniform distribution
over F. P_F, P_τ and D_cond are read off the exact state, and Q_τ(S) follows by the formula. The
uniform-over-F sampler draws K distinct assets uniformly without optimizing the objective. It is a
control that uses the constraint directly, matched to QAOA on the number of candidates generated
rather than on computational cost, and exact enumeration is used to score its success probability,
not to define it. No optimization is re-run.

**E3 — optimizer controls.** A block of 300 runs was fixed before any E3 result was seen (instances
0–9, seeds 42–46, {A_heur, A_margin}, at each N), and re-run under two other settings: **A**, ADAM
with step 0.03 for 100 steps; **B**, SciPy's L-BFGS-B [46], [47] on the analytic expectation with
adjoint gradients, stopped at SciPy's own criteria or at 100 gradient evaluations, whichever comes
first. Each run pairs with its E1 baseline on (N, instance, A, seed) and the same initial angles, so
⟨H_norm⟩ is comparable within a pair. Neither setting is proposed as a better QAOA; the block asks
whether the baseline's results are an artifact of one weak setting.

**E4-A — the same angles on four backends.** Twelve circuits were fixed before any E4-A result was
seen: N ∈ {12, 16, 20} × instances {0, 1} × seeds {42, 43} at A_heur, taking the final angles of the
stored E1 runs. They are evaluated without optimization on `lightning.gpu` at double precision (the
reference), `lightning.gpu` at single precision (the precision of every E1 run), `lightning.qubit`
at double precision on one thread, and SV1 at double precision. Each arm returns the exact
probabilities of the output distribution and 10 independent batches of 1,000 shots, each batch a
separate circuit execution and, on SV1, a separate task. The probabilities are the full 2^N vector
locally; on SV1, which refuses the `Probability` result type at zero shots, they are the
feasible-set amplitudes from one analytic task. Locally the sampler is re-seeded per batch through
the device's own generator (§VI-C), and SV1's sampler cannot be seeded. The design answers two
questions: whether the output distribution at fixed angles is the same, and what repeated sampling
from it returns. It does not answer whether an optimization run on SV1 reaches the same angles.

**E5 — the E1 block on a second price window.** The E3 block (300 runs) was re-run with the baseline
solver on the window that immediately precedes the paper's, from a second cached snapshot of the
same 50 tickers under the same instance-drawing rule; the feasible set, A_crit, A_margin and the
uniform-sampler baselines are recomputed for the new window, which was fixed before any E5 result
was seen. The question is whether the direction of the main contrast, the deflation, the
disagreement of the two success rules and the E2 conclusion depend on the market period the
instances were drawn from. It is a check under a second input condition, not a backtest.

**E6 — the constraint in the ansatz instead of the cost.** Every run above carries the cardinality
constraint in the cost, as a quadratic penalty, and lets the circuit put amplitude anywhere in 2^N.
E6 reverses that choice and changes nothing else. The initial state is the Dicke state |D^N_K⟩ =
C(N, K)^{−1/2} Σ_{|x| = K} |x⟩, of equal amplitude on every feasible bitstring. The cost is the
objective alone, A = 0, normalized by its own s_H as every other block is. The mixer layer is a ring
of XX + YY terms, each of them exp(i (β/2) (X_i X_j + Y_i Y_j)) and each commuting with the total
Hamming weight [11], [12]; the pair order, the one Trotter sweep it amounts to and the gate the
implementation calls are in Supplementary Sec. S-X. The feasible set is then invariant under the
whole ansatz, so P_F = 1 holds by construction rather than by optimization, and what the angles can
still change is the distribution *within* F — the quantity §V-A measures and §V-C tracks. ε_F is not
1 here: with no penalty, s_H is the objective's own largest Ising coefficient, and Δ_F / s_H is
1.27, 1.67 and 2.26 in the instance median (Table III).

The rest is the baseline: 30 instances per size, five initial-angle seeds, ADAM at 0.1 for 50 steps,
1,000 shots, the same evaluator and the same output rule. "The same budget" throughout means those
three: depth, optimizer steps and shots. It does not mean the same circuit resources: the Dicke
preparation, a known construction [48] set here as amplitudes, and the N pair gates of each mixer
layer are extra to the penalty ansatz's rotation count (§V-G). A second block of 450 runs changes
only the cost: A = 0 with the baseline transverse-field ansatz, the uniform state over 2^N and the
transverse mixer exp(i β Σ_i X_i). It separates the two halves of the change, since E6 moves the
constraint into the ansatz *and* takes the penalty out of the cost at the same time.

**Earlier single-seed series.** The single-instance benchmark at N = 8–30, the depth sweep and the
steps × p × seed grid, the SV1 cross-backend, matched-precision and repeat runs, the four IonQ
Forte-1 tasks, and the classical annealing arms are single-seed or single-instance measurements, and
are read as such. Their tables are in the Supplementary Material, and the main text keeps a
paragraph on each.

### D. Reproducibility of the inputs, and the record

All conditions are in code, seeds are named and the window is pinned. Four conditions had to be met,
each learned by losing results to it. (i) Prices are pinned by *value*, not by date: `yfinance`
revises adjusted closes between calls, so the raw series are cached at full precision and read back
with round-trip parsing. The snapshots are third-party data and are not redistributed. The released
package carries what the pipeline actually reads: the annualized μ and Σ of each 50-ticker window,
≈54 kB each. Every instance's μ and Σ is a slice of those, bit for bit, so every number reproduces
from a clone with no network access. (ii) Every instance slices one snapshot, so no two sizes see
different revisions of a price. (iii) Statistics are computed on a C-contiguous float64 copy,
because `DataFrame.cov()` sums in memory-layout order. (iv) `lightning.qubit` is run at
`OMP_NUM_THREADS=1`, without which it drifts by ~10⁻¹³ between runs; `lightning.gpu` is bitwise
reproducible regardless. Figures are regenerated from the committed CSVs with timestamps suppressed.
The failures behind each rule — including the 10⁻¹⁰-level input changes that single-precision QAOA
amplified into different solutions — are in Supplementary Sec. S-IX.

Every run is recorded as one CSV row (initial and final angles, evaluation counts, shot counts,
exact-state scores, wall-clock and, on metered backends, task identifiers) and every instance as one
row with its reference solution, A_crit, A_margin and the uniform-sampler baselines. What E1 retains
per run is not the full state vector: 2,565 dense initial and final vectors at these sizes would
occupy about 16 GB. It keeps the exact probability of every feasible bitstring under the initial and
the final state, the per-cardinality mass, the angles, the trajectory and the shot counts, ≈100 MB
compressed. The E6 block and its control keep the same record for their 900 runs, ≈34 MB. Every
E2 quantity, D_cond included, is recomputed from those, and the full vector for any run is
reconstructed from its recorded angles and seed in seconds.

### E. Statistics

The unit of generalization is the instance. Initial-angle seeds are repeats within an instance, and
shots and offline draws are neither. A paired contrast (treatment − control on the same instance,
seed and, for E3, initial angles) is first reduced to one value per instance by the median over
seeds. The instance-level values are then bootstrapped by resampling instances (2,000 draws,
percentile 95 % interval), with the number of instances on which the treatment was better, tied or
worse. Two conditions on what the intervals estimate are stated rather than left to be inferred.
Because the 30 subsets share one market period, every interval is conditional on that universe, that
window and that drawing rule. And because every instance uses the same five seeds, the
instance-level value is a contrast *summarized by the median over that fixed set of initial angles*:
resampling instances propagates the uncertainty in which subsets were drawn, not the uncertainty in
which angles were. Supplementary Table S19 shows what that costs. At each of the two larger sizes,
one of the five seeds reverses the sign of the main contrast on its own, while dropping any one seed
leaves it unchanged, so a claim about initializations drawn from a wider distribution would need a
design that samples them. The primary contrast is A_margin against A_heur, and the primary
acquisition quantity is Q_0.01(1000). P_F, g_F of the best feasible shot, the rate of exact optima
and wall-clock are reported beside it, and 5 × A_crit against A_heur is a secondary contrast on the
instances with A_crit > 0. All other cuts are exploratory. Failed runs, runs without a feasible
shot, and runs whose lowest-energy shot is infeasible are counted rather than dropped. A contrast on
g_F is formed only where both sides have a feasible shot, and the number of such pairs is printed
next to the estimate. The design was not sized from a minimum detectable effect. The instance count
was fixed by the compute budget of the cross rather than by a power calculation: 30 subsets per
size, with five seeds and six penalty settings, is 2,565 runs and 13.9 h on one GPU. The resolution
that affords is shown with every contrast as "better in k of 30", and five seeds show that a
within-instance tail exists, not how probable it is.

### F. What ran

**Table II. Experiment counts, execution environments and cost.** E1 rows: nominal is the full
cross; "expected" applies the A_crit = 0 collapse (nine N = 12 instances run three conditions rather
than six); "failed" counts execution failures (a run that raised or wrote no result), not optimizer
stopping reasons — an L-BFGS-B line-search failure is a recorded termination, not a failed run
(Supplementary Sec. S-VI). "No feasible shot" counts completed runs whose 1,000 shots held no
feasible bitstring; "min-energy shot infeasible" counts runs whose lowest-energy shot violated the
constraint although a feasible shot may exist. Time is wall-clock on the host of §IV-B (E1 includes
0.06, 0.82 and 10.3 s of reference enumeration per size); metered charges are AWS list price as
billed, and a "—" in that column is an unmetered local run, not a zero cost. The earlier single-seed
series are listed in Supplementary Table S33.

| Experiment | Runs nominal / expected / done / failed | No feasible shot | Min-energy shot infeasible | Environment (precision) | Time | Metered tasks | Metered charge |
|---|---:|---:|---:|---|---:|---:|---:|
| E1, N = 12 / 16 / 20 (K = 2/3/4, \|F\| = 66/560/4,845; 9 of 30 N = 12 instances have A_crit = 0) | 2,700 / 2,565 / 2,565 / 0 | 0 / 56 / 1 | 8 / 124 / 95 | lightning.gpu (single) | 13.9 h | — | — |
| E0 re-scoring (354 rows) and E2 (from E1 states) | 0 new runs | — | — | CPU (double) | minutes | — | — |
| E3, ADAM 0.03 × 100 | 300 / 300 / 300 / 0 | 7 | 8 | lightning.gpu (single) | 3.2 h | — | — |
| E3, L-BFGS-B ≤ 100 gradients | 300 / 300 / 300 / 0 | 10 | 12 | lightning.gpu (single) | 2.3 h | — | — |
| E5, second window | 300 / 300 / 300 / 0 | 11 | 13 | lightning.gpu (single) | 1.6 h | — | — |
| E6, constraint in the ansatz (XY mixer, A = 0) | 450 / 450 / 450 / 0 | 0 | 0 | lightning.gpu (single) | 3.5 h | — | — |
| E6 control, A = 0 with the baseline transverse-field ansatz | 450 / 450 / 450 / 0 | 0 | 406 | lightning.gpu (single) | 2.5 h | — | — |
| E4-A, local arms (3 arms × 12 circuits × 11) | 396 / 396 / 396 / 0 | 46 batches | — | lightning.gpu / .qubit (double, single) | 3 min | — | — |
| E4-A, SV1 (12 circuits × 11 + 2 pilot) | 134 / 134 / 134 / 0 | 14 batches | — | Braket SV1 (double) | 21 min | 134 | $0.50 |
| IonQ Forte-1, 100 shots per task | 4 / 4 / 4 / 0 | 0 | — | Forte-1 | — | 4 | $33.20 |
| Earlier SV1 series (Supplementary Sec. S-VII) | 30 | — | — | Braket SV1 (double) | — | ≈1,300 | ≈$4.2 |
| Deployment and smoke test | — | — | — | SV1 | — | 1 | $0.004 |

The Braket total is $37.90 against a $50 budget.

## V. Results

The sections follow the question rather than the order the experiments were run. §V-A asks what
optimization buys at one weight and §V-B whether that reaches a sampler which only respects the
constraint. §V-C asks what sets the difference between weights and §V-D what the reported number
does over the same range. §V-E asks whether another optimizer setting or another price window
changes it, §V-F what a second draw from one distribution returns, and §V-G what the whole
measurement costs in resources. §V-A to §V-C and §V-E are the cross and its controls; §V-D is
re-scoring and enumeration, for which no QAOA is re-run; §V-F is fixed-angle evidence from four
backends and single-seed evidence from metered ones, read as such.

### A. What optimization changes at the baseline weight (E1)

*Within the feasible set, the heuristic-weight sampler is close to uniform.* P_τ factors as P_F ·
q_τ with q_τ = Pr[g_F ≤ τ | x ∈ F]. At A_heur the second factor is the share u_τ = |F_τ|/|F| that a
uniform draw from F would give, and the ratio R_τ = q_τ/u_τ is 0.999, 0.998 and 1.009 at N = 12, 16
and 20, with interquartile ranges of 0.02 or less (Supplementary Table S17). That ratio, however,
only checks the mass on one subset. A distribution such as (½, 0, ½, 0) on a four-element F has R_τ
= 1 on a two-element F_τ and is not uniform. The retained probabilities of every feasible bitstring
were therefore used to compute the conditional distance D_cond of Table I directly, per run (Table
III; Supplementary Table S18). At A_heur it is 0.001–0.002 in the median at every size, and never
above 0.038 (N = 12), 0.009 (N = 16) or 0.005 (N = 20) in 450 runs. For comparison, a point mass
sits at 0.985–1.000, and a 1,000-shot empirical draw from the uniform distribution itself would show
0.10–0.81 at these |F|. Small is not zero: the distribution is *near*-uniform, and because D_cond
carries no direction, the departures from uniform that it measures are read through the R_τ above.
At A_margin the conditional distribution departs further from uniform (D_cond 0.17, 0.14 and 0.09 in
the median). Enrichment of the primary quality set is clear at N = 12 and 16 (median R_τ 1.40 and
1.55), whereas at N = 20 the median R_τ stays close to one (1.01) although the distribution is not
uniform. The multiples of A_crit depart more the closer A is to A_crit (D_cond 0.37–0.49 at
1.1·A_crit).

*What optimization does at a fixed weight, and what the weight does.* The two are separated by
pairing each run's final state with its initial state (Supplementary Table S34). At A_heur,
optimization raises the feasible probability P_F: P_F goes from 0.114, 0.035 and 0.087 in the median
initial state to 0.179, 0.229 and 0.167, a paired median increase of +0.070, +0.128 and +0.134 on 30
of 30 instances at N = 12, 16 and 20. The expected constraint violation E[V] falls with it, from
21.9, 34.8 and 45.7 to 10.6, 4.0 and 14.3. The conditional distribution stays where it started:
D_cond moves from 0.002–0.004 to 0.001–0.002 in the median, and the initial states that were as far
as 0.41 from uniform on F for some seeds end within 0.005 of it in every N = 20 run. At A_heur,
optimization therefore raised the feasible probability P_F with little systematic enrichment of the
primary quality set.

Separately, changing the weight from A_heur to A_margin after optimization raises the median P_F
from 0.167 to 0.303 at N = 20 (Table IV). At A_margin, optimization raises P_F as well (+0.18, +0.22
and +0.24), and the final conditional distribution departs from uniformity, with the enrichment of
Table III at N ≤ 16. As a complementary multiplicative summary of the same matched runs, P_τ = P_F ·
u_τ · R_τ on each (instance, seed) pair splits the geometric-mean ratio of P_τ into a feasible-mass
factor and a conditional-enrichment factor. The mass accounts for 60 %, 21 % and 70 % of the mean
log-ratio at N = 12, 16 and 20, and the R_τ ratio exceeds one on 113, 125 and 88 of the 150 pairs
(Supplementary Table S38). Those shares are of the mean log-ratio, so they complement the median
contrasts of Table IV rather than decomposing them, and the geometric-mean R_τ ratio at N = 20 is a
different statistic from the median R_τ of Table III.

The paired increases at A_heur are nearly the same on every instance (interquartile range ≤ 0.001),
which is what a circuit dominated by the permutation-symmetric penalty term would give. With f
removed, the cost, the mixer and the initial state are all invariant under qubit permutations, so
every bitstring of one cardinality has the same probability at any angles, and the conditional
distribution on F is exactly uniform. The measured ε_F(A) = Δ_F / s_H puts the objective's feasible
range in units of s_H, the largest Ising coefficient of the cost before it is divided by s_H
(§III-A); equally, ε_F is the spread of the normalized cost over F. In the instance median it is
0.015, 0.015 and 0.016 at A_heur, with the objective's own largest Ising coefficient at 0.7–1.2 % of
s_H (Supplementary Table S35), against 0.45–0.52 at A_margin and 1.27–2.26 at A = 0, where the
objective is the whole cost. That scale is consistent with the near-uniformity, and it is offered as
context rather than as a demonstration: the symmetry argument is exact only for the penalty alone,
and how far a 1.5 % objective perturbation is resolved by a given optimizer at finite A is what
D_cond measures. Most of the A_heur state's mass remains far from cardinality K after optimization,
and P_F is what best-of-shots has to rescue.

*The seed axis, within every instance.* The within-instance spread over five seeds of Q_τ(1000) has
a median over instances of 0.86, 0.49 and 0.19 at A_heur, which is most of the quantity's range at
the two smaller sizes. At N = 16 and A_heur, 14 of 150 runs returned no feasible bitstring among
1,000 shots (9 %; under 1 % at N = 12 and N = 20), a within-instance tail rather than a size trend.
Across settings, the lowest-energy shot is infeasible in 8, 124 and 95 runs (Table II), and in 55 %
of N = 20 runs at 1.1·A_crit. At that setting the best *feasible* shot nevertheless has the smallest
g_F of any setting (−3.3 points of Δ_F against A_heur, 29 of 30 instances), so the selection rule of
§III-C is not a formality.

### B. Comparison with uniform feasible sampling (E2)

*What the best shot means (E2; Fig. 1).* The same exact states answer the question that the
best-of-shots rule begs: how often does a batch of S shots contain a solution within τ, and how does
that compare with not optimizing at all? Optimization *works*, in the sense that the literature's
convergence curves show. At A_heur the optimized state raises Q_τ(1000) over the unoptimized state
at the same initial angles, from 0.062 to 0.336 at N = 16 and from 0.019 to 0.059 at N = 20, and it
beats the uniform-over-2^N sampler by one to two orders of magnitude. Its one-shot success
probability at the primary threshold τ = 1 %, however, never exceeds that of a uniform draw from the
feasible set: P_τ < P_τ^{uniform-F} in 450 of 450 A_heur runs, and therefore at every S from 10 to
10,000. That is one direct comparison of P_τ, shown at four operating points because Q_τ(S) is
monotone in P_τ. It is a measured inequality rather than a consequence of the small D_cond above
(§III-C), and it holds at the two sensitivity thresholds as well, 0 of 450 at the exact optimum and
at τ = 5 % (Supplementary Table S14).

At S = 1,000 and N = 20, a uniform feasible draw succeeds with probability 0.34 and the optimized
state with 0.06. A benchmark that reported its best-of-1,000 gap would not show the difference at N
= 12, where both samplers usually contain a good shot (median Q_τ(1000) 0.93 and 1.00). At N = 16
the gap between them is already wide (0.34 against 0.83), and at N = 20 neither sampler reliably
contains one (levels at every S: Supplementary Table S14). The comparison depends on the penalty
setting: at A_margin the optimized state beats the feasible-uniform sampler in 7–9 % of runs, and at
1.1·A_crit in 41 % of N = 20 runs, so the 0 of 450 is a result at A_heur. At these sizes and
budgets, then, a best-of-shots figure alone cannot confirm that the optimization contributed, while
the exact-state comparison with explicit control samplers can. What that comparison shows at A_heur
and the evaluated thresholds is that optimization improves acquisition relative to the unoptimized
state, and that the resulting sampler stays far better than an uninformed one and below uniform
feasible sampling. The uniform-over-F sampler is not a computational peer of QAOA (§IV-C). It is the
baseline that a constraint-respecting method has to clear, and the classical annealers of §V-D clear
it by a wide margin.

*Multistart at equal shots.* The five E1 seeds can be re-read as a five-start strategy: optimize
five times, draw 1,000 shots from each state, and keep the best feasible candidate of the 5,000. The
comparison is with what a user can run at the same shot budget, which is to optimize once from a
random seed and sample 5,000 times, with Q computed per seed and then averaged. By the AM–GM
inequality the pooled strategy is never worse at equal shots, so the question is the increment
against four extra optimizations. It is +0.046 to +0.336 in the instance median over the six cells,
above 0.01 on 30 of 30 instances in every cell but one, for 41–118 s of extra optimization per
instance (Supplementary Table S24). Five initial angles do buy acquisition probability that five
thousand shots from one start do not. None of the strategies reaches uniform sampling from the
feasible set at the same 5,000 candidates, nor the annealing baseline of §V-D.

### C. What sets the preference: the weight, and where the constraint lives (E1, E6)

The two factors answer for one weight. The cross ran six, and the weight is the one formulation
constant that moves them: Supplementary Fig. S13 shows the four quantities of §III-C against the
penalty setting, one series per size. What the weight sets is how wide the objective's feasible
range is next to s_H, which is the ε_F = Δ_F / s_H of §V-A and equally the spread of the normalized
cost over F, and the preference inside the feasible set follows it.

**Table III. Feasibility, conditional preference and sampling success across penalty settings and
ansatzes.** The threshold is τ = 1 % of the feasible range throughout. Per (N, A) over the runs of
that cell: 150 runs, 30 instances × 5 seeds, except the positive multiples of A_crit at N = 12,
which have 105 over the 21 instances with A_crit > 0, since A_crit = 0 collapses the multiples to
the same A = 0 setting. Rows are ordered by ε_F = Δ_F / s_H, the objective's spread over F divided
by s_H = max(max_i |h_i|, max_{i<j} |J_ij|), measured per instance (`coefficient_scale.py`) and
taken as the median over the instances the cell ran. ε_F is a ratio and not a share: it passes 1
wherever the objective's range exceeds that coefficient, as the A = 0 rows do. P_F and R_τ are the
two factors of P_τ / P_τ^{uniform-F} = P_F · R_τ (§III-C), as medians over runs; "above uniform-F"
counts runs whose one-shot success probability exceeds a uniform draw's from F, the comparison E2
makes at A_heur; "infeasible minima" counts runs whose lowest-energy shot is infeasible, which the
feasibility-aware output rule of §III-C absorbs. g_F is of the best feasible shot of the 1,000;
D_cond is from the same runs (Supplementary Tables S17–S18 give quartiles and the initial state).
Every column is a median taken separately over the runs of its cell, so multiplying the printed P_F
and R_τ need not reproduce the printed ratio. |F| is 66, 560 and 4,845. The last six rows are E6
(§IV-C). In both of them the cost is the objective alone, so their ε_F is the A = 0 value, measured
on the same 30 instances per size; D_cond is computed from that block's own artifacts. They differ
in the ansatz, and therefore in P_F, which is 1 by construction for the constraint-preserving one
and measured for the published one. The N = 12 cells do not all cover the same instances, so the
monotonicity there is read on the 21 that ran every weight (`paper01_e1_penalty_ladder_common.csv`,
§V-C); at N = 16 and N = 20 the two files agree row for row.

| N | A | ε_F | P_F | R_τ | D_cond | P_τ ÷ uniform-F | Above uniform-F | Q_τ(1000) | g_F | Infeasible minima |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | A_heur | 0.015 | 0.179 | 1.00 | 0.002 | 0.17 | 0/150 (0 %) | 0.928 | 0.000 | 0/150 |
| 12 | 10·A_crit | 0.248 | 0.190 | 1.13 | 0.108 | 0.17 | 3/105 (3 %) | 0.929 | 0.000 | 0/105 |
| 12 | 5·A_crit | 0.396 | 0.244 | 1.23 | 0.207 | 0.26 | 6/105 (6 %) | 0.992 | 0.000 | 0/105 |
| 12 | A_margin | 0.520 | 0.328 | 1.40 | 0.173 | 0.47 | 13/150 (9 %) | 1.000 | 0.000 | 1/150 |
| 12 | 2·A_crit | 0.663 | 0.277 | 1.40 | 0.308 | 0.46 | 10/105 (10 %) | 0.999 | 0.000 | 3/105 |
| 12 | 1.1·A_crit | 0.780 | 0.257 | 1.83 | 0.382 | 0.47 | 12/105 (11 %) | 0.999 | 0.000 | 3/105 |
| 16 | A_heur | 0.015 | 0.229 | 1.00 | 0.001 | 0.23 | 0/150 (0 %) | 0.336 | 0.017 | 14/150 |
| 16 | 10·A_crit | 0.184 | 0.309 | 1.10 | 0.080 | 0.36 | 1/150 (1 %) | 0.488 | 0.017 | 15/150 |
| 16 | 5·A_crit | 0.328 | 0.359 | 1.27 | 0.106 | 0.47 | 6/150 (4 %) | 0.584 | 0.013 | 18/150 |
| 16 | A_margin | 0.455 | 0.301 | 1.55 | 0.139 | 0.42 | 11/150 (7 %) | 0.536 | 0.015 | 14/150 |
| 16 | 2·A_crit | 0.621 | 0.275 | 1.65 | 0.192 | 0.43 | 22/150 (15 %) | 0.553 | 0.003 | 16/150 |
| 16 | 1.1·A_crit | 0.868 | 0.245 | 2.26 | 0.369 | 0.58 | 33/150 (22 %) | 0.661 | 0.000 | 47/150 |
| 20 | A_heur | 0.016 | 0.167 | 1.01 | 0.002 | 0.17 | 0/150 (0 %) | 0.059 | 0.062 | 1/150 |
| 20 | 10·A_crit | 0.176 | 0.219 | 0.97 | 0.036 | 0.30 | 1/150 (1 %) | 0.083 | 0.049 | 0/150 |
| 20 | 5·A_crit | 0.325 | 0.233 | 0.97 | 0.061 | 0.36 | 5/150 (3 %) | 0.085 | 0.048 | 1/150 |
| 20 | A_margin | 0.473 | 0.303 | 1.01 | 0.090 | 0.43 | 10/150 (7 %) | 0.108 | 0.042 | 0/150 |
| 20 | 2·A_crit | 0.663 | 0.224 | 1.55 | 0.198 | 0.45 | 37/150 (25 %) | 0.132 | 0.032 | 11/150 |
| 20 | 1.1·A_crit | 0.976 | 0.238 | 3.10 | 0.494 | 0.80 | 62/150 (41 %) | 0.224 | 0.019 | 82/150 |
| 12 | XY, A = 0 | 1.266 | 1.000 | 4.49 | 0.470 | 4.49 | 145/150 (97 %) | 1.000 | 0.000 | 0/150 |
| 16 | XY, A = 0 | 1.669 | 1.000 | 4.74 | 0.515 | 4.74 | 134/150 (89 %) | 1.000 | 0.000 | 0/150 |
| 20 | XY, A = 0 | 2.258 | 1.000 | 9.89 | 0.535 | 9.89 | 141/150 (94 %) | 0.941 | 0.000 | 0/150 |
| 12 | A = 0, transverse | 1.266 | 0.149 | 3.03 | 0.519 | 0.41 | 38/150 (25 %) | 0.999 | 0.000 | 106/150 |
| 16 | A = 0, transverse | 1.669 | 0.091 | 4.32 | 0.520 | 0.40 | 31/150 (21 %) | 0.556 | 0.000 | 150/150 |
| 20 | A = 0, transverse | 2.258 | 0.058 | 8.45 | 0.539 | 0.45 | 28/150 (19 %) | 0.137 | 0.030 | 150/150 |

*When the circuit orders the feasible set, and when it does not.* Asking E2's question at every
weight that the cross ran makes the near-uniformity at A_heur one end of a monotone relation rather
than an isolated fact (Table III and Fig. 2, from `analyze_ladder.py`). Ordered by ε_F, the share of
runs whose one-shot success probability exceeds a uniform draw from F rises from 0 % at A_heur to 11
%, 22 % and 41 % at 1.1·A_crit (N = 12, 16, 20), monotonically at every size and over six weights.
At N = 12 the six cells do not cover the same instances, because A_crit = 0 on 9 of the 30. On
the 21 that ran every weight the ordering is the same, and the shares are 0 %, 2.9 %, 5.7 %, 7.6 %,
9.5 % and 11.4 %, with R_τ from 1.00 to 1.83 (`paper01_e1_penalty_ladder_common.csv`). At N = 16 and
N = 20 that file equals Table III row for row. R_τ rises with the share: 1.00 to 1.83 at N = 12 and
1.00 to 2.26 at N = 16, while at N = 20 it stays within 3 % of one up to ε_F = 0.47 and then reaches
3.10 at the weakest weight. The best feasible shot's g_F falls overall with the same ordering at N =
16 and 20, though not at every step (Table III). At N = 20, ε_F is 0.016 at A_heur and 0.98 at
1.1·A_crit.

Three facts keep this from being a recommendation. The median one-shot probability stays below that
draw even at the weakest weight (0.80 of it at N = 20), so what rises with ε_F is the share of runs
that cross it, not the median run. A_crit is computed by enumeration (§III-B), so a weight set as a
multiple of it is a reference condition rather than a procedure that scales. And the weakest weight
is where the batch minimum is infeasible most often, in 55 % of runs at N = 20 and 31 % at N = 16,
so it is usable only under a rule that selects the best feasible shot rather than the lowest-energy
one (§III-C). What the ladder does establish is that the preference the circuit expresses inside F
tracks ε_F, and that the dominate-the-objective rule puts the weight at the end of that range where
the preference is absent and the offset-normalized gap reads at its smallest.

*With the constraint in the ansatz, the circuit clears the baseline.* The ladder varies one
formulation constant and keeps the encoding. E6 changes the encoding instead (§IV-C): the Dicke
state, a weight-preserving ring mixer, and the objective as the whole cost. P_F = 1 then holds by
construction, while ε_F is measured at the A = 0 scales like every other row of Table III. It is
1.27, 1.67 and 2.26 in the instance median, above 1 because the objective's feasible range is wider
than its own largest Ising coefficient. Over 450 runs on the same instances and seeds at the
baseline budget, the one-shot success probability is 4.5, 4.7 and 9.9 times that of the uniform
feasible draw in the median (interquartile 3.1–5.8, 2.5–7.5 and 6.0–13.4 at N = 12, 16 and 20), and
above that draw in 145, 134 and 141 of 150 runs. The conditional distribution sits 0.47–0.54 from
uniform on F, where the dominant weight leaves it at 0.002. The 1,000-shot batch returns the exact
optimum in 149, 142 and 111 of 150 runs, against 115, 51 and 4 of 150 at the dominant weight. No
batch of the block had an infeasible lowest-energy shot, because no amplitude leaves F at any angle
(`tests/test_xy_mixer.py` pins that property rather than the numbers). The uniform feasible draw
therefore bounds what this penalty encoding leaves the circuit able to do, not on what a quantum
sampler can do on this problem.

The comparison has two halves, and the control separates them. Removing the penalty from the cost
but keeping the baseline transverse-field ansatz (A = 0, transverse mixer, 450 runs) buys the
*preference*: R_τ is 3.03, 4.32 and 8.45 in the median, within the spread of E6's 4.49, 4.74 and
9.89, and the conditional distribution sits 0.52–0.54 from uniform. What it does not buy is the
*mass*. P_F falls to 0.149, 0.091 and 0.058, since nothing now pushes the state toward F, so the
product stays at 0.41, 0.40 and 0.45 of that draw's and crosses it in 97 of 450 runs. The
lowest-energy shot is infeasible in 106 of 150 runs at N = 12 and in every run at N = 16 and 20,
which the feasibility-aware output rule absorbs (§III-C). The two factors of §III-C therefore moved
apart between these two encodings: with the penalty gone, the preference is there and the mass is
not, and only the product clears the control. That is not a demonstration that the two can be set
independently, since the ladder moves both at once, with P_F varying from 0.167 to 0.359 across the
penalty settings of Table III. It is a measurement of what each encoding gave.

### D. The penalty weight and the reported gap (E0, E1)

The same runs carry the number a benchmark would report, and it moves over this range in the
opposite direction. Two measurements separate what the number does from what the solver does:
re-scoring *fixed* feasible solutions as only A changes (E0, Fig. 3), and the paired contrasts of
the cross at two reference weights (Table IV, by the statistics of §IV-E).

**Table IV. Main effects of the penalty setting.** The two level columns and the contrast column
answer different questions: levels are medians over the runs of a setting, the contrast is the
median of within-instance median paired differences, and the difference of two levels is therefore
not the paired contrast. Paired A_margin − A_heur on the same instance and seed, reduced to an
instance median, then bootstrapped over instances (2,000 resamples, 95 % percentile interval);
"better" counts instances on which A_margin is better, with ties in parentheses. Gap differences are
in percentage points, and a count of pairs is of runs with a feasible shot on both sides, not of
instances. Gaps are of the best *feasible* shot and are formed only where both sides have one (pairs
shown); P_F, P_τ and Q_τ(1000) are exact-state quantities and exist for every run.

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
A_margin makes g_off *worse* in 29 of 30 instances at N = 20 (median +0.81 percentage points) and
23 of 30 at N = 16. Every acquisition quantity moves the other way: P_F, P_τ and Q_τ(1000) are
better in 30 of 30 instances at N = 20 and in 27–30 of 30 at the other sizes. The feasible-range gap
of the best feasible shot is better in 24 of 30 at N = 20 (median −1.6 points of Δ_F) and unchanged
in the median at N = 16 (better in 13, tied in 8, worse in 9). The secondary contrast, 5·A_crit
against A_heur on the instances with A_crit > 0, agrees on Q_τ(1000) (better in 29 of 30 at N = 16,
+0.094 [+0.069, +0.113]; 30 of 30 at N = 20, +0.019 [+0.002, +0.045]) and on the direction of g_off.
It is mixed on the best feasible shot's g_F (worse in 17 of 30 at N = 16, better in 23 of 30 at N =
20). Which weight yields the best single shot and which yields the highest Q_τ(1000) are different
questions.

*What moves the gap is the denominator.* The paired ratio of offset-normalized gaps factors, pair by
pair, into a solution part (f(x_A₂) − f\*)/(f(x_A₁) − f\*) and a denominator part |f\* − A₁K²|/|f\*
− A₂K²|. The identity does not survive a median, so the decomposition is taken in the geometric mean
over the same pairs. At N = 20, on the 128 pairs with a positive gap on both sides, the A_margin run
reports a gap 27.8× that of the A_heur run in the geometric mean (29.0× in the median), of which
36.6× is the denominator and 0.76× is the solution — which is *better*, on 59 % of pairs. The
reported gap moves because the denominator moves, with the solution pulling the other way (all
sizes: Supplementary Table S13).

*Two success rules disagree, and the disagreement belongs to the weight.* Under the
offset-normalized rule g_off ≤ 0.1 %, which is the line this implementation's own tables drew, the
A_heur runs succeed at 81 %, 85 % and 82 % (N = 12, 16, 20); under the feasible-range rule g_F ≤ 1 %
they succeed at 78 %, 40 % and 6 %. The share of runs that pass the first and fail the second is 3
%, 46 % and 76 % at A_heur, 0 % at every other setting at N = 16, and ≤ 5 % at N = 20. Read at
four gap thresholds and three quality thresholds (Supplementary Table S16), the disagreement is
monotone in the gap threshold, from 0.06 to 0.89 at N = 20 and A_heur over 0.01 % to 0.2 %, and it
stays between 0.03 and 0.07 at A_margin throughout. The 0.1 % line is not a threshold at which
anything changes character.

*The same solutions, three denominators (E0; Fig. 3).* Fig. 3 (left) takes one fixed feasible
portfolio of the reference N = 20 instance, the one of median rank, 2,423 of 4,845, and scores it
as only A changes. Its feasible-range gap g_F is 37.8 % at every A, because neither f(x) nor Δ_F
depends on A. Its offset-normalized gap falls from 33.6 % at 1.1·A_crit to 3.9 % at 10·A_crit and
to 0.22 % at A_heur, and the all-states approximation gap 1 − AR of [22] falls from 1.36 % to
0.014 % over the same range. Nothing about the portfolio changed. Fig. 3 (middle) is the deflation
D(A) = |f\* − A·K²|/Δ_F for every size: below 1 near A_crit at N ≤ 12, linear in A, and 57×, 81×,
121×, 174×, 207×, 240× and 279× at the A_heur of N = 8 to 30 (Supplementary Table S3). The
mechanism is the one that §III states. For feasible x the numerator f − f\* is independent of A
while the denominator |f\* − A·K²| is not, so two studies using the same formulation with
different penalty weights report gaps that differ by roughly the ratio of their weights for
identical portfolios. With A = N + 1 and K = ⌊N/5⌋ the constant grows as Θ(N³), while the
objective's measured spread over F rises only from 1.27 at N = 16 to 4.00 at N = 30, against A·K²
rising from 153 to 1,116. The same solution quality reads as a smaller number at larger N.

*Over 30 instances (Fig. 3, right; Supplementary Table S9).* The median deflation at A_heur rises
monotonically with N, from 68× at N = 12 to 282× at N = 30, and no instance at any size escapes it:
the smallest value observed anywhere is 36× and the largest 411×. The heuristic weight sits a median
103–179× above the threshold at every size, so the overshoot is a property of the rule rather than
of one covariance matrix. The threshold is the exact A_crit at N ≤ 20, over the 21 N = 12 instances
with A_crit > 0, and the upper bound A_safe at N > 20, where A_heur/A_safe is a lower bound. On the
reference instance the overshoot is 148–1213×, because its A_crit is unusually small at N ≤ 12 and
the overshoot is a ratio against a threshold that can be near zero. That instance is near the median
in deflation and below it in the Sharpe spread of its 0.1 % band (next paragraph), so the
single-instance results understate the effect rather than exaggerate it.

*Feasible solutions admitted by an offset-normalized threshold.* Exact enumeration makes the set
that a reported gap admits countable. On the reference instances, a 0.1 % offset-normalized band
holds 5.4 % of the feasible set at N = 16, 7.6 % at N = 20 and 24.7 % at N = 30, and the band is a
step function of the threshold: at N = 30 it holds 1.5 % at 0.05 % and 94.7 % at 0.2 %
(Supplementary Tables S4–S5). The solutions it admits are not near-degenerate in any
penalty-independent sense, since at a deflation of 279× the band reaches 28 % of Δ_F from the
optimum, and they are admitted because the threshold was set on a compressed scale. The single-seed
QAOA solutions at N = 20–30, reported at 0.024–0.053 %, sit at 5.7–9.2 % of Δ_F and at ranks 43 to
844, and a fixed universe-wide normalization widens the band rather than narrowing it (Supplementary
Table S6).

The domain metric is an auxiliary consequence rather than a solver error, since the Sharpe ratio is
not what the QUBO optimizes (§III-A). Similar objective values did not guarantee similar Sharpe
ratios. The N = 30 band spans 0.172 to 1.695; the single-seed N = 24–30 solutions sit 15–22 % below
that of the optimum at gaps of 0.024–0.031 %; and of the 47 sub-optimal solutions of the seed-42
sweep at N ≤ 20, 17 are *above* that of the optimum while the worst deviations reach −61 %
(Supplementary Sec. S-IV, Fig. S7). The deviations are of both signs, and the reported gap does not
carry the sign.

*The prescribed all-states ratio.* Applied literally to this QUBO, the range-normalized ratio of
[22] removes the offset but inherits the scale. C_max over all 2^N assignments is attained by a
violating assignment and grows with A, while the spread over F does not, so every feasible
solution's ratio tends to 1. At A_heur the QAOA solution and the *worst* feasible portfolio score
1.000000 and 0.999560 at N = 16 (Supplementary Table S8). [22] calibrates the metric on
unconstrained instances and does not say which range a constrained problem should use.

*The pattern does not need a quantum sampler.* Simulated annealing on the *feasible set*
(cardinality-preserving swaps, five seeds × 30 instances × 6 sizes, 900 runs) reaches the exact
optimum in 895 of 900 runs at a median 3.4 s, so it leaves the metric nothing to hide. The same
annealer on the penalty-encoded Q with single-bit flips, five seeds at N ∈ {12, 16, 20}, reproduces
the QAOA arm's pattern: every run feasible, offset-normalized gaps of 0.00–0.59 %, Sharpe ratios up
to 64 % below that of the optimum, and ranks as deep as 4,307 of 4,845 (Supplementary Table S11).
What produces the pattern is the combination of the encoding's offset with an offset-sensitive
score, and a classical solver scored the same way inherits it. The two arms are not on matched
budgets (50 ADAM steps against 50,000 annealing moves), so this is a check on whether the reporting
critique needs an imperfect solver, not a verdict on which method is better. The seed-42 QAOA arm
over the same 30 instances reaches the exact optimum in 26, 17 and 0 of 30 at N = 12, 16 and 20
(Supplementary Table S12), and the configuration measured here trails the classical baseline on
quality as on runtime.

### E. Another optimizer setting, another window (E3, E5)

The baseline runs ADAM at a fixed step of 0.1 for 50 steps. One objection to §V-A through §V-C is
that a better optimizer would change the picture. The single-seed depth sweep and the steps × p ×
seed grid (Supplementary Sec. S-VI) had already shown three things: that p = 2 is where depth stops
helping monotonically, that ⟨H_norm⟩ saturates by 50–100 ADAM steps and does not improve with an 8×
larger budget, and that a run can barely optimize and still report a small best-of-1,000 gap
(⟨H_norm⟩ = −0.91 against −5.01 in the same cell, gaps 0.038 % and 0.022 %). Which of
initialization, local minima and expressivity produces the non-monotonicity in p is not identified
by those runs. E3 asks the narrower question for p = 2: on a block fixed before it was run, do two
other optimizer settings change the acquisition quantities? The levels, the paired differences with
their intervals, the budgets and the optimum hits are in Supplementary Table S22, and the text below
reports what they show.

The controls did not yield a uniform improvement in acquisition probability. *L-BFGS-B* reduced the
optimized expectation modestly and consistently, below that of the baseline in 59 of 60
instance-cells and at 1.2–1.6× the wall-clock. Its effect on the batch probability Q_τ(1000) is of
both signs and not uniformly small, from +0.009 [+0.004, +0.009] at N = 12 and A_heur to −0.016
[−0.027, +0.002] at N = 16 and A_margin (per-cell levels, intervals and budgets: Supplementary
Tables S22, S23). Its paired median changes to the one-shot P_τ are within ±4·10⁻⁴ in every cell. Q
compounds P over a thousand draws (∂Q_τ(S)/∂P_τ = S(1 − P_τ)^(S−1)), so a change in P_τ of 4·10⁻⁴
can move Q_τ(1000) by a tenth, and a paper that reports only P_τ has not reported the quantity that
its own selection rule uses. "Stopped" is not "converged": the block ran at single precision, and a
diagnostic re-run of a fixed sub-block at both precisions shows every single-precision run that
stopped on a tolerance stopping on the *objective* criterion, several of them after a failed line
search, against 11 of 24 on the gradient criterion in double precision. What the precisions agree
on, and the two pairs that reach different optimization outcomes from the same initialization, are
in Supplementary Table S36 and Sec. S-VI. In both of those pairs P_τ stays below the instance's
uniform-F probability at either precision, so the comparison against uniform feasible sampling keeps
its direction.

*ADAM at step 0.03 with twice the budget* did not improve on the baseline within the tested budget
at N = 20. Its expectation after 100 steps is higher than the baseline's after 50 in 10 of 10
instances at both weights, by 0.32 at A_margin. Its Q_τ(1000) is lower by 0.006 and by 0.106
[−0.187, −0.060], worse on 10 of 10 instances in both cells, which is the largest effect any control
has on any quantity in this block, and it is an optimizer setting. Whether those runs were still
descending or had settled in a different local minimum is not identified by the trajectories, which
record no stopping reason for ADAM.

*The scale of the seed.* The baseline's own spread of ⟨H_norm⟩ over the five seeds *within an
instance* has a median of 3.4–5.5 at every N and both weights, and moves Q_τ(1000) by 0.2–0.9
(§V-A). At this budget and p = 2, a large within-instance variation across initial angles thus
remained, alongside optimizer-dependent changes in acquisition probability. What is compared are
observed ranges: a paired median against a within-instance max − min. It is not a ranking of factor
contributions. Where a control moves a run by order 1, as ADAM at 0.03 does at N = 20, it has
reached a different optimization outcome, in either direction. Whether a different initialization
rule [32], [33] or a larger budget changes the picture is not tested.

*A second market period (E5; Supplementary Table S15).* Everything above is conditional on one
three-year window. E5 re-ran the pre-fixed 300-run block on the adjacent window, where Δ_F is 23–34
% wider at every size and one N = 12 instance whose A_crit is zero in the paper's window has a
positive threshold. The main contrast, the disagreement of the two success rules, and the uniform-F
comparison keep their direction (Supplementary Table S15; the conditional-distribution and initial →
final analyses were not repeated on this window). At N = 20, A_margin against A_heur raises
Q_τ(1000) on 10 of 10 instances in both windows (+0.09 and +0.12), lowers g_F on 9 and on 7, and
worsens g_off on 9 and on 10, with the denominator carrying the sign of the g_off ratio in both. The
two success rules disagree in both windows: 74 % and 66 % of N = 20 A_heur runs pass g_off ≤ 0.1 %
and fail g_F ≤ 1 %. The optimized state beats the uniform-over-F sampler in 0 of 50 A_heur runs per
size in the earlier window, as it does in 0 of 150 in the paper's. The levels move, which is what a
different instance set produces; the signs and orderings do not. This is one further window from the
same universe and asset class rather than a sample of market periods (§VII), and the full comparison
is Supplementary Table S15.

### F. Numerical agreement and sampling variability (E4-A, hardware)

*Numerics or sampling (E4-A; Supplementary Fig. S14, Table S30).* Single runs of the same
configuration on the local GPU and on SV1 return bitwise-identical portfolios at N ≤ 12 and
different feasible portfolios at N = 16 and N = 20. Three submissions of one configuration to SV1
return three different portfolios at N ≥ 16, and matching the precision does not restore agreement
(Supplementary Sec. S-VII). Those single runs cannot say whether the output distribution differs
between implementations or the same distribution was sampled differently. E4-A fixes the angles and
asks both questions separately. What is compared is the exact probability of each computational
basis state rather than the quantum state, since equal basis probabilities leave the relative phases
unconstrained and probabilities are what the selection rule consumes. The comparison is on F ∪ {⊥}:
the feasible bitstrings together with one symbol for "this shot is infeasible", p_⊥ = 1 − P_F. That
p_⊥ is a one-shot quantity. The probability that a whole batch of 1,000 shots holds no feasible
solution is (1 − P_F)^1000, and the two are not the same event.

*The double-precision implementations agree; single precision differs by 10⁻⁷.* On F ∪ {⊥} the
managed cloud simulator's exact probabilities differ from the GPU double-precision reference by at
most 2.6·10⁻¹⁴ in total variation at every one of the 12 circuits, and those of the single-threaded
CPU by at most 4.3·10⁻¹⁵. Single precision, the precision of every E1 run, is the only arm that
moves the distribution, by up to 6.1·10⁻⁷ (per-state and relative distances: Supplementary Table
S30). How much such a distance can change the distribution of the *selected* output is bounded
directly: for one-shot distributions at total variation d and a deterministic selection rule, a
maximal coupling of the S shots puts the selected outputs within 1 − (1 − d)^S ≤ S·d (derivation in
Supplementary Sec. S-VII). At S = 1,000 the selected-output distributions differ by at most
2.6·10⁻¹¹ for SV1 and 6.1·10⁻⁴ for the single-precision arm, relative to the double-precision GPU
reference. These bounds quantify how far the arithmetic can move the distribution of the returned
portfolio at fixed angles. They do not bound how often two *independently* sampled batches return
different portfolios, which is a different quantity: two identical fair coins are at d = 0 and still
disagree on half of independent draws. That is what the next paragraph measures.

*Repeated batches return different portfolios.* Ten batches of 1,000 shots from one state on one
backend return 9 or 10 distinct best-feasible portfolios in 10 batches at N = 20, on every arm and
in every circuit: 10 in 15 of the 16 arm × circuit cells, and 9 feasible portfolios in the one SV1
cell where a batch held no feasible shot (10 selected outcomes there if ⊥ is counted). At the
smaller sizes the count depends on the state as much as on the size: at N = 12 it is 1–2 for the
seed-42 states and 5–10 for the seed-43 states, and at N = 16 it is 3–5 and 1–5 (Supplementary Table
S30). In no N = 20 batch is the best feasible shot the exact optimum. The spread is what i.i.d.
sampling from the arm's own exact distribution predicts. A Monte Carlo of 10 batches from each arm's
distribution puts the observed number of distinct portfolios inside its 95 % interval in 46 of 48
arm × circuit cells, and the feasible-shot count of 456 of 480 batches lies within two binomial
standard deviations of 1,000·P_F.

Repeated sampling from a fixed distribution was therefore sufficient to produce portfolio
disagreement of the magnitude that the single runs showed, while the arithmetic can move the
selected-output distribution by at most the bounds above. No individual disagreement is attributed
to either channel. E4-A does not test the third possibility, which is different final angles from
optimization runs whose trajectories differ by ~10⁻⁶ per step on the two backends, so sampling is
shown to be sufficient rather than sole. The consequence for a device-abstraction layer is that
backend-independence of the QUBO does not imply backend-independence of the returned portfolio, and
that agreement at small sizes does not establish it at larger ones, where the feasible set is large.

*Hardware: four tasks on IonQ Forte-1.* Each task samples 100 shots from a circuit whose angles
were optimized on the local GPU (two seeds at N = 8, one at N = 12, one at N = 16; $33.20 in all).
In the N = 12 task, 89 of 100 shots violated the constraint and the remaining 11 contained the exact
optimum, so a best-of-shots report would record a 0.0000 % gap. At N = 16, 18 shots were feasible
and the best was 0.0438 % away. The two N = 8 tasks, differing only in initial angles, returned 25 %
and 7 % feasible shots. The four tasks illustrate the distinction between the quality of the best
feasible outcome and the frequency of feasible outcomes. They identify no size limit and no ranking
of noise against instance structure, which four tasks at one calibration could not, and the
product-fidelity model that predicted a ceiling near N = 12 was contradicted by the N = 16 task
(Supplementary Sec. S-VIII).

### G. Resources and how the measured quantities scale

*Scale and reference.* The single-instance, single-seed benchmark (N = 8–30, seed 42;
Supplementary Table S2) locates the simulator envelope: classical runtime is flat in the seconds
range, while QAOA grows from 30 s at N = 20 to 3.14 h at N = 30. At N = 30, single precision uses
41.3 of 49 GB of VRAM, double precision does not fit, and N = 50 (≈ 9 PB for a dense statevector) is
beyond any memory this study could reach. Its gaps are 0.000 % at N ≤ 16 and 0.024–0.053 % at N =
20–30, with every row feasible, and §V-D measures what those numbers denominate.

*How the measured quantities scale.* Over the sizes the cross covers, the quantities of §III-C fall
with N at about the rate at which the constraint-respecting control falls. The median one-shot
success probability at A_heur drops 43-fold from N = 12 to N = 20 and that of the uniform-feasible
draw drops 37-fold, while the feasible set grows 73-fold (66, 560 and 4,845). The circuit's resource
count grows as N qubits and p·[C(N, 2) + 2N] parameterized rotations, 460 at N = 20 with p = 2, and
the evaluation's own cost grows with the feasible set, since the exact optimum, the exact rank and
Δ_F come from enumerating C(N, K) candidates (5.9·10⁵ of them in 5.7 s at N = 30). Both the rotation
count and the enumeration are reported as what this study measured, not as native gate counts, as
fault-tolerant resource estimates, or as an extrapolation to sizes where the feasible set cannot be
enumerated (§VI-A, §VII).

## VI. Engineering Implications

The measurements identify three engineering requirements: formulation-independent scoring,
feasibility-aware acquisition metrics, and a recorded execution configuration. Runtime comparisons
give context for the evaluated implementation rather than evidence about quantum advantage: on a
statevector simulator, the exact classical baseline dominated on runtime by three to six orders of
magnitude, and on solution quality at every size we could run (§V-D, §V-G). What the measurement
leaves behind is reusable: what to report, what the same experiment cost on metered backends, the
implementation defects that changed results before they were caught, and an artifact that lets a
reader re-run any of it (§IV-D and the availability statement). The infrastructure, its cost guards,
and the provider-specific defects met in deploying it are documented in the artifact rather than
here.

### A. What to report

Three reporting proposals exist [22], [23], [24], and none of them is a formal standard. We adopt
the fields of [22], [23] where they apply; Table V collects what the measurements here showed a
report has to carry, as recommendations rather than as a standard. Table V lists what each result of
§V showed to be necessary, linking each reporting item to the measurement that motivates it.

Which of those items survive where the feasible set cannot be enumerated is the next question, since
every quantity above that names f\*, Δ_F or a rank in F is computed here by enumeration. The
feasible fraction of a batch, the original objective of the returned solution, the task counts and
the cost, and a comparison against a feasible classical baseline need none of it. A success
probability can still be estimated against a target fixed in advance rather than against the
optimum, with its sampling uncertainty reported; what cannot be had are g_F, the rank, and D_cond,
which are defined against the whole feasible set. An estimate of the first kind should be labeled
as such, and not read as one of the exact quantities this paper reports.

**Table V. Reporting items, and the measurement that motivates each.**

| Evaluated quantity | Report | Result that showed the need |
|---|---|---|
| Solution quality | g_F against the objective's spread over F, or the rank in F; the penalty weight A alongside any offset-normalized figure | E0: the same portfolio reads 33.6 % or 0.22 % as A alone changes; the all-states ratio of [22] is non-discriminative at A_heur (§V-D) |
| Optimization | the optimized expectation beside the best-of-shots figure ([22]'s mean/best split) | a run at ⟨H_norm⟩ = −0.91 and one at −5.01 both report small offset-normalized gaps, 0.038 % and 0.022 % (§V-E) |
| Acquisition | P_F, P_τ and Q_τ(S) at the budget used, against a constraint-respecting control sampler | E2: optimization raises the median Q_τ(1000) from 0.06 to 0.34 at N = 16, yet at A_heur and τ = 1 % beats a uniform feasible draw's P_τ in 0 of the 450 runs over the three sizes (§V-B) |
| Execution conditions | the initial-angle seed distribution, precision, backend and sampler seeding, as factors | one seed reverses the main contrast (§IV-E); single runs on two backends disagree at N ≥ 16 (§V-F) |
| Cost | measured task counts and billed time per run | one sampling call was billed as 177 tasks, not one, until its angles were marked non-trainable (§VI-B) |
| Domain metric | the finance quantity as an auxiliary, with its definition | similar objective values do not guarantee similar Sharpe ratios (§V-D) |

The sharpest form of the first three rows is that the penalty weight is a free parameter of the
reported metric (Table IV). Re-scoring fixed feasible solutions at a larger penalty weight reduces
the offset-normalized gap without changing the solutions or their sampling distribution (E0). In the
re-optimization contrast, the larger weight also yielded a lower acquisition probability under the
tested settings (E1). Those are two effects, one of the number and one of the run, and neither is a
law that a larger A always degrades the sampler. Reporting A is the minimum. Reporting against the
penalty-free objective, or reporting a rank, removes the degree of freedom, and reporting Q_τ(S)
against the uniform-feasible control says whether the optimizer did anything that a constraint-aware
guess would not. The case of Mancilla et al. [27] makes the cost of under-reporting concrete. A
realized Sharpe of 1.81 for QAOA against 1.31 for annealing is read there as a property of the
energy landscape, with no objective value for a single selected portfolio and no exact optimum — and
C(10, 5) = 252 candidates is small enough to enumerate directly.

### B. Compute and billing

Table II reports, for every metered run, the task count and the dollars billed, and the totals of
the study ($37.90 on Braket). Task counts are reported because one of them was wrong by two orders
of magnitude before it was measured: in the N = 8 SV1 pilot, the final 1,000-shot sampling call was
billed as 177 tasks rather than one. The optimizer's angles reached that call still marked
trainable, so the framework returned 176 shifted tapes for a gradient that no one reads (2·p·[C(N,
2) + 2N] at N = 8, reproduced locally with no device; 920 at N = 20). The run's 227 tasks became 51
once the angles were frozen. The mechanism is inferred from the count and the fix rather than from
an instrumented re-run, and the per-run ledger is in the artifact. On a metered backend, the
executed circuit count is not implied by the algorithm's description, so it has to be measured and
reported per run.

### C. Checks on the recorded results

The results of §V rest on four correspondences that were checked rather than assumed, each of them
because an implementation defect had at some point broken it. (i) *Objective against QUBO.* The
evaluator of §III-C scores from the returns and covariance, not from the QUBO matrix, and it agrees
with the QUBO path on every state at N = 8 and 12 (E0). (ii) *State against batch.* The analytic
state quantities and the final sampling call receive the same final parameters, which regression
tests pin on both optimizer paths. Every arm is reported with two consistency checks: the batch's
feasible fraction must track the final state's P_F, and the batch counts must fall within what the
recorded states predict under independent draws (Supplementary Table S22). (iii) *Sampler against
seed.* E4-A seeds the device's own generator, because the Lightning devices draw a seed from NumPy's
global generator once, at construction. (iv) *Hardware records against device schema.* The four IonQ
task records are read from the device's nested result schema (Supplementary Sec. S-VIII).

One of the seven defects reached a result reported in §V: the E3 L-BFGS-B arm's sampler read the
initial parameter vector, so its batch columns described unoptimized states. That arm was re-run
with the fix on the same host. Every state of the 300 runs reproduced bit for bit, and the re-run's
counts sit within 3.0 hits and 2.7 no-feasible runs of the prediction in every cell. The defects in
the order found, what each would have reported, and the checks that caught them are in Supplementary
Sec. S-IX.

## VII. Limitations and Threats to Validity

**Formulation and data.** Every measurement is on one constrained problem, K of N by cardinality,
under two encodings of it: the quadratic penalty at six weights, and the constraint-preserving
ansatz of E6 at A = 0. Slack variables or a Lagrangian update would place the constant differently
or remove it, and would change P_F by construction; an implementation that restores the constant and
scores on the original objective would not report g_off at all. The cross covers N ≤ 20, where the
feasible set is enumerable. The census extends to N = 30 by enumeration on 30 instances per size
(Supplementary Table S9), the QAOA runs above N = 20 are single, and N = 50 was not enumerated in
this study. All E0–E4 instances are subsets of one 50-ticker Nikkei universe over one three-year
window, so every interval in §V is conditional on that universe, that window, and that drawing rule
(§IV-E). E5 adds one adjacent window of the same universe, chosen by adjacency rather than sampled,
and it does not make the intervals unconditional. Thirty subsets is fewer than the 100 instances of
Uotila et al. [18] or the 250 of Stopfer and Wagner [13]. The Sharpe ratio is auxiliary: it is
measured on the raw series for the equal-weight portfolio, it is not what the QUBO optimizes, and
its deviations from those of the optimum are reported as domain-level consequences rather than as
predictions.

**Optimization.** Five seeds per instance show that a within-instance tail exists; they do not
estimate how often it occurs. The design was not sized from a minimum detectable effect, and the
intervals are conditional on the fixed seed set, one member of which reverses the main contrast on
its own at each of the two larger sizes (Supplementary Table S19). E3 tests two settings, not the
optimizer space. One of them did not improve on the baseline within its budget, and the other's
single-precision terminations include line-search failures (§V-E). A schedule, a larger budget, a
different initialization rule [32], [33], or depth beyond 2 with a warm start could change the
relative sizes of the effects that §V-E summarizes, and the depth non-monotonicity remains an
observation whose cause is not identified. The multistart comparison is at equal shots, with the
optimization cost stated beside it, rather than at equal total cost. A_crit and A_margin are
reference conditions computed by enumeration, not procedures that scale. The critical penalty is a
ground-state feasibility threshold, not a guarantee of reliable finite-shot acquisition: near it,
infeasible batch minima are common (55 % of N = 20 runs at 1.1·A_crit), which is what makes the
feasibility-aware selection rule essential. These observations do not by themselves rule out
near-threshold weights under that rule, whose utility is assessed through the acquisition
probability and the solution quality at the chosen budget (§V-C). E2's strongest control is matched
on candidates generated, not on cost. That the optimized state at A_heur beat a uniform draw from F
in none of 450 runs at the tested thresholds says that the best-of-shots figure cannot confirm the
optimization's contribution at these sizes; it does not say that a constraint-aware quantum method
would fail to. Q_τ(S) assumes i.i.d. shots.

**Backend and hardware.** E4-A separates the output distribution from the draws, not the
optimization: 12 circuits at one weight, evaluated at fixed angles, bound how far the arithmetic can
move the selected-output distribution, and they show sampling to be sufficient for the observed
disagreement. Whether an optimization run on SV1 reaches the same angles as the GPU run does was not
tested (E4-B, ≈1,500 tasks), so the backend attribution is partial. Hardware is four tasks at one
calibration: two seeds at N = 8, one each at N = 12 and 16, 100 shots each. They establish no size
limit, no noise ranking, and no distribution. Continuous weights and richer constraints are
deferred.

## VIII. Conclusion

The contribution of a constrained quantum optimizer reads as two factors: how often it satisfies the
constraint and which of the feasible solutions it prefers. Under the dominate-the-objective rule
that this implementation uses, the second was near-uniform. The median preference ratio inside the
feasible set remained within 1% of unity, where the run's initial state already was, and across 900
runs, over three optimizer settings and two price windows, the optimized state's one-shot success
probability never exceeded that of a uniform draw from that set. What optimization changed was the
first factor: the feasible probability P_F rose on 30 of 30 instance medians at each size. The
preference appears when the objective carries more of the cost, and the share of runs exceeding the
control rises monotonically with ε_F, to 41% at 1.1 times the feasibility threshold at 20 assets.
Moving the constraint into the ansatz instead puts 420 of 450 runs above the control, whereas a
block that only removes the penalty gains the preference without the mass. It is the product of the
two factors, not either one alone, that rises above the control.

The evidence comes from one constrained problem at N ≤ 20 and p = 2, under a penalty weight swept
over six settings and a constraint-preserving ansatz beside it, on two adjacent windows of one
50-ticker universe, with two optimizers in three settings. Within it, the weaker weights are not a
recommendation; the median run stays below the control even at the weakest, the threshold is
computed by enumeration, and 55% of batches there have an infeasible minimum at 20 assets. None of
these limits bounds QAOA or argues for or against a quantum speedup, a question that these sizes
cannot address. By contrast, enumeration solves instances of this size outright.

What these measurements do influence is how such a run is reported. The offset-normalized gap
registers none of this and moves the other way: lowering the weight worsens it in 29 of 30 instances
at 20 assets, while the acquisition probability improves in 30 of 30, because the weight sits in the
gap's denominator and also sets how much of the objective the circuit resolves. In our comparison,
the gap ranked best the setting whose optimized state showed no preference. A benchmark that
emphasizes what its quantum part contributes therefore carries the feasible and the acquisition
probability at the budget used, alongside the gap and the weight, with a constraint-respecting
control (Table V), and §VI-A marks which of those items survive where the feasible set cannot be
enumerated.

## Acknowledgment

This work received no specific grant from any funding agency. Claude (Anthropic), via Claude Code
[49], assisted with research discussions, initial implementation, and the drafting and revision of
this manuscript. The author takes responsibility for the final manuscript and the accompanying
artifacts.

## References

[1] H. Markowitz, "Portfolio selection," *J. Finance*, vol. 7, no. 1, pp. 77–91, Mar. 1952, doi:
10.1111/j.1540-6261.1952.tb01525.x.
[2] D. Bienstock, "Computational study of a family of mixed-integer quadratic programming problems,"
*Mathematical Programming*, vol. 74, pp. 121–140, 1996, doi: 10.1007/BF02592208.
[3] A. Lucas, "Ising formulations of many NP problems," *Frontiers in Physics*, vol. 2, art. 5,
2014, doi: 10.3389/fphy.2014.00005.
[4] E. Farhi, J. Goldstone, and S. Gutmann, "A quantum approximate optimization algorithm,"
arXiv:1411.4028, 2014.
[5] R. Orús, S. Mugel, and E. Lizaso, "Quantum computing for finance: Overview and prospects,"
*Reviews in Physics*, vol. 4, art. 100028, 2019, doi: 10.1016/j.revip.2019.100028.
[6] D. J. Egger *et al.*, "Quantum computing for finance: State-of-the-art and future prospects,"
*IEEE Trans. Quantum Eng.*, vol. 1, pp. 1–24, 2020, doi: 10.1109/TQE.2020.3030314.
[7] D. Herman *et al.*, "Quantum computing for finance," *Nature Reviews Physics*, vol. 5, pp.
450–465, 2023, doi: 10.1038/s42254-023-00603-1.
[8] S. Brandhofer, D. Braun, V. Dehn, G. Hellstern, M. Hüls, Y. Ji, I. Polian, A. S. Bhatia, and T.
Wellens, "Benchmarking the performance of portfolio optimization with QAOA," *Quantum Inf.
Process.*, vol. 22, no. 1, art. 25, 2023; arXiv:2207.10555.
[9] J. S. Baker and S. K. Radha, "Wasserstein solution quality and the quantum approximate
optimization algorithm: A portfolio optimization case study," arXiv:2202.06782, 2022.
[10] L. Lozano, "A penalty-free pipeline for direct quantum-annealer portfolio optimization," *arXiv
preprint* arXiv:2605.17628, May 2026.
[11] S. Hadfield, Z. Wang, B. O'Gorman, E. G. Rieffel, D. Venturelli, and R. Biswas, "From the
quantum approximate optimization algorithm to a quantum alternating operator ansatz," *Algorithms*,
vol. 12, no. 2, art. 34, 2019, doi: 10.3390/a12020034.
[12] Z. Wang, N. C. Rubin, J. M. Dominy, and E. G. Rieffel, "XY mixers: analytical and numerical
results for the quantum alternating operator ansatz," *Phys. Rev. A*, vol. 101, art. 012320, 2020,
doi: 10.1103/PhysRevA.101.012320.
[13] E. Stopfer and F. Wagner, "Quantum portfolio optimization: An extensive benchmark,"
arXiv:2509.17876, 2025 (rev. Jul. 2026).
[14] D. Goldberg, "What every computer scientist should know about floating-point arithmetic," *ACM
Comput. Surv.*, vol. 23, no. 1, pp. 5–48, Mar. 1991, doi: 10.1145/103162.103163.
[15] G. Rosenberg, P. Haghnegahdar, P. Goddard, P. Carr, K. Wu, and M. López de Prado, "Solving the
optimal trading trajectory problem using a quantum annealer," *IEEE J. Sel. Topics Signal Process.*,
vol. 10, no. 6, pp. 1053–1060, Sep. 2016, doi: 10.1109/JSTSP.2016.2574703.
[16] S. Mugel *et al.*, "Dynamic portfolio optimization with real datasets using quantum processors
and quantum-inspired tensor networks," *Phys. Rev. Research*, vol. 4, art. 013006, 2022, doi:
10.1103/PhysRevResearch.4.013006.
[17] D. Venturelli and A. Kondratyev, "Reverse quantum annealing approach to portfolio optimization
problems," *Quantum Machine Intelligence*, vol. 1, no. 1, pp. 17–30, 2019, doi:
10.1007/s42484-019-00001-w.
[18] V. Uotila, J. Ripatti, and B. Zhao, "Higher-order portfolio optimization with quantum
approximate optimization algorithm," in *Proc. IEEE Int. Conf. Quantum Computing and Engineering
(QCE)*, 2025, pp. 2238–2249, doi: 10.1109/QCE65121.2025.00244.
[19] A. Verma and M. Lewis, "Penalty and partitioning techniques to improve performance of QUBO
solvers," *Discrete Optimization*, vol. 44, art. 100594, 2022, doi: 10.1016/j.disopt.2020.100594.
[20] M. Ayodele, "Penalty weights in QUBO formulations: permutation problems," in *Proc. EvoCOP
2022*, LNCS 13222, pp. 159–174; preprint arXiv:2206.11040.
[21] J. A. Montanez-Barrera, D. Willsch, A. Maldonado-Romo, and K. Michielsen, "Unbalanced
penalization: a new approach to encode inequality constraints of combinatorial problems for quantum
optimization algorithms," *Quantum Sci. Technol.*, vol. 9, no. 2, art. 025022, 2024;
arXiv:2211.13914.
[22] A. Abbas *et al.*, "Challenges and opportunities in quantum optimization," *Nature Reviews
Physics*, vol. 6, no. 12, pp. 718–735, Dec. 2024, doi: 10.1038/s42254-024-00770-9.
[23] T. Koch *et al.*, "Quantum optimization benchmarking library — The intractable decathlon,"
arXiv:2504.03832, 2025.
[24] L. Lozano, "Where the quantum lives in D-Wave hybrid portfolio optimization: an operational
decomposition audit," *arXiv preprint* arXiv:2605.17623, 2026.
[25] M. Hodson, B. Ruck, H. Ong, D. Garvin, and S. Dulman, "Portfolio rebalancing experiments using
the Quantum Alternating Operator Ansatz," *arXiv preprint* arXiv:1911.05296, Nov. 2019.
[26] A. Oralkhan and T. Zhaxalykov, "Investigation of hardware architecture effects on quantum
algorithm performance: A comparative hardware study," arXiv:2601.05286, 2026.
[27] J. Mancilla, T. D. Bouloumis, and F. Goguikian, "Constrained portfolio optimization via quantum
approximate optimization algorithm (QAOA) with XY-mixers and Trotterized initialization: A hybrid
approach for direct indexing," *arXiv preprint* arXiv:2602.14827, Feb. 2026.
[28] N. N. Hegade, P. Chandarana, K. Paul, X. Chen, F. Albarrán-Arriagada, and E. Solano, "Portfolio
optimization with digitized counterdiabatic quantum algorithms," *arXiv preprint* arXiv:2112.08347,
Dec. 2021.
[29] B. Wu and L. Wang, "A two-step quantum approximate optimization algorithm for portfolio
optimization and risk assessment," *Quantum Reports*, vol. 8, no. 2, art. 45, May 2026, doi:
10.3390/quantum8020045.
[30] N. Innan, A. Saleem, A. Marchisio, and M. Shafique, "Quantum portfolio optimization with expert
analysis evaluation," *arXiv preprint* arXiv:2507.20532, Jul. 2025.
[31] P. K. Barkoutsos, G. Nannicini, A. Robert, I. Tavernelli, and S. Woerner, "Improving
variational quantum optimization using CVaR," *Quantum*, vol. 4, art. 256, 2020, doi:
10.22331/q-2020-04-20-256.
[32] L. Zhou, S.-T. Wang, S. Choi, H. Pichler, and M. D. Lukin, "Quantum approximate optimization
algorithm: Performance, mechanism, and implementation on near-term devices," *Phys. Rev. X*, vol.
10, art. 021067, 2020, doi: 10.1103/PhysRevX.10.021067.
[33] S. H. Sack and M. Serbyn, "Quantum annealing initialization of the quantum approximate
optimization algorithm," *Quantum*, vol. 5, art. 491, Jul. 2021, doi: 10.22331/q-2021-07-01-491.
[34] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, "Barren plateaus in
quantum neural network training landscapes," *Nat. Commun.*, vol. 9, art. 4812, 2018, doi:
10.1038/s41467-018-07090-4.
[35] L. Bittel and M. Kliesch, "Training variational quantum algorithms is NP-hard," *Phys. Rev.
Lett.*, vol. 127, no. 12, art. 120502, 2021, doi: 10.1103/PhysRevLett.127.120502.
[36] G. G. Guerreschi and A. Y. Matsuura, "QAOA for Max-Cut requires hundreds of qubits for quantum
speed-up," *Scientific Reports*, vol. 9, art. 6903, 2019, doi: 10.1038/s41598-019-43176-9.
[37] Amazon Web Services, "Amazon Braket Developer Guide," AWS documentation. [Online]. Available:
https://docs.aws.amazon.com/braket/ (accessed Nov. 7, 2026).
[38] V. Bergholm *et al.*, "PennyLane: Automatic differentiation of hybrid quantum-classical
computations," arXiv:1811.04968, 2018.
[39] H. Bayraktar *et al.*, "cuQuantum SDK: A high-performance library for accelerating quantum
science," in *Proc. IEEE Int. Conf. Quantum Computing and Engineering (QCE)*, 2023, pp. 1050–1061,
doi: 10.1109/QCE57702.2023.00119.
[40] T. Jones and J. Gacon, "Efficient calculation of gradients in classical simulations of
variational quantum algorithms," arXiv:2009.02823, 2020.
[41] C. Collberg and T. A. Proebsting, "Repeatability in computer systems research," *Communications
of the ACM*, vol. 59, no. 3, pp. 62–69, Mar. 2016, doi: 10.1145/2812803.
[42] G. K. Sandve, A. Nekrutenko, J. Taylor, and E. Hovig, "Ten simple rules for reproducible
computational research," *PLoS Computational Biology*, vol. 9, no. 10, art. e1003285, 2013, doi:
10.1371/journal.pcbi.1003285.
[43] Association for Computing Machinery, "Artifact review and badging — version 1.1," ACM policy,
Aug. 2020. [Online]. Available:
https://www.acm.org/publications/policies/artifact-review-and-badging-current (accessed Nov. 7,
2026).
[44] S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi, "Optimization by simulated annealing,"
*Science*, vol. 220, no. 4598, pp. 671–680, 1983, doi: 10.1126/science.220.4598.671.
[45] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf.
Learning Representations (ICLR)*, 2015. arXiv:1412.6980.
[46] R. H. Byrd, P. Lu, J. Nocedal, and C. Zhu, "A limited memory algorithm for bound constrained
optimization," *SIAM J. Sci. Comput.*, vol. 16, no. 5, pp. 1190–1208, 1995, doi: 10.1137/0916069.
[47] P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python,"
*Nature Methods*, vol. 17, pp. 261–272, 2020, doi: 10.1038/s41592-019-0686-2.
[48] A. Bärtschi and S. Eidenbenz, "Deterministic preparation of Dicke states," in *Fundam. Comput.
Theory*, LNCS 11651. Springer, 2019, pp. 126–139, doi: 10.1007/978-3-030-25027-0_9.
[49] Anthropic, "Claude Code," 2026. [Online]. Available: https://claude.com/claude-code (accessed
Nov. 7, 2026).

---

### Figure manifest

Main-text figures. Files carry experiment names; the mapping here is the one the text uses.
- **Fig. 1** — `notebooks/figures/fig_e2_shots.pdf` — E2. Q_τ(S) at τ = 1 % against the shot budget
  S ∈ {10, 100, 1,000, 10,000} for four samplers: the optimized QAOA state, the unoptimized state at
  the same initial angles, uniform over F, and uniform over all bitstrings. Panels are (a) N = 12,
  (b) N = 16 and (c) N = 20, all at A_heur. Each instance is first summarized by the median over its
  five initial-angle seeds; the line is the median of those instance values over the 30 instances
  and the band their interquartile range. The two uniform samplers are instance properties evaluated
  on the same instances, and Q_τ(S) follows from the exact one-shot probabilities under i.i.d.
  shots.
- **Fig. 2** — `notebooks/figures/fig_ladder_preference.pdf` — Conditional preference and sampling
  success against a uniform feasible draw, over the penalty settings and the two A = 0 blocks, at τ
  = 1 % (§V-C). The horizontal coordinate is ε_F = Δ_F / s_H, the objective's feasible range divided
  by the largest Ising coefficient of the cost before normalization, one series per N. (a) The
  preference ratio R_τ inside the feasible set, median over the runs of each setting with the
  interquartile range, and the line R_τ = 1 that a uniform draw would give. (b) The share of runs
  whose one-shot success probability exceeds a uniform draw from the feasible set. Each series is
  labeled at its ends, A_heur at the left and 1.1, meaning 1.1·A_crit, at the right. The stars are
  the two A = 0 blocks at their measured ε_F, filled for the constraint-preserving ansatz and hollow
  for the published one. Both panels draw the cells of Table III as they stand, whose N = 12 cells
  cover different instances (§V-C); the A = 0 cell of the cross is left out there, because at a
  threshold of zero every multiple is the same A = 0 setting.
- **Fig. 3** — `notebooks/figures/fig_e0_rescore.pdf` — E0. (a) The median-rank feasible solution of
  the reference N = 20 instance scored three ways as only A changes: the offset-normalized gap g_off
  = (f − f\*)/|f\* − AK²|, the all-states range gap 1 − AR = (E_A − E_A\*)/(C_max − E_A\*) of [22],
  where C_max is the largest E_A over all 2^N assignments, and the feasible-range gap g_F = (f −
  f\*)/Δ_F. (b) The deflation D(A) = |f\* − A·K²|/Δ_F per size on the reference instances, with
  A_heur marked by a star; N > 20 uses A_safe and is drawn hollow. (c) D(A_heur) over 30 instances
  per size (box = IQR, whisker = range), with the reference instance marked.

### Data/artifact availability

The code, the derived input statistics of the two price windows (μ and Σ of the 50-ticker universe,
from which every instance is sliced; the price snapshots themselves are third-party data and are not
redistributed), the census and its per-paper cells, every result CSV this paper reads (one row per
run and one per instance, with the aggregates the tables are checked against), the figure scripts
and the table-checking script are at `https://github.com/tobitaQ/qaoa-portfolio-penalty-benchmark`,
release `v1.2`, archived at doi:10.5281/zenodo.24032941; the README indexes the files by experiment.
A clone reproduces every number without network access to a market-data provider, at three levels:
every figure and table from the committed CSVs (seconds, no GPU); one N = 8 or N = 12 instance end
to end (minutes on a CPU); the cross from the manifest (13.9 GPU h). The per-run artifacts (exact
probabilities on F under the initial and final state, angles, trajectory, shot counts) are deposited
in the same archive: 2,565 files and ≈100 MB compressed for E1, 900 files and ≈34 MB for E6 and its
control. They are needed only to recompute the E2 columns and D_cond, which the committed CSVs
already hold; any full state vector is reconstructed from the recorded angles and seed. The four
IonQ task records are committed in the device's own nested schema. Supplementary Figs. S1–S14 and
Tables S1–S38 are generated by the same scripts from the same committed files. Cloud and hardware
paths require an explicit backend name and a results tag.