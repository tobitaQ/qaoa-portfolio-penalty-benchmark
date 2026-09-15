# Supplementary Material: Measuring Metric and Sampling Distortions in Penalty-Encoded QAOA Portfolio Benchmarks

**Draft v3.2 — 2026-09-13.** Supplementary material for the paper of the same title. It holds the
literature census, the single-instance and single-seed measurements that preceded the cross, the
full versions of the tables the main text prints in compact form, and the diagnostics and
reproduction notes the main text cites. Every table is regenerated from the same committed result
files by the same scripts, and `papers/check_tables.py` checks these tables as it checks the main
text's. Section, figure and table numbers without an S prefix, and bracketed citations, refer to the
main text and its reference list.

---

## S-I. The literature census

Sec. II summarises what fourteen published QAOA and annealing portfolio benchmarks report. The
claim is a measurement, so it carries a method and a population.

*Sources.* arXiv (quant-ph, q-fin), Google Scholar, IEEE Xplore, Springer Link, and the reference
lists of every paper already included. *Queries.* The cross product of {QAOA, quantum approximate
optimization, quantum alternating operator ansatz, quantum annealing} with {portfolio optimization,
portfolio selection, portfolio rebalancing}, each also run with "benchmark" and with "QUBO".
*Window.* Through 2026-08-10. *Inclusion*, all four required: the paper runs QAOA — including
constraint-preserving mixers in the alternating-operator family — or quantum annealing; on a
portfolio selection or rebalancing problem; on a quantum device or simulator; and reports at least one
solution-quality metric of its own. *Exclusion.* Surveys, quantum-inspired or classical-only methods,
other quantum algorithm families (VQE, quantum-walk, Grover-type search), and documents that propose a
reporting protocol rather than use one, which are read with the reporting proposals instead. One paper is named
but unread for want of institutional access.

*Selection flow.* The searches surfaced 26 candidate papers, of which twelve were excluded. Eleven fall under four
disjoint grounds: four are surveys that report no runs of
their own; three use a quantum algorithm outside the QAOA and annealing families
(a VQE with a Dicke ansatz, a quantum-walk optimizer, a Grover-type search); one
is quantum-inspired with no device or simulator run; and three propose a
reporting protocol rather than use one, so they are read with the reporting proposals
instead. The twelfth is named rather than silently dropped: a conference paper we could not obtain for want of institutional access. It is outside the population, so every count below is over the remaining fourteen, and would be over fifteen if that paper proved to qualify.

*Coding.* One coder, reading each paper's formulation rather than its abstract,
recording one row per paper with a verbatim quote for each cell that becomes a
criticism. There is no second coder — this is a single-author study — so the
per-paper cells and their quotes are committed, and a re-check of them found two
errors, both in our own favour, which are recorded with their corrections. That
is the accountability a second coder would otherwise provide, offered as an
artifact rather than as an assurance.

That protocol identifies fourteen benchmarks — [8]–[15], [22]–[27]. We read each paper's *formulation* rather than its
prose, recorded one row per paper, and committed the per-paper cells with quotes alongside the script
that derives the counts below. What follows is therefore a claim about this population; whether it
generalizes to the wider quantum-optimization literature is untested.

**Table S1. What fourteen published QAOA/annealing portfolio benchmarks report.** Counts derived from
the committed census, not written out by hand.

| Reported quantity | Papers reporting it |
|---|---:|
| An optimality gap, approximation ratio or regret | 4 of 14 |
| Task counts, billed device time, or monetary cost | 0 of 14 |
| Citation of [19], [20] or [21] | 0 of the 7 published after [19], [20] existed (0 of 14 overall) |
| Repeats per configuration | 3 of 14 |
| A finance/domain metric as a *result* | 6 of 14 |
| Solution degeneracy, observed | 4 of 14 |

The first row counts papers reporting *some* normalised quality number, not papers using this paper's
denominator: of the four, one normalises over the feasible set [9], one takes a relative distance to
a brute-force optimum [22], one a regret against an offline reference [8] and one a min–max over the
compared methods [15]. None divides by an energy carrying the penalty constant. These benchmarks have largely abandoned the gap, and the replacements are all different — a Wasserstein distance, a time-to-solution hit
rate, a perturbation-tolerant success rate, a ground-state success probability, a feasibility rate, an
arctan-compressed unnormalized energy, a realized Sharpe, a raw objective value. So **the third row is
the one that matters**: proposals exist that would restore comparability, and none of the seven
benchmarks that could have used them cites one. Not citing is what the row counts; whether a paper
departs from a proposal's content is a separate question that the census answers per cell. Sec. VI-A of the main text sets what the results add to what they ask for.

## S-II. The single-instance benchmark

The benchmark the paper's single-instance measurements come from (N = 8–30, seed 42, p = 2), kept
because it locates the simulator envelope (Sec. V-A) and because every later table qualifies it.
After the conditioning fix of Sec. VI-C, ⟨H_norm⟩ converges within ~20 ADAM steps at every size
(Fig. S1); classical runtime is flat in the seconds range while QAOA grows from 30 s at N = 20 to
3.14 h at N = 30 (Fig. S2), with no crossover anywhere in the simulable range. N = 30 at single
precision uses 41.3 of 49 GB of VRAM (adjoint differentiation keeps ≈3 statevector copies); double
precision would need ≈51.6 GB and does not fit; N = 50 needs 2^50 × 8 B ≈ 9 PB and is infeasible for
any statevector simulator. The ⟨H_norm⟩ column is the objective the optimizer actually descended:
at N = 20 the run reached −0.91 against −5.00 one row above and −8.49 one row below and still
reports a 0.053 % gap, the second-best number in the column, because best-of-1,000 recovered a good
sample from a poor state. Sec. V-B measures that effect across the cross. The reported gap per size
is Fig. S3 and the Sharpe ratio of the selected portfolio Fig. S4.

**Table S2. QAOA vs. classical on the reference instances (instance 0; seed 42, p = 2, single precision).**

| N | K | Classical E | QAOA E | Offset-normalized gap g_off | ⟨H_norm⟩ | QAOA runtime | Feasible |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 8 | 2 | −36.25 | −36.25 | 0.000 % | −2.01 | 5.3 s | ✓ |
| 12 | 2 | −52.25 | −52.25 | 0.000 % | −1.51 | 10 s | ✓ |
| 16 | 3 | −153.27 | −153.27 | 0.000 % | −5.00 | 17 s | ✓ |
| 20 | 4 | −336.15 | −335.97 | 0.053 % | **−0.91** | 29 s | ✓ |
| 24 | 4 | −400.15 | −400.03 | 0.031 % | −8.49 | 150 s | ✓ |
| 28 | 5 | −725.17 | −724.99 | 0.024 % | −9.53 | 42 min | ✓ |
| 30 | 6 | −1115.96 | −1115.64 | 0.029 % | −9.42 | 3.14 h | ✓ |
| 50 | 10 | −5098.08 | — | — (infeasible) | — | — | ✓ (classical) |

Sharpe ratios agree exactly wherever the energies do (N ≤ 16) and diverge far more than the gap
suggests once QAOA is merely near-optimal: at N = 20 the 0.053 % gap comes with a *higher* Sharpe
than the energy-optimal selection (1.53 vs. 1.47), while at N = 24, 28 and 30 gaps of 0.024–0.031 %
come with Sharpe ratios 15–22 % below it (1.20 vs. 1.47, 1.36 vs. 1.67, 1.30 vs. 1.55; Fig. S4).
Sec. S-III measures why.

## S-III. The metric on the reference instances

Sec. V-A reports the deflation, the band a 0.1 % offset-normalized gap admits, and the all-states
ratio of [19] on the reference instances. This section holds the tables behind those numbers: the
census by exact enumeration (Table S3), the relocation of the single-seed QAOA solutions of Table S2 in the full
ranking (Table S4), the threshold sweep (Table S5), the census under a universe-wide normalization
(Table S6), the same QAOA solution under three denominators as the penalty weight is moved to its
exact threshold (Table S7) and the all-states ratio (Table S8). Fig. S5 shows the Sharpe ratios
spanned by the 0.1 % band and the single-seed QAOA solutions under both denominators.

**Table S3. What the reported gap denominates (exact enumeration, seed 42).** "Spread" is the range of
the penalty-free objective over the feasible set; "deflation" is \|E*\| divided by it; the last two
columns describe the solutions lying within a 0.1 % reported gap of the optimum.

| N | C(N,K) | Spread | \|E*\| | Deflation | In 0.1 % band | Sharpe there |
|---:|---:|---:|---:|---:|---:|---|
| 16 | 560 | 1.271 | 153.3 | 121× | 30 (5.4 %) | 1.149 – 1.511 |
| 20 | 4,845 | 1.930 | 336.2 | 174× | 368 (7.6 %) | 0.634 – 1.539 |
| 24 | 10,626 | 1.930 | 400.2 | 207× | 998 (9.4 %) | 0.539 – 1.539 |
| 28 | 98,280 | 3.023 | 725.2 | 240× | 13,737 (14.0 %) | 0.428 – 1.707 |
| 30 | 593,775 | 4.000 | 1116.0 | 279× | 146,484 (24.7 %) | 0.172 – 1.695 |

The deflation factor rises monotonically with N, as the growth of A·K² against the measured spread
predicts. Its consequence for the numbers in Table S2 is direct: **the QAOA gaps of 0.024–0.053 % are 5.7–9.2 % of the objective's
own spread over the feasible set.** At N = 30 the QAOA solution ranks 844th of 593,775 — the top
0.14 %, which sounds excellent — and its Sharpe ratio is 16.1 % below the optimum's.

**Table S4. The single-seed QAOA solutions of Table S2 relocated in the full ranking.**

| N | Gap as reported | Gap vs objective spread | Rank | Percentile | Sharpe (optimum → QAOA) |
|---:|---:|---:|---:|---:|---|
| 20 | 0.0528 % | 9.20 % | 80 / 4,845 | 1.65 % | 1.473 → 1.532 (+4.0 %) |
| 24 | 0.0312 % | 6.47 % | 43 / 10,626 | 0.40 % | 1.473 → 1.201 (−18.5 %) |
| 28 | 0.0239 % | 5.74 % | 111 / 98,280 | 0.11 % | 1.666 → 1.363 (−18.2 %) |
| 30 | 0.0286 % | 7.97 % | 844 / 593,775 | 0.14 % | 1.553 → 1.303 (−16.1 %) |

**Why 0.1 %, and what happens either side of it.** The band is a step function
of the threshold, so a single threshold invites the reading that it was chosen
to flatter the argument. It was not; it is the conservative choice. Sweeping the
threshold over the same enumeration:

**Table S5. The near-optimal band as a function of the reported-gap threshold.**
Fraction of the feasible set inside the band, by size. The figure quoted in the text uses
0.1 %.

| Threshold | N = 8 | N = 12 | N = 16 | N = 20 | N = 24 | N = 28 | N = 30 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 % | 3.6 % | 1.5 % | 0.5 % | 0.1 % | 0.1 % | 0.0 % | 0.0 % |
| 0.05 % | 3.6 % | 1.5 % | 2.0 % | 1.5 % | 1.2 % | 1.1 % | 1.5 % |
| **0.1 %** | 3.6 % | 1.5 % | 5.4 % | 7.6 % | 9.4 % | 14.0 % | **24.7 %** |
| 0.2 % | 3.6 % | 4.5 % | 18.0 % | 42.5 % | 57.1 % | 80.8 % | **94.7 %** |
| 0.5 % | 21.4 % | 25.8 % | 80.7 % | 99.8 % | 100 % | 100 % | 100 % |
| 1.0 % | 50.0 % | 81.8 % | 100 % | 100 % | 100 % | 100 % | 100 % |

At N = 30 the band holds 22 solutions at 0.01 %, 146,484 at 0.1 %, and 562,148 —
94.7 % of everything feasible — at 0.2 %. A benchmark quoting 0.2 % rather than
0.1 % would have a stronger version of this paper's finding, not a weaker one.
What the sweep shows is that there is no plateau to stand on: between 0.05 % and
0.2 % the band goes from 1.5 % of the feasible set to 94.7 %, so in exactly the
range where reported gaps live, the metric has almost no resolution. The
threshold also matters more as N grows — 3.6 % of the feasible set at N = 8
against 24.7 % at N = 30 for the same 0.1 % — which is the same constant
appearing in the band count rather than in the denominator.

**Is the degeneracy an artefact of the normalization?** §III scales μ and Σ by
*this instance's* max-abs value, so a reader can reasonably ask whether the
near-optimal band looks crowded because every instance has been squeezed into
the same objective range. We repeated the enumeration with the scale held fixed
instead — taken once from the fifty-ticker universe and applied to every subset,
which lowers |μ̂| and |Σ̂| below one and therefore lowers A as well.

**Table S6. The census under two normalizations.** Left: each instance scaled
by its own max-abs value, as everywhere else in this paper. Right: one scale
from the fifty-ticker universe, applied to every size.

| N | Deflation, per-instance | Deflation, global | Band, per-instance | Band, global | Sharpe range, global |
|---:|---:|---:|---:|---:|---|
| 16 | 121× | 109× | 5.4 % | 3.4 % | 0.716 – 1.511 |
| 20 | 174× | 162× | 7.6 % | 7.1 % | 0.274 – 1.539 |
| 24 | 207× | 193× | 9.4 % | 9.2 % | 0.274 – 1.539 |
| 28 | 240× | 222× | 14.0 % | **15.5 %** | 0.148 – 1.707 |
| 30 | 279× | 262× | 24.7 % | **27.1 %** | 0.070 – 1.695 |

It is not an artefact. Deflation survives at 109–262× against 121–279×, eight to
ten per cent lower and rising with N for the same reason. The band
survives too, and at the two largest sizes it is *wider* under the fixed scale —
27.1 % of the feasible set at N = 30 against 24.7 %. The Sharpe range inside the
band widens at every size, its floor falling from 0.172 to 0.070 at N = 30. If
anything the per-instance normalization the rest of this paper uses understates
the effect, so the choice is conservative rather than convenient.

**The penalty weight sets the scale of the reported gap.** The heuristic A = 2·s_obj·N + 1 of
Sec. III-B exceeds the exact threshold A_crit by two to three orders of magnitude on the
reference instances; Table S7 scores the same QAOA solution at the weight used, at the
threshold, and against the objective's own spread. For N > 20 the threshold is the bounded
A_safe, which understates the overshoot.

**Table S7. The penalty weight silently rescales the reported gap.**

| N | K | A used | A_crit (A_safe for N > 20) | Overshoot | Gap at A used | Gap at A_crit | Gap vs objective spread |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 2 | 9 | 0.0107 | 840× | 0.0000 % | 0.00 % | 0.00 % |
| 12 | 2 | 13 | 0.0107 | 1213× | 0.0000 % | 0.00 % | 0.00 % |
| 16 | 3 | 17 | 0.0441 | 385× | 0.0000 % | 0.00 % | 0.00 % |
| 20 | 4 | 21 | 0.1148 | 183× | 0.0528 % | 8.93 % | 9.20 % |
| 24 | 4 | 25 | 0.1148 | 218× | 0.0312 % | 6.28 % | 6.47 % |
| 28 | 5 | 29 | 0.1657 | 175× | 0.0239 % | 4.02 % | 5.74 % |
| 30 | 6 | 31 | 0.2099 | 148× | 0.0286 % | 4.24 % | 7.97 % |

The last three columns are the same QAOA solution measured three ways. The heuristic penalty reports
0.024–0.053 %; the threshold penalty reports 4.0–8.9 % for those identical portfolios; and the
penalty-independent measure against the objective's own spread gives 5.7–9.2 %. The second and third
are different denominators — |f\* − A_crit·K²| against the spread Δ_F — and they are close where
A_crit·K² happens to be near Δ_F (N = 20, 24) and apart where it is not (4.24 % against 7.97 % at
N = 30, where the bounded A_safe also overstates the threshold). What the table shows is that the
discrepancy between the first column and either of the others is the overshoot; it does not show
the two alternative denominators agreeing, and they need not.

**Table S8. The all-states approximation ratio of [19] evaluated on our own instances.** AR = (C_max − C)/(C_max − C_min),
with C_max taken over all 2^N assignments. For N ≤ 20, C_max is the exact maximum over the
enumerated 2^N states (it falls at cardinality N); at N = 30 it is the energy of the all-ones
assignment, which we have not proven maximal, so that row is a lower bound on C_max and the ratios
there are, if anything, understated. "Worst feasible" is the highest-objective
member of the enumerated feasible set — the worst answer a solver could return while still satisfying
the constraint.

| N | Penalty A | AR, our QAOA solution | AR, worst feasible |
|---:|---:|---:|---:|
| 16 | 17 (heuristic) | 1.000000 | 0.999560 |
| 16 | 170 (10×) | 1.000000 | 0.999956 |
| 20 | 21 (heuristic) | 0.999967 | 0.999642 |
| 20 | 210 (10×) | 0.999997 | 0.999964 |
| 30 | 31 (heuristic) | 0.999982 | 0.999777 |
| 30 | 310 (10×) | 0.999998 | 0.999978 |

The metric does not degrade as the penalty grows; **at the penalty our formulation already uses, it
has already collapsed**. At N = 16 the QAOA solution and the worst feasible portfolio in the entire
set score 1.000000 and 0.999560 — the whole feasible set occupies four decimal places, and the
solutions §V-A shows spanning Sharpe ratios from 0.172 to 1.695 are, on this axis, indistinguishable.
Multiplying A by ten shrinks the remaining separation by a further order of magnitude. This is not a
criticism of [19], whose calibration problems have no feasibility constraint to violate; it is a statement that the one prescribed comparable metric is practically non-discriminative for the
constrained family studied here.

## S-IV. Thirty instances per size, and classical annealing

Sec. V-A summarises the instance axis and the two annealing arms. Table S9 is the census repeated
over thirty instances per size; Table S10 the annealing-on-the-feasible-set arm beside the seed-42
QAOA arm; Table S11 annealing on the penalty-encoded QUBO; Table S12 the seed-42 QAOA arm in full,
with the gap/Sharpe association of Fig. S7. Fig. S6 shows the deflation and the Sharpe spread inside
the 0.1 % band over the thirty instances, with the reference instance marked.

**Table S9. The census repeated over 30 instances per size.** Medians over the thirty unless noted;
"Ref. inst." is the deflation of the reference instance (instance 0, the one the rest of the paper
uses), "in 0.1 % band" counts feasible solutions
within a 0.1 % reported gap, and "overshoot" is the heuristic penalty divided by the minimum that keeps
the optimum feasible.

| N | Deflation med. (range) | Ref. inst. | In 0.1 % band | Sharpe spread | Overshoot |
|---:|---:|---:|---:|---:|---:|
| 12 | 68× (36–126) | 81× | 2 | 12 % | 179× (over the 21 with A_crit > 0; 9/30 have A_crit = 0) |
| 16 | 118× (71–210) | 121× | 22 | 43 % | 136× |
| 20 | 162× (105–253) | 174× | 354 | 71 % | 118× |
| 24 | 184× (121–282) | 207× | 1,052 | 96 % | 150× |
| 28 | 239× (193–332) | 240× | 26,670 | 131 % | 124× |
| 30 | 282× (220–411) | 279× | 261,202 | 142 % | 103× |

Both mechanisms survive as distributions rather than anecdotes. The median
deflation rises monotonically with N, as the growth of A·K² against the spread predicts, and no
instance at any size escapes it — the *smallest* deflation we observed anywhere is
36× at N = 12. The median penalty overshoot stays between 103× and 179× at every
size, so it is a property of the heuristic rather than of a particular covariance
matrix. The reference instance is the exception, and for a reason worth stating:
its A_crit is 0.0107 at N = 8 and N = 12, small enough that the same heuristic
weight overshoots it 840× and 1213× (Table S7). Overshoot is a ratio against a
threshold that can itself be near zero, so at small N it is far more dispersed
than the deflation factor, which is why the two are quoted differently —
121–279× for deflation on the reference instance against 36–411× across
instances, but 148–1213× for overshoot on the reference instance against a
103–179× median. At N = 12, 9 of 30 instances have A_crit = 0 — their unconstrained
optimum already has cardinality K — and for them the overshoot is undefined rather than large, so
the N = 12 median is taken over the other 21. (The exact all-states A_crit of Sec. III-B is used here and everywhere in Sec. V; the bounded
A_safe would count seven, because it is 0.008–0.009 on two instances whose exact A_crit is 0.)

The reference instance is unremarkable in deflation — near the median at every
size — but it is *favourable* in the quantity that matters to a reader. Ranked by
the Sharpe spread inside a 0.1 % gap it is below the median at *every* size
(18th, 17th, 37th, 13th, 3rd and 3rd percentile for N = 12 … 30), and at N = 28
and N = 30 it has the narrowest spread of all thirty instances. Fig. S6 shows both
quantities across the sweep: the deflation factor, which no instance escapes and
whose smallest observed value anywhere is 36×, and the Sharpe spread inside the
0.1 % band, where the reference instance sits at the bottom of the range rather
than in the middle of it. The median instance at N = 30 has
261,202 feasible solutions inside that band, 44 % of the entire feasible set,
spanning a Sharpe range of 142 % of the optimum's. The single-instance results
understate the effect rather than exaggerate it.

**Simulated annealing on the feasible set.** The gap's denominator, the band and the
objective/Sharpe decoupling are properties of the QUBO and its scoring rather than of the solver.
Simulated annealing with cardinality-preserving swaps, five seeds on each of the same 30 instances
at each size (900 runs), scored against the same enumerated optima, reached the exact optimum in
895 of 900 runs; all five misses are at N = 30, the largest gap is 0.0022 % and the worst Sharpe
deviation 3.4 %. The two arms are not on matched budgets (50 ADAM steps against 50,000 annealing
moves). That arm searches the feasible set directly and never enters the penalty landscape, so it
bounds what a competent classical baseline achieves on the *task*, not what a stochastic solver
scores on the *encoding*; Table S11 runs the same annealer on the penalty-encoded Q with single-bit
flips.

**Table S10. The same 30 instances per size, scored against the same enumerated optima.**

| N | Simulated annealing | median runtime | QAOA (p = 2, 50 steps) | median runtime |
|---:|---:|---:|---:|---:|
| 12 | 150/150 | 3.4 s | 26/30 | 9.9 s |
| 16 | 150/150 | 3.4 s | 17/30 | 16.4 s |
| 20 | 150/150 | 3.4 s | 0/30 | 29.0 s |
| 24 | 150/150 | 3.4 s | — | — |
| 28 | 150/150 | 3.4 s | — | — |
| 30 | 145/150 | 3.4 s | — | — |

**Table S11. Classical annealing on the penalty-encoded QUBO, five seeds per size.**
Same Q as the QAOA arm, same enumerated optimum; moves are single-bit flips, so
the search walks the penalty landscape rather than the feasible set.

| N | Exact optimum | Reported gap | ΔSharpe | Worst rank |
|---:|---:|---|---|---:|
| 12 | 1 of 5 | 0.00 – 0.59 % | 0 to **−64.2 %** | 24 of 66 |
| 16 | 0 of 5 | 0.14 – 0.52 % | −3.7 to −36.3 % | 466 of 560 |
| 20 | 0 of 5 | 0.10 – 0.33 % | −23.5 to **−49.9 %** | 4,307 of 4,845 |

Every one of the fifteen runs returned a feasible portfolio, and every one of
them reproduces the pattern the QAOA arm shows: a reported gap under one per
cent alongside a Sharpe ratio tens of per cent below the optimum's, and a rank
deep inside the feasible set. At N = 20, seed 46 reports 0.33 % while sitting
4,307th of 4,845. So the four effects that do not depend on the sampler — the
denominator, the near-degeneracy, the objective/Sharpe decoupling and the
instance axis — are properties of the encoding, and a classical solver scored the
same way inherits all of them. The pattern belongs to the encoding scored this way, not to the sampler.

**Running QAOA over the same instances qualifies Table S2's headline.** We repeat the
QAOA configuration of Table S2 — p = 2, 50 ADAM steps, 1000 shots, initial-angle
seed 42 — on 30 instances at each of N = 12, 16, 20, and locate every returned
solution in the exhaustively ranked feasible set.

**Table S12. QAOA over 30 instances per size (p = 2, seed 42).** "Gap/spread" restates the reported gap
against the objective's own range over the feasible set; "rank" is the solution's position in that set;
ΔSharpe is relative to the exact optimum's.

| N | Exact optimum | Gap med. (max) | Gap/spread med. | Rank med. (max) | ΔSharpe med. (worst) |
|---:|---:|---:|---:|---:|---:|
| 12 | 26/30 | 0.0000 % (0.2900 %) | 0.00 % | 1 (2) | +0.0 % (−31.3 %) |
| 16 | 17/30 | 0.0000 % (0.0781 %) | 0.00 % | 1 (6) | +0.0 % (−34.3 %) |
| 20 | 0/30 | 0.0582 % (0.2236 %) | 8.94 % | 52 (504) | −7.3 % (−61.4 %) |

Every one of the 90 runs returned a feasible lowest-energy shot; §V-B shows that this
is a property of seed 42 rather than of the configuration — across the five seeds of the cross,
14 of 150 N = 16 runs at A_heur returned no feasible bitstring at all. Table S2's "exact agreement for N ≤ 16" is true of the reference
instance and holds in 26 of 30 instances at N = 12 but only 17 of 30 at N = 16;
at N = 20 it never happens. Reporting it from one instance overstated it.

The 47 runs that missed carry the paper's sharpest evidence that an objective gap
is not a portfolio guarantee, and the direction of the error is not fixed. At
N = 12 every miss is *rank 2* — the second-best portfolio in the entire feasible
set — and the four of them sit at ΔSharpe = **+60.2 %, −2.2 %, −20.5 % and
−31.3 %**: the same rank, in the same instance family, is worth anything from a
60 % improvement to a third of the Sharpe ratio. At N = 16 the worst miss reports
a gap of 0.0228 %, while its
Sharpe falls from 1.089 to 0.716, a 34.3 % loss, at rank 4 of 560. At N = 20 one
instance reports 0.1050 % and loses 61.4 %. Across all 47 misses, 17 have a
Sharpe ratio *above* the exact optimum's, so the correct statement is not that
small gaps hide losses but that they carry no sign.

The association has to be characterised carefully, and it is weaker than
"uncorrelated" but not absent. All correlations below are against |ΔSharpe|, the
magnitude of the deviation, not its signed value. Pooled over the 47 misses,
Pearson r = 0.17 (p = 0.25) but Spearman ρ = 0.34 (p = 0.019). Part of that
is a size confound — N = 20 supplies both the larger gaps (median 0.058 %) and
the larger Sharpe deviations (median 7.3 %) — but not all of it: *within* N = 20
alone, Spearman ρ = 0.43 (p = 0.019, n = 30), against Pearson +0.22
(p = 0.24) on the same 30 points. At the two smaller sizes the rank association
is absent or negative (ρ = −0.60 at N = 12, n = 4; +0.06 at N = 16, n = 13;
neither significant). The two p-values being equal to three decimals is a coincidence of rounding, not a
transcription error: they are 0.0187 and 0.0189 under SciPy's asymptotic t
approximation. Reporting only the Pearson coefficients within size would have dismissed a relationship that a
reader recomputing from the committed CSV will find. Two things keep it from
rescuing the metric: it is one of three within-size tests, so its p = 0.019
becomes 0.057 under a Bonferroni correction, and a rank correlation of 0.43 on
30 points is a loose ordering, not a predictor. The rank of the
returned solution within the feasible set does no better, at Spearman 0.13
against |ΔSharpe| (p = 0.40).

The defensible claim is therefore narrower than "the gap says nothing", and
still damaging: **the reported gap predicts neither the sign of the Sharpe
deviation nor, usefully, its size**. The sign result is unconditional — 17 of the
47 misses beat the exact optimum's Sharpe ratio, so a small gap does not even
tell a reader which direction they moved. The magnitude survives only as a weak
rank tendency at one size, next to individual pairs that contradict it outright:
0.0228 % against a 34.3 % loss, and four solutions of identical rank spanning
+60.2 % to −31.3 %. A quantity that ranks 30 points loosely while mispredicting
individual ones by that much is not a portfolio guarantee, which is what
Fig. S7 shows directly.

## S-V. The cross in detail (E1, E2, E5)

The full tables behind Sec. V-B: the decomposition of the paired gap ratio at every size (Table
S13), the four samplers at every shot budget (Table S14), the two-window comparison in full (Table
S15), the disagreement of the two success rules over both thresholds (Table S16), the split of P_τ
into feasible mass and selection within it (Table S17), the conditional distance of every run from
uniform on F (Table S18), and the design subsets on which the main contrast depends (Table S19).

**Table S13. Decomposition of the A_margin / A_heur ratio of reported gaps.** Pairs with a zero gap
on either side, or without a feasible shot, are counted and excluded from the ratio. The identity
holds pair by pair, and a median does not carry it: the median of a product is not the product of the
medians. The medians are therefore reported as levels and the decomposition is taken in the
geometric mean, which is multiplicative over the same pairs because log turns the product into a sum.

| N | Pairs used (zero gap / no feasible shot) | Medians: g_off ratio / solution (IQR) / denominator | Geometric mean: g_off ratio = solution × denominator | Solution part < 1 |
|---:|---:|---|---|---:|
| 12 | 16 (134 / 0) | 29.4 / 0.54 (0.42–0.91) / 46.8 | 34.7 = 0.71 × 49.1 | 75 % |
| 16 | 49 (80 / 21) | 58.1 / 1.00 (0.49–2.18) / 48.2 | 45.8 = 1.00 × 45.6 | 43 % |
| 20 | 128 (21 / 1) | 29.0 / 0.81 (0.39–1.34) / 40.0 | 27.8 = 0.76 × 36.6 | 59 % |

At N = 20 the A_margin run reports a gap that is 29 times larger in the median, and 27.8 times larger
in the geometric mean, than the A_heur run on the same instance and seed. The decomposition assigns
36.6 of that to the denominator and 0.76 to the solution, which is *better*: it is better in the
median too (0.81) and on 59 % of pairs. The reported gap moves because the denominator moves, with
the solution pulling the other way.

**Table S14. Q_τ(S) for four samplers at A_heur (medians over instances of the within-instance
median over seeds; τ = 1 % of Δ_F).** The two uniform samplers are instance properties. The last
three columns count runs whose optimised state beats the uniform-over-F sampler, decided on the
one-shot probability so the count is the same at every S: at τ = 1 %, at the exact optimum and at
τ = 5 %.

| N | S | Optimised QAOA | Unoptimised (same initial angles) | Uniform over F | Uniform over 2^N | QAOA beats uniform-F, τ = 1 % | exact optimum | τ = 5 % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 10 | 0.026 | 0.017 | 0.142 | 0.0024 | 0 / 150 | 0 / 150 | 0 / 150 |
| 12 | 1,000 | 0.928 | 0.817 | 1.000 | 0.217 | 0 / 150 | 0 / 150 | 0 / 150 |
| 16 | 100 | 0.040 | 0.006 | 0.164 | 0.0015 | 0 / 150 | 0 / 150 | 0 / 150 |
| 16 | 1,000 | 0.336 | 0.062 | 0.833 | 0.015 | 0 / 150 | 0 / 150 | 0 / 150 |
| 16 | 10,000 | 0.983 | 0.470 | 1.000 | 0.142 | 0 / 150 | 0 / 150 | 0 / 150 |
| 20 | 100 | 0.006 | 0.002 | 0.040 | 0.0002 | 0 / 150 | 0 / 150 | 0 / 150 |
| 20 | 1,000 | 0.059 | 0.019 | 0.338 | 0.0019 | 0 / 150 | 0 / 150 | 0 / 150 |
| 20 | 10,000 | 0.458 | 0.171 | 0.984 | 0.019 | 0 / 150 | 0 / 150 | 0 / 150 |

**Table S15. The E1 block on two price windows (E5).** Instances 0–9 × seeds 42–46 × {A_heur,
A_margin}, 300 runs per window; "2023–26" is the paper's window (2023-08-06 – 2026-08-05) and
"2020–23" the adjacent earlier one (2020-08-05 – 2023-08-05). Levels are medians over the 50 runs
per cell; paired rows are A_margin − A_heur reduced to an instance median and bootstrapped over the
ten instances (95 % interval), with the count of instances on which A_margin is better (ties in
parentheses); g_off and g_F are of the best feasible shot. The ratio rows give two aggregations of the A_margin / A_heur g_off ratio: the medians of the
ratio, of its solution part and of its denominator part, and the geometric-mean decomposition,
which is multiplicative over the same pairs (Sec. V-B). The E2 row counts A_heur runs whose
optimised state beats the uniform-over-F sampler at S = 1,000.

| Quantity | N = 12, 2023–26 | N = 12, 2020–23 | N = 16, 2023–26 | N = 16, 2020–23 | N = 20, 2023–26 | N = 20, 2020–23 |
|---|---:|---:|---:|---:|---:|---:|
| Δ_F, median | 0.705 | 0.942 | 1.296 | 1.736 | 2.195 | 2.710 |
| A_heur / A_crit, median (instances with A_crit = 0) | 92 (4) | 199 (3) | 139 (0) | 115 (0) | 99 (0) | 92 (0) |
| A_heur: Q_τ(1000) | 0.926 | 0.933 | 0.336 | 0.395 | 0.060 | 0.098 |
| A_heur: g_off (%) / g_F (%) | 0.000 / 0.00 | 0.000 / 0.00 | 0.023 / 2.19 | 0.017 / 1.64 | 0.043 / 6.48 | 0.036 / 4.64 |
| A_heur: pass g_off ≤ 0.1 %, fail g_F ≤ 1 % | 0.04 | 0.04 | 0.47 | 0.49 | 0.74 | 0.66 |
| A_heur: runs with no feasible shot | 0 | 0 | 3 | 7 | 0 | 0 |
| Paired Q_τ(1000) [CI]; better of 10 | +0.020 [+0.000, +0.067]; 8 | +0.010 [+0.000, +0.061]; 9 | +0.041 [+0.004, +0.062]; 10 | +0.057 [+0.029, +0.074]; 10 | +0.090 [+0.067, +0.202]; 10 | +0.122 [+0.062, +0.219]; 10 |
| Paired g_off (pt) [CI]; better of 10 | 0 [0, 0]; 0 (9) | 0 [0, 0]; 0 (10) | +0.28 [−0.00, +1.41]; 2 (2) | +0.19 [−0.01, +0.76]; 4 (0) | +0.63 [+0.41, +0.95]; 1 (0) | +0.52 [+0.20, +0.90]; 0 (0) |
| Paired g_F (pt) [CI]; better of 10 | 0 [0, 0]; 0 (9) | 0 [0, 0]; 0 (10) | 0.00 [−2.31, +0.07]; 4 (4) | −0.22 [−2.05, +0.13]; 5 (3) | −1.75 [−4.49, −0.73]; 9 (0) | −1.93 [−3.80, 0.00]; 7 (2) |
| g_off ratio / solution / denominator, medians (pairs) | 27.5 / 0.50 / 54.2 (6) | 44.4 / 0.92 / 48.4 (3) | 60.8 / 1.26 / 47.8 (20) | 44.7 / 1.62 / 34.3 (17) | 26.3 / 0.74 / 39.5 (42) | 18.0 / 0.67 / 27.8 (44) |
| g_off ratio = solution × denominator, geometric means | 25.5 = 0.50 × 51.3 | 18.7 = 0.38 × 49.6 | 42.7 = 0.98 × 43.7 | 69.1 = 2.05 × 33.7 | 22.3 = 0.64 × 34.6 | 12.4 = 0.45 × 27.4 |
| E2: A_heur beats uniform-F at S = 1,000 | 0 / 50 | 0 / 50 | 0 / 50 | 0 / 50 | 0 / 50 | 0 / 50 |

Sec. V-B reports that 76 % of N = 20 heuristic-weight runs pass a 0.1 %
offset-normalized gap and miss a 1 % feasible-range criterion. Both rules have a
threshold, and one pair of thresholds is one point on a surface. Table S16 reads
both rules at four gap thresholds and three quality thresholds on the same runs.

The disagreement is monotone in the gap threshold and concentrated on the
heuristic weight, which is what the deflation account predicts: at A_heur and
N = 20 it runs 0.06, 0.49, 0.76, 0.89 across gap thresholds of 0.01 %, 0.05 %,
0.1 % and 0.2 %, while at A_margin it stays between 0.03 and 0.07 at every one
of them. The 0.1 % figure is neither the largest available nor a threshold at
which something special happens; the tightest threshold nearly removes the
disagreement, because a gap of 0.01 % under a 174-fold deflation is still a
demanding requirement.

**Table S16. Disagreement between the two success rules, over both thresholds.**
Share of runs in which exactly one of the two rules passes. Columns are the
offset-normalized gap threshold; "quality" is the feasible-range criterion
g_F ≤ τ, with the exact optimum as its limiting case.

| N | A | quality rule | 0.01 % | 0.05 % | 0.1 % | 0.2 % |
|---:|---|---|---:|---:|---:|---:|
| 12 | A_heur | 1 % | 0.01 | 0.01 | 0.03 | 0.06 |
| 12 | A_heur | 5 % | 0.03 | 0.01 | 0.01 | 0.04 |
| 12 | A_heur | exact | 0.01 | 0.03 | 0.05 | 0.07 |
| 12 | A_margin | 1 % | 0.01 | 0.01 | 0.01 | 0.01 |
| 12 | A_margin | 5 % | 0.05 | 0.05 | 0.05 | 0.04 |
| 12 | A_margin | exact | 0.00 | 0.00 | 0.00 | 0.01 |
| 16 | A_heur | 1 % | 0.05 | 0.33 | 0.46 | 0.51 |
| 16 | A_heur | 5 % | 0.24 | 0.04 | 0.16 | 0.22 |
| 16 | A_heur | exact | 0.07 | 0.35 | 0.48 | 0.54 |
| 16 | A_margin | 1 % | 0.04 | 0.04 | 0.03 | 0.01 |
| 16 | A_margin | 5 % | 0.26 | 0.26 | 0.24 | 0.23 |
| 16 | A_margin | exact | 0.00 | 0.00 | 0.01 | 0.03 |
| 20 | A_heur | 1 % | 0.06 | 0.49 | 0.76 | 0.89 |
| 20 | A_heur | 5 % | 0.32 | 0.11 | 0.38 | 0.52 |
| 20 | A_heur | exact | 0.09 | 0.52 | 0.79 | 0.93 |
| 20 | A_margin | 1 % | 0.07 | 0.07 | 0.06 | 0.03 |
| 20 | A_margin | 5 % | 0.43 | 0.43 | 0.43 | 0.39 |
| 20 | A_margin | exact | 0.00 | 0.00 | 0.01 | 0.04 |

*Feasible mass and selection within it.* The one-shot quality probability factors
as P_τ = P_F · Pr[g_F ≤ τ | x ∈ F], and the second factor can be compared with
what an indifferent draw from the feasible set would give, |F_τ|/|F|. Their
ratio R_τ is 1 for a state that finds the feasible set but sorts nothing inside
it and greater than 1 for one that concentrates on the good part. It is computed
per run and then summarised; dividing the medians of P_τ and P_F would be a
different quantity.

Table S17 measures it, and the heuristic weight comes out at R_τ = 1.00: 0.998,
0.998 and 1.009 at N = 12, 16 and 20, with interquartile ranges of 0.02 or less.
Conditioned on landing in the feasible set, the optimised state at A_heur picks
a solution the way a uniform draw from F would. The margin weight does sort, at
the two smaller sizes — R_τ = 1.40 (IQR 1.04–1.97) at N = 12 and 1.55
(1.21–2.04) at N = 16, above 1 on 114 and 125 of 150 runs — and stops doing so
at N = 20 (1.01, IQR 0.80–2.10).

R_τ checks the mass on one subset of F only. A ratio of 1 is consistent with the
conditional distribution being uniform on F and also with its being anything else
that puts the same mass in F_τ, so the uniformity Sec. V-B reports is established by
the conditional distance of Table S18, not by this table.

**Table S17. P_τ split into feasible mass and selection within the feasible set.**
Medians over the 150 runs of each cell. |F_τ|/|F| is the share of the feasible
set within τ = 1 % of the feasible range, a property of the instance. R_τ is the
per-run ratio of the conditional rate to that share.

| N | A | P_F | Pr[g_F ≤ τ \| x ∈ F] | \|F_τ\|/\|F\| | R_τ (IQR) | R_τ > 1 |
|---:|---|---:|---:|---:|---|---:|
| 12 | A_heur | 0.179 | 1.52·10⁻² | 1.52·10⁻² | 1.00 (0.99–1.01) | 68/150 |
| 12 | A_margin | 0.328 | 2.20·10⁻² | 1.52·10⁻² | 1.40 (1.04–1.97) | 114/150 |
| 16 | A_heur | 0.229 | 1.78·10⁻³ | 1.79·10⁻³ | 1.00 (1.00–1.00) | 50/150 |
| 16 | A_margin | 0.301 | 3.01·10⁻³ | 1.79·10⁻³ | 1.55 (1.21–2.04) | 125/150 |
| 20 | A_heur | 0.167 | 4.07·10⁻⁴ | 4.13·10⁻⁴ | 1.01 (0.99–1.01) | 90/150 |
| 20 | A_margin | 0.303 | 4.19·10⁻⁴ | 4.13·10⁻⁴ | 1.01 (0.80–2.10) | 85/150 |

### The conditional distribution on the feasible set

For every E1 run the artifact holds the exact probability p(x) of every feasible bitstring under
the final state and under the initial state. The distance of the conditional distribution from
uniform on F,

> D_cond = ½ Σ_{x∈F} | p(x)/P_F − 1/|F| |,

is 0 for a uniform conditional distribution and 1 − 1/|F| for a point mass; it is undefined at
P_F = 0, and runs with P_F below 10⁻⁹ are counted rather than normalised (none occurred). Two
reference distances are printed beside it: the same distance for the *initial* state of the run,
and the mean distance a 1,000-shot empirical distribution drawn from uniform-on-F would show at
that |F| (200 Monte Carlo draws), which is what uniform looks like at the experiment's own shot
resolution — 0.10 at |F| = 66 and 0.81 at |F| = 4,845, so the question could not have been settled
from the shots and is settled from the exact probabilities. Table S18 gives the distribution over
runs per cell; the per-run file is `paper01_e1_conditional_tv.csv`.

At A_heur the optimised state is close to uniform on F — within 0.04 at N = 12 and 0.01 at N ≥ 16
in every run, 0.001–0.002 in the median. The initial state is already close in the median but not
in every run (as far as 0.41 at N = 20 for some seeds), so the optimisation removes the ordering
the random angles carried rather than adding one (Table S34 pairs each run's initial and final
state). Every other weight departs further from uniform: the closer A is to A_crit the more, and
A_margin sits between. D_cond carries no direction; whether the departure favours the good
solutions is read from R_τ in Table S17 (above 1 at A_margin for N ≤ 16).

**Table S18. Distance of the conditional distribution on F from uniform, per cell.** Median, IQR
and range over the runs of each cell; "initial" is the median for the unoptimised state at the same
angles; "uniform, 1,000 shots" is what a finite-shot draw from the uniform distribution would show;
"point mass" is the distance of a single-bitstring state. Runs with P_F = 0 would be counted in the
"undefined" column; there were none.

| N | A | Runs (undefined) | P_F | D_cond median (IQR) | D_cond min – max | Initial-state D_cond | Uniform, 1,000 shots | Point mass |
|---:|---|---:|---:|---|---|---:|---:|---:|
| 12 | A = 0 | 45 (0) | 0.199 | 0.554 (0.480–0.603) | 0.303 – 0.690 | 0.456 | 0.102 | 0.985 |
| 12 | 1.1·A_crit | 105 (0) | 0.257 | 0.382 (0.208–0.498) | 0.053 – 0.747 | 0.302 | 0.102 | 0.985 |
| 12 | 2·A_crit | 105 (0) | 0.277 | 0.308 (0.148–0.456) | 0.036 – 0.969 | 0.238 | 0.102 | 0.985 |
| 12 | 5·A_crit | 105 (0) | 0.244 | 0.207 (0.070–0.389) | 0.012 – 0.891 | 0.172 | 0.102 | 0.985 |
| 12 | 10·A_crit | 105 (0) | 0.190 | 0.108 (0.030–0.267) | 0.008 – 0.844 | 0.090 | 0.102 | 0.985 |
| 12 | A_margin | 150 (0) | 0.328 | 0.173 (0.089–0.393) | 0.027 – 0.701 | 0.195 | 0.102 | 0.985 |
| 12 | A_heur | 150 (0) | 0.179 | 0.002 (0.001–0.013) | 0.000 – 0.038 | 0.003 | 0.102 | 0.985 |
| 16 | 1.1·A_crit | 150 (0) | 0.245 | 0.369 (0.201–0.545) | 0.072 – 0.844 | 0.354 | 0.301 | 0.998 |
| 16 | 2·A_crit | 150 (0) | 0.275 | 0.192 (0.117–0.321) | 0.035 – 0.888 | 0.369 | 0.301 | 0.998 |
| 16 | 5·A_crit | 150 (0) | 0.359 | 0.106 (0.050–0.209) | 0.020 – 0.749 | 0.342 | 0.301 | 0.998 |
| 16 | 10·A_crit | 150 (0) | 0.309 | 0.080 (0.022–0.136) | 0.003 – 0.465 | 0.122 | 0.301 | 0.998 |
| 16 | A_margin | 150 (0) | 0.301 | 0.139 (0.087–0.214) | 0.014 – 0.464 | 0.377 | 0.301 | 0.998 |
| 16 | A_heur | 150 (0) | 0.229 | 0.001 (0.001–0.003) | 0.000 – 0.009 | 0.002 | 0.301 | 0.998 |
| 20 | 1.1·A_crit | 150 (0) | 0.238 | 0.494 (0.239–0.614) | 0.051 – 0.867 | 0.354 | 0.813 | 1.000 |
| 20 | 2·A_crit | 150 (0) | 0.224 | 0.198 (0.080–0.401) | 0.008 – 0.827 | 0.228 | 0.813 | 1.000 |
| 20 | 5·A_crit | 150 (0) | 0.233 | 0.061 (0.022–0.114) | 0.005 – 0.479 | 0.095 | 0.813 | 1.000 |
| 20 | 10·A_crit | 150 (0) | 0.219 | 0.036 (0.014–0.067) | 0.003 – 0.366 | 0.054 | 0.813 | 1.000 |
| 20 | A_margin | 150 (0) | 0.303 | 0.090 (0.022–0.148) | 0.006 – 0.387 | 0.145 | 0.813 | 1.000 |
| 20 | A_heur | 150 (0) | 0.167 | 0.002 (0.001–0.003) | 0.001 – 0.005 | 0.004 | 0.813 | 1.000 |

### Which parts of the design the main contrast depends on

The interval Sec. V-B quotes for A_margin − A_heur is over the thirty instances
of a size, with the same five initial-angle seeds inside every one and instance
0 — the reference instance — deliberately among them. Table S19 re-runs the same
paired contrast on subsets of that design. The subsets are of the design and not
of the outcome: nothing is selected on the value being summarised.

Three of them change nothing. Dropping instance 0 moves the N = 20 median from
+0.086 to +0.083 and the N = 16 median from +0.0397 to +0.0390. Splitting the
N = 12 instances by whether A_crit is zero gives +0.011 on the twenty-one
positive ones and +0.013 on the nine zero ones, both with intervals clear of
zero. Dropping any one seed leaves the N = 20 median between +0.044 and +0.112,
improved on 30 of 30 instances every time.

The fourth changes the sign. Conditioned on a *single* seed, one of the five
reverses the contrast at each of the two larger sizes: seed 45 at N = 16
(−0.157, better on 0 of 30 instances) and seed 44 at N = 20 (−0.096, 0 of 30).
Which seed it is differs by size, and at N = 12 none reverses though two have
intervals containing zero. This is the paper's own finding about initial angles
arriving in its statistics: the instance-level interval is an interval for the
contrast *averaged over these five initial angles*, and it is not an interval
for the contrast at an angle drawn from any wider distribution. Readers wanting
the latter need a design that samples initialisations, which this one does not.

**Table S19. The A_margin − A_heur contrast in Q_τ(1000) under subsets of the
design.** Instance-level median difference with its 95 % bootstrap interval, and
the count of instances on which A_margin is better. "A_crit = 0" exists only at
N = 12.

| Subset | N = 12 | N = 16 | N = 20 |
|---|---|---|---|
| all | +0.011 [+0.005, +0.034]; 27/30 | +0.040 [+0.031, +0.061]; 30/30 | +0.086 [+0.071, +0.116]; 30/30 |
| without instance 0 | +0.011 [+0.007, +0.026]; 26/29 | +0.039 [+0.029, +0.061]; 29/29 | +0.083 [+0.067, +0.111]; 29/29 |
| instance 0 only | +0.074; 1/1 | +0.043; 1/1 | +0.231; 1/1 |
| A_crit > 0 instances | +0.011 [+0.000, +0.026]; 18/21 | +0.040 [+0.030, +0.061]; 30/30 | +0.086 [+0.072, +0.116]; 30/30 |
| A_crit = 0 instances | +0.013 [+0.001, +0.063]; 9/9 | — | — |
| seed 42 only | +0.067 [−0.053, +0.073]; 20/30 | +0.037 [+0.025, +0.043]; 26/30 | +0.088 [+0.080, +0.153]; 30/30 |
| seed 43 only | +0.095 [+0.074, +0.173]; 30/30 | +0.003 [+0.002, +0.004]; 30/30 | +0.001 [+0.001, +0.002]; 30/30 |
| seed 44 only | +0.000 [+0.000, +0.000]; 26/30 | +0.384 [+0.359, +0.414]; 30/30 | −0.096 [−0.119, −0.065]; 0/30 |
| seed 45 only | +0.006 [−0.130, +0.074]; 17/30 | −0.157 [−0.162, −0.152]; 0/30 | +0.130 [+0.096, +0.155]; 30/30 |
| seed 46 only | +0.011 [+0.010, +0.012]; 29/30 | +0.367 [+0.302, +0.428]; 30/30 | +0.124 [+0.109, +0.170]; 30/30 |
| without seed 42 | +0.006 [+0.006, +0.039]; 29/30 | +0.188 [+0.159, +0.208]; 30/30 | +0.055 [+0.047, +0.070]; 30/30 |
| without seed 43 | +0.006 [+0.003, +0.022]; 24/30 | +0.194 [+0.154, +0.222]; 30/30 | +0.112 [+0.083, +0.130]; 30/30 |
| without seed 44 | +0.040 [+0.010, +0.057]; 24/30 | +0.022 [+0.017, +0.029]; 27/30 | +0.112 [+0.083, +0.130]; 30/30 |
| without seed 45 | +0.031 [+0.006, +0.041]; 29/30 | +0.217 [+0.172, +0.228]; 30/30 | +0.045 [+0.039, +0.065]; 30/30 |
| without seed 46 | +0.035 [+0.008, +0.056]; 24/30 | +0.022 [+0.017, +0.029]; 27/30 | +0.044 [+0.040, +0.061]; 30/30 |

## S-VI. Depth, optimizer budget and the optimizer controls (E3)

The single-seed depth sweep (Table S20) and the steps × p × seed grid (Table S21, Fig. S10) that
Sec. V-C summarises; the optimizer controls in full (Table S22, Fig. S11), the dispersion behind
each signed median (Table S23), the stopping-criterion diagnostic at both precisions, and the
multistart re-analysis at equal shots (Table S24).

*Depth, optimizer budget, and what the reported gap actually measures (single-seed grid).*

**Table S20. Optimality gap and runtime vs. QAOA depth p (seed 42, exact classical reference).**

| N | p = 1 | p = 2 | p = 3 | p = 4 | runtime p = 1 → 4 |
|---:|---:|---:|---:|---:|---|
| 12 | 0.1949 % | 0.0000 % | 0.0000 % | 0.0000 % | 5.6 s → 18.6 s |
| 16 | 0.2197 % | 0.0000 % | 0.0516 % | 0.0427 % | 9.3 s → 30.0 s |
| 20 | 0.2651 % | 0.0528 % | 0.0089 % | 0.0563 % | 16.9 s → 54.7 s |

Runtime is essentially linear in p (Fig. S9, right), as expected: depth multiplies the per-step circuit
work while the 50-step optimizer budget stays fixed. Quality, however, is not monotone (Fig. S9, left).
Going from p = 1 to p = 2 removes the gap outright at N = 12 and N = 16 — both reach the proven
optimum — and cuts it 5.0× at N = 20, which justifies the p = 2 used throughout §V-A. Beyond p = 2 the
ordering is not stable and does not repeat across sizes: p = 3 improves N = 20 (0.0528 % → 0.0089 %)
but loses the exact hit at N = 16 (0.0000 % → 0.0516 %), while p = 4 recovers part of that loss at
N = 16 (0.0427 %) and regresses N = 20 to 0.0563 %. Depth beyond p = 2 changes which basin the run
lands in without systematically improving it, at roughly double the runtime. Table S20 holds the ADAM
budget at 50 steps, so it cannot by itself say whether the p ≥ 3 non-monotonicity is the ansatz or
simply too few optimizer iterations for the larger parameter vector. We resolve that with a grid.

**The steps × p × seed grid.** We ran QAOA over N ∈ {12, 16, 20} × p ∈ {1, 2, 3, 4} ×
steps ∈ {50, 100, 400} × five seeds (180 runs), recording both the reported gap and the optimized
objective ⟨H_norm⟩ (Fig. S10). Two things separate cleanly.

*The optimizer is not budget-starved.* The optimized objective saturates by 50–100 ADAM steps and does
not improve with an 8× larger budget; several cells are marginally *worse* at 400 steps because ADAM
with a fixed 0.1 step size wanders near the minimum rather than settling. Table S21 shows seed 42.

**Table S21. Optimized expectation at each optimizer budget, seed 42, three cells of the grid.**

| final ⟨H_norm⟩ | steps = 50 | 100 | 400 |
|---|---:|---:|---:|
| N = 16, p = 2 | −4.9959 | −5.0669 | −5.0674 |
| N = 16, p = 4 | −4.3513 | −4.9416 | −4.9796 |
| N = 20, p = 3 | −3.5908 | −3.9047 | −3.8730 |

So the p ≥ 3 non-monotonicity in Table S20 is not a shortage of iterations *for this optimizer
setting*: ADAM at a fixed step of 0.1 does not recover it with eight times the budget. Whether a
smaller step or a quasi-Newton method would is what E3 below tests at p = 2.

*What does move is the initial angle.* At a fixed, saturated budget the depth that lands in a bad basin
changes with N: at N = 16 it is p = 3 (⟨H_norm⟩ = −2.89, against p = 2's −5.07), while at N = 20 it is
p = 2 (−0.91, against p = 3's −3.87). A monotone property of the ansatz would not flip which depth is
unlucky from one size to the next; a 2p-dimensional non-convex landscape with a fixed random start
would. This is the seed variance of §V-B appearing on the depth axis — more parameters give the
optimizer more bad basins to fall into — which is consistent with an initialisation effect; the runs do not separate
initialisation, local minima and expressivity, and the cause is not identified. A fixed-step ADAM that stalls in the same basin at 50 and at 400 steps has not
been shown to explore, so on the grid alone "not budget" meant "not iteration count", not "not
optimizer"; E3 closes that gap for p = 2.

*The reported gap can hide all of this.* Because the gap is the best of 1000 final samples, it is a
minimum over samples, not the quantity the optimizer descended. A run can barely optimize and still
report a small gap: at N = 20, steps = 400, p = 2 the optimizer effectively failed
(⟨H_norm⟩ = −0.91, scarcely below p = 1's −0.63) yet its reported gap is 0.038 %, indistinguishable
from the well-optimized p = 4 in the same cell (⟨H_norm⟩ = −5.01, gap 0.022 %). Fig. S10 therefore plots
both rows — optimized ⟨H_norm⟩ on top, reported gap below — precisely so the disagreement is visible.
The methodological point is that **a QAOA benchmark that reports only a best-of-shots gap can present a
failed optimization as a success**; the optimized objective must be reported alongside it. At this optimizer setting p = 2 is where depth stops helping monotonically; beyond it the
iteration count is saturated, and the remaining variation is not attributed to a cause.

### The optimizer controls in full

**Table S22. Optimizer controls on the E3 block.** Levels are medians over the 50 runs per cell;
the paired columns are the instance-level median difference from the baseline with its 95 %
bootstrap interval, and the count of instances (of 10) on which the control reached a lower ⟨H_norm⟩.
Budget is the median number of gradient evaluations actually used; every ADAM step is one objective
and one gradient evaluation, and for L-BFGS-B the objective and gradient counts were equal in every
one of the 300 runs, n_f = n_∇, line-search evaluations included (each trial point evaluates both),
so the two budgets are comparable as printed. The last column is the median optimisation wall-clock. "Optimum hit" counts runs whose best feasible shot is the
exact optimum over *all fifty* runs of the cell, with the number that produced no feasible shot in
parentheses; a run that returns nothing usable is a failure to hit, not an absent observation.
Conditioning on a feasible shot instead would raise N = 16, A_heur, L-BFGS-B from 0.34 to 0.40 — eight
of its fifty runs found no feasible shot, against three of the baseline's — and widen a margin of
three runs over the baseline (0.28) into one of ten points (0.30 conditioned the same way).
"Predicted", in the same layout (hits, then no-feasible runs in parentheses), is what the cell's
fifty recorded exact states give for the same two counts under independent draws, Σ_j [1 − (1 − P_0,j)^1000] hits and Σ_j (1 − P_F,j)^1000 runs without a feasible
shot, with P_0 the one-shot probability of the exact optimum (`expected_optimum_hits` and
`expected_no_feasible_shot_runs` in `paper01_e3_summary.csv`); each run has its own probability, so
the count is a sum of Bernoulli variables with different rates, not a binomial. The ADAM cells
observe within four of the predicted hits and within three of the predicted no-feasible runs. This
comparison is a check on the batch, not on the angles — it is what exposed the L-BFGS-B handoff
defect of Sec. VI-C (the pre-fix arm observed 6 hits against 14.8 predicted at N = 16, A_heur, and 9
runs without a feasible shot against 0.2 at N = 20, A_heur) — and the argument check is the
regression test named in Sec. S-VI. The re-run's L-BFGS-B cells observe within 3.0 of the predicted
hits and within 2.7 of the predicted no-feasible runs, as the ADAM cells do.
ΔP_τ and ΔQ_τ(1000) are separate measurements on the same pairs and are not interchangeable: Q is a
batch quantity, ∂Q_τ(S)/∂P_τ = S(1 − P_τ)^(S−1), so a difference in P_τ too small to print can be
worth a tenth of a batch. Table S23 adds the median |Δ|, its 95th percentile and the
spread behind each signed median. The ⟨H_norm⟩ *levels* at N = 16 and N = 20 sit on the
boundary between two basins the runs fall into (≈ −2.2 and ≈ −5.1 at N = 16), so a shift of 10⁻³ in
individual runs can move the median across it; the paired column is the measurement.

| N | A | Setting | Budget (∇, s) | ⟨H_norm⟩ | Δ⟨H_norm⟩ vs. baseline [CI]; lower in | P_τ | ΔP_τ [CI] | Q_τ(1000) | ΔQ_τ(1000) [CI] | Optimum hit / 50 (no feas.) | Predicted |
|---:|---|---|---:|---:|---|---:|---|---:|---|---:|---:|
| 12 | A_heur | ADAM 0.1 × 50 | 50, 10 | −2.121 | — | 2.6·10⁻³ | — | 0.926 | — | 38/50 (0) | 39.4 (0.0) |
| 12 | A_heur | ADAM 0.03 × 100 | 100, 20 | −2.124 | −2.4·10⁻³ [−2.5, −2.3]·10⁻³; 10/10 | 2.7·10⁻³ | −2.4·10⁻⁶ [−9.4·10⁻⁵, +1.1·10⁻⁵] | 0.934 | +0.0010 [−0.0001, +0.0036] | 42/50 (0) | 39.2 (0.0) |
| 12 | A_heur | L-BFGS-B | 52, 13 | −3.937 | −3.0·10⁻³ [−3.1, −2.4]·10⁻³; 10/10 | 5.4·10⁻³ | +3.8·10⁻⁴ [+1.2, +3.9]·10⁻⁴ | 0.995 | +0.0087 [+0.0044, +0.0094] | 40/50 (0) | 39.6 (0.0) |
| 16 | A_heur | ADAM 0.1 × 50 | 50, 17 | −5.065 | — | 4.1·10⁻⁴ | — | 0.336 | — | 14/50 (3) | 14.1 (5.8) |
| 16 | A_heur | ADAM 0.03 × 100 | 100, 35 | −2.173 | −6.5·10⁻³ [−6.8, −6.2]·10⁻³; 10/10 | 4.6·10⁻⁴ | +2·10⁻⁸ [−6·10⁻⁹, +3·10⁻⁸] | 0.370 | +0.0000 [−0.0000, +0.0000] | 14/50 (4) | 13.6 (5.8) |
| 16 | A_heur | L-BFGS-B | 53, 21 | −2.173 | −3.6·10⁻³ [−3.7, −3.6]·10⁻³; 9/10 | 4.7·10⁻⁴ | +1.6·10⁻⁷ [+1.3, +2.5]·10⁻⁷ | 0.377 | +0.0002 [+0.0001, +0.0002] | 17/50 (8) | 14.8 (5.3) |
| 20 | A_heur | ADAM 0.1 × 50 | 50, 30 | −4.556 | — | 6.2·10⁻⁵ | — | 0.060 | — | 2/50 (0) | 2.1 (0.1) |
| 20 | A_heur | ADAM 0.03 × 100 | 100, 61 | −2.211 | **+2.9·10⁻³** [+2.8, +3.2]·10⁻³; 0/10 | 2.5·10⁻⁵ | −6.1·10⁻⁶ [−9.3, −2.7]·10⁻⁶ | 0.025 | −0.0060 [−0.0092, −0.0027] | 1/50 (0) | 1.4 (0.2) |
| 20 | A_heur | L-BFGS-B | 51, 36 | −2.258 | −1.4·10⁻³ [−1.5, −1.2]·10⁻³; 10/10 | 7.6·10⁻⁵ | −1.5·10⁻⁷ [−2.5, −0.7]·10⁻⁷ | 0.074 | −0.0002 [−0.0002, −0.0001] | 1/50 (0) | 2.2 (0.2) |
| 12 | A_margin | ADAM 0.1 × 50 | 50, 10 | −4.069 | — | 6.5·10⁻³ | — | 0.998 | — | 38/50 (0) | 39.5 (0.0) |
| 12 | A_margin | ADAM 0.03 × 100 | 100, 20 | −3.796 | +1.3·10⁻³ [−1·10⁻⁴, +9.6·10⁻²]; 4/10 | 4.9·10⁻³ | −3·10⁻⁶ [−2.8, +2.6]·10⁻⁴ | 0.993 | −0.0000 [−0.0030, +0.0014] | 36/50 (0) | 37.7 (0.0) |
| 12 | A_margin | L-BFGS-B | 65, 17 | −4.114 | −4.3·10⁻³ [−15.9, −1.5]·10⁻³; 10/10 | 5.4·10⁻³ | +8.5·10⁻⁵ [−2.5, +4.7]·10⁻⁴ | 0.996 | +0.0000 [−0.0029, +0.0010] | 36/50 (0) | 39.0 (0.0) |
| 16 | A_margin | ADAM 0.1 × 50 | 50, 17 | −5.242 | — | 7.4·10⁻⁴ | — | 0.523 | — | 20/50 (2) | 21.2 (4.0) |
| 16 | A_margin | ADAM 0.03 × 100 | 100, 35 | −5.017 | −1.0·10⁻³ [−1.2·10⁻³, +5.5·10⁻²]; 7/10 | 4.6·10⁻⁴ | −6.1·10⁻⁵ [−10.4, −1.9]·10⁻⁵ | 0.368 | −0.0281 [−0.0710, −0.0128] | 12/50 (3) | 15.9 (1.6) |
| 16 | A_margin | L-BFGS-B | 68, 27 | −5.229 | −2.1·10⁻² [−2.4, −1.2]·10⁻²; 10/10 | 7.8·10⁻⁴ | −3.0·10⁻⁵ [−11.0, +0.2]·10⁻⁵ | 0.540 | −0.0160 [−0.0271, +0.0020] | 17/50 (2) | 19.1 (1.8) |
| 20 | A_margin | ADAM 0.1 × 50 | 50, 30 | −6.139 | — | 1.4·10⁻⁴ | — | 0.130 | — | 6/50 (0) | 4.3 (0.0) |
| 20 | A_margin | ADAM 0.03 × 100 | 100, 62 | −5.826 | **+0.318** [+0.292, +0.331]; 0/10 | 3.4·10⁻⁵ | −1.2·10⁻⁴ [−2.1, −0.6]·10⁻⁴ | 0.033 | −0.1060 [−0.1866, −0.0604] | 0/50 (0) | 1.1 (0.0) |
| 20 | A_margin | L-BFGS-B | 63, 45 | −5.900 | −4.1·10⁻³ [−60.2, −2.0]·10⁻³; 10/10 | 1.4·10⁻⁴ | +5·10⁻⁷ [−1.0·10⁻⁵, +5.5·10⁻⁶] | 0.127 | +0.0005 [−0.0064, +0.0049] | 3/50 (0) | 4.3 (0.0) |

Fig. S11 puts every paired change in ⟨H_norm⟩ next to the baseline's own spread over the five
seeds within an instance (the grey band is that seed range, not a factor contribution): the median
of max − min over seeds is 3.4–5.5 at every N and both weights.

### The spread behind each signed median

A signed median can be small because every change is small or because the
changes cancel. Table S23 separates the two for the two acquisition quantities
of Table S22: the typical size of a change, its 95th percentile, and how the
ten instances split between better and worse. Read
with the main table, it is what supports the claim that L-BFGS-B leaves P_τ
alone (median |ΔP_τ| ≤ 3.8·10⁻⁴ everywhere) while not leaving Q_τ(1000) alone
(95th percentile of |ΔQ| reaches 0.079 at N = 16, A_margin), and that the one
control with a large effect is ADAM at 0.03 at N = 20, A_margin.

**Table S23. Dispersion of the E3 paired differences.** Instance-level paired
differences against the ADAM 0.1 × 50 baseline, ten instances per cell; the signed
medians and their intervals are in Table S22; these are the columns that
median cannot recover. "Better/worse/tied" counts
instances, not runs.

| N | A | Setting | median \|ΔP_τ\| | p95 \|ΔP_τ\| | P_τ better/worse | median \|ΔQ\| | p95 \|ΔQ\| | Q_τ better/worse |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 12 | A_heur | ADAM 0.03 × 100 | 1.3·10⁻⁵ | 2.7·10⁻⁴ | 5/5 | 0.0012 | 0.0048 | 5/5 |
| 12 | A_heur | L-BFGS-B | 3.8·10⁻⁴ | 7.4·10⁻⁴ | 10/0 | 0.0087 | 0.0109 | 10/0 |
| 16 | A_heur | ADAM 0.03 × 100 | 2.0·10⁻⁸ | 4.3·10⁻⁸ | 7/3 | 0.0000 | 0.0000 | 7/3 |
| 16 | A_heur | L-BFGS-B | 1.7·10⁻⁷ | 9.1·10⁻⁵ | 9/1 | 0.0002 | 0.0715 | 9/1 |
| 20 | A_heur | ADAM 0.03 × 100 | 6.1·10⁻⁶ | 1.1·10⁻⁵ | 0/10 | 0.0060 | 0.0102 | 0/10 |
| 20 | A_heur | L-BFGS-B | 1.5·10⁻⁷ | 2.8·10⁻⁷ | 0/10 | 0.0002 | 0.0003 | 0/10 |
| 12 | A_margin | ADAM 0.03 × 100 | 2.0·10⁻⁴ | 1.3·10⁻³ | 4/6 | 0.0021 | 0.0286 | 4/6 |
| 12 | A_margin | L-BFGS-B | 2.9·10⁻⁴ | 9.4·10⁻⁴ | 6/4 | 0.0001 | 0.0234 | 6/4 |
| 16 | A_margin | ADAM 0.03 × 100 | 6.1·10⁻⁵ | 1.5·10⁻⁴ | 1/9 | 0.0281 | 0.0834 | 1/9 |
| 16 | A_margin | L-BFGS-B | 3.2·10⁻⁵ | 4.8·10⁻⁴ | 3/7 | 0.0217 | 0.0792 | 3/7 |
| 20 | A_margin | ADAM 0.03 × 100 | 1.2·10⁻⁴ | 3.1·10⁻⁴ | 0/10 | 0.1060 | 0.2566 | 0/10 |
| 20 | A_margin | L-BFGS-B | 1.1·10⁻⁶ | 5.2·10⁻⁵ | 8/2 | 0.0010 | 0.0469 | 8/2 |

### What "stopped" means at single precision

The E3 block ran at single precision, as every run in
this paper does, and recorded only how many gradient evaluations each run used — a count from which
convergence cannot be inferred. SciPy's default tolerances are ftol = 2.2·10⁻¹⁶ ·
factr = 2.2·10⁻⁹ on the relative reduction of f and gtol = 10⁻⁵ on the projected gradient; the first
sits two orders of magnitude below the resolution of a single-precision expectation. Re-running a
fixed block of the same configuration at both precisions (2 instances × 2 seeds × 2 weights at
N = 12 and N = 16, 32 runs, local CPU) shows what the counts were hiding: in single precision **2 of
16 runs failed their line search and 3 exhausted the gradient budget**, and of the eleven that did
stop on a tolerance, all eleven stopped on the *objective* criterion and none on the gradient one,
with a final |∇|∞ between 2.7·10⁻⁵ and 1.2·10⁻¹ over the sixteen. In double precision none of the 16
failed, seven of the sixteen stopped on the gradient criterion, and |∇|∞ ran 1.7·10⁻⁶ to 3.4·10⁻⁴ —
one to three orders of magnitude smaller — using 15–73 gradient evaluations against 27–100. The same block at N = 20 (16 runs,
on the simulation host) had no failures and no budget exhaustion at either precision, but the same
asymmetry: all eight single-precision runs stopped on the objective criterion with |∇|∞ between
5.8·10⁻⁵ and 1.9·10⁻² after 27–85 gradient evaluations, while four of the eight double-precision runs
stopped on the gradient criterion with |∇|∞ between 7.5·10⁻⁷ and 1.3·10⁻⁴ after 17–27. The claim that
survives is therefore "terminated under SciPy's stopping criteria at single precision", not "converged".

What does *not* change is the result. The optimised ⟨H_norm⟩ agrees between the two precisions to
6·10⁻⁴ in 14 of the 16 paired conditions; the largest disagreement is 0.053, a hundredth of the
3.4–5.5 spread the same instance shows across its five initial angles, and in both of the two
largest the single-precision run took more gradient evaluations (62 against 53; the whole budget of
100 against 68) and ends *lower*, not higher, than its double-precision partner. At N = 20 the eight
pairs agree to 4.7·10⁻⁵.
A similar expectation does not by itself imply similar mass on the success set — two distributions
with the same mean can differ on P_τ — so the diagnostic block also scores each run's exact final
state on what the paper reports (Table S36): over the 24 pairs the two precisions agree on P_F,
P_τ, Q_τ(1000) and D_cond to 1.0·10⁻⁵, 7·10⁻⁸, 3·10⁻⁵ and 2·10⁻⁵ in the median, and in 22 of the 24
pairs to 3.6·10⁻³, 1.7·10⁻⁵, 4.4·10⁻³ and 1.0·10⁻³ at worst; the two exceptions are the two pairs
above whose expectations differ by 0.04–0.05, both at A_margin with seed 42. At N = 12, instance 0,
both precisions stopped on the objective criterion (62 against 53 gradient evaluations, ⟨H_norm⟩
−4.317 against −4.264) and the single-precision state has P_F 0.443 against 0.377, P_τ
7.0·10⁻³ against 3.1·10⁻³ and Q_τ(1000) 0.999 against 0.956; at N = 16, instance 1, the
single-precision run exhausted the 100-gradient budget while the double-precision run stopped on
the objective criterion after 68 (⟨H_norm⟩ −5.569 against −5.528), with P_F 0.465 against 0.415,
P_τ 1.4·10⁻³ against 0.7·10⁻³ and Q_τ(1000) 0.747 against 0.520. In both pairs the two precisions
reached different optimisation outcomes on the same start — the size of the difference is that
of a different local minimum, not of rounding — and in both the reading the paper draws does not
change: P_τ stays below the uniform-F success probability of the instance (1.5·10⁻² and
1.8·10⁻³) at either precision. The
1,000-shot batch's within-τ verdict differs on 8 of the 24 pairs; on every one of them the two
states' P_τ agree to 3.0·10⁻⁶ and their Q_τ(1000) to 6.4·10⁻⁴, so the disagreement is the draw
(Sec. S-VII). The
conclusion of this section rests on those comparisons, not on the stopping reason. Runs now record
SciPy's status, message, iteration count, final gradient norm, tolerances and version, which is
what this diagnostic had to be run to recover. The block itself is not portable across hosts at
single precision: re-running the N = 12 and N = 16 runs on a second CPU host reproduced every
double-precision row (the same gradient-evaluation counts, ⟨H_norm⟩ to 4·10⁻¹⁴), while 14 of the 16
single-precision rows changed their evaluation count and 7 their stopping reason, with ⟨H_norm⟩
within 4.2·10⁻⁴; the N = 20 rows, re-run on the same host, reproduced bit for bit at both
precisions. The second-host rows are the `_n12_16_mac.csv` file of the termination diagnostic,
beside the `_n12_16.csv` file they are compared with.

The batch columns of the L-BFGS-B arm — the E3 block's "no feasible shot" and "min-energy shot
infeasible" counts, Table S22's "optimum hit", and this diagnostic's within-τ verdict — come from a
re-run of that arm: until 2026-09-13 the solver's L-BFGS-B path handed the sampler the initial
angles rather than the optimised ones (Sec. VI-C of the main text), which the recorded states
exposed (the batch's feasible fraction followed `initial_P_F`, r = 0.99, not `final_P_F`,
r = −0.30, over the 300 runs). The optimisation itself was untouched by the fix, and the
re-run on the same host reproduced every state column of the 300 + 48 runs bit for bit — ⟨H_norm⟩,
P_F, P_τ, D_cond, the evaluation counts and SciPy's status and message, at both precisions of the
diagnostic (its Q_τ columns differ from the superseded file by at most 2.8·10⁻¹³ because the
cancellation-free evaluation of Q was adopted after that file was written). Two regression tests
pin both optimizer paths: `test_the_sampler_gets_the_final_angles` inspects the angles the sampler
receives (the argument, not the shots), and `test_reproduces_the_state_solve_recorded` re-evaluates
the recorded final angles from scratch against the recorded state; the predicted counts of Table
S22 compare every batch with its state.

### Multistart at equal shots

The five E1 seeds of an instance can be re-read as a five-start strategy: optimise five times, draw
1,000 shots from each state, keep the best feasible candidate of the 5,000, with
Q_multi,i = 1 − ∏_k (1 − p_ik)^1000 from the exact one-shot probabilities p_ik. The single-start
strategy a user can run at the same shot budget draws one seed at random before optimising and
samples that state 5,000 times, Q_single,i = (1/5) Σ_k [1 − (1 − p_ik)^5000] — the Q of each seed,
then the mean, not the mean p put into Q, and not the median seed's state, which is a summary no
one can select in advance. With a_k = (1 − p_ik)^1000 the two are ∏_k a_k and mean_k(a_k^5), so
by the AM–GM inequality Q_multi,i ≥ Q_single,i on every instance identically; the increment is the
measurement, set against the four extra optimisations it costs. The same pair of strategies is
evaluated for reaching the exact optimum. Uniform sampling from the feasible set at 5,000 shots is
carried as the no-optimisation control at the same candidate count.

**Table S24. Five starts against one, at equal shots.** Medians over 30 instances of Q_τ(S) at
τ = 1 %. "1×1k" is one optimisation sampled 1,000 times (the E1 baseline; median seed); "random
1×5k" one optimisation from a randomly drawn seed sampled 5,000 times; "5×1k pooled" five
optimisations with 1,000 shots each and all batches pooled. The gain is the per-instance difference of the last two, with its
instance-level 95 % bootstrap interval and the number of instances on which it exceeds 0.01; "unif-F
5k" is uniform sampling from F at 5,000 shots. The two "Opt" columns ask the same question of the
exact optimum, for the random single start and the pooled five starts.
Optimisation time is one seed against five.

| N | A | 1×1k | random 1×5k | 5×1k pooled | Gain [95 % CI] | > 0.01 of 30 | unif-F 5k | Opt, random | Opt, pooled | Opt. time |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| 12 | A_heur | 0.927 | 0.905 | 1.000 | +0.095 [+0.095, +0.095] | 30 | 1.000 | 0.905 | 1.000 | 10 → 51 s |
| 12 | A_margin | 1.000 | 0.945 | 1.000 | +0.055 [+0.026, +0.076] | 23 | 1.000 | 0.942 | 1.000 | 10 → 51 s |
| 16 | A_heur | 0.335 | 0.684 | 0.836 | +0.153 [+0.152, +0.153] | 30 | 1.000 | 0.683 | 0.835 | 17 → 84 s |
| 16 | A_margin | 0.534 | 0.639 | 0.983 | +0.336 [+0.316, +0.344] | 30 | 1.000 | 0.634 | 0.979 | 17 → 84 s |
| 20 | A_heur | 0.067 | 0.305 | 0.353 | +0.046 [+0.015, +0.048] | 30 | 0.873 | 0.181 | 0.196 | 30 → 148 s |
| 20 | A_margin | 0.120 | 0.463 | 0.520 | +0.060 [+0.036, +0.074] | 30 | 0.873 | 0.321 | 0.350 | 30 → 148 s |

## S-VII. SV1 against the local GPU, and E4-A in full

The single-run cross-backend, matched-precision, seed-pair and repeat comparisons on Amazon Braket
SV1 that Sec. V-D summarises in a sentence (Tables S25–S29, Fig. S12), the full E4-A comparison
(Table S30), and the finite-shot bound Sec. V-D applies.

**Cross-backend reproducibility: SV1 versus the local GPU.** A device-abstraction layer is only worth the name if changing the device changes nothing but the
device. We therefore re-ran the Table S2 experiment for N ∈ {8, 12, 16, 20} on AWS Braket SV1 — a
different simulator, by a different vendor, at double rather than single precision, reached over the
network — selected by nothing but a backend name. Table S25 reports the result.

**Table S25. Same code, two backends (p = 2, seed 42, 50 ADAM steps, 1000 shots).** Every row reads the
single-snapshot cache of §IV-D. Table S28 below shows that
the SV1 column is one draw, not a measurement: Three runs of each row return three
different portfolios at N ≥ 16. We print one run here because that is what a benchmark normally
prints, and we then measure what that convention costs.

| N | E, lightning.gpu | E, SV1 | Bitwise equal | Gap, GPU | Gap, SV1 | Sharpe, GPU | Sharpe, SV1 | t, GPU | t, SV1 |
|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| 8 | −36.2526116808 | −36.2526116808 | ✓ | 0.000 % | 0.000 % | 1.322 | 1.322 | 5.3 s | 172 s |
| 12 | −52.2526116807 | −52.2526116807 | ✓ | 0.000 % | 0.000 % | 1.322 | 1.322 | 10 s | 179 s |
| 16 | −153.2671105966 | −153.2406894624 | ✗ | 0.000 % | 0.017 % | 1.385 | 1.492 | 17 s | 257 s |
| 20 | −335.9747295240 | −335.8084974079 | ✗ | 0.053 % | 0.102 % | 1.532 | 1.222 | 29 s | 858 s |

At N = 8 and N = 12 the two committed artifacts agree to the last bit of a double — not to plotting
accuracy, but exactly. At N = 16 and N = 20 they return different feasible portfolios. The N = 16 row
is the more instructive of the two: SV1 misses the exact optimum by 0.017 %, and the portfolio it
returns has a Sharpe ratio 7.7 % higher than the optimum's (1.492 vs. 1.385). The backend that
solved the stated problem *worse* produced the better portfolio, which is §V-A's decoupling arriving
by a third route.

The explanation that the cloud simulator is less accurate at larger N does not fit the data. Fig. S12 (right) plots the per-step distance between the two optimizer trajectories:
|⟨H⟩_SV1 − ⟨H⟩_GPU| is ≈1e-6 at *every* size and from the first step, and the largest disagreement
occurs at N = 16 (up to 2.1e-5), which is also the smallest size at which the answers diverge — no
trajectory point out of 50 coincides there. The perturbation does not grow with N; what grows is the
density of near-degenerate solutions
that a fixed 1e-6 can separate. This is the same mechanism as the N = 30 basin change of §IV-D, reached
by a different route: there a 1e-10 change in the *input* moved the answer, here a 1e-6 difference in
*floating-point evaluation order* does, and both times the objective moves by a few 1e-4 while the
Sharpe ratio moves by tens of percent.

**Is it the backend, or is it the precision?** The GPU arm of Table S25 runs at
single precision and SV1 at double, so a reader can reasonably ask whether a
≈1e-6 trajectory difference is just fp32 and the backend attribution collapses.
We re-ran the GPU arm at double precision to find out.

**Table S26. The same experiment at matched precision.** Reported gap against the
enumerated optimum; the last column counts how many distinct portfolios the three
arms returned between them.

| N | GPU, single | GPU, double | SV1, double | Distinct answers |
|---:|---:|---:|---:|---:|
| 8 | 0.0000 % | 0.0000 % | 0.0000 % | 1 |
| 12 | 0.0000 % | **0.1024 %** | 0.0000 % | 2 |
| 16 | 0.0000 % | 0.0000 % | 0.0172 % | 2 |
| 20 | 0.0528 % | 0.0938 % | 0.1023 % | **3** |

Matching the precision does not make the answers agree. At N = 12 the two double-
precision arms *disagree with each other* while the single-precision GPU matches
SV1 exactly, and at N = 20 all three arms return different portfolios. The
higher-precision arm is also the worse one at N = 12 — 0.1024 % against
0.0000 % — which is not a behaviour a precision explanation predicts.

So the disagreement is not fp32. Precision, vendor and evaluation order are three
ways of perturbing the same arithmetic at the 1e-6 level, and on this landscape
any of them is enough to select a different member of the near-optimal set. The
useful statement is not which backend is right but that the *returned solution*
is not determined by the formulation at these sizes.

**Is it the backend, or is it the sampler?** A near-degenerate landscape makes the returned portfolio a
*draw* from the near-optimal set, and a benchmark run has more than one unpinned source of randomness:
it can differ from another because the optimizer reached different angles, or because the same state
was sampled differently. A "shot-RNG control" — the seed sweep re-run with the angles fixed and only
the sampling seed changed, which agreed 10 of 10 — was first read as ruling the second out. That
control was vacuous: the seed it changed was consulted by no sampler
(§VI-C), so the two sweeps re-drew identical shots and agreed because the samples were the same. The
separation is made properly below, with the angles fixed and ten genuinely independent batches on
each backend (E4-A), after the remaining single-run evidence.

**Table S27. The ten cross-backend pairs behind "1 of 10".** Same instance, same
seed, same configuration on each side; only the backend differs.

| N | seed | Same portfolio | Gap, SV1 | Gap, GPU |
|---:|---:|:--:|---:|---:|
| 16 | 42 | **yes** | 0.0000 % | 0.0000 % |
| 16 | 43 | no | 0.1703 % | 11.2246 % |
| 16 | 44 | no | 0.0000 % | 0.0041 % |
| 16 | 45 | no | 0.0172 % | 0.0041 % |
| 16 | 46 | no | 0.0270 % | 0.0230 % |
| 20 | 42 | no | 0.0860 % | 0.0528 % |
| 20 | 43 | no | 0.0526 % | 0.0979 % |
| 20 | 44 | no | 0.0177 % | 0.0089 % |
| 20 | 45 | no | 0.0712 % | 0.0049 % |
| 20 | 46 | no | 0.0035 % | 0.0220 % |

The one agreement is the reference seed, 42 — and it disagrees with Table S25, which
reports SV1 missing the optimum at N = 16 for what is nominally the same
configuration. That is not a contradiction between the tables; it is the
same finding twice. The two SV1 columns are separate submissions, and Table S28
shows what that costs: three runs of one configuration return three different
portfolios at this size. Table S25 prints the submission made for it, Table S27
the one made for the seed sweep, and they differ by 2.6e-2 in energy. A reader
comparing the two rows is looking at the paper's own claim about the cloud
simulator, not at an error.

The row worth dwelling on is N = 16, seed 43: the run of §V-B that returned no feasible shot on the GPU
(11.22 % by its lowest-energy shot) does not happen on SV1,
which reports 0.17 % for the same configuration. A single-backend seed sweep
would have found either the tail or its absence depending on where it ran.

Ruling that out raised a sharper question, because SV1 also disagrees with *itself*. We therefore ran
the identical configuration three times at each size — same seed, same depth, same optimizer budget,
same shot count, same committed price cache, nothing varied but the submission — and counted how many
distinct portfolios came back.

**Table S28. Three SV1 runs of one configuration (N, seed 42, p = 2, 50 steps, 1000 shots, double).**
Nothing is varied between runs. The GPU column is the same configuration on `lightning.gpu`.

| N | C(N,K) | Distinct answers in 3 SV1 runs | Sharpe range across runs | GPU |
|---:|---:|---:|---|---:|
| 8 | 28 | 1 (bitwise identical) | 1.3219 (no spread) | 1.3219 |
| 12 | 66 | 1 (bitwise identical) | 1.3219 (no spread) | 1.3219 |
| 16 | 560 | 3 | 1.3768 – 1.4920 (8.4 %) | 1.3851 |
| 20 | 4,845 | 3 | 1.0256 – 1.2608 (22.9 %) | 1.5323 |

**Every run at N ≥ 16 returned a different portfolio.** At N = 20 the four runs in that row — three on
SV1, one on the GPU — produced four distinct answers, spanning Sharpe ratios from 1.026 to 1.532. And
the small sizes are the control that makes this a mechanism rather than an anecdote: at N = 8 and
N = 12 the three cloud runs are bitwise identical to each other *and* to the GPU, so the cloud is not
simply unstable. What changes between N = 12 and N = 16 is not the service, it is C(N,K) — 66 feasible
portfolios against 560 — and with them the density of solutions a floating-point perturbation can
separate.

The convergence traces show the optimizer trajectories differ: at N = 20 across 50 recorded steps 2
agree and 48 do not, with a maximum difference of 5.2e-15 and a disagreement already present at step
0. At N = 16 the two runs differ by 4.4e-16 at step 0 — one unit in the last place of a double — and
end in portfolios whose Sharpe ratios differ by 7.7 %. SV1's analytic evaluation is therefore not
bit-reproducible between runs. Observing the ULP difference and the different final portfolio in the
same pair of runs does not by itself show that the one caused the other: SV1's final sampling is a
second unpinned source, and we have not re-sampled a fixed state on SV1 to separate them (§VII).

That completes a hierarchy worth stating plainly, because it is not the one a user would assume
(Table S29).

**Table S29. Run-to-run reproducibility of the three execution environments.**

| Environment | Reproducible? | Evidence |
|---|---|---|
| `lightning.gpu` (local) | *yes, bitwise* | 3 of 3 runs identical |
| `lightning.qubit` (local) | only if pinned | bitwise at `OMP_NUM_THREADS=1`; ~1e-13 drift otherwise |
| SV1 (managed cloud) | *no* | 48 of 50 trace points differ; 3 of 3 runs give different answers at N ≥ 16 |

The managed cloud simulator is the least reproducible of the three we tested, and the only one whose randomness
cannot be pinned from the client. A cross-backend check therefore measures backend *and* run, and
5.2e-15 of run-to-run drift in the trajectory accompanies a changed answer on this landscape — the
same sensitivity §IV-D measures at 4e-10 in the input. The fixed-angle experiment that follows shows
the sampler alone changes the answer as often.

### E4-A in full

**Table S30. Fixed angles on four backends (E4-A), full comparison.** Left: distance between each
arm's exact output probabilities and the GPU double-precision ones at the same angles, maximum over
the twelve circuits — the largest per-state difference on F, total variation on F ∪ {⊥} (one
outcome per feasible bitstring and one symbol for "no feasible shot", p_⊥ = 1 − P_F), total
variation on F alone (half the L¹ distance of two unnormalised restrictions, for reference), and
the relative change in P_τ. These are probabilities, not quantum states. Right: distinct
best-feasible portfolios over ten 1,000-shot batches of one state, range over the four circuits of
each size, the Monte Carlo 95 % interval under i.i.d. draws from the arm's own distribution in the
last row; batches with no feasible shot at N = 16, seed 43, in the last column.

| Arm | max \|Δp\| on F | TV on F ∪ {⊥} | TV on F | \|ΔP_τ/P_τ\| | Distinct best, N = 12 (seed 42 / 43) | N = 16 (42 / 43) | N = 20 | No-feasible-shot batches, N = 16 seed 43 |
|---|---:|---:|---:|---:|---|---|---|---|
| GPU double (reference) | — | — | — | — | 1–2 / 5–6 | 3–4 / 1–5 | 10 | 5–9 |
| CPU double, 1 thread | 2.7·10⁻¹⁷ | 4.3·10⁻¹⁵ | 2.1·10⁻¹⁵ | 2.1·10⁻¹⁴ | 1–2 / 6–10 | 3–5 / 3 | 10 | 7 |
| GPU single | 5.0·10⁻⁹ | 6.1·10⁻⁷ | 3.0·10⁻⁷ | 5.5·10⁻⁶ | 1 / 6–7 | 3–4 / 1 | 10 | 9 |
| SV1 double (cloud) | 5.3·10⁻¹⁶ | 2.6·10⁻¹⁴ | 1.3·10⁻¹⁴ | 2.8·10⁻¹³ | 2 / 7–8 | 3–4 / 3–4 | 9–10 | 6–7 |
| i.i.d. from own distribution (MC 95 %) | | | | | 1–3 / 5–9 | 2–6 / 1–7.5 | 9–10 | — |

### The finite-shot bound on the selected output

Let P and P′ be two one-shot output distributions on the same outcome set with total variation
distance d = TV(P, P′), and let a batch be S independent shots to which a deterministic selection
rule (here: the lowest-E_A feasible shot, or "no output") is applied. By the coupling
characterisation of total variation there is a joint distribution of one shot from P and one from
P′ under which the two shots differ with probability exactly d; drawing the S shots of both batches
jointly under S independent copies of that coupling, the two batches are identical with probability
(1 − d)^S, and identical batches give identical selected outputs. Hence

> TV(selected output under P, selected output under P′) ≤ 1 − (1 − d)^S ≤ S·d.

The bound is conservative — it counts any differing shot as a differing output — and it needs no
assumption about the structure of F. With the maxima of Table S30 at S = 1,000 it gives
≤ 4.3·10⁻¹² for the single-threaded CPU, ≤ 2.6·10⁻¹¹ for SV1 and ≤ 6.1·10⁻⁴ for the single-precision
arm: these are bounds on how far the arithmetic can move the *distribution* of the selected
portfolio at fixed angles. They are not bounds on how often two independently sampled batches
return different portfolios — at S = 1 two identical fair coins have d = 0 and disagree on half of
independent draws — which is why ten independent batches of one state can return nine or ten
distinct portfolios at N = 20 while the distributions they were drawn from are within 10⁻¹⁴ of
each other. `tests/test_e4_bounds.py` checks the inequality numerically and keeps the two readings
apart. This is a derivation from the paper's own definitions, not a new measurement, and it is
what makes E4-B (optimising on SV1 to compare final angles) unnecessary for the claim Sec. V-D
makes; E4-B would be needed only to attribute the disagreement of the single runs to
backend-dependent optimisation trajectories, which the paper does not do.

## S-VIII. IonQ Forte-1: the four tasks, the fidelity model and two defects

Four tasks on IonQ Forte-1 (us-east-1), $33.20 total. Each is the configuration
of Table S2 — p = 2, 50 ADAM steps — with the optimizer on the local GPU and a
single 100-shot sampling circuit on hardware. That split is not a convenience:
a QPU cannot use adjoint differentiation, so an optimization loop submitted there
becomes one billable task per parameter-shift circuit — 1 + 2·p·[C(N, 2) + 2N] per evaluation,
177 at N = 8 and 921 at N = 20 for p = 2 (Sec. VI-B) — and the optimizer was never submitted to
hardware.

**The fidelity model.** Forte-1 reports a two-qubit fidelity of 0.9905 and a native ZZ gate, so each
of the QUBO's C(N, 2) quadratic terms is one two-qubit gate per QAOA layer and the dense
mean–variance QUBO needs 2·C(N, 2) entangling gates at p = 2 — 56, 132, 240 and 380 at N = 8, 12,
16 and 20. Multiplying the published per-gate fidelity over that count (with SPAM at 0.9938 per
qubit) gives the predicted circuit fidelities of Table S31 — a prediction from a simplified
product model, not a measured state, process or output-distribution fidelity, and none was
measured. The cardinality penalty makes the interaction graph complete regardless of the covariance
structure, so the gate count is a property of the encoding; [11] reaches the same point from the
annealing side via chain-break fractions of 83–92 %. The model did not predict usefulness: at 9 %
predicted fidelity the N = 16 task returned 18 % feasible shots, a higher feasible fraction than at
N = 12, and a portfolio 0.0438 % from the exact optimum. What the four tasks establish is narrow: at
N = 16 the prescribed circuit executed, 18 of its 100 shots were feasible, and the best of them was
0.0438 % from the exact optimum. Whether that is typical of N = 16, and where the usable range ends,
four tasks cannot say — the two N = 8 runs differ by more between themselves (25 % and 7 % feasible,
intervals [17.5, 34.3] and [3.4, 13.7]) than N = 8 differs from N = 16.

**Table S31. Predicted circuit fidelity on IonQ Forte-1 at p = 2, from its published calibration.**

| N | 2-qubit gates | Predicted fidelity |
|---:|---:|---:|
| 8 | 56 | 56 % |
| 12 | 132 | 26 % |
| 16 | 240 | 9 % |
| 20 | 380 | 2 % |

**Table S32. IonQ Forte-1, 100 shots per run.** Feasible-shot counts carry a Wilson 95 % interval for
100 independent draws; the counts themselves are exact (see the provenance note below), the interval
is the sampling uncertainty in reading them as a rate.

| N | seed | Distinct strings | Feasible shots [95 % CI] | Noiseless feasible | Retention | Gap | Sharpe vs exact optimum |
|---:|---:|---:|---|---:|---:|---:|---:|
| 8 | 42 | 60 | 25 % [17.5, 34.3] | 40.2 % | 0.62 | 0.2809 % | 1.3380 vs 1.3219 (+1.2 %) |
| 8 | 43 | 79 | 7 % [3.4, 13.7] | 9.9 % | 0.71 | 0.3917 % | 1.0889 vs 1.3219 (−17.6 %) |
| 12 | 42 | 92 | 11 % [6.3, 18.6] | 16.4 % | 0.67 | 0.0000 % | 1.3219 vs 1.3219 (exact) |
| 16 | 42 | 95 | 18 % [11.7, 26.7] | 39.2 % | 0.46 | 0.0438 % | 1.2836 vs 1.3851 (−7.3 %) |

*Where these integers come from.* Forte-1 does not return a shot sequence. Each task's result is the
nested `program_set_task_result` schema, whose executable record holds `measurementProbabilities` —
a map from bitstring to frequency — and no `measurements` array. Every probability in all four tasks
is an exact multiple of 1/100 and each map sums to exactly 1, against `requestedShots` =
`successfulShots` = 100 and `deviceParameters` = None in the task metadata, so the counts in this
table are the empirical frequencies of 100 shots recovered without loss, not estimates and not
post-processed by any error-mitigation setting we requested. What is not recoverable is the *order*
of the shots, which nothing here uses. The four task records are committed with the paper, so this
chain can be re-walked; §VI-C reports what the client library did with the same records before they
were read this way.

Three observations follow, and none of them is a hardware limit. Hardware retained 0.46–0.71 of the
noiseless feasible fraction across the four runs, with the lowest value at the largest size; the
noiseless fraction itself ranges from 40.2 % to 9.9 % between two seeds at the *same* size, so over
these four tasks the initial angles moved the feasible fraction more than the device did — an
observation about four tasks at one calibration, which cannot rank noise against initialisation in
general. At N = 12, 89 of 100 shots violated the cardinality constraint and the remaining 11
contained the exact optimum, so a benchmark reporting only the best shot would record a 0.0000 %
gap from a circuit whose product-model fidelity is 26 %; the quality of the best feasible outcome
and the frequency of feasible outcomes are different quantities and are reported apart. And the two
N = 8 runs report gaps 0.11 percentage points apart (0.2809 % and 0.3917 %) and Sharpe ratios 19
percentage points apart, one above the exact optimum's and one 17.6 % below it — same device, same
instance, same shot count; only the initial angles differ.

**Two defects worth reporting, and the second is the first one's fix.** The
first three hardware runs returned an energy of exactly 0.0 at a 100 % gap. The
device returns its shots under a nested result schema that the client library we
used does not read and does not raise on; it returned an array that was not the
measurements, while the real shots sat in object storage the whole time. A
framework that turns an $8.30 hardware result into a plausible-looking zero is a
reproducibility hazard of a different kind from those in §IV-D: not an unstable
input, but a silent misread of a correct output. We now read hardware shots from
the task record rather than through the library.

That fix then failed on its first contact with hardware, for a reason worth more
than the API detail behind it. It located the completed task through a handle
the library sets on one execution path and not on the path hardware sampling
actually takes. The N = 16 run submitted its task, the task completed, $8.30 was
billed, and the run died reading a handle that was never set — the code written
to stop losing hardware results lost one, because it had only ever been
exercised against a simulator. **A recovery path reachable only by spending
money will not be tested, and will therefore be wrong when it is needed.** The
current version asks the service which task this run created, requires the
answer to be exactly one, and refuses to guess otherwise; a wrong guess would
attribute another run's shots to this row, which is worse than an exception
because the number would look plausible. Five tests now cover it against a
stubbed client, at no cost. Both defects, and the code, are in the repository.

## S-IX. Reproducibility notes, the defects that changed results, and resources

### The four input-reproducibility rules, and what each cost

All conditions are in code, seeds are named and the window is pinned. Four prerequisites were learned by losing results to them, and are stated because each is
cheap to satisfy and easy to miss. (i) *Pin values, not dates.* `yfinance` revises adjusted closes
between calls, so two runs on the identical date range produced QUBOs differing at the 10⁻¹⁰ level
and single-precision QAOA occasionally amplified that into a different local minimum. The raw close
series are cached to disk at full precision and read back with round-trip parsing (two fifty-ticker
snapshots, ≈1.2 MB); the released package carries the derived μ and Σ of each window rather than the
third-party series, and every instance's inputs are a bit-exact slice of them, so every number
reproduces from a clone with no network access. Re-running the
single-instance benchmark on the cache moved fourteen of its fifteen rows by ~10⁻⁷ (the same assets)
and the N = 30 QAOA row into a different near-degenerate basin — 5·10⁻⁵ in objective, 9 % in Sharpe
ratio — under an input perturbation four orders of magnitude below any reported digit. (ii) *One snapshot, not
one per experiment.* A per-size cache held a sixteen-asset file fetched seven minutes after the
others and revised by up to 4.2·10⁻⁷; every instance now slices one fifty-ticker snapshot.
Re-deriving that size turned the defect into a controlled measurement: under a 4·10⁻¹⁰ relative
change in the QUBO, 9 of 60 grid runs changed solution and the worst gap moved from 0.49 % to
11.07 %. (iii) *Canonical memory layout.* `DataFrame.cov()` sums in layout order, so identical prices
gave last-bit-different covariances on two code paths; statistics are computed on a C-contiguous
float64 copy. (iv) *Thread count.* `lightning.qubit` reduces in thread order and drifts by ~10⁻¹³
between runs unless `OMP_NUM_THREADS=1`; `lightning.gpu` is bitwise reproducible regardless. The
experiment scripts pin the thread count before importing a simulator and a test asserts it. Figures
are regenerated from the committed CSVs with embedded timestamps suppressed, so a changed figure file
means changed data.

### Implementation defects that changed results, in the order found

Seven implementation defects changed reported numbers; six were found before any number in §V
of the main text was produced, the seventh after. Sec. VI-C of the main text states the four
correspondences they left as checks; this is the record of the defects themselves, in the order found. *Before §V.* (1) The angles were a plain array rather than a
differentiable tensor, so the optimizer returned them unchanged — a flat "convergence" curve that
would have been reported as optimized QAOA. (2) The Braket device rejects computational-basis
`sample(wires=...)`; PauliZ is sampled per wire and mapped to bits. (3) The expectation varies in
γ at frequencies set by the differences between H_C's eigenvalues, which grow with the
coefficients (and with their number); at N = 12 the expectation varied faster in γ than one ADAM
step of 0.1 moved it, and the cost Hamiltonian is therefore divided by its largest coefficient
(§III-A) — a choice of parameterisation, not a scale law for the spectrum. At A_heur the
objective's whole feasible range is 1.5–1.6 % of that coefficient s_H in the median (§V-B), so the
normalised landscape is almost entirely the constraint, which is the context for the near-uniform
conditional distribution of §V-B. (4), (5) On the hardware path, a nested result schema the client
library misread into an energy of exactly 0.0, and a recovery path that had only ever been
exercised against a simulator (Sec. S-VIII). (6) While building E4-A, a "shot-RNG
control" had re-seeded NumPy's global generator before the sampling circuit, but the Lightning
devices draw a seed from that generator once, at construction, and sample from a private generator
thereafter, so the re-seeding changed nothing and two sweeps reported as varying the shots re-drew
identical ones. A control that cannot fail is not a control, and a reproducibility claim about a
sampler has to name the generator it seeds; E4-A seeds the device's own.

*After §V.* (7) An integration error in the L-BFGS-B wrapper left the final sampling call
referencing the initial parameter vector, whereas the analytic state quantities were evaluated at
the returned final parameters: the ADAM loop rebinds the variable the sampler reads to the array
each step returns, but the L-BFGS-B wrapper received a copy of that variable and returned its
result in another. The batch columns of the E3 L-BFGS-B arm ("no feasible shot" and "min-energy
shot infeasible" in Table II, "optimum hit" in Table VI) therefore described unoptimised states
while its state columns (⟨H_norm⟩, P_F, P_τ, Q_τ) did not. It was caught because every run records
its exact final state beside its batch: the batch's feasible fraction tracked the initial state's
P_F (r = 0.99) and not the final state's (r = −0.30), and the batch counts fell outside what the
recorded states predict under independent draws (Table S22) — a comparison the
unaffected ADAM arms pass. The handoff was corrected, regression tests check on both optimizer
paths that state evaluation and sampling receive the same final parameters, and the affected
finite-shot records were regenerated by re-running the arm with the fix on the same host: every
recorded state of the 300 runs — ⟨H_norm⟩, P_F, P_τ, the initial state and the evaluation counts —
reproduced bit for bit, the batch columns printed are the re-run's (the batch's feasible fraction
now tracks the final state's P_F, r = 0.998), and the re-run's counts sit within 3.0 hits and 2.7
no-feasible runs of the independent-draw prediction in every cell, as the ADAM cells do.

### Resources in full

**Table S33. Resources used.** GPU time is wall-clock on the host of §IV-B; dollars
are AWS list price as billed.

| Experiment | Runs | Environment | Precision | Time | Tasks | Cost |
|---|---:|---|---|---:|---:|---:|
| Single-instance benchmark, N = 8–30 (Table S2) | 7 QAOA + 8 classical | lightning.gpu | single | 3 h 55 min | — | $0 |
| Seed and depth sweeps, steps × p × seed grid (§V-C) | 20 + 12 + 180 | lightning.gpu | single | 4.6 h | — | $0 |
| Degeneracy census and instance sweep, N ≤ 30 (§V-A) | 30 instances × 6 sizes; 900 + 15 annealing runs; 90 QAOA | CPU / lightning.gpu | double / single | ≈1 h | — | $0 |
| E0 re-scoring (Fig. 1) | 354 rows | CPU | double | seconds | — | $0 |
| E1 cross (Table II) | 2,565 | lightning.gpu | single | 13.9 h | — | $0 |
| E2 (from E1 states) | 0 new | CPU | double | minutes | — | $0 |
| E3 controls | 300 + 300 | lightning.gpu | single | 3.2 h + 2.4 h | — | $0 |
| E5 second window (Table S15) | 300 | lightning.gpu | single | 1.65 h | — | $0 |
| SV1 cross-backend, seed pairs, matched precision, three repeats (§V-D) | 4 + 10 + 4 + 12 | Braket SV1 | double | — | ≈1,300 | ≈$4.2 (incl. a $0.85 pilot that exposed the fallback of §VI-B) |
| E4-A fixed angles, local arms (§V-D) | 3 arms × 12 circuits × (1 + 10) | lightning.gpu / lightning.qubit | double, single | 3 min | — | $0 |
| E4-A fixed angles, SV1 (§V-D) | 12 circuits × (1 + 10) + 2 (pilot) | Braket SV1 | double | 21 min | 134 | $0.50 |
| IonQ Forte-1 (§V-D) | 4 | Forte-1, 100 shots | — | — | 4 | $33.20 |
| Deployment and smoke test | — | SV1 | — | — | 1 | $0.004 |

The Braket total is $37.90 against a $50 budget; the account carries no idle charge, so the deployed
footprint is retained.

## S-X. Supporting re-analyses

Five re-aggregations of data the paper already held, each answering one question the main text
leaves open, and one evidence table. None re-ran a QAOA optimisation except the precision diagnostic of Sec. S-VI,
whose optimised values no claim depends on.

### Initial → final at a fixed penalty weight

Sec. V-B separates what optimisation does at a fixed A from what changing A does after
optimisation. For every completed E1 run the exact initial and final states are paired within the
run; the difference is reduced to one value per instance by the median over the five seeds and
summarised over the thirty instances by its median and bootstrap interval (Sec. IV-E). R_τ is
q_τ / u_τ from P_τ / P_F against the instance's share |F_τ|/|F|; D_cond is from
`paper01_e1_conditional_tv.csv`. From `paper01_e1_initial_final.csv`.

**Table S34. The same run before and after optimisation, at A_heur and A_margin.** Medians over
instances of the seed-median initial and final levels, the paired median difference with its
95 % interval, and the count of instances on which the quantity increased.

| N | A | Quantity | Initial | Final | Paired diff [95 % CI] | Increased of 30 |
|---:|---|---|---:|---:|---|---:|
| 12 | A_heur | P_F | 0.114 | 0.179 | +0.070 [+0.069, +0.070] | 30 |
| 12 | A_heur | P_τ | 1.7·10⁻³ | 2.6·10⁻³ | +1.3·10⁻³ [+1.2·10⁻³, +1.3·10⁻³] | 30 |
| 12 | A_heur | Q_τ(1000) | 0.816 | 0.927 | +0.163 [+0.162, +0.163] | 30 |
| 12 | A_heur | E[V] | 21.9 | 10.6 | −11.1 [−11.1, −11.1] | 0 |
| 12 | A_heur | R_τ | 1.001 | 0.997 | +0.004 [+0.002, +0.005] | 22 |
| 12 | A_heur | D_cond | 0.002 | 0.002 | +0.001 [+0.001, +0.001] | 28 |
| 12 | A_margin | P_F | 0.134 | 0.357 | +0.177 [+0.089, +0.206] | 23 |
| 12 | A_margin | P_τ | 2.5·10⁻³ | 7.9·10⁻³ | +6.7·10⁻³ [+3.1·10⁻³, +7.7·10⁻³] | 25 |
| 12 | A_margin | Q_τ(1000) | 0.921 | 1.000 | +0.055 [+0.010, +0.078] | 25 |
| 12 | A_margin | E[V] | 16.7 | 1.5 | −11.5 [−11.7, −11.3] | 0 |
| 12 | A_margin | R_τ | 1.384 | 1.432 | +0.446 [+0.201, +0.543] | 26 |
| 12 | A_margin | D_cond | 0.223 | 0.175 | +0.043 [+0.028, +0.056] | 28 |
| 16 | A_heur | P_F | 0.035 | 0.229 | +0.128 [+0.128, +0.129] | 30 |
| 16 | A_heur | P_τ | 6.3·10⁻⁵ | 4.1·10⁻⁴ | +2.3·10⁻⁴ [+2.3·10⁻⁴, +2.3·10⁻⁴] | 30 |
| 16 | A_heur | Q_τ(1000) | 0.061 | 0.335 | +0.179 [+0.173, +0.182] | 30 |
| 16 | A_heur | E[V] | 34.8 | 4.0 | −17.2 [−17.2, −17.2] | 0 |
| 16 | A_heur | R_τ | 1.008 | 0.997 | −0.009 [−0.011, −0.008] | 0 |
| 16 | A_heur | D_cond | 0.002 | 0.001 | −0.001 [−0.001, −0.001] | 0 |
| 16 | A_margin | P_F | 0.093 | 0.313 | +0.221 [+0.199, +0.279] | 30 |
| 16 | A_margin | P_τ | 3.9·10⁻⁴ | 7.6·10⁻⁴ | +4.5·10⁻⁴ [+3.8·10⁻⁴, +5.8·10⁻⁴] | 28 |
| 16 | A_margin | Q_τ(1000) | 0.324 | 0.534 | +0.255 [+0.206, +0.310] | 28 |
| 16 | A_margin | E[V] | 25.7 | 1.7 | −21.1 [−21.7, −20.2] | 0 |
| 16 | A_margin | R_τ | 2.033 | 1.627 | −0.225 [−0.628, +0.146] | 11 |
| 16 | A_margin | D_cond | 0.388 | 0.139 | −0.174 [−0.197, −0.132] | 2 |
| 20 | A_heur | P_F | 0.087 | 0.167 | +0.134 [+0.133, +0.135] | 30 |
| 20 | A_heur | P_τ | 3.6·10⁻⁵ | 6.9·10⁻⁵ | +5.7·10⁻⁵ [+2.9·10⁻⁵, +5.8·10⁻⁵] | 30 |
| 20 | A_heur | Q_τ(1000) | 0.035 | 0.067 | +0.054 [+0.028, +0.055] | 30 |
| 20 | A_heur | E[V] | 45.7 | 14.3 | −18.9 [−18.9, −18.9] | 0 |
| 20 | A_heur | R_τ | 0.983 | 1.009 | +0.026 [+0.025, +0.031] | 29 |
| 20 | A_heur | D_cond | 0.004 | 0.002 | −0.002 [−0.002, −0.002] | 0 |
| 20 | A_margin | P_F | 0.048 | 0.310 | +0.242 [+0.234, +0.253] | 30 |
| 20 | A_margin | P_τ | 2.8·10⁻⁵ | 1.3·10⁻⁴ | +1.0·10⁻⁴ [+9.2·10⁻⁵, +1.5·10⁻⁴] | 30 |
| 20 | A_margin | Q_τ(1000) | 0.028 | 0.120 | +0.095 [+0.088, +0.138] | 30 |
| 20 | A_margin | E[V] | 37.0 | 3.6 | −33.3 [−34.4, −32.6] | 0 |
| 20 | A_margin | R_τ | 1.524 | 1.012 | −0.560 [−0.593, −0.337] | 5 |
| 20 | A_margin | D_cond | 0.145 | 0.092 | −0.045 [−0.062, −0.028] | 3 |

### The objective's scale in the normalised Hamiltonian

Sec. V-B and Sec. VI-C give the scale of the objective against the coefficient that normalises the
Hamiltonian at A_heur; three different numbers could be meant by that. The largest entry of the QUBO matrix and the scale s_H = max(max|h_i|, max|J_ij|) that normalises
the circuit's Hamiltonian are different numbers, since each h_i sums several entries of Q, and
the quantity the argument needs is neither: it is ε_F(A) = Δ_F / s_H(A), the objective's whole
range over the feasible set in units of the Hamiltonian the optimiser descends (on F the penalty
vanishes and the constant cancels in differences). `coefficient_scale.py` computes all three per
instance and setting with no circuit; from `paper01_coefficient_scale_summary.csv`.

**Table S35. Objective against penalty, three ways.** Median [min, max] over the thirty instances
of each size. "QUBO" is max|Q_objective| / max|Q|; "Ising" is s_H(objective only) / s_H(A); ε_F is
Δ_F / s_H(A). The 0.5 % of the earlier text was the QUBO ratio at N = 16.

| N | A | QUBO ratio | Ising ratio | ε_F = Δ_F / s_H |
|---:|---|---|---|---|
| 12 | A_heur | 1.16 % [0.66, 1.74] | 1.24 % [0.72, 2.28] | 1.46 % [0.79, 2.71] |
| 12 | A_margin | 46.51 % [33.52, 66.69] | 41.40 % [33.05, 52.92] | 51.99 % [23.98, 84.19] |
| 12 | 1.1·A_crit | 72.52 % [47.12, 95.14] | 69.06 % [47.45, 92.16] | 78.04 % [30.02, 126.30] |
| 16 | A_heur | 0.54 % [0.34, 0.83] | 0.88 % [0.61, 1.55] | 1.51 % [0.85, 2.49] |
| 16 | A_margin | 20.41 % [12.97, 37.22] | 28.68 % [22.72, 35.25] | 45.48 % [35.14, 75.81] |
| 16 | 1.1·A_crit | 50.69 % [36.10, 89.05] | 52.26 % [42.87, 83.21] | 86.85 % [53.54, 178.95] |
| 20 | A_heur | 0.34 % [0.19, 0.49] | 0.72 % [0.51, 1.24] | 1.63 % [1.05, 2.50] |
| 20 | A_margin | 10.17 % [7.39, 15.97] | 21.73 % [17.62, 25.94] | 47.30 % [34.45, 67.34] |
| 20 | 1.1·A_crit | 28.61 % [17.71, 69.87] | 43.43 % [35.67, 70.49] | 97.56 % [64.38, 198.26] |

### The precision diagnostic, scored on acquisition

**Table S36. The precision diagnostic scored on acquisition.** Single minus double precision on the
same (N, instance, weight, seed) of the diagnostic block (48 runs, 24 pairs), from
`paper01_e3_precision_pairs.csv`: each cell is the median (maximum) of the absolute
difference over the pairs, and the last column counts the pairs on which the best feasible shot's
within-τ verdict agrees (on each of the 8 disagreeing pairs the two states' P_τ agree to 3.0·10⁻⁶ or
better and Q_τ(1000) to 6.4·10⁻⁴, so the disagreement is the draw).

| N | Pairs | \|Δ⟨H_norm⟩\| | \|ΔP_F\| | \|ΔP_τ\| | \|ΔQ_τ(1000)\| | \|ΔD_cond\| | Verdict agrees |
|---:|---:|---|---|---|---|---|---:|
| 12 | 8 | 2.8·10⁻⁶ (5.3·10⁻²) | 1.2·10⁻⁵ (6.6·10⁻²) | 2.0·10⁻⁷ (3.8·10⁻³) | 1.4·10⁻⁴ (4.3·10⁻²) | 1.9·10⁻⁵ (9.0·10⁻²) | 4 |
| 16 | 8 | 1.1·10⁻⁴ (4.1·10⁻²) | 2.0·10⁻⁵ (5.0·10⁻²) | 1.1·10⁻⁷ (6.4·10⁻⁴) | 1.7·10⁻⁵ (2.3·10⁻¹) | 2.8·10⁻⁵ (8.3·10⁻²) | 5 |
| 20 | 8 | 4.2·10⁻⁶ (4.7·10⁻⁵) | 1.0·10⁻⁵ (9.4·10⁻⁴) | 2.3·10⁻⁹ (5.3·10⁻⁷) | 2.3·10⁻⁶ (5.1·10⁻⁴) | 7.9·10⁻⁶ (9.8·10⁻⁴) | 7 |

### The exact-optimum share per E1 setting

**Table S37. Runs whose best feasible shot is the exact optimum, per setting (E1).** The population
Fig. 2 draws: every completed E1 run, from `paper01_e1_optimum_hits.csv`. Each cell is hits / runs
with a feasible shot (the gap panels of Fig. 2 plot only those) and the share among them; the
completed runs per cell are 150, except 45 at A = 0 and 105 at the A_crit multiples of N = 12
(the nine A_crit = 0 instances run only A = 0, A_margin and A_heur).

| N | A = 0 | 1.1·A_crit | 2·A_crit | 5·A_crit | 10·A_crit | A_margin | A_heur |
|---:|---|---|---|---|---|---|---|
| 12 | 41/45 (91 %) | 99/105 (94 %) | 89/105 (85 %) | 76/105 (72 %) | 73/105 (70 %) | 118/150 (79 %) | 115/150 (77 %) |
| 16 | — | 89/150 (59 %) | 72/150 (48 %) | 59/134 (44 %) | 54/135 (40 %) | 56/139 (40 %) | 51/136 (38 %) |
| 20 | — | 25/150 (17 %) | 23/150 (15 %) | 8/150 (5 %) | 17/150 (11 %) | 18/150 (12 %) | 4/149 (3 %) |

### Which factor carried the paired P_τ gain

**Table S38. P_τ(A_margin) / P_τ(A_heur) as P_F ratio × R_τ ratio, on the same pairs.** From
P_τ = P_F · u_τ · R_τ with u_τ an instance constant, the ratio of the two weights' one-shot success
probabilities on the same (instance, seed) factorises exactly, and log turns the product into a
sum; the geometric mean over the 150 pairs per size is therefore the decomposition (the medians are
levels and do not multiply). No pair has a zero probability on either side. From
`paper01_e1_acquisition_factors.csv`.

| N | Pairs | P_τ ratio, geometric mean | = P_F ratio × R_τ ratio | Share of the log gain in P_F | Medians (P_τ / P_F / R_τ ratio) | Pairs with R_τ ratio > 1 |
|---:|---:|---:|---|---:|---|---:|
| 12 | 150 | 1.64 | 1.35 × 1.22 | 60 % | 1.69 / 1.26 / 1.37 | 113 |
| 16 | 150 | 1.59 | 1.10 × 1.45 | 21 % | 2.41 / 1.26 / 1.56 | 125 |
| 20 | 150 | 2.29 | 1.79 × 1.28 | 70 % | 2.52 / 1.61 / 1.02 | 88 |

### Figure manifest

- **Fig. S1** — `notebooks/figures/fig1_convergence.pdf` — ⟨H_norm⟩ vs. ADAM step, one curve per N (Table S2).
- **Fig. S2** — `notebooks/figures/fig3_runtime_scaling.pdf` — runtime vs. N, classical (exact enumeration to N = 30, annealing at N = 50) vs. QAOA (log y).
- **Fig. S3** — `notebooks/figures/fig2_optimality_gap.pdf` — reported gap vs. N (Table S2).
- **Fig. S4** — `notebooks/figures/fig4_sharpe.pdf` — Sharpe ratio of the selected portfolio vs. N (Table S2).
- **Fig. S5** — `notebooks/figures/fig9_degeneracy.pdf` — Sharpe ratios spanned by every feasible solution within a 0.1 % offset-normalized gap, per N, and the single-seed QAOA solutions under both denominators.
- **Fig. S6** — `notebooks/figures/fig10_instances.pdf` — deflation and the Sharpe spread inside the 0.1 % band over 30 instances per size, with the reference instance marked.
- **Fig. S7** — `notebooks/figures/fig11_gap_vs_sharpe.pdf` — every sub-optimal QAOA solution of the seed-42 instance sweep, reported gap against Sharpe deviation; and the exact-optimum rate per size.
- **Fig. S8** — `notebooks/figures/fig5_seed_variance.pdf` — the five-seed sweep (a sub-block of E1): gap across initial-angle seeds (symlog), the infeasible run marked separately.
- **Fig. S9** — `notebooks/figures/fig6_depth_sensitivity.pdf` — gap and runtime vs. QAOA depth p (Table S20).
- **Fig. S10** — `notebooks/figures/fig7_steps_depth_grid.pdf` — the steps × p grid over N ∈ {12, 16, 20}: optimised ⟨H_norm⟩ (top) and best-of-1,000 gap (bottom), median with IQR over five seeds.
- **Fig. S11** — `notebooks/figures/fig_e3_optimizer.pdf` — E3. Top: Q_τ(1000) under each optimizer setting for A_heur and A_margin, every run as a point. Bottom: the paired change in ⟨H_norm⟩ against the baseline (symlog), with the baseline's within-instance seed spread as a grey band and the budget actually used under each tick.
- **Fig. S12** — `notebooks/figures/fig8_cross_backend.pdf` — SV1 vs. lightning.gpu: gap per N and the per-step distance between the two optimizer trajectories (log y).
