# A Verifiable and Cost-Controlled Experiment Platform for Cloud Quantum Optimization Benchmarks

*Draft v0.3 — 2026-09-20. Target: ACM Transactions on Quantum Computing (acmsmall). Companion to the
numerical study [1], which this paper cites for its data and does not repeat.*

**Author.** Hiroaki Tobita, Advanced Institute of Industrial Technology, Tokyo, Japan.

## Abstract

A cloud quantum benchmark has inputs the provider revises, task counts the framework decides, and
records too numerous for a mix-up to show in an aggregate; at each, a script writes an inference
where a fact belongs. We present three design principles, implemented in the platform behind 2,565
runs of the quantum approximate optimization algorithm: bind records to inputs through
machine-checkable relations, reject unsafe configurations before submission, and provide
reconstruction levels with measured costs. The platform detects all ten injected faults, against one under schema
checks and eight under our reimplementation of declarative data-quality checks over tabular
records; the two the declarative checks miss require verification against pinned input statistics
and run artifacts.
Structural guards prevent an optimizer misconfiguration whose measured per-step circuit count
extrapolates to 9,027 tasks and $74,924 at list prices. A model of the provider's documented $50
cap permits six tasks before rejection; the cap is unavailable for simulators, where the
unintended expansion was observed. Reconstruction takes 24 seconds for figures and table checks,
18 seconds for a minimal rerun, and 222 seconds for a 24-run block; tested interruption lost only
the run in flight. Measuring the three principles caught the platform leaving a fact unwritten
three more times.

**Keywords.** reproducibility; experiment management; data provenance; quantum cloud services;
QAOA; cost control; fault injection; software engineering for quantum computing.

---

## 1. Introduction

A benchmark of a variational quantum algorithm on a cloud service is an experiment with three
moving parts that a laboratory instrument does not have. The *input* is not a fixed dataset but a
query — in the finance setting, adjusted closing prices that the provider revises between calls, so
that two runs over the same date window need not be the same problem. The *execution* is metered by
the task and the shot, and the number of tasks is decided not by the algorithm's description but by
the framework's choice of differentiation method, which it makes from the trainability flag of a
tensor and not from whether anyone asked for a gradient. The *record* is large — thousands of runs,
each with angles, a state, a shot batch and a bill — and a mix-up between any two of its parts
(a row pasted from a different window, a batch drawn from the wrong state, a column dropped by a
writer that takes its header from the first row) leaves the aggregate looking plausible.

What these three have in common is the failure they invite: at each of them an experiment script
records an *inference* where a *fact* belongs. A date range stands in for the values it retrieved;
the algorithm's description stands in for the circuits the service executed; a printed "done" stands
in for how a job ended; a plausible aggregate stands in for the agreement of the records beneath it.
The numerical study this platform served [1] — 2,565 recorded QAOA runs across three problem sizes,
thirty instances, five seeds and six penalty settings, with control arms on a cloud simulator and a
trapped-ion device — met seven implementation defects that changed reported numbers, and every one
of them had this shape: a window pinned by date that the provider revised underneath; a sampling call
billed as 177 tasks because the framework, not the algorithm, decided the count; a "shot-seed
control" that re-seeded a generator the device did not read; a device's answer replaced by a client
library's misparse of it; a batch drawn from the initial state while the record said the final one.
Six of the seven were found by inspection or by an ad hoc comparison and the last by a check between
two records that should have agreed and did not (Section 6): detection depended on luck, because the
inference was never written down as something that could be checked.

The present paper asks how the experiment platform should be designed so that the inference is
replaced by a record at each of the three places, and the records are checked against one another by
a machine — so that mix-ups are detected, mis-billing is prevented before submission, and a third
party can rebuild the results at a known cost. It answers with three design principles, one per
place, each stated with what it does and does not prevent, and each evaluated against a control
configuration that keeps the inference. The principles live in the experiment platform's own code and
records, not in the cloud provider's settings: no provider setting binds a run to its market
statistics, checks a batch against its state, or records how a job ended, and the one provider
control that bears on billing — a per-device dollar cap — enters the evaluation as a *control* for
the second principle, not as its implementation. The setting is deliberately small: one researcher,
one GPU workstation, a commercial cloud account, and a total cloud bill of $42.59 for the two papers'
runs. Systems reported from dedicated quantum infrastructure with a team behind them can be read but
not repeated by most of their readers; the unit of reproduction here is a person with a laptop and a
credit card, and the design is what makes that unit safe to use.

**P1 — Bind each record to its inputs and its outputs, and make the relations between records
checkable.** A run row carries the instance's input digest, the penalty constant and its label, the
initial angles, the final angles, the exact-state quantities, the shot summary and the cost; a
per-run artifact carries the trajectory, the stored probabilities and the shot counts; a reference
row per instance carries the enumerated optimum and the penalty scales; the derived statistics are
the pinned input. Thirteen checks connect these records: consistency conditions that a correct record
satisfies, numerical recomputations within stated tolerances, and a few diagnostics that mark a run
to look at. A fault appears as a check that reports.

**P2 — Refuse billing by configuration, before submission.** A quantum processing unit (QPU) cannot
be named as the optimizer's backend: the constructor refuses it before any cloud object exists and the
command-line driver refuses it before constructing a solver. The final sampling call receives frozen
angles, so that the framework's fallback from adjoint to parameter-shift differentiation — which
would submit 1 + 2p·[C(N,2) + 2N] tasks for one sampling call — cannot occur. A cloud run must name
itself and cannot overwrite the committed baseline. A dollar cap is layered on top, not relied on.

**P3 — Guarantee reconstruction at three levels and record the cost of each.** Tables and figures
from committed aggregates (seconds); one instance end to end (minutes on a CPU); the cross experiment
from its manifest (GPU hours). The cross driver flushes after every run, restarts idempotently with
`--resume`, and writes how it ended beside its output; the single-instance driver flushes after
every size and has neither — it merges into an existing file rather than resuming, and a long run of
it is covered by the detached wrapper's status file (Section 3.3) rather than by one of its own.

Each principle is stated with what it prevents and what it does not (Section 3), and each is
evaluated against a control (Section 4): the record with its verification relations against a
schema-only pipeline (E-S1, fault injection); the structural guards against no guard and against a
spending limit alone (E-S2, guard-stage measurement on a stand-in device); the reconstruction levels
against a package stripped of its management features (E-S3, a clean-room clone). The evaluation
touches the cloud account in two places: 178 SV1 tasks to observe on the service two facts the guard
harness had modelled, and 1,071 SV1 tasks across six runs of an account-level cost control to measure
how far past its own limit it lets a loop go (Section 5.2; $4.69 in all, Table 13).

**What form these problems take in a quantum experiment.** None of what this platform binds is
unique to quantum computing: an input the provider revises, a job whose end state is inferred, a
metered call whose cost the framework decides, a result that is a sample, a package a reader has to
rebuild. What is worth stating is the shape each takes here, because the shape is what the design
answers. The billable unit is decided by the trainability of a tensor, because a gradient of a
quantum circuit *is* more circuits: one optimizer step on a device without adjoint support executes
1 + 2p·[C(N,2) + 2N] tasks, so a flag in the client library moves the bill by two orders of
magnitude. A result is a sample, and the distribution it should have come from is not a model but an
object the simulator computed and the run stored, so testing the sample against it is a relation
*within* the record rather than an appeal to theory outside it — and the same relation has no force
on hardware, where a mismatch is as likely to be noise. Three more, none of them quantum in
themselves: the provider's dollar cap exists for hardware and not for the simulator on which this
project's real fallback occurred; the device this paper needed answered `OFFLINE` to 336 queries over
seven days, so a hardware observation is a waiting problem; and single-precision state quantities are
not portable across hosts, so the README must say in advance which columns a reader should expect to
move. A provider's policy, a queue and a floating-point fact — recognisable anywhere, and each with a
particular edge here.

**Contributions.** Three, one per principle; each names the sections that state it and evidence it.

1. **P1, and how far a record has to reach.** Thirteen verification relations across the records of
   one metered run, with their scope and their implementation (Section 3.1), and a fault-injection
   evaluation of a 2,565-run record (Section 5.1): ten faults detected of ten, against one for a
   schema-only pipeline and eight for the same expectations stated in a data-quality library's
   vocabulary over the run table, with no check firing on any of five committed clean records. Over
   600 random mutations, half drawn after the checks were frozen, the two are close wherever a
   headline number moves — 51 against 49 — and separate on the 168 that touch a file the run table
   does not contain, which the relations catch 135 times and the control never.
2. **P2, and where a guard has to sit.** A configuration in which the expensive call cannot be
   expressed, acting where the framework fixes the billable unit rather than after a threshold is
   crossed (Section 3.2), and a guard-stage evaluation on a device that, like a QPU, offers no
   derivative method (Section 5.2): the circuits one optimizer step sends, counted
   (177 / 361 / 609 / 921 at N = 8 / 12 / 16 / 20), which a metered device bills as one task each;
   the stopping stage of five misconfigurations under four guard configurations; the division of
   labour between a structural guard and a spending limit; and the fallback re-observed on the
   service.
3. **P3, and what a reconstruction costs.** Levels of stated cost, an interruption that loses only
   the run in flight, and an end state written rather than inferred — and compared with the file it
   describes (Section 3.3) — with a costed clean-room reconstruction of the public package at three
   levels, including interruption and idempotent resumption, against a control that lacks the
   derived statistics (Section 5.3).

Two results came out of the work rather than out of its design. A record bound to its inputs
answers questions that were not asked when it was written: two hypotheses about the platform's next
use were settled from the committed record alone, in minutes of CPU and with no quantum run, each
re-scoring checked against values the record already held (Section 5.4). And measuring the three
principles kept finding the platform recording an inference of its own — a digest of asset *names*
standing in for the statistics' *values*, a failure that did not say which input was missing, a job
whose end state was read off a log line for nine hours — with four more found by reading the
drivers against the sentences that describe them, and a fifth by a reader who had only the package
(Section 7).

## 2. Background and related work

### 2.1 Related work

**Provenance and experiment tracking for quantum software.** QProv [2] defined provenance
attributes for quantum computation and a one-record-per-job system that links what was executed,
where, under which calibration, with what result. Stirbu and colleagues adapt MLflow to quantum
development [3], instrument Qiskit's transpiler so that compilation provenance is captured
automatically [4], and propose a provider-independent provenance contract with adapters for Amazon
Braket, IBM Quantum and IonQ after analysing fifteen platforms [5]. Qunicorn [6] and QUARK [7]
address unified execution and application benchmarking across providers; Metriq [8] unifies
benchmark definition, execution, data collection and public presentation so that "the same
benchmark" means the same experiment across platforms.
Two recent systems are closer in shape. QEF [9] expands a concise experiment specification into
the Cartesian product of its variants, partitions the sweep into asynchronous jobs and replays it,
which is the shape of E1 in Section 3; it stops at reproducing the sweep — the specification is not
bound to the values of the application's inputs, and what a run costs is outside it.
A three-month field report on commercial cloud quantum environments [10] records connection metrics,
job execution and tool behaviour as operational metadata — the kind of observation Section 5.2 makes
about a device that stayed offline and a policy that took 7–146 s to propagate — and reports what the
runs cost, but not what refuses a run that should not be paid for. These works define *what to record* and *where to put it*. None binds the record to the
application's input values (a market-data statistic, a penalty constant) as well as to the billed
execution, none states relations between records that a machine can check, and none measures whether
the record detects a mix-up. P1 is a design on top of such a container, and E-S1 is the measurement
they do not report.

**Experiment management and provenance in machine learning.** MLflow [11] gives the container;
MLflow2PROV [12], ProvLake and related work [13] give a W3C PROV representation of ML pipelines;
Ursprung [14] captures provenance transparently. Lineage-driven fault injection [15] uses lineage to
decide which faults to inject into a distributed system; E-S1 borrows the evaluation type — inject a
fault, observe whether the lineage detects it — and applies it to a scientific record rather than a
protocol.

**Declarative data validation, testing without an oracle, and packaging.** Closer to P1's mechanism
than to its subject, declarative data validation states expectations about a dataset and has a
machine check them: Deequ [16] compiles constraints on completeness, uniqueness, admitted values,
ranges and simple predicates into a single pass over the data, and pandera [17] and comparable tools
do the same for data frames. Their declarative part is a vocabulary of constraints over one table: completeness, type,
admitted set, uniqueness, size, range, a predicate over the columns of a row, and — in pandera — a
check over the whole frame. Both are extensible, and a user may register an arbitrary Python
function as a check, so nothing here is beyond what such a tool can *run*. What P1 adds is a
different thing: the expectations themselves, stated between the heterogeneous records of one run —
the row, its artifact, the instance's reference, the pinned inputs and the provider's task ledger —
and several of them leave the declarative vocabulary behind, because they recompute a quantity from
a stored state, hash an external input, or compare a shot batch with the distribution it should have
been drawn from. Written inside such a tool those would be custom functions that open another file:
the tool supplies the runner and the report, and the relation is still ours to write. Section 4.1
therefore makes the declarative vocabulary the second control of E-S1 — stating its boundary as the
comparison's definition rather than as a limit of any product — and Section 5.1 reports which faults
it reaches. The wider
question behind it is the oracle problem of scientific software, where the expected output is not
known and testing proceeds by relations that must hold between outputs [18]; the relations here are
of that kind, but between the *records* of a run rather than between runs of a program. The line is
current rather than historical: PrismaDV [19] generates data unit tests from the code that consumes
the data, by the group that built Deequ. Packaging, workflow and experiment-management systems
answer a different half: Snakemake [20] makes an analysis re-runnable from a declarative workflow,
RO-Crate [21] makes an artefact's metadata portable, ReproZip [22] packs an execution with the
environment it needs, Sacred [23] records the configuration of every run, and DVC versions data
beside code. Each of them keeps a run reproducible or a configuration recoverable; none states what
must be true of the numbers the run produced, which is what a reader of a benchmark has to check.

**Reproducibility in quantum computing research.** Köster et al. [24] screened 127 papers by hand
and about 5,000 automatically: 24.4 % release code, 64.5 % of released code fails to execute in a clean
environment, 54.8 % of released code has a machine-readable environment specification, 2.4 % of
papers specify hardware completely. Mauerer and Scherzinger [25] propose a generator for DOI-safe,
dependency-free reproduction packages. Ye et al. [26] audited 455 comparative claims in 119 quantum
software papers and found that eight could be audited without reconstruction. Köster and Mauerer
[27] show that error-mitigation conclusions reverse with configuration and with hardware drift. A
longitudinal study of a quantum defect dataset across 21 versions of the core library [28] finds its
reproducibility falling from 62 % to 16 %, 94 % of the failures dependency-related. These studies measure the field; E-S3 measures one package against their criteria (clean
environment, pinned dependencies, single entry point, executability) and adds what they do not
measure: the cost of each reconstruction level and the behaviour under interruption.

**Cost control on cloud quantum services.** The provider used here offers a per-device dollar cap
for hardware tasks, checked at task creation against a list-price estimate, and none for its
simulators [29] (Appendix A.1). Account-level controls are post-hoc: AWS Budgets [30] notifies after
a threshold is crossed, and AWS's deployable cost-control solution for Braket [31] (2023) turns
task-state events into a near-real-time running cost with an optional IAM denial, for simulators and
QPUs alike. Adaptive shot allocation [32] reduces the cost of a *correct* run; from the other side,
per-shot pricing has been shown to be exploitable through mid-circuit reset, so that shots are
obtained without being paid for [33]; and FinOps policy-as-code gates infrastructure provisioning
before deployment. One attempt to price the first segment of a small quantum neural network arrived
at $2,000, $60,000 and $1,000,000 on three services for the same work [34]. P2 is orthogonal to cost
optimisation: it concerns runs that should not happen, and E-S2 measures where a dollar cap and a
structural guard each stop them.

**What is proposed here, against those lines.** Each line above answers part of the problem:
experiment-management and packaging systems keep a run re-runnable and its configuration recoverable;
data-validation libraries state what must be true of a table; the reproducibility studies of quantum
software measure whether a package executes at all. Three things this paper proposes are not in any
of them, and the claim is the combination rather than any one — the three are P1, P2 and P3 of
Section 3. First, a set of verification relations *across* the records of one metered quantum run —
the pinned input statistics, the reference enumerated from them, the run row, and the stored state
and shot batch — including one that tests the batch against the state it was drawn from. Second, a
configuration in which the expensive call cannot be expressed, acting where the framework decides the
billable unit rather than after a threshold is crossed. Third, a reconstruction designed as levels of
stated cost, with an interruption that loses only the run in flight and a job whose end state is
written rather than inferred. The three evaluations are the evidence for what each of the three
reaches and where it stops; the fault injection is evidence about the relations, not the proposal
itself.

### 2.2 Vocabulary

We use *reproduction* for obtaining the published results with the author's
artifacts, following ACM's badging terminology [35] and the lexicon of Goodman et al. [36] as applied
to computational work by Plesser [37]. The three levels of P3 are levels of *cost* within
reproduction, not of independence. Against ACM's badges [35], the package's deposit is
*Artifacts Available*, and the README's per-OS instructions and pinned environment are what
*Reusable* asks for. The three levels are not badge claims: *Results Reproduced* is defined as
reproduction by a *different* team, and every run reported here is the authors' own, on a host of
their own. What the levels measure is the work such a team would face — what has to be installed,
what has to be decided, how long each level takes, and which columns agree — with E-S3 run from a
clean clone by a harness that may use nothing but the README. Read that way the three levels are
evidence for *Artifacts Evaluated — Functional* and a cost estimate for the badge we cannot award
ourselves; level 3's agreement is claimed only for the columns the README says are portable — the
reference quantities and the initial angles — and explicitly not for the rest.

### 2.3 The companion study

**The numerical study.** The optimisation data in this paper are all from [1]: the E1 cross (2,565 runs on a
local GPU at single precision: N ∈ {12, 16, 20} × 30 instances × seeds 42–46 × up to six penalty
settings), its controls, the SV1 and IonQ arms, and the derived statistics of two market windows.
This paper contributes no new optimisation runs; its evidence is what the record, the guards and the
package do under injected faults, misconfigurations and a clean-room reconstruction, plus 1,249 SV1
tasks of its own: 178 that re-observe one framework behaviour on the service, and 1,071 that measure
an account-level cost control over six runs (Section 5.2, Table 13).

**What is in that study and what is here.** Four subjects appear in both papers, and the division is
the same in each: the study reports a fact about its runs, and this paper reports the mechanism that
was built in response and what that mechanism was measured to do.

- *The sampling call billed as 177 tasks.* There, the count, the cause and the fix. Here, one of five
  misconfigurations whose stopping stage and bill are measured under four guard configurations at two
  sizes (Table 7), re-observed on the service for this paper (Section 5.2).
- *The seven implementation defects.* There, defects that changed reported numbers. Here, each is
  placed beside the check that now holds its class — or beside a dash, for the four that no check
  holds (Section 6, Table 12) — and two of them are replayed as
  injected faults whose detection is measured. Two further defects of the study, which concerned the
  record's writer and its inputs rather than a run, are replayed as well, so four of the ten injected
  faults come from something that happened.
- *The revised price snapshot.* There, the incident and the single-snapshot fix. Here, the fault that
  exposed the gap between naming an input and binding its values, and the reason for the content
  digest that now sits on every record (Section 5.1).
- *The reproduction package.* There, a statement of availability. Here, a subject: cloned at its tag,
  rebuilt at three levels, interrupted and resumed, and run on four machines, with the cost of each
  step (Section 5.3).

No run of the study is repeated and no number of it is restated as a result of this paper: the record,
not the experiment, is this paper's data.

## 3. The platform

The platform is a Python code base with a solver abstraction over PennyLane [38] devices
(`lightning.qubit`, `lightning.gpu`, the Braket local simulator, Braket SV1, IonQ Forte-1),
command-line drivers per experiment, aggregation scripts, a manuscript checker, and an AWS CDK
footprint (a results bucket, an IAM role scoped to it, two Budgets). This section states the three
principles as design decisions, each with its scope.

### 3.1 P1 — Bound records and checkable relations

**The quantities the record holds.** The relations below recompute these, so this paper states them
rather than sending a reader to the numerical study. An instance is N assets of which exactly K must be selected; F is the
set of bitstrings with that cardinality, f(x) the portfolio objective of x on the pinned statistics
μ̂ and Σ̂ (a risk-adjusted return, no penalty in it), f\* its minimum over F, and Δ_F the feasible range, the largest f over F less f\*. The penalised objective is H_A(x) = f(x) + A(Σx − K)², which on F is just f(x).
What the solver minimises, however, is a QUBO, and a QUBO matrix carries no constant: expanding the
penalty and dropping its constant A·K² gives the energy the solver reports,
E_A(x) = xᵀQx = H_A(x) − A·K², which on F is f(x) − A·K².
E_A is the energy the record stores (`energy_no_offset_solver`), and the constant is stored beside it
(`QUBOProblem.offset`), so the two are recoverable from each other. |E_A(x\*)| = |f\* − A·K²| is the
magnitude the solver's own optimum reports, and it depends on the penalty weight A rather than on the
solution — which is the distortion that study is about. From these: the *feasible-range gap*
g_F = (f(x) − f\*)/Δ_F, free of A; the *offset-normalized gap* g_off = (f(x) − f\*)/|f\* − A·K²|, the
one a solver's own energies give; and the *deflation* D(A) = |f\* − A·K²|/Δ_F, which equals g_F/g_off
wherever that ratio is defined (at the optimum both gaps are 0 and the ratio is not). Under the state a run ends in, P_F = Pr[x ∈ F],
P_opt = Pr[x is an optimum], P_τ = Pr[x ∈ F and g_F(x) ≤ τ] with τ = 0.01, and
Q_τ(S) = 1 − (1 − P_τ)^S is the success probability of an S-shot batch of i.i.d. draws. Three penalty
weights recur: A_heur = 2·s_obj·N + 1, the formulation's own heuristic; A_crit, the smallest weight
at which the constrained optimum is a global minimiser of H_A; and A_margin, which places the best
infeasible assignment at the midpoint between f\* and the feasible mean. [1] derives all of them; what
matters here is that each is a function of the pinned statistics alone, so a relation can recompute
it.

**The record.** Four kinds of file describe a run.

- *Derived statistics* (`data/derived/stats_<start>_<end>_<n>_<digest>.csv`): the annualised mean
  return vector and covariance matrix of the fifty-ticker universe for one price window, written with
  17 significant digits and read back with round-trip parsing. Every instance is a slice of one file
  (a per-size cache once held a sixteen-asset file fetched seven minutes later than the others and
  revised by the provider by up to 4.2·10⁻⁷; one snapshot per window removes that class of
  inconsistency). The price snapshots themselves are third-party data and are not distributed; the
  statistics are the only input the pipeline reads.
- *Reference row* (one per (N, instance)): the ticker set and its digest, the content digest of the
  instance's μ and Σ (`input_digest`, a hash of the float64 arrays; added after the first pass of
  E-S1, Section 5.1), the enumerated feasible
  set's size, optimum f\*, range Δ_F and mean, the penalty scales A_crit, A_heur, A_margin with their
  derivations, and the uniform-sampler baselines.
- *Run row* (one per run, 101 columns): the instance and penalty keys (label and value), the input
  digest, the seed,
  backend, precision, optimizer and step count, the reference quantities copied from the reference
  row, the shot summary (counts, best shot and its energy from the solver and from the scorer, the
  best feasible shot's f, g_F, g_off, rank), the exact-state quantities at the final and the initial
  angles (P_F, P_opt, P_τ, Q_τ(S) at four budgets), the expectations, both angle vectors, the
  evaluation counts and the wall-clock of each phase; on metered backends the task identifiers.
- *Per-run artifact* (`.npz`): the initial and final angles, the trajectory, the convergence
  curve, the feasible set's indices and objective values, the stored probability of every feasible
  bitstring under the final and the initial state, the per-cardinality mass, the shot indices and
  counts, the penalty and the Hamiltonian scale.

![Fig. 1. The four files that describe a run and the thirteen checks that bind them. A solid double-headed arrow is a relation between two files, named on the arrow; the relations that hold within one file are listed inside its box. The one dotted arrow is not a relation but the schema-only control of Section 4.1.](figures/fig_record_relations.pdf)

**The relations.** Figure 1 places them. Thirteen, each implemented as a function that returns the
rows on which it does not hold. Most are consistency conditions that a correct record satisfies
outright; three clauses are diagnostics that a correct record can in principle trip, and they are
marked as such below, so the set is thirteen checks rather than thirteen invariants. Section 4.1 lists them as the
checks of E-S1 and gives the stage at which each runs. Each is marked with what a violation means:
**[E]** exact — the two sides are equal or the record is inconsistent; **[N]** numerical — one side
is recomputed and compared within the tolerance given; **[S]** statistical — a rule a correct record
can fail, so a finding is a suspicion and not a proof. Several relations hold clauses of more than
one kind and carry more than one mark; where they do, the clauses are named.

1. *schema* **[E]** — every required column is present and filled on every completed run.
2. *manifest* **[E]** — each (N, instance) block holds exactly the (penalty, seed) cells its
   reference row prescribes, and the file carries one window and one specification version.
3. *penalty* **[N]** — the penalty value is the one its label names (relative 10⁻¹²), and the
   recorded A/A_crit is their quotient (10⁻⁹).
4. *reference-row* **[E]** — the instance quantities copied onto a run row equal the reference row's,
   bit for bit: both are the same float64 written twice.
5. *input-digest* **[E]** — the digest on the reference row and on every run row is the digest of the
   pinned statistics.
6. *reference-input* **[E]** — the reference row is what those statistics give when the feasible set
   is enumerated again. The harness compares bit for bit, with no tolerance, which holds because the
   enumeration is deterministic on one host. Across hosts the numerical study reports differences of
   a few ulp [1], so on another machine this comparison has to be relaxed to a relative 10⁻⁹; the
   harness does not relax it, and a reader running it elsewhere should expect the difference rather
   than a pass.
7. *scorer* **[N]** — the solver's best shot, its energy and the scorer's decomposition agree
   (relative 10⁻⁶ between the solver's energy and the scorer's, 10⁻⁹ within the scorer), and g_F and
   g_off follow from f, f\*, Δ_F and A (10⁻⁹).
8. *row-npz* **[E/N]** — the row and its artifact describe the same run: the angle vectors and the
   shot totals bit for bit, P_F and P_opt as sums of the stored probabilities at a relative 10⁻¹².
9. *npz-internal* **[E/N/S]** — the artifact is self-consistent: the trajectory starts at the initial
   angles (bit for bit), the convergence curve ends at the final expectation (10⁻¹²), the stored
   probabilities are a sub-distribution, and the Hamiltonian scale on the row is the one in the
   artifact. One further clause is a diagnostic rather than an identity: the final angles must differ
   from the trajectory's first point, which a run stopped at a stationary point could legitimately
   violate but which no run of this record does, and which is what an optimizer handed a
   non-differentiable array produces (Section 6, defect 1).
10. *identities* **[N]** — every column that is a function of other recorded values equals that
    function, at a relative 10⁻⁹ with an absolute floor of 10⁻¹⁵, which is the ulp scale the
    numerical study reports between libm builds:
    Q_τ(S) of P_τ; P_τ and P_opt of the stored probabilities and f; the rank of the best feasible
    shot; the shot-level success counts of the stored batch; D and the offset magnitude |E_A(x\*)|; the best expectation of the
    stored convergence; the portfolio metrics of the pinned statistics; and, on the reference row,
    the penalty scales of its own cardinality minima and the uniform baselines of its counts.
11. *state-batch* **[S]** — the shot batch is a draw from the *final* state: the smaller one-sided
    exact binomial tail of the feasible and of the optimum count under S × P_F and S × P_opt is at
    least 10⁻⁹. That is one threshold per *count*, so a count is rejected with probability up to
    2 × 10⁻⁹ and a row, which carries two, with probability up to 4 × 10⁻⁹. And within each
    (N, penalty) block of at least ten runs whose final P_F actually varies, the feasible fraction
    correlates with final P_F above 0.9 and at least as well as with initial P_F.
12. *seed-distinct* **[E/S]** — two runs of one cell that differ only in seed have different initial
    angles and different shot batches. Neither clause follows from the mathematics of a continuous
    distribution: a pseudo-random generator is a finite map, and nothing in it forbids two seeds from
    agreeing. The angle clause is exact *in the scope of this record* and no wider — the generator is
    `numpy.random.default_rng` (PCG64), the draw is `uniform(0, π, 2p)`, and for the five seeds
    42–46 at p = 1…4 the vectors are pairwise distinct, which a test asserts. Outside that scope it
    is a diagnostic. The batch clause is a diagnostic everywhere: two batches are discrete objects and
    can coincide without a defect when the state is concentrated on few outcomes. It is the batch
    clause that fires on `shot_seed_noop`, in three instances; the angle clause has never fired,
    which is what the `shot_seed` defect predicts, since that defect left the angles alone.
13. *state-recompute* **[N]** — the state re-simulated at the recorded final angles reproduces the
    stored probabilities to 10⁻⁵ absolute; E1 ran in single precision on a GPU and the check in
    double precision on a CPU, which agree to about 10⁻⁶, while a substituted state is off by O(1).

**What the three kinds mean for a reader of a finding.** A violation of an exact relation is a
defect in the record: nothing but a wrong write produces it. A violation of a numerical relation is a
defect unless the record was produced under conditions the tolerance does not cover, which is why
each tolerance above is stated with the arithmetic it allows for. A violation of a statistical rule
is a *suspicion*: a correct record can produce one, and the threshold says how unlikely that is. For
the binomial tails the bound is a union bound over the counts actually tested, at 2 × 10⁻⁹ each:
this record carries two counts on each of 2,565 rows, so 5,130 tests bound the chance that any fires
on a correct record at 1.03 × 10⁻⁵. The true rate is smaller, because the tails of a discrete
distribution do not reach their threshold, but the bound is what the design guarantees. It covers
those counts and nothing else. The three diagnostics carry no such bound and are not meant to. The correlation floor
of 0.9 was read off the clean record's nineteen blocks, whose correlations run from 0.983 to 0.999,
and is a property of blocks of this size and spread rather than a constant: two guards are part of
the rule — a block of fewer than ten runs, or one in which final P_F does not vary at all, is not
tested — and a third condition is not enforced and should be, since a block whose runs all have
nearly the same P_F carries little information about the correlation and a correct record of that
shape would fail the rule. Two seeds carrying the same shot batch is possible without a defect
whenever the state is concentrated. And a run whose final angles equal its trajectory's first point
is what an optimizer handed a non-differentiable array produces, but also what a run stopped at a
stationary point would look like. Each reports something to inspect. Section 5.1 reports the one
false alarm the binomial test produced before its threshold was corrected, and the five clean records
on which no check fires at all.

**Where the batch-against-state rule applies.** *state-batch* compares a batch with the distribution the
same run recorded, which requires that distribution to be exact — it is, because the state comes
from a state-vector simulator that computes it. On hardware the recorded ideal state is not the
distribution the device sampled, and a failure would mean noise as readily as a mix-up; the relation
holds for the simulator arms this paper checks and is not claimed for hardware, where it would have
to be restated against a noise model or dropped.

**What P1 prevents, automatically and otherwise.** Before it reads a number, the aggregation runs
six validation routines over the two CSVs, and a record that fails one does not reach a table:
`analyze_e1` names the routine and exits non-zero unless told `--allow-inconsistent`. Against the
numbering above, what runs is

- five checks entire: 1 *schema*, 3 *penalty*, 4 *reference-row*, 7 *scorer*, 11 *state-batch*; and
- a sixth routine for the range and ordering clauses that 7 and 10 carry and that Section 4.1's
  control states as families — probabilities between 0 and 1, optimum ≤ within-τ ≤ feasible ≤ shots,
  P_opt ≤ P_τ ≤ P_F, f\* ≤ f of the best feasible shot.

It is six routines over selected clauses, then, not six of the thirteen. What does not run: 2
*manifest*, which needs no other file but which a reproduction of one block fails by design; 5, 6 and
13, which re-derive from the pinned statistics or by re-simulation; 8 and 9, which read the artifact;
and the clauses of 10 and 12 that do — the parts of *identities* that recompute from stored
convergence, and the batch half of *seed-distinct*. Those are the separate command of Appendix C, and
the README gives the order. One implementation serves both (`src/record_checks.py`), so the gate and
the checker cannot drift. Stopping on 11 is conservative: it is a statistical rule, so a refusal there
is a suspicion, and `--allow-inconsistent` is how a reader who has looked proceeds. Section 6 lists the seven defects of the numerical study, how each was
found at the time, and which check holds its class now — for four of them; three are held by none,
and the table says so.

**What P1 does not prevent.** A fault that preserves every relation is invisible: a consistent
substitution of the input statistics *and* the reference *and* every run computed from them is a
different, internally consistent experiment. The relations bound the record to itself and to the
pinned inputs, not to the world. The first pass of the evaluation exposed one narrower gap — the
instance digest was a hash of the ticker set, not of the statistics' values, so a revised statistic
passed every row-level relation and was caught only when the reference was re-enumerated — which the
`input_digest` closed (Section 5.1).

### 3.2 P2 — Structural refusal of billing, before submission

Three mechanisms, each measured in E-S2.

**Hardware is unrepresentable as the optimizer's backend.** `BraketSolver(backend="braket_ionq")`
raises `ValueError` in the constructor, before an AWS session, a device or a bucket reference is
created; the drivers refuse the same setting before constructing a solver. A QPU is accepted only as
`sample_backend`, the device of the single final sampling call, and a test spies on device
construction to assert that the metered device is built exactly once per run. The reason is the
framework's: a QPU device offers no adjoint differentiation, so PennyLane's `best` method resolves to
parameter shift and one objective-and-gradient evaluation of the p-layer ansatz on a dense N-asset
cost Hamiltonian executes 1 + 2p·[C(N,2) + 2N] circuits — 177 at N = 8, 921 at N = 20 for p = 2 —
each a billable task.

**Sampling angles are frozen.** The final sampling call receives `frozen_angles(params)`, a plain
array with the trainability flag removed. A QNode that returns one sample per wire cannot be
differentiated by the adjoint method; with trainable inputs the metered device's execution path
evaluated a parameter-shift gradient that nothing read. This was measured, not predicted: the N = 8
SV1 pilot of 2026-08-09 shows 227 tasks in the service's log — 50 analytic tasks for fifty optimizer
steps and 177 for one sampling call — and 51 once the angles were frozen (the per-task records are
archived and reduced to a per-run ledger, `paper01_sv1_task_ledger.csv`).

**Cloud runs name themselves.** A cloud or hardware backend requires `--results-tag`; an untagged
run finding the committed baseline files in place is refused unless `--write-canonical` is passed;
`--resume` appends only missing rows.

**The acknowledged estimate is part of the record.** A metered invocation prints its expected task count and price floor from its arguments and does not start without an acknowledgement flag; since this revision the acknowledged estimate is also written next to the tagged results, and a check compares it with the task count the provider's own records show for that invocation (`check_metered.py`). This is one relation across the billing boundary rather than one of the thirteen inside the record: the pilot of [1] that was acknowledged at 51 tasks and billed for 227 fails it, and the corrected run (204 acknowledged, 204 billed) passes. The relation is checked after the fact — it cannot stop the bill — but it turns a defect that was found by reading an invoice into one that a script reports. E-S2 does not evaluate this one: it is stated as a design decision and exercised only by the historical pair it reproduces (51 acknowledged against 227 billed, and 204 against 204).

**Three kinds of control, and which one P2 is.** A metered backend can be guarded at three points:
*in the configuration*, before a call is built (a wrong call is unrepresentable); *at the service*,
per device, by a dollar cap that rejects a task whose estimate exceeds it; and *at the account*, by
a detective control that watches task creation and notifies or revokes. P2 is the first kind. The
second is a provider setting where one exists — for the provider used here, only for hardware, not
for the simulator on which this project's one real fallback occurred — and Section 5.2 treats it as a
fourth configuration to show what it adds and what it misses. The third exists as a deployable
solution for this provider; it was not part of the platform's footprint, and Section 5.2 deploys it
once to measure what it adds. Provider specifics are in Appendix A.

**What P2 prevents.** The optimizer loop on hardware (stopped at the command line and at the
constructor), the sampling fallback (removed before the call reaches the device), and the overwrite
of the baseline by a trial run (refused before a solver exists).

**What P2 does not prevent.** A correctly built call that is expensive by its parameters: one
sampling task on IonQ at the default 1,000 shots costs $0.30 + 1,000 × $0.08 = $80.30 and passes every
structural guard, because nothing in its structure is wrong. Only a cap sees the shot count — and
only while the cap is below the price of the one task: at $50 the 1,000-shot call is refused because
it would cost $80.30, and a $100 cap would let it through
(Section 5.2). Nor does P2 prevent the *repetition* of correct calls — a script that invokes the solver
in a loop, a scheduler that re-issues a finished run — the failure a recursive serverless function
makes familiar. A structural guard sees the shape of one call, not the count of calls. The platform
bounds one invocation (a metered invocation prints its task count and price floor from its arguments
and does not start without an acknowledgement flag; the cross driver accepts local backends only) and
bounds how long it waits for one task (a one-hour polling limit in place of the client library's
five-day default, which frees the process but does not cancel the task or stop its charge),
but a bound per invocation is not a bound per account. That bound is the second and third kind of
control above; the platform's own footprint had only a post-hoc budget notification, whose lag is
hours against a pilot that submitted 227 tasks in fourteen minutes, and Section 5.2 measures the
third kind's lag in tasks. Nor does P2 address a provider's own failure modes — the four IonQ task records were
parsed from S3 directly after the client library misread the device's nested result schema into an
energy of exactly 0.0 [1, Sec. VI-C].

**The general form of the defect P2 guards against.** The sampling fallback is one instance of a
class: *the framework decides the billable unit from a property of an object rather than from what
the user asked for*. Here the property is a tensor's trainability flag and the unit is a
parameter-shift tape; elsewhere in the same kind of stack it is a primitive's default shot count, an
error-mitigation setting that multiplies the circuits submitted, or a hybrid job billed by wall-clock
while a classical loop waits. What makes the class dangerous is not the default but that the cost is
decided below the level at which the experiment is described, so that reading the experiment's
description does not tell anyone what it will cost. The structural answer is the same in each case:
make the expensive configuration unrepresentable, and write the acknowledged estimate into the record
so that the bill can be checked against it. We have measured this for one stack; the other instances
are named here as observations, not as results.

### 3.3 P3 — Three reconstruction levels with recorded cost

The public package documents three commands: (1) regenerate the figures and check every manuscript
cell against the committed CSVs; (2) run one small instance end to end on a CPU; (3) run the E1 cross
from its manifest on a GPU. Each driver writes its CSV and its artifact directory after every
run, and `--resume` reads the manifest, skips the (N, instance, penalty, seed) cells
already present, refuses a manifest written by a different specification version, and appends the
rest, so that a restart after any interruption converges to the same file as an uninterrupted run.
The reference row of an instance is written once and reused. The README states the environment that
produced the numbers (`requirements-repro.txt`, a `pip freeze`), the minimum versions the code runs
with, and which columns are expected to differ across hosts (single-precision state quantities and
every shot-derived column) and which are not (reference quantities, initial angles).

**A flush is only a bound if it lands.** The per-run flush rewrites the whole manifest, so it
bounds the loss at the run in flight only if the rewrite cannot be interrupted half-way. In the
released `v1.0` it could, and the run's line was printed before the row was written — the
log-line-as-fact pattern this paper is about, in the platform's own loop. Both are now the other way
round: every result file, the cross driver's manifest included, is written by one function that
lands it with a single rename, and the run's line follows the record rather than preceding it; a
test kills that function inside the write and asserts that what was on disk before is still there.
The heartbeat is written after the rename, so a status of `RUNNING k/n` always names k rows that are
on disk. Fault injection against the record cannot find this class of fault: the failure is in the
*writer*, in a window no injected fault occupies.

**The end state of a job is a written fact.** Flushing bounds the loss on interruption to one run;
it says nothing about how long the interruption goes unnoticed, nor whether the file a reader
aggregates is whole. The cross driver therefore keeps a status record beside its manifest
(`<manifest>.status.json`: `RUNNING` with a heartbeat after every run, the rows done of the rows
planned and the last run id; `EXIT 0` or `FAILED code` at the end; a kill leaves `RUNNING` with a
stale heartbeat, which is the stall; the heartbeat is written after the manifest, so the count it
carries is a count of rows on disk), prints the previous invocation's state on `--resume`, and the
aggregation refuses a manifest whose job is not `EXIT 0` and complete unless told to proceed.

A status record is itself a claim about a file, and a claim nothing compares to its subject is the
inference the record was written to remove. So the aggregation reads the manifest before it trusts
the record: the row count against the rows the record says were written, a run id that occurs twice,
a last run the file does not hold. A review of this driver deleted one row from a committed
thirty-row block, left `EXIT 0, 30/30` beside it, and the aggregation printed `30/30` and `29 done`
on adjacent lines and published the medians of twenty-nine; the same review duplicated a row and got
thirty-one. The comparison is exact where the job ended normally and allows one row where it did not,
because the driver writes the row before it beats and a kill between the two leaves the file one
ahead. It is not waived by the flag that aggregates an unfinished run: that flag is for a job known
to be incomplete, not for a file the record does not describe.

`EXIT 0` says the driver reached the end of its loop with every planned run attempted and recorded,
not that every run succeeded. A run that raises is recorded as a row with an `error:` status and
counted in the record's `failed`, which the state line prints; `--resume` fills the gaps the
manifest does not hold and does not retry those rows, because a recorded failure is an observation
about a configuration and re-running it would replace that observation with a second attempt's.
Any other long job runs under a wrapper that keeps the same four states in a one-line file. Both were
added after the evaluation's own harness had failed unnoticed for nine hours (Section 7), and the
driver's record is exercised in Section 5.3.

**A device's availability is a written fact too.** A hardware observation that depends on a device being online is not retried by hand: a poller records the device status at a fixed interval, launches the observation once when the status is `ONLINE`, and gives up after a stated number of days; its log is the record of how long the device was unavailable (Section 5.2 reports one such wait).

**What P3 prevents.** Loss of completed work on interruption beyond the run in flight; divergence
between a resumed and an uninterrupted run; a reader's dependence on the author's machine for levels
1 and 2; a reader's dependence on the market-data provider at all.

**What P3 does not prevent.** Reproduction of the metered arms without an AWS account and spend; the
N = 20 timing columns without the same GPU class; bit-identity of single-precision state quantities
across hosts, which [1, Supplementary Sec. S-VI] showed does not hold. Nor does the platform reconcile
its task ledger with the provider's invoice: the dollar figures in this paper were compared with the
billing console by hand (once finding a $4.67 omission in an earlier tally), because the billing API
is itself metered and an invoice settles days after the tasks.

### 3.4 What is not claimed

The platform is an experiment platform. It makes no claim about production trading systems, market
risk, security beyond least-privilege IAM for a results bucket, or regulatory compliance. Its
threat model is the experimenter's own mistakes and the provider's drift, not an adversary.

## 4. Evaluation design

Three evaluations, one per principle, each with a control. The cloud calls are 178 SV1 tasks for
the sampling-fallback observation, 1,071 SV1 tasks across the six repetitions of the account-level
cost control, and the two spending-limit requests of Section 4.2 (Table 13); no quantum processor
was billed.

### 4.1 E-S1 — Does the record catch a mix-up? (P1)

**Subject.** The committed E1 record: 2,565 run rows, 90 reference rows, 2,565 artifacts, the derived
statistics of the 2023-08-06 → 2026-08-05 window.

**Checks.** Thirteen functions, one per relation of Section 3.1, run in stage order so that the
cheapest that can see a fault sees it first:

- *file* — schema, manifest.
- *row* — penalty, reference-row, scorer.
- *input* — input-digest recomputes the content digest of μ and Σ from the derived statistics and
  compares it with the one on the reference row and on every run row; reference-input re-enumerates
  the feasible set from those statistics alone, as a clone would.
- *artifact* — row-npz, npz-internal.
- *row*, again — identities recomputes every column that is a function of other recorded values.
- *statistical* — state-batch takes the smaller exact one-sided binomial tail, flagging below 10⁻⁹
  over the 5,130 counts this record carries, and per (N, penalty) block of 45 to 150 runs requires the feasible fraction to
  correlate with final P_F above 0.9 and above its correlation with initial P_F. These are
  thresholds chosen on the clean record, not hypothesis tests.
- *block* — seed-distinct.
- *recompute* — state-recompute re-simulates the reference instance's A_heur seed-42 run and four
  spaced runs per N, at their recorded final angles, on `lightning.qubit` in double precision, and
  compares the feasible probabilities with the stored ones at 10⁻⁵. E1 ran in single precision on a
  GPU, so agreement to about 10⁻⁶ is expected and a swapped state is off by O(1).

**Faults.** Ten, each applied to an in-memory copy of the record (Table 1). Four reproduce defects
that occurred in this project; six are mix-ups a user of the code base can make. The harness's first
pass is the clean record, on which every check must return nothing.

**Table 1. Injected faults. Each changes only the records its row names, leaving the rest of the copy
as committed; "the row" is the run row and "the artifact" its `.npz`. Section 5.1 says where that
partial reach matters.**

| Fault | What is changed | Origin |
|---|---|---|
| price_perturbation | one asset's mean return in the derived statistics × (1 + 10⁻⁶) | the provider's 4.2·10⁻⁷ revision, 2026-08 |
| window_mixin | one run of the prior-window replication (E5) pasted over the same cell of the main file, run id included | plausible merge error |
| optimizer_mixin | one run of the E3 L-BFGS-B arm pasted the same way | plausible merge error |
| label_swap | the A_margin and A_heur rows of one (instance, seed) relabelled | setting mix-up |
| angle_swap | initial and final angles exchanged in the row and in the artifact | handoff error |
| batch_from_initial | every row's shot summary — feasible, within-τ and optimum counts — redrawn as one multinomial over the *initial* state, keeping the counts nested; the artifact's shot arrays left as recorded | the E3 L-BFGS-B wrapper defect [1, Sec. VI-C] |
| evaluator_from_Q | f of the best feasible shot replaced by its penalised energy, g_F recomputed | an evaluator fed the QUBO matrix |
| column_drop | the `final_P_F` column removed | the E4 `write_csv` defect |
| shot_seed_noop | seed 43's shot batch replaced by seed 42's in three instances, summary columns copied | the `shot_seed` defect [1, Sec. VI-C] |
| reference_stale | the reference f\* of one instance × (1 + 10⁻⁹) | a stale regeneration |

**Controls and measures.** Three configurations. The *naive* pipeline reads the CSV and computes
the study's headline numbers with no check but the schema. The *declarative* control is the
vocabulary of a data-quality library, applied generously and in eight families:

- presence and completeness of every required column;
- the declared type of each column;
- the set of admitted values for a categorical column;
- uniqueness of the run id, of the (N, instance, penalty, seed) key, and of the shot summary within
  one cell;
- the expected number of rows, and the columns that must hold one value across the file (the window,
  the specification version, the optimizer, the backend, the precision, the shot count);
- numeric ranges: probabilities between 0 and 1, counts non-negative, a probability sum of 1 ± 10⁻⁴;
- relations between two columns of one row: feasible shots ≤ shots, optimum shots ≤ within-τ shots ≤
  feasible shots, P_opt ≤ P_τ ≤ P_F, rank ≤ |F|, f\* ≤ f of the best feasible shot, the penalty equal
  to the column its own label names, and the feasible-shot fraction within five binomial standard
  deviations of final P_F;
- and, after a join with the reference table — which such a library can do, though Deequ's
  single-table vocabulary cannot — equality of the instance quantities copied onto each run row with
  the reference row they came from.

**What the declarative control is, exactly.** It is our re-implementation of that vocabulary over
the run table, not a run of Deequ or pandera: the record is a CSV, a reference CSV and 2,565 `.npz`
files, and putting a JVM or a schema object in front of the CSV would change the dependency list and
not the comparison. The vocabulary was taken from Deequ's constraint set [16] and from the built-in checks of pandera
0.33.1 [17], the current release in September 2026 (`equal_to`, `not_equal_to`, the four order
comparisons, `in_range`, `isin`/`notin`, `str_*`, `is_monotonic`, `unique_values_eq`, with
nullability, dtype, uniqueness and strictness declared on the schema). The reading is deliberately
generous: every
constraint either tool can state over these columns is granted, including pandera's whole-frame
checks and a join with the reference table, which Deequ's single-table vocabulary does not offer. Two
things are *not* granted, and they are the boundary the comparison measures: an expectation on an
aggregate, because in this record the aggregates are the results being checked; and a custom Python
function that opens another file — the artifact, the pinned statistics — because that function is the
verification relation itself, and granting it would make the control a re-implementation of the
treatment. What the control therefore measures is not what pandera can execute, which is anything,
but how far an expectation stated in the declarative vocabulary over the run table reaches into this
record. The *relations* configuration runs all thirteen. For each fault
we record which checks fail, the first failing check and its stage, and how many of thirty headline
numbers moved (per N and per penalty setting: the median g_off, median g_F, mean feasible fraction,
median final P_F, and the number of runs with an optimum hit) — that is, whether the fault would have
changed a conclusion had nothing flagged it.

**Random mutations and column coverage.** The ten faults are ours, which leaves the
objection that the checks were written for them. Two further measurements use the same record and
the same checks. *Column coverage*: for every column of the run row, the reference row and the
artifact (133 in all; the one that is empty on every row is reported as not applicable, and a
mutation that lands on it in the random draw below is counted as undetected), one random cell is
perturbed — a value by 10⁻³ relative, a count by ±1, a flag
by negation, a string by a different value — and a column counts as *covered* if some check fails.
*Random mutations*: 300 mutations are drawn with a seeded generator from ten operators (a cell
perturbed by δ ∈ {10⁻⁹, 10⁻⁶, 10⁻³, 10⁻¹}; a cell replaced by another row's; a row's metrics copied
onto another row; a row deleted; the penalty labels of a block permuted; a reference cell perturbed;
an artifact element perturbed; two artifacts exchanged; one entry of the derived statistics
perturbed; a column dropped), each applied to a copy and run through the cheap checks. Both were run
three times, because the first pass's uncovered list told us which columns a further identity could
bind; the passes are reported in that order. Because the checks were developed against that draw, a
fourth run of the mutation harness was made after they were frozen: 300 further mutations from the
same operators with a different seed, run once, with nothing changed in the checks afterwards.

### 4.2 E-S2 — Does a misconfigured run stop before it is billed? (P2)

**Stand-in device.** `default.qubit` subclassed to declare that it supports no derivative method.
PennyLane's `best` differentiation method then resolves to parameter shift exactly as it does for the
Braket QPU device, and `qml.Tracker` counts the tapes the committed solver's own QNode sends. The
count is cross-checked on `braket.local.qubit`, the Braket plugin's own local simulator, which also
lacks adjoint support, and against the parameter-shift transform's tape count, which is what the
SV1 ledger recorded for the sampling call (177 at N = 8). Three independent sources — the ledger
(service), the plugin's device (plugin), the stand-in (framework) — agree.

**Misconfigurations.** Five (Table 2). Four are structural errors; the fifth is a correct call with an
expensive default, included so that the table shows what the structural guards do not see.

**Table 2. Misconfigurations a user of the code base can write.**

| Misconfiguration | What is written | Consequence if nothing stops it |
|---|---|---|
| QPU as optimizer backend | `--backend braket_ionq` | 1 + 2p·[C(N,2) + 2N] tasks per optimizer step for 50 steps, plus the sampling call's own |
| Trainable angles at sampling, SV1 | the optimizer's tensors handed to the 1,000-shot QNode | 1 + 2p·[C(N,2) + 2N] tasks for one sampling call that returns one shot batch (the 2026-08-09 pilot) |
| Trainable angles at sampling, IonQ | the same, on hardware at 100 shots | the same count at $8.30 per task |
| QPU as sampling backend, default shots | `--sample-backend braket_ionq` without `--shots` | one correct task at $0.30 + 1,000 × $0.08 |
| Untagged cloud run | `--backend braket_sv1` without `--results-tag` | the committed baseline behind the paper's tables is overwritten |

**Configurations.** *none* (every guard of the code base disabled by monkey-patching: the QPU set
emptied, the cloud backends declared local, the canonical-file refusal disabled, `frozen_angles`
made the identity); *cap* (no structural guard; a $50 per-QPU spending limit, the project's budget
figure, modelled as documented — tasks accepted while the cumulative list-price estimate stays within
the cap, the first task that would exceed it rejected; simulators outside its scope); *structural*
(the committed guards, with the device factory returning the stand-in for every cloud backend and
the AWS session factory raising if called); *both*.

**Observed on the service.** Two of the harness's inputs were then checked against the service
itself ($0.67 in total, `es2_observe_cloud.py`): a $1.00 spending limit was created on IonQ Forte-1
and requested for SV1, and the N = 8, p = 2 sampling QNode was sent to SV1 twice, with trainable and
with frozen angles, the tasks the service created in each window counted through
`SearchQuantumTasks`. The 100-shot probe against the IonQ cap was submitted while the device was
offline, and a seven-day poll found no window in which to repeat it (Section 5.2). The third kind of control was then
exercised: AWS's cost-control solution [31] was deployed as documented (one region, a $0.30 monthly
limit, the denial targeted at the project's IAM user; Lambda images built for x86_64 instead of
arm64, which changes nothing in behaviour), and the cheapest SV1 task — a Bell circuit at 10 shots,
billed at the 3 s minimum, $0.00375 — was submitted sequentially, each after the previous had
completed, as a variational loop would, until the service denied `CreateQuantumTask` or 400 tasks
had been created (`es2_observe_cost_control.py`). The measures are the tasks created after the
list-price running total crossed the limit, after the alarm fired, and after the first denial.

**Measures.** The stage at which the run stops — `cli`, `constructor`, `pre-submit` (the
misconfiguration is removed before the call reaches the device), `service` (the cap rejects a task),
`submitted` (every task executes) — the tasks that would reach the service, the tasks billed and the
list-price bill (IonQ: $0.30 per task + $0.08 per shot; SV1: $0.075 per minute with a 3 s minimum
per task, and a sampling task ran 0.11 s), whether a usable result exists, and whether the baseline
file survived. The structural cells run the committed code; the `none` cells run the same code with
the guards removed, so a guard that stopped firing would move a cell.

### 4.3 E-S3 — What does a reader need, and what does it cost? (P3)

**Protocol.** A clean-room harness plays the reader: a script that has only what the package
publishes, on a machine with none of the working tree. It clones the published repository over the
network at its release tag (`v1.0`, commit `c804a63`, 21 MB of files), creates a fresh virtual
environment, installs
`requirements-repro.txt`, runs the package's test suite, then runs the README's three levels as
written: level 1 (`generate_tqe_figures`, then `check_tables.py`), level 2 (`run_experiment.py` on
N ∈ {8, 12}, CPU), level 3 as a 24-run block of the cross (N = 12, instances 0–1, seeds 42–43, six
penalty settings, CPU, double precision). Every step records its command, wall-clock, exit code,
outcome and the inputs the reader had to supply. Level-1 figures are compared byte for byte with
the committed renders; level-2 rows with the committed file; level-3 rows column by column with the
committed GPU rows, separating reference quantities, state quantities, shot-derived columns and
angles.

**Interruption.** The same block is run with `--resume`, killed with `SIGKILL` after seven runs have
printed, and re-issued. The rows on disk at the kill are the work the per-run flush saved; the
resumed file is compared row for row with the uninterrupted block (both CPU double precision on the
same host, so identity is the expectation); the control without flush and resume is what a restart
would have redone.

**Controls for missing material.** The level-2 command in a copy of the clone with `data/derived`
removed and the network unreachable (an unreachable proxy), which is a package without the derived
statistics in the hands of a reader without the price snapshot; and the E2 aggregation without the
per-run artifacts, which are deposited separately (Zenodo, ≈100 MB).

**Other machines.** Level 1 was then run by hand on three further environments — Windows (Git Bash,
Python 3.12), macOS (3.12) and Google Colab (3.13.15, three times) — from the same tag and the same
written instructions, installing `requirements.txt` (the minimum versions) rather than the pinned
`requirements-repro.txt`, which is CUDA-Linux only. These runs test whether the instructions carry to
a machine that is not the one they were written on. They do not test independence: **the author ran
all of them** (Section 7).

### 4.4 Threats to validity

*Construct.* E-S2's dollar figures are list prices applied to measured circuit counts; nobody paid
them, except the $0.67 of the service observation and the $4.02 of the six account-level runs. The counts for the optimizer loop are measured on a stand-in that shares the framework's
execution path, not on hardware; the count for the sampling fallback was re-observed on SV1 for this
paper (177 and 1 tasks), because the local devices execute that call once whatever the angles' flag
(the harness prints this). The *structural* column of Table 7 is, by construction, the guards of Section 3.2 refusing the
settings they were written to refuse: taken alone it is a unit test, and it is reported because the
comparison across the four configurations — which stage each reaches, what each costs, and what each
does not see — is what the table is for. *Internal.* The ten faults of E-S1 are our own choice; four are
replays of real defects, six are our judgement of plausible mistakes, and a fault designed to preserve
every relation would pass. The false-positive side is measured on five committed records rather than
on the one the faults enter (Section 5.1), which is a statement about these records and this code
base, not a false-positive rate for the relations in general. The random mutations answer the choice of faults but not the choice of
operators, which are also ours, and their detection rate depends on tolerances we set. The spending-limit model is the documented behaviour; its simulator half was observed, its QPU
rejection not yet (Section 5.2).
*External.* One code base, one provider, one application; the principles are stated generally but
are evidenced here. E-S3's reader is a clean-room script on the author's own host with a warm package
cache (`pip` took 150 s; a cold cache downloads several gigabytes of CUDA libraries); it measures the
package, not a stranger's understanding of the README. Level 1 was also run by hand on Windows, macOS
and Colab (Section 5.3), which extends the evidence across machines but not across people: every run
in this paper is the author's own.

## 5. Results

Three evaluations, three results. The record's checks caught ten of ten injected faults where a
schema-only pipeline caught one and a data-quality vocabulary over the table caught eight, and fired
on none of five committed clean records. The structural guards stopped the two ruinous
misconfigurations at the command line, where a dollar cap stops them after six tasks and does not
see the simulator fallback at all. The package rebuilt at three levels in 24 s, 18 s and 222 s,
losing one run in flight to an interruption. Each subsection reports one of these against its
control.

### 5.1 E-S1: fault detection and coverage

The clean record passes all thirteen checks. Table 3 gives the matrix.

**Table 3. Detection matrix (N ∈ {12, 16, 20}; 2,565 runs). "First" is the first failing check in
stage order, "moved" how many of the thirty headline numbers changed. Naive and Declarative are the
controls of Section 4.1; the declarative entry names the families that fire (single = one value per
file, fk = the join with the reference table, pair = an inequality within a row).**

| Fault | First failing check (stage) | Also failing | Naive | Declarative | Moved |
|---|---|---|---|---|---:|
| clean | — | — | — | — | 0 |
| price_perturbation | input-digest (input) | reference-input, identities | no | no | 0 |
| window_mixin | manifest (file) | reference-row, input-digest, identities, state-recompute | no | single;fk | 2 |
| optimizer_mixin | manifest (file) | — | no | single | 3 |
| label_swap | penalty (row) | — | no | conditional | 4 |
| angle_swap | npz-internal (artifact) | state-recompute | no | no | 0 |
| batch_from_initial | row-npz (artifact) | identities, state-batch | no | band | 12 |
| evaluator_from_Q | scorer (row) | row-npz, identities | no | range;pair | 6 |
| column_drop | schema (file) | row-npz, state-batch | **yes** | schema | 6 |
| shot_seed_noop | state-batch (statistical) | seed-distinct | no | unique;band | 2 |
| reference_stale | reference-row (row) | reference-input | no | fk | 0 |

Three observations.

*Detection is at the cheapest stage that can see the fault.* Six faults stop at the file or row
stage, before any artifact is opened; one (price_perturbation) stops at the input stage, where the
content digest is recomputed from the derived statistics in 1.2 s for all 90 instances and 2,565
rows; the two that require the artifacts (angle_swap, batch_from_initial) stop at the artifact stage;
one (shot_seed_noop) reaches the statistical stage, where it fails both the batch's binomial test and
the diagnostic that two seeds should carry two batches. The harness records what each check costs on
each record (`paper02_es1_finding_counts.csv`), and the spread is three orders of magnitude: on the
clean record the schema takes 0.009 s, the whole set 46.6 s, and most of that is the three checks
that derive rather than compare — reference-input re-enumerates every feasible set from the pinned
statistics (10.5 s), identities recomputes every derived column (10.3 s), and state-recompute
re-simulates the sampled states (13.1 s). The two that re-derive from outside the record —
reference-input and state-recompute — are redundant for every fault: each confirmed a fault a
cheaper check had already flagged.

*Seven of ten faults move headline numbers silently.* The largest is the replay of the E3 L-BFGS-B
defect: drawing every batch from the initial state moves twelve of thirty headline numbers (the
feasible-fraction mean and the optimum-hit count of every size and penalty setting), and the naive
pipeline reports them as the study's results. label_swap moves four — the A_heur-versus-A_margin contrast that is the numerical study's
main finding — and passes every check a schema-only pipeline has. Only column_drop is visible to the
naive pipeline, because a missing column is the one fault that a CSV reader cannot ignore.

*The digest gap, found and closed.* In the first pass of this harness the record carried only the
instance digest, a hash of the ticker set, and a 10⁻⁶ relative change to one mean return in the
derived statistics passed the schema, manifest, penalty, reference-row, scorer, artifact and
statistical checks — every relation that reads the record — and was caught only by re-enumerating the
reference from the statistics (f\* of instance 7 at N = 12 moves from −0.24331838 to −0.24331874;
Δ_F, A_crit and A_margin move on the instances that hold the asset). The digest bound the record to
*which* inputs, not to their values, which is less than P1 as stated in Section 3.1 requires. A
content digest of μ and Σ (`input_digest`) was then added to the reference row and to every run row —
appended to the committed files from the pinned statistics, without re-running anything, and written
by the driver for every new run — together with the input-digest check. In every pass after the fix (Table 3
reports the last), the same fault fails on 1,009 records at the input stage in 1.2 s, and
reference-input, which enumerates every feasible set again from the statistics, confirms it in
9.7 s. This was the first of
three gaps.

*A false positive, removed.* A first version of the state-batch check used a five-sigma band around
S × P_opt with a floor of one shot. On the clean N = 20 block it flagged one run whose batch held two
optimum shots against an expectation of 0.1 — a 0.5 % event, not a five-sigma one, and with 2,565
runs an expected occurrence. The check was replaced by the exact binomial tail with a threshold of
10⁻⁹ over the record's 5,130 counts; it still reports 2,299 findings under batch_from_initial and three
under shot_seed_noop.

*Where an expectation stated over the table stops.* The declarative control of Section 4.1 — the
vocabulary of a data-quality library over the run table, applied generously — catches eight of the
ten faults where the schema-only pipeline catches one. It sees both pasted rows, because each makes a
column take two values where the file must hold one: the prior-window row does it to `data_start`
and `data_end`, the L-BFGS-B row to `optimizer` and `steps`. It sees the stale reference, after a
join with the reference table, and the dropped column. It sees the evaluator fed the QUBO matrix,
where f of the best feasible shot falls below the instance's optimum on 2,463 rows and g_F turns
negative. It sees the relabelled penalty, as a conditional equality: the value a label names is a
column of the same row. It sees the re-used shot batch, as two runs of one cell carrying the same
shot summary. And it sees the batch drawn from the initial state, by a tolerance band computed from
the row itself: the feasible-shot fraction must sit within five binomial standard deviations of
final P_F, which holds on every clean row and fails on 2,319 of the faulted ones.

Two faults it does not see, and the reason is not the vocabulary but where the value lives. The
revised input statistic is caught by recomputing a content digest of the pinned statistics, which
are a different file. The exchanged angles are caught inside the artifact: the fault exchanges the
two angle vectors in the row *and* in the `.npz`, so the row and its artifact still agree, and what
breaks is the artifact's own relation that the stored trajectory begins at the recorded initial
angles (`npz-internal`), confirmed by re-simulating the state at the recorded final angles
(`state-recompute`). Neither value is in the table, so a predicate over the table's columns cannot
reach either, however expressive it is. That is the increment the relations buy over the setting
evaluated here, and it is narrow: everything else the control reaches is a relation we would now
recommend declaring that way rather than writing by hand. As Section 4.1 sets the comparison up, the
eight-against-ten says where a *declared* expectation over the run table reaches in this record — not
that either library could not run these two checks, which a custom Python function in it would, and
which would be our relation wearing the library's decorator.

*A replay has to be faithful in the right respect.* batch_from_initial draws the three nested
counts — optimum ⊆ within τ ⊆ feasible — from one multinomial over the initial state, because the
defect drew one batch from one wrong state and its counts were therefore mutually consistent. Drawn
independently instead, they break that ordering, and a pipeline catches the replay on an
inconsistency the defect never produced — a control that fires on the injection rather than on the
fault it stands for measures the injection. The faithful replay is still within the declarative
control's reach, by the tolerance band above, but now for the reason the defect gives.

*How far the replay reaches, and what Table 3 therefore says for this fault.* The injection rewrites
the row's shot summary and leaves the artifact's shot arrays as recorded, and this is a limit of the
record rather than a choice: the `.npz` stores the probabilities of the feasible set, not the full
state, so a batch over all 2^N outcomes cannot be redrawn from the initial state out of the record
alone. The defect itself wrote a batch its artifact agreed with. The consequence is visible in
Table 3: `row-npz` and `identities` fail on the mismatch the injection creates between the row and
the artifact, and `row-npz` is therefore the first failing check for a reason the defect never gave
anyone. The check that sees the defect as it occurred is `state-batch`, which compares the batch with
the state the same run recorded and reports 2,299 findings — 2,252 of the 2,565 rows, twenty-eight of
them on both their feasible and their optimum count, and all nineteen (N, penalty) blocks, whose
feasible fraction tracks the initial state rather than the final one; it is the check that caught the
defect in the numerical study (Table 12, defect 7) and the one that stays when the artifact is made
to agree. For this row of Table 3 the *Also failing* column, not the first one, carries the claim.

*What kind of rule catches each fault.* The faults divide in the sense of Section 3.1. Six are
caught by an exact relation, where the violation is a plain inconsistency between two recorded things
and no tolerance is involved: both pasted rows against the manifest, the stale reference against the
run rows, the revised statistic against the digest of the pinned file, the exchanged angles against
the artifact's own trajectory, and the missing column against the schema. Two are caught by a
numerical relation, where every record agrees and the value is nevertheless wrong, so only a
recomputation finds it: the relabelled penalty and the evaluator fed the QUBO matrix. Two are caught
by a statistical rule: the copied shot batch, by the binomial tail and by the diagnostic that two
seeds should carry two batches, and the batch drawn from the initial state, for which no pair of
records disagrees at all once the replay is faithful. That is the ordering of difficulty, and it is
why the record stores the state and not only the summary. It also fixes what those last two findings
are — a suspicion a reader then confirms, which is how the `shot_seed` defect was actually settled,
by comparing two sweeps that should have differed and did not.

*Five clean records, not one.* Zero false positives above is a statement about the one record the
faults were injected into. The same thirteen checks were then run, with nothing injected, on the
four other records the same driver produced and committed: the prior-window replication (300 runs
whose market window, pinned statistics, references and digests all differ), the two E3 optimizer
arms (300 runs each, ADAM 0.03 × 100 and L-BFGS-B), and the pilot block (36 runs, one seed). Over
3,501 runs, 186 reference rows and 3,501 artifacts, across two market windows and three optimizer
settings, **no relation reports a finding**. What that does and does not establish: the tolerances
and the statistical threshold were set on the first of these records, and at a binomial threshold of
10⁻⁹ over a few thousand counts per record a rate of zero is what the threshold was chosen to produce. The
other four records add that relations calibrated on one record do not fire on four they were not
calibrated on, one of them from a different market window with its own statistics, references and
digests. Two things had to change for that measurement to mean
anything, and both were in the checker rather than in the records: the manifest relation compared
every block against E1's own six-setting penalty grid, where an arm may run two of the six, and now
reads the grid the arm's own manifest records; and one identity read a column the pilot, written
before that column existed, does not carry, where absence is the schema relation's business. Neither
changes a cell of Table 3.

**Column coverage and random mutations.** Before the evaluation the relations bound 47 of the
record's 132 value-carrying columns (of 133 columns, one is empty on every row); the checks written
in response to the first coverage pass (the identity check of Section 3.1, in two rounds) raised that
to 103 and then 114 (Table 4). The last column of that table counts the columns that *only* the
identity check binds, which is smaller than the increase: four columns (`p_layers`, `steps`,
`stepsize`, `status`) were bound in the same pass by adding them to the manifest relation's list of
columns that must hold one value across the file, so 47 + 4 + 52 = 103 and 47 + 4 + 63 = 114. The 18 columns that
remain uncovered are of three kinds. Seven only a re-simulation can bind: the full-state expectations
of f and of the constraint violation at the initial and at the final angles, the expectation at the
final angles, and the interior points of the trajectory and of the convergence curve — the artifact
stores the probabilities on the feasible set, not the full state. Eight record *how long* or *how
many* rather than *what*: five timings on the run row, the reference row's build time, and the two
evaluation counts. Three could be bound and are not: the index of the best shot including infeasible
ones, the objective of every feasible solution (which Section 5.4 re-enumerates from the pinned
statistics and finds equal to 2·10⁻¹⁶, though this harness does not check it), and a best-expectation
column that is empty on all but ten rows. Of the 300 random mutations, 236 (79 %) were detected; of the 26 that moved a
headline number — the mutations that would have changed something the paper reports — **25 were
caught**, and the one that was not moved a feasible-shot fraction by 10⁻⁹ relative, which moved a
mean by 10⁻¹² (Table 5). Those 26 are not evenly spread — 17 are permuted penalty
labels, 5 deleted rows and 2 copied metrics — so the figure is mostly about three operators that move
a block of the record at once.

The declarative control detects 81 of the 300, and **24 of the 26 that move a headline** against the
relations' 25. On this mutation set it is barely worse, and Table 5 shows why: what moves a headline
here is a block of the table changing at once, which an expectation over the table sees. The two separate
elsewhere. Of the 52 perturbed artifacts, the 15 exchanged artifacts and the 24 perturbed inputs —
91 mutations that touch a file the table does not contain — the control detects none and the
relations detect 71. None of those 91 moved a headline number in this draw, which is a fact about
the operators rather than a defence of the control: the two defects of this project that did move
reported numbers, a batch drawn from the wrong state and a revised price snapshot, are of exactly
those kinds. Detection depends on magnitude as the tolerances say it should: 100 % for a count moved
by one, 79–81 % at 10⁻⁶ to 10⁻¹, 47 % at 10⁻⁹, which is the relative tolerance itself. Of the 64
missed, 38 touch the unbindable columns above, 19 are 10⁻⁹ perturbations of bound columns, and seven
sit inside a stated tolerance (a probability sum of 1 ± 10⁻⁴ replaced by another row's; an interior
value of the stored f that is neither an extremum nor the selected solution). Every operator that
changes *which* record a value belongs to — a swapped artifact, a copied row, a deleted row, a
permuted label, a perturbed input — was detected in 100 % of its draws.

**What these rates are a rate of, and a held-out draw.** The 300 mutations were drawn once and then
used three times, and between passes the relations changed: the uncovered list of the first pass is
what told us which columns a further identity could bind. The 79 % is therefore the rate of the final
relations on the mutation set they were developed against, not an estimate of their rate on faults
nobody has seen. So, with the relations frozen and nothing changed in them afterwards, the same
harness drew 300 more from the same operators with a different seed and ran them once: **245 (82 %)
detected**, and of the 26 that moved a headline number, **26 were caught** (Table 5, last row;
`paper02_es1_random_faults_heldout.csv`). The control detects 86 and 25 of the 26. The 77 mutations
that touch a file the run table does not contain — a perturbed artifact, an exchanged artifact, a
perturbed input — are caught 64 times by the relations and never by the control, the same separation
the ten faults show. Taking the two draws together: of 600 mutations, 52 move a headline number and
the relations catch 51 against the control's 49, which is not a difference worth a claim; of the 168
that reach outside the table, the relations catch 135 and the control none, which is. What this does not remove is that the mutation operators are ours, so both rates
are conditioned on a distribution of faults we chose, and that the tolerances were set on this
record.

**Table 4. Column coverage of the relations over the 133 columns of the run row, the reference row
and the artifact; the one column empty on every row is excluded. Each pass is a committed scan of the
whole record. The last column counts the columns no other relation binds.**

| Pass | Checks | Covered | Covered by `identities` alone |
|---|---|---:|---:|
| 1 | 12 relations without the identity check | 47 / 132 | 0 |
| 2 | + identities (Q_τ, P_τ/P_opt, rank, D and the offset magnitude, best expectation, uniform baselines, Sharpe constancy) | 103 / 132 | 52 |
| 3 | + shot-level successes, fractions, distinct shots, probability sums, portfolio metrics from the pinned statistics, reference row against its own cardinality minima and stored f | 114 / 132 | 63 |

**Table 5. 300 random mutations by operator, final pass. "Declarative" is the control of Section 4.1
on the same mutations; "Moved" counts the mutations that changed a headline number and "Missed"
those of them the relations did not catch. The last row is the held-out draw.**

| Operator | n | Detected | Declarative | Moved | Missed |
|---|---:|---:|---:|---:|---:|
| cell perturbed (δ ∈ {10⁻⁹, 10⁻⁶, 10⁻³, 10⁻¹}) | 80 | 52 | 16 | 1 | 1 |
| cell replaced by another row's | 30 | 20 | 2 | 0 | 0 |
| row's metrics copied onto another row | 13 | 13 | 13 | 2 | 0 |
| row deleted | 15 | 15 | 15 | 5 | 0 |
| penalty labels of a block permuted | 18 | 18 | 18 | 17 | 0 |
| reference cell perturbed | 26 | 22 | 5 | 0 | 0 |
| artifact element perturbed | 52 | 32 | 0 | 0 | 0 |
| two artifacts exchanged | 15 | 15 | 0 | 0 | 0 |
| derived statistic perturbed | 24 | 24 | 0 | 0 | 0 |
| column dropped | 27 | 25 | 12 | 1 | 0 |
| **all** | **300** | **236 (79 %)** | **81** | **26** | **1** |
| **all, held-out draw (seed 7, relations frozen)** | **300** | **245 (82 %)** | **86** | **26** | **0** |

### 5.2 E-S2: execution guards and cost control

**Measured counts.** One optimizer step of the committed solver on the stand-in device executes
177 / 361 / 609 / 921 tapes at N = 8 / 12 / 16 / 20 (Table 6) — 1 + 2p·[C(N,2) + 2N] in every case,
177 also on `braket.local.qubit` at N = 8, and 177 is the sampling-call count of the SV1 ledger's
first run. The committed sampling path hands the device one tape with the angles frozen. **One step
is measured and the loop is multiplied out**: the task count of the unguarded optimizer loop in
Table 7 is that measured per-step count times the driver's fifty steps, plus the sampling call
(50 × 177 + 177 = 9,027 at N = 8; 50 × 921 + 921 = 46,971 at N = 20). Running fifty steps on the
stand-in would measure the same arithmetic at fifty times the cost, and the stand-in is not the
service in any case — what is measured is the framework's decision, not a bill anyone received.

**Table 6. Tapes per optimizer step on a device without a derivative method.**

| N | stand-in (`default.qubit`, no derivatives) | `braket.local.qubit` | transform | formula |
|---:|---:|---:|---:|---:|
| 8 | 177 | 177 | 177 | 177 |
| 12 | 361 | — | 361 | 361 |
| 16 | 609 | — | 609 | 609 |
| 20 | 921 | — | 921 | 921 |

**Stopping stages.** Figure 2 shows the N = 8 cells; Table 7 gives N = 8 and N = 20, and the CSV
holds N = 12 and 16, which lie between.

![Fig. 2. Where each misconfiguration stops under the four guard configurations (N = 8, list prices). Nothing is carried by colour: the marker's shape is the guard configuration and its fill the outcome — open for nothing wasted, grey for a correct call with a costly default, solid for wasted spend or the baseline file lost. Nothing stops at the constructor: the command line refuses first, and the constructor's own refusal (Section 3.1) is a second line that nothing here reaches. The marks are Table 7's: ‡ observed on the service, § a task count obtained by counting circuits, † the provider's documented behaviour, unmarked what the harness ran and counted.](figures/fig_guard_stages.pdf)

**Table 7. Stage at which each misconfiguration stops, tasks billed and list-price bill, under four
guard configurations (cap = $50 per QPU). Where a cell comes from: ‡, observed on Amazon Braket;
unmarked, the harness ran the committed code and counted what it did — the stage it stopped at, or
the tapes a device received; §, a task count obtained by counting *circuits* rather than tasks (the
parameter-shift transform's tapes for one sampling call, or one step's tapes times the driver's fifty
steps plus that call), which a metered device turns into one task each — an identity confirmed at
N = 8 on SV1 and against the 2026-08-09 ledger, and assumed at N = 20 and on hardware; †, the
provider's documented behaviour, not observed. Where a cell ends in "—" nothing usable comes back,
and where it ends in "?" a value does and a gradient does not. Every dollar figure is a list price
applied to a task count, not a bill. The optimizer loop is fifty steps; the sampling call is
1,000 shots on SV1 and 100 on IonQ.**

| Misconfiguration | N | none | cap only | structural | both |
|---|---:|---|---|---|---|
| QPU as optimizer backend | 8 | § submitted; 9,027 tasks; **$74,924** | † service; 6 tasks; $49.80; — | **cli**; 0 tasks; $0 | cli; 0; $0 |
| | 20 | § submitted; 46,971 tasks; **$389,859** | † service; 6; $49.80; — | cli; 0; $0 | cli; 0; $0 |
| Trainable angles at sampling, SV1 | 8 | ‡ submitted; 177 tasks; $0.66 | ‡ submitted; 177; $0.66 (**cap blind**) | ‡ **pre-submit**; 1 task | ‡ pre-submit; 1 |
| | 20 | § submitted; 921; $3.45 | § submitted; 921; $3.45 | pre-submit; 1 | pre-submit; 1 |
| Trainable angles at sampling, IonQ | 8 | § submitted; 177; $1,469 | † service; 6; $49.80; ? | pre-submit; 1; $8.30 | pre-submit; 1; $8.30 |
| | 20 | § submitted; 921; $7,644 | † service; 6; $49.80; ? | pre-submit; 1; $8.30 | pre-submit; 1; $8.30 |
| QPU as sampling backend, default shots | 8 | submitted; 1 task; $80.30 | † **service; 0; $0** | submitted; 1; **$80.30** | † service; 0; $0 |
| Untagged run over the baseline | 8 | submitted; **baseline overwritten** | overwritten | **cli**; baseline intact | cli; intact |

The division of labour is the result. The structural guards stop the two runs that would have been
ruinous at the command line and before the call reaches a device, and they do so at every N, because
they act on the configuration and not on a running total. The counterfactual bill of the unguarded
optimizer loop is $74,924 at N = 8 (the numerical study's "about $73,000" excluded the sampling
call's own 177 tasks) and $389,859 at N = 20.

The cap stops the same runs after six tasks of $8.30 with nothing to show — the first gradient of
the first step needs 177 — which at $49.80 is a cheap failure but a failure; and the cap is blind to
the simulator fallback that actually occurred in this project (177 to 921 tasks, $0.66 to $3.45, and
4.7× the wall-clock of the run). Conversely, the one correctly built call with an expensive default —
one IonQ sampling task at 1,000 shots — passes every structural guard, because there is nothing
structurally wrong with it, and only the cap refuses it — because $80.30 exceeds $50, not because
the cap can see that a thousand shots was a slip. The untagged run costs $0.01 and destroys
the file the paper reads; no cap sees it.

**What the service confirmed.** The sampling fallback is no longer an inference from a month-old
ledger: on 16 September 2026, under PennyLane 0.45.1 and amazon-braket-pennylane-plugin 1.35.1, the
N = 8 sampling QNode sent to SV1 with the optimizer's angles still marked trainable created **177
tasks in 672 s**, and the same QNode with frozen angles created **1 task in 4 s**; both returned a
full 1,000 × 8 sample. The two N = 8 SV1 cells of that row therefore rest on a service observation;
their N = 20 counterparts and the IonQ row are the transform's circuit counts, which Table 7 marks as
such. The
transform's count and the 2026-08-09 ledger agree with it. Of the cap, one half is observed and one
half still modelled: the request for a spending limit on SV1 was refused by the service ("Spending
limits can not be created for simulator devices"), which is the row "cap blind" of Table 7; the
rejection of an over-cap QPU task at creation was attempted with a $1.00 limit on IonQ Forte-1 and a
100-shot ($8.30) circuit, and the service answered `DeviceOfflineException` because the device was
outside its availability — the task was not created and nothing was billed, but the refusal observed
was the device's, not the cap's. The probe was then left to a poller that queried the device every
30 minutes for seven days (2026-09-17 09:45 to 2026-09-24 09:28 UTC, 336 queries,
`paper02_ionq_availability_probe.log`): the device answered `OFFLINE` every time and the probe never
ran. That is itself the point of P3's device-availability record — a benchmark that needs this QPU
is not runnable on demand, and the platform writes that fact rather than inferring it — but it leaves
the `service` cells of Table 7 as the provider's documented behaviour, not as an observation of ours.

**The third kind of control, observed twice.** Deployed as documented, the account-level solution
metered nothing. Its task-logger function failed on import in every invocation CloudWatch recorded
for the first run — 392 of 392, against the 400 tasks that run created, and the record does not say
where the other eight events went — because the Braket SDK pulls in a simulator whose numba-compiled
functions try to write a cache beside the source and `/var/task` is read-only in Lambda. The
running-cost alarms stayed `INSUFFICIENT_DATA`, no denial was attached, and 400 tasks ($1.50 at list
price) were created against a $0.30 limit.
The solution's own "task logger errors" alarm did fire, four minutes in — a detective control for the
detective control, which nothing in the documented procedure says to watch. With one environment
variable added (`NUMBA_CACHE_DIR=/tmp/numba`; Appendix A.2) the control behaved as designed, and the
measurement was repeated five times with the running cost reset between runs (Table 8). In every
run the list-price total crossed $0.30 at task 81 and the alarm fired 91–96 s later (one 60 s
evaluation period plus its phase); and the first `AccessDeniedException` came 7–146 s after the
alarm — that interval covers the alarm action attaching the deny policy and IAM's eventually
consistent propagation of it, neither of which the record separates. The tasks created after the
limit was crossed were **41–92 (median 47)**, after the alarm **6–56 (median 10)**, and after the
first denial a further 1–5 slipped through before the denial held. The residue the documentation
calls "a few" is therefore, for a sequential loop at 3–4 s per task, about 47 tasks in the median and
twice that in one run of five — $0.18 to $0.35 on the simulator. Priced at IonQ's $8.30 per task the
same counts would be $390 to $760, which is an illustration of what a residue of that size costs on
hardware and not a prediction of what a QPU loop would overrun by: the residue is a count of tasks a
3–4 s submission loop got in before a denial propagated, and on hardware both the submission rate and
the queue are different. It is the order of magnitude that matters here, and it is what a cap of the
second kind — had it existed for the device — would have stopped at
$50, and the structural guard at $0. Two cells of the first repetition are not in the summary that
run wrote, because the aggregation gained those fields afterwards; `es2_cost_control_aggregate.py`
recomputes them from the run's own task records and checks the recomputation against every field the
four later summaries do hold.

**Table 8. Five repetitions of the account-level control, one column each (limit $0.30; SV1 Bell
circuit, 10 shots, $0.00375 per task, sequential). Every cell is an observation on Amazon Braket;
nothing here is modelled. Two cells of run 1 are recomputed from its task records by
`es2_cost_control_aggregate.py`.**

| Quantity | 1 | 2 | 3 | 4 | 5 | median |
|---|---:|---:|---:|---:|---:|---:|
| Tasks created | 124 | 127 | 172 | 121 | 127 | 127 |
| Created past the limit | 44 | 47 | 92 | 41 | 47 | 47 |
| Limit → alarm | 96 s | 95 s | 94 s | 91 s | 94 s | 94 s |
| Created past the alarm | 6 | 10 | 56 | 6 | 11 | 10 |
| Alarm → first denial | 16 s | 14 s | 146 s | 7 s | 19 s | 16 s |
| Created past the first denial | 2 | 5 | 1 | 4 | 5 | 4 |

### 5.3 E-S3: reconstruction cost and recovery

**Table 9. The clean-room harness's steps: levels 0–3 and the two controls on the public package
`v1.0`, then the same checks and the interruption on `v1.1`, the release this paper ships. A fresh
environment for each, on one host: Ubuntu, one RTX A6000, the GPU host of Table 11. The byte
comparison of the figures and the wall-clock of level 3 are that machine's; Table 11 is what
three machines without a GPU do with the same level 1.**

| Level | Command (from the README) | Wall-clock | Outcome |
|---|---|---:|---|
| 0 | `git clone --branch v1.0` | 1.5 s | commit c804a63, 21 MB |
| 0 | `python -m venv`; `pip install -r requirements-repro.txt` | 8 s; 150 s | PennyLane 0.45.1, NumPy 2.4.6, Matplotlib 3.11.1 |
| 0 | `pytest -q` | 77 s | 364 passed, 4 skipped |
| 1 | `python -m notebooks.generate_tqe_figures` | 22 s | 5/5 regenerated PNG renders byte-identical to the committed ones |
| 1 | `python check_tables.py` | 2.6 s | 2,578 manuscript cells agree, 0 disagree, 0 pending |
| 2 | `run_experiment --backend lightning_cpu --classical-n 8 12 --quantum-n 8 12 --results-tag es3` | 18 s | classical and QAOA rows: energy and Sharpe identical to the committed (GPU) file, at N where the best shot is the optimum on both |
| 3 | `run_e1_cross --backend lightning_cpu --precision double --n 12 --instances 0 1 --seeds 42 43 --results-tag es3block` | 222 s (9.1 s/run) | 24 rows; see Table 10. The committed cross records 13.9 h of run time for 2,565 runs on the GPU |
| 3 | same with `--resume`, `SIGKILL` after 7 runs | 68 s | 6 rows on disk: the run in flight is lost; control without flush: 0 rows, 68 s to redo |
| 3 | re-issue with `--resume` | 168 s | "6 rows already done"; 24 rows; 94/94 non-timing columns identical to the uninterrupted block |
| control | level 2 without `data/derived`, network unreachable | 6 s | fails: `TypeError: '<' not supported between instances of 'int' and 'NoneType'` |
| control | E2 aggregation without the artifacts | 1.2 s | fails: "no run artifacts under …/e1_runs" |
| 0 (v1.1) | `git clone --branch v1.1`; `python -m venv`; `pip install -r requirements-repro.txt` | 0.3 s; 8 s; 146 s | commit a32d427, 24 MB |
| 0 (v1.1) | `pytest -q -rs` | 102 s | 417 passed, 33 skipped: 29 need the per-run artifacts, 3 the price snapshots, 1 a host without `lightning.gpu` |
| 1 (v1.1) | `python -m notebooks.generate_tqe_figures`; `python check_tables.py` | 26 s; 2.6 s | 5/5 byte-identical; 2,578 cells agree, 0 disagree, 0 pending |
| 2 (v1.1) | the level-2 command above | 17 s | energy and Sharpe identical to the committed file, to 0 |
| 3 (v1.1) | the level-3 block with the job status record: `SIGKILL` after 5 runs | 51 s | status file `RUNNING 5/12, last N12_i00_5xAcrit_s42`, matching the 5 rows on disk, which the manifest parses with no artifact left without a row; `analyze_e1` refuses the manifest |
| 3 (v1.1) | re-issue with `--resume` | 70 s | driver prints "previous job: RUNNING pid … 5/12"; status `EXIT 0 12/12`; `analyze_e1` runs |

The inputs the harness supplied were the repository URL and tag, a name for `--results-tag`, and
the choice of CPU over the README's GPU for level 3; every other token came from the README.

**Table 10. Level 3: a CPU double-precision block against the committed GPU single-precision rows
(24 runs).**

| Group | Column | Identical rows | Max abs. difference |
|---|---|---:|---:|
| reference | f\*, Δ_F, A_crit, A_heur, A_margin, Hamiltonian scale | 24/24 each | 0 |
| angles | initial_angles | 24/24 | 0 |
| angles | final_angles | 0/24 | — |
| state | final_expectation | 0/24 | 1.1·10⁻⁴ |
| state | expectation_at_final_angles | 0/24 | 1.4·10⁻⁴ |
| state | final_P_F | 0/24 | 6.2·10⁻⁵ |
| state | final_P_opt, final_P_τ | 0/24 | 1.1·10⁻⁶ |
| state | initial_P_F | 0/24 | 4.5·10⁻⁷ |
| state | final_Q_τ(1000) | 0/24 | 3.3·10⁻⁶ |
| shots | feasible_shots | 0/24 | 31 shots |
| shots | shots_optimal | 10/24 | 5 shots |
| shots | best_all_E, g_F, g_off | 14/24 | 0.19, 0.28, 0.11 |

The block changes two things at once — the host and the precision — so it bounds what a reader
should expect rather than separating the two causes; a CPU run at single precision would separate
them and is not part of the level the README defines. What reproduces bit for bit across hosts is
what P1 binds the record to: the pinned inputs, the reference derived from them, and the seeded
initial angles. What does not is what the numerical
study said would not: fifty ADAM steps in single precision on one host and double on another end at
different angles, the state quantities differ at the 10⁻⁴ level, and the shot columns are a
different draw from a slightly different state. The README states this division in advance; the
table measures it.

*Interruption.* Seven runs had printed their completion line when the process was killed; six rows
were on disk. That measurement is of the released `v1.0`, whose flush rewrote the manifest in place,
so the six rows survived because the kill fell outside the rewrite rather than because it could not
fall inside it; the `v1.1` driver, which lands the file with a rename (Section 3.3), is measured in
the same way in the last block of Table 9, where the manifest parses and holds exactly the rows the
status file claims. What an interruption costs either way is the run in flight and its
artifact: the `.npz` is written before the row, so a kill between the two leaves an artifact with no
row — the resumed run rewrites it, and the row-npz relation would flag it if it did not. The seventh
had completed its computation and not yet its flush, so the run in flight is the unit of loss and the
flush granularity bounds it at one. The resumed run read the six rows, skipped their cells, and
produced a file identical to the uninterrupted block in all 94 non-timing columns; the timing columns
differ, as they must.

*The same checks on the release this paper ships.* E-S3's three levels are an evaluation of `v1.0`,
because that is the package a reader of the numerical study has and the one whose gaps this section
reports. The fixes those gaps produced are in `v1.1`, so the last block of Table 9 clones `v1.1` and
repeats what can be repeated: the suite (417 passed, 33 skipped), level 1 (5/5 figures byte-identical,
2,578 manuscript cells agreeing), level 2 (identical to the committed file, to zero), and the
interruption and resumption — which `v1.0` cannot be asked for at all, since the job status record
was written in response to this evaluation. That last step is also where the record gate of
Section 3.1 is exercised end to end: it calls the aggregation twice, and the aggregation refuses the
partial manifest and then, on the resumed twelve rows, runs the six CSV-level validation routines of
Section 3.1 and finds nothing. The `v1.0` rows of Table 9 clone over the network from
the published repository and the `v1.1` rows clone the same release from a local mirror, because the
tag becomes public on submission; that is why the second clone takes 0.3 s rather than 1.5 s, and it
is the only difference between the two passes. What the 33 skips are: 29 tests of the E-S1 harness need
the per-run artifacts, about 100 MB of `.npz` that the git repository does not carry and the Zenodo
deposit of [1] does, so a reader who clones without fetching the deposit does not exercise the
fault-injection relations against real artifacts; three need the price snapshots, which are
third-party data the package deliberately does not redistribute (it ships the derived statistics
instead); and one is the test that a GPU backend does not silently fall back, which has nothing to
fall back from on a host where `lightning.gpu` is installed and runs. Every claim this paper makes about the final
implementation is therefore a claim about a tagged commit (`a32d427`) that a reader can clone, and the
`CHANGELOG` says which paper was evaluated against which release. What is *not* repeated on `v1.1` is
level 3's full block and its column-by-column agreement (Table 10), which measures a host-and-precision
difference the release does not change.

*The second gap.* The package without its derived statistics and without network access fails, as it
should — but in `v1.0` it fails with a `TypeError` raised inside pandas on the empty frame the
provider returned, which tells the reader nothing about which of the three input sources (snapshot,
derived statistics, provider) is missing. The loader now raises a `RuntimeError` naming all three and
the file to restore; a test pins it. The E2 aggregation without the artifacts fails with a message
that names the directory and, in the README, the deposit.

*Three other machines.* Table 11 gives level 1 from the same tag on Windows, macOS and Colab. What
travels is the manuscript check: the 2,578 cells agree in every one. What is not claimed is the
figures — level 1 regenerates them and the command ran on all three, but only the GPU host's run
compared the files with the committed ones byte for byte, and the three ran at the minimum library
versions rather than the pinned ones, which is enough on its own to change a render. The test suite
differs from the GPU host by a single case: the test that a GPU backend does not silently fall back is skipped where `lightning.gpu` is
installed and runs — and passes — where it is not, so the hosts without a GPU simulator report 365
passed and 3 skipped against the GPU host's 364 and 4. Getting there on Windows needed four things
the README did not say, and two of them are defects of the package rather than of the reader's
machine. (i) The package reads its own text files with the locale encoding; where that is not UTF-8
— a Japanese Windows, where it is cp932 — fifteen tests fail on the first em dash of the manuscript,
and `PYTHONUTF8=1` is the reader's workaround. Every text open in the package now names its encoding;
under a forced ASCII locale the suite goes from 15 failures to none. (ii) The README's setup line,
`python -m venv .venv && . .venv/bin/activate`, assumed one shell and one layout. The Windows run was
in Git Bash, where `&&` is fine and `bin/activate` is not — Windows puts the environment in
`Scripts/` — so the line appears to work, activates nothing, and the reader installs into and then
tests the system interpreter. The next release states one path per shell, with the Git Bash block the
one these runs used. The other two are the reader's
environment: a Python newer than the libraries (3.14 at the time), and pytest's own temporary
directory, whose `pytest-current` symlink it cannot remove on Windows, which ends the session with a
`PermissionError` after the tests have passed (`--basetemp` avoids it).

**Table 11. Level 1 of the `v1.0` tag on four machines (the author's own; `requirements.txt`, the
minimum versions, except on the GPU host). The suite is `v1.0`'s, which is why the counts are those
of Table 9's first block and not of its `v1.1` block.**

| Machine | Shell | Python | Test suite | `check_tables.py` |
|---|---|---|---:|---|
| GPU host (Ubuntu, RTX A6000) | bash | 3.11.9 | 364 passed, 4 skipped | 2,578 agree, 0 disagree |
| Windows | Git Bash | 3.12 | 365 passed, 3 skipped | 2,578 agree, 0 disagree |
| macOS | bash | 3.12 | 365 passed, 3 skipped | 2,578 agree, 0 disagree |
| Google Colab (three runs) | notebook | 3.13.15 | 365 passed, 3 skipped | 2,578 agree, 0 disagree |

### 5.4 Secondary analysis from the record

A record bound to its inputs (P1) has a use the three evaluations do not measure: it can answer
questions that were not asked when it was written. This section is a case study of that reuse, not a
result about portfolios. Two questions arose while planning the work that follows this paper; both
were settled from the committed record alone — no quantum run, no cloud task, minutes of CPU — and
the method is the point: re-score the stored solutions under the changed conditions, verify the
re-scoring against the record it came from, and read the answer off.

*Does a stored portfolio survive a change of conditions?* If the market window, the risk aversion,
the cardinality or the asset universe changes, is it better to hand back the stored portfolio, to
repair it classically, or to run the sampler again? What the record supplies is the best feasible
portfolio of every run, the stored final state, and the enumerated feasible set of every instance —
so the regret under a changed problem is exact rather than estimated. Over the 295 runs of the ten
instances the prior-window replication also covers, the stored portfolio is within τ = 1 % of the new
optimum in 4 % of cases under a window change, 6–10 % under a change of risk aversion and 20 % when
carried to a universe that contains its assets; a fresh 1,000-shot batch from the stored final state
matches the repaired portfolio with median probability 0.2; and a greedy repair of one to four
single-asset swaps reaches the exact new optimum in **every case of every kind of change**. The same
search reaches the optimum from a *random* feasible start too (100 of 100 at N = 24 and at N = 30,
median 13 swaps), so it is the search and not the stored start doing the work. There was no decision
to make, and a planned paper on the choice was dropped before a line of it was written.

*Does a second constraint change the picture of [1]?* The question is what "at most one asset per
sector" does to the cardinality-constrained problem, with a pairwise quadratic penalty, on the same
instances. Here the record supplies the enumerated feasible sets, the penalty scales with their
derivations, and the stored probability of every feasible bitstring, so the new feasible set, the new
critical weights and the mass the stored states keep under both constraints all follow without a run.
The feasible set shrinks to 80 %, 54 % and 33 % of its size at N = 12, 16 and 20, yet the
cardinality-only optimum already satisfies the new constraint in 8–9 of 10 instances, and the
competition between two penalty weights that a second constraint could create appears only where the
constraint binds — one or two instances in ten, by 10–58 %. The pairwise form adds no constant
offset, so the denominator that [1] showed to inflate the conventional gap does not grow. The stored
states keep exactly the ratio of the feasible-set sizes, to within 0.1 pt, which is what enforcing the
second constraint by filtering rather than by a penalty would cost. A sector cap is, for a Markowitz
objective that diversifies on its own, mostly a constraint that does not bind; the extension study
was re-scoped on that basis.

Both answers were checked the way Section 4.1 checks the record: re-scoring the stored portfolios
under *unchanged* conditions reproduced the recorded g_F, rank and f on all 826 rows with no
mismatch; the enumerated objective agreed with every stored feasible-set value to 2·10⁻¹⁶; and the
single-constraint critical weight recomputed by the new code equalled the reference file's on all
30 instances. The scripts and their result files are in the package
(`experiments/a1_reuse/`: `a1_probe_rows.csv`, `a1_probe_summary.csv`, `a1_repair_scaling.csv`;
`experiments/a3_constraints/`: `a3_probe_instances.csv`, `a3_probe_frontier.csv`,
`a3_probe_repair.csv`). What the record bought is the cost of the answers: two studies were re-scoped
in minutes of CPU, because every run was already bound to its inputs and to an exact reference, and
the re-scoring could be checked against the values the record already held.

## 6. Implementation defects: discovery and verification coverage

Seven implementation defects changed reported numbers during the numerical study [1, Sec. VI-C and
Supplementary S-IX]. Table 12 lists how each was found at the time and which check of Section 3.1
would flag it now, or a dash where none would; the last column names the E-S1 fault that replays it
where one does.

**Table 12. The seven defects: how each was found, and what of it a relation holds now. A dash is a
defect no relation detects; the paragraphs below say what the record holds instead where that needs
saying.**

| # | Defect | How it was found | Relation that holds it now | E-S1 replay |
|---|---|---|---|---|
| 1 | angles passed as a plain array; the optimizer returned them unchanged | a flat convergence curve, by inspection | npz-internal, diagnostic clause | — |
| 2 | computational-basis sampling unsupported on the Braket device | an exception on first use | — (a failure, not a wrong number) | — |
| 3 | the γ-landscape's period at N = 12 smaller than one ADAM step | divergence of the expectation, by inspection | — | — |
| 4 | IonQ's nested result schema misread by the client library into an energy of exactly 0.0 | the shots were found in S3 and did not match | — | — |
| 5 | a hardware recovery path exercised only against a simulator | on the first real task | — (documented manual step) | — |
| 6 | `shot_seed` a no-op: the Lightning devices seed once at construction | two sweeps with "different" shots had identical energies, by comparison | seed-distinct (batch clause) and state-batch | shot_seed_noop |
| 7 | the L-BFGS-B wrapper sampled from the initial angles | the batch's feasible fraction tracked initial P_F (r = 0.99), not final (r = −0.30) | state-batch | batch_from_initial |

Four of those cells need a sentence. *Defect 1*: npz-internal's clause is that the final angles equal
the trajectory's first point. It is not an identity — a run stopped at a stationary point looks the
same — so it reports a run to look at rather than a fault, which is what this defect needs.
*Defect 3*: no relation detects it. A step larger than the period leaves a record that agrees with
itself in every way the thirteen test; what npz-internal holds is the neighbouring inconsistency,
the Hamiltonian scale on the row against the one in the artifact, so a run scored under a scale it
was not optimised under is visible. *Defect 4*: the record does not catch this one either. Solver and
scorer read the same misparsed bitstring, so they agree with each other, and a batch with no feasible
shot is a legitimate outcome elsewhere in the study. What changed is upstream: the hardware records
are parsed from the device's own nested schema (`src/solvers/braket_results.py`), and a test pins
the shot, distinct-bitstring and feasible counts of all four committed tasks. *Defect 7*: the defect
wrote a batch its artifact agreed with, so row-npz does not see it; row-npz sees the *injection* of
Section 5.1, which rewrites the row's summary and leaves the artifact. Its replay is the largest of
the ten — 2,299 findings on 2,252 rows and all nineteen (N, penalty) blocks, moving 12 of 30 headline
numbers.

Only the seventh was found by a relation that already existed as a check; the others were found by
inspection, by exception or by an ad hoc comparison, and the relations were written afterwards so that
the same class of defect would not need luck again. Four of the seven are still not held by any of
the thirteen, and the table says so rather than claiming coverage it does not have: an exception on
first use is a failure and not a wrong number, a documented manual step is a step, the misread
hardware schema produces a record that agrees with itself, and a step larger than the landscape's
period does too. For the last two the record holds a *neighbouring* relation — solver against
scorer, row against artifact — and the table distinguishes the two rather than letting a relation
that would catch something nearby stand for one that catches this. Two further defects concerned the record's writer
and its inputs rather than a run: a CSV writer that took its header from the first row dropped twelve
of nineteen columns of one file without a message (replayed as column_drop; the writer now takes the
union of keys), and a per-size price cache that held one file revised by the provider seven minutes
after the others (replayed as price_perturbation; one snapshot per window now, and — from this
evaluation — the content digest of Section 5.1).

## 7. Limitations

The evaluation is of one platform for one application family on one provider, by the platform's
authors. The faults of E-S1 are ours, four replays and six judgement, and the mutation operators are ours
too: the held-out draw holds the operators fixed and changes only the seed, so it answers the
objection that the relations were tuned to a particular draw and not the one that they were written
for a particular kind of fault. Of E-S2's cap, the simulator half is observed and the QPU rejection is still the documented
behaviour; the fallback counts rest on the transform, a month-old ledger and one re-observation. E-S3's reader is a clean-room script with the
author's package cache, and the three further machines of Table 11 were run by the author as well:
what is evidenced is that the written instructions carry to a machine they were not written on, not
that an independent person can follow them. No one outside the project has run the package; the
question of whether a stranger understands the README is a question about people, and answering it
would need their consent and an ethics review rather than another harness. Three design gaps were found by the evaluation, and a fourth — a per-run flush that was not
atomic, so that the bound it was supposed to give held only by luck — by checking the driver against
the claim this paper makes about it; they are stated above, the releases they are in and are not in are Section 5.3's last paragraph, and
the digest was appended to the released record files from the pinned statistics rather than produced
by a re-run. A review of the drivers against the claims this paper makes about them then found three
more, all of the same kind and all fixed in `v1.1` with a regression test each: the thirteen checks
did not gate the aggregation at all — `analyze_e1` verified the job's exit status and not the record,
so an edited cell reached a published median — `--results-tag ""` was falsy where the file name is
built and not-None where the overwrite guard looks, so an empty shell variable replaced the committed
baseline without a word, and `--resume` skipped cells by key without asking whether the manifest had
been written under the same arguments, so a resumed run with different steps left two experiments in
one file and exited 0. None was found by the three evaluations; all three were found by reading a
driver beside the sentence that describes it, which is the same instrument as the non-atomic flush
above.

A reader of the package then found a fourth of that kind, and it is the sharpest of them, because
it is this paper's own thesis turned on the platform: the status record of Section 3.3 was a claim
about a manifest that nothing compared to the manifest. Deleting one row from a committed
thirty-row block and leaving `EXIT 0, 30/30` in place, the aggregation printed the two numbers on
adjacent lines and published the medians of twenty-nine; duplicating a row gave thirty-one the same
way. The comparison described in Section 3.3 is the fix. The same reading found the release missing
a file this paper cites — the device-availability log, tracked in the working repository and matched
by the package's own ignore rule, so the export left it out while printing one count for the files
it selected and a smaller one for the files it committed. Both are fixed in `v1.1`: a check that
refuses, rather than two numbers nobody compared. That reading is a second instrument, and it is not
ours: of the defects in the platform's *writers*, three came from us reading our own drivers and
this one from a reader who had only the package. The third gap was found not by a harness but by operating one: the
two-hour coverage scan of Section 5.1 crashed in its first minute on a column that is empty on every
row, the shell script that ran it had no `set -e` and printed its completion marker anyway, and success
was read off that marker for nine hours. The status record of Section 3.3 is the consequence. The dollar figures are list prices applied to
counts; the project's total spend on the provider was $42.59. P2 is preventive: it does not detect a
loop of correct calls; the account-level control that does was measured five times, on one provider's
sample deployment, at one submission rate, and its residue varied twofold between runs. The
thirteen checks are written by hand, one expression per fact; generating them from a declared
record schema is prototyped and reported elsewhere, and is not what E-S1 evaluated.

One binding cannot close, and it is this paper's own. The package holds this manuscript *and* the
record of the pass that evaluated the package, so writing that record back into the package changes
the package the pass examined. What a reader needs is the difference: the pass cloned the released
tag exactly, and the tree at that tag differs from the submitted version in four files and no others
— this manuscript as Markdown, LaTeX and PDF, and the step record the pass itself wrote. The
relations bind a record to its inputs; they do not bind it to the act of writing it down.

One limit of P1 is not ours to remove here. A record is bound to its inputs by a digest of those
inputs, which lets a reader detect a mix-up but not re-derive the inputs: the price snapshots
behind the derived statistics are third-party data that the package may not redistribute
(Section 3.1), so the reproduction levels of Section 3.3 start from the statistics we publish and
trust us for the step before them. The binding is verifiable only where the inputs are, and closing that gap needs a
different data source rather than a different design: inputs a reader can resolve independently — a
public ledger's block height and transaction hash, say — would make the same record, relations and
reconstruction levels checkable end to end. We have not built that.

## 8. Conclusion

A cloud quantum benchmark's record can be designed so that a mix-up between its parts is a violated
relation rather than a plausible number, its billing can be refused by configuration before a task
exists, and its results can be rebuilt from the published package at a cost that package states. Measured
against controls on a 2,565-run record, what a check can settle is set by how far the record it
reads extends. Expectations declared over the run table catch eight of the ten injected faults, and
over six hundred random mutations they come within two of the relations wherever a headline number
moves — 51 against 49. The two faults they cannot reach, and the 168 mutations of a file the run
table does not contain, need the relations that cross from the table to the pinned statistics and
the stored state: those catch 135 of the 168 and the declarative control none. A schema-only
pipeline catches one of the ten, and five committed clean records raise nothing. The two ruinous
misconfigurations stopped before any device is built, whereas a model of the provider's documented dollar cap stops them after six tasks, and
that cap is unavailable for the simulator where the fallback occurred; and the published package's three reconstruction levels rebuilt from a clone in
24 s (21.7 s of figures and 2.6 s of table checks), 18 s, and — for the 24-run block that stands
for the 13.9-GPU-hour cross — 222 s, with the loss on interruption bounded at the run in flight once
the file is written by a single rename. Measuring the three principles found the platform recording an inference in
three places, reading its drivers found four more, and a reader who had only the package found a
fifth — a status record that nothing compared to the file it described; every fix is measured or
exercised above, and Section 7 says which instrument found which. Future work will extend fault injection to the execution and
persistence paths, including interrupted writes, aggregation gates, and resume conditions, to test
failures that mutations of stored records alone cannot expose.

## Acknowledgment

**AI assistance disclosure.** The author used Claude (Anthropic), accessed through the Claude Code
command-line interface, to assist with initial drafts of code and prose, generation of the tables and
figures from committed result files, and checks of numerical statements against those files. The
author formulated the research questions, designed the study, reviewed and revised the AI-assisted
code and text, and interpreted the results. The author takes responsibility for the final manuscript
and accompanying artifacts.

## Availability

The code, the evaluation results and the commands that rebuild them are in one
package, release `v1.1` (https://github.com/tobitaQ/qaoa-portfolio-penalty-benchmark,
tag `v1.1`, commit `a32d427`, doi:[[v1.1 DOI]]): the platform; the three harnesses
(`experiments/system_paper/es1_fault_injection.py`, `es1_random_faults.py`, `es2_guard_stages.py`,
`es2_observe_cloud.py`, `es2_observe_cost_control.py`, `es3_reconstruction.py`) and the cost-control
aggregation (`es2_cost_control_aggregate.py`); the cost-control patch (`cdk/cost_control_patch/`); the job status module
(`src/run_status.py`, `scripts/run_detached.sh`); their tests; the result files behind every table
(`paper02_es1_*.csv`, `paper02_es2_*.csv`, `paper02_es3_*.csv`, `paper02_level1_machines.csv`); and
`papers/system_paper/check_tables.py`, which binds each cell of Tables 3–11 and 13 to one of them and
regenerates both figures for a byte comparison. Appendix C lists the commands each
table came from. Two inputs are not
in it: the 2,565 per-run artifacts (`.npz`, about 100 MB), which the Zenodo deposit of [1] carries
and which the checks that read an artifact and E-S1's injection need (Section 5.3), and the price
snapshots behind the derived statistics, which are third-party data the package may not
redistribute (Section 3.1). Every reproduction level starts from the statistics it does ship.

The package E-S3 evaluated at all three levels is the *earlier* release, `v1.0` (same repository,
commit `c804a63`, doi:10.5281/zenodo.22743582), which is the version the numerical study [1] was
submitted with and
which does not contain this paper's material. `v1.1` adds it, together with the fixes this paper
reports — the locale encoding of Section 5.3, the non-atomic flush of Section 3.3, and the three
driver defects of Section 7: the aggregation that checked the job and not the record, the empty
`--results-tag`, and the resume that did not compare its conditions — and an `input_digest` column
appended to the committed record from the pinned statistics, leaving every column of `v1.0`
unchanged. `CHANGELOG.md` states which version each paper was evaluated against.

The cloud spend of this paper, and of the project behind it, is Table 13.

**Table 13. Cloud tasks paid for, by what they were paid for. Each cost is the row's task count at SV1's $0.00375 per task — the three-second minimum — rounded to the cent, and *this paper*'s total is the sum of those rounded figures; the rounded sum of 1,249 tasks is $4.68. The last two rows are the provider's billing console and are not derived from a count.**

| What | Tasks | Cost |
|---|---:|---:|
| the sampling-fallback observation on SV1 (Section 5.2) | 178 | $0.67 |
| the account-level cost control as documented, the deployment that metered nothing (Section 5.2) | 400 | $1.50 |
| the same control with its import fixed, the five repetitions of Table 8 | 671 | $2.52 |
| the IonQ spending-limit probe: limit created, device never online (Section 5.2) | 0 | $0.00 |
| **this paper** | **1,249** | **$4.69** |
| the numerical study [1]: SV1 and IonQ arms, 2026-08 to 2026-09 | — | $37.90 |
| **the project on Amazon Braket, to date** | — | **$42.59** |

All other figures in this paper are list prices applied to task counts — some observed on the
service, some counted as circuits, as Table 7 distinguishes; nobody paid them.

## References

[1] H. Tobita, "Measuring metric and sampling distortions in penalty-encoded QAOA portfolio
benchmarks," submitted to *IEEE Trans. Quantum Eng.*, 2026; reproduction package
doi:10.5281/zenodo.22743582.

[2] B. Weder, J. Barzen, F. Leymann, M. Salm, and K. Wild, "QProv: A provenance system for quantum
computing," *IET Quantum Communication*, vol. 2, no. 4, 2021, doi:10.1049/qtc2.12012.

[3] M. Gamage, O. Kinanen, J. Muff, and V. Stirbu, "Enhancing quantum software development process
with experiment tracking," arXiv:2507.06990, 2025.

[4] V. Stirbu and A. Meijer-van de Griend, "Observing the quantum compiler through automatic
experiment tracking for Qiskit," arXiv:2608.05041, 2026.

[5] J. Peltonen, V. Stirbu, T. Mikkonen, and C. Pautasso, "Toward standardized quantum provenance:
A cross-provider analysis, unified API, and reference prototype," arXiv:2608.08272, 2026.

[6] B. Weder, J. Barzen, M. Beisel, F. Bühler, D. Georg, F. Leymann, and L. Stiliadou, "Qunicorn: A
middleware for the unified execution across heterogeneous quantum cloud offerings,"
arXiv:2411.06889, 2024.

[7] J. R. Finžgar, P. Ross, L. Hölscher, J. Klepsch, and A. Luckow, "QUARK: A framework for quantum
computing application benchmarking," in *Proc. IEEE Int. Conf. Quantum Computing and Engineering
(QCE)*, pp. 226–237, 2022, doi:10.1109/QCE53715.2022.00042.

[8] A. Cosentino, C. Li, V. Russo, B. A. Chase, T. Lubinski, S. Niu, N. Patel, N. Shammah, and
W. J. Zeng, "Metriq: A collaborative platform for benchmarking quantum computers," arXiv:2603.08680,
2026.

[9] V. Gierisch and W. Mauerer, "QEF: Reproducible and exploratory quantum software
experiments," arXiv:2511.04563, 2025.

[10] D. Teegarden, A. Casey, F. G. Serpa, P. Becker, A. Brahme, S. Kataria, and P. Lopata, "Three
months in the life of cloud quantum computing," arXiv:2601.09943, 2026.

[11] M. Zaharia et al., "Accelerating the machine learning lifecycle with MLflow," *IEEE Data Eng.
Bull.*, vol. 41, no. 4, pp. 39–45, 2018. http://sites.computer.org/debull/A18dec/p39.pdf

[12] M. Schlegel and K.-U. Sattler, "MLflow2PROV: Extracting provenance from machine learning
experiments," in *Proc. 7th Workshop on Data Management for End-to-End Machine Learning
(DEEM@SIGMOD)*, 2023, doi:10.1145/3595360.3595859.

[13] R. Souza et al., "Workflow provenance in the lifecycle of scientific machine learning,"
*Concurrency and Computation: Practice and Experience*, vol. 34, no. 14, 2022,
doi:10.1002/cpe.6544.

[14] L. Rupprecht, J. C. Davis, C. Arnold, Y. Gur, and D. Bhagwat, "Improving reproducibility of
data science pipelines through transparent provenance capture," *Proc. VLDB Endow.*, vol. 13,
no. 12, 2020, doi:10.14778/3415478.3415556.

[15] P. Alvaro, J. Rosen, and J. M. Hellerstein, "Lineage-driven fault injection," in *Proc. ACM
SIGMOD*, 2015, doi:10.1145/2723372.2723711.

[16] S. Schelter, D. Lange, P. Schmidt, M. Celikel, F. Biessmann, and A. Grafberger, "Automating
large-scale data quality verification," *Proc. VLDB Endow.*, vol. 11, no. 12, pp. 1781–1794, 2018,
doi:10.14778/3229863.3229867.

[17] N. Bantilan, "pandera: Statistical data validation of pandas dataframes," in *Proc. 19th Python
in Science Conf. (SciPy)*, 2020, pp. 116–124, doi:10.25080/Majora-342d178e-010.

[18] U. Kanewala and J. M. Bieman, "Testing scientific software: A systematic literature review,"
*Information and Software Technology*, vol. 56, no. 10, pp. 1219–1232, 2014,
doi:10.1016/j.infsof.2014.05.006.

[19] H. Chen, A. Phani, and S. Schelter, "PrismaDV: Automated task-aware data unit test generation,"
arXiv:2604.21765, 2026.

[20] F. Mölder et al., "Sustainable data analysis with Snakemake," *F1000Research*, vol. 10, art. 33,
2021, doi:10.12688/f1000research.29032.2.

[21] S. Soiland-Reyes et al., "Packaging research artefacts with RO-Crate," *Data Science*, vol. 5,
no. 2, pp. 97–138, 2022, doi:10.3233/DS-210053.

[22] F. Chirigati, R. Rampin, D. Shasha, and J. Freire, "ReproZip: Computational reproducibility with
ease," in *Proc. ACM SIGMOD*, 2016, doi:10.1145/2882903.2899401.

[23] K. Greff, A. Klein, M. Chovanec, F. Hutter, and J. Schmidhuber, "The Sacred infrastructure for
computational research," in *Proc. 16th Python in Science Conf. (SciPy)*, 2017,
doi:10.25080/shinma-7f4c6e7-008.

[24] D. Köster, M. Franz, B. Zec, N. Hoess, R. Ramsauer, and W. Mauerer, "Works on my QPU:
Reproducibility in quantum computing research," arXiv:2607.08348, 2026.

[25] W. Mauerer and S. Scherzinger, "1-2-3 reproducibility for quantum software experiments," in
*Proc. IEEE Int. Conf. Software Analysis, Evolution and Reengineering (SANER)*, 2022,
doi:10.1109/SANER53432.2022.00148.

[26] B. Ye, P. Liang, M. Tavassoli Sabzevari, and A. A. Khan, "Auditing empirical comparisons in
quantum software," arXiv:2607.00516, 2026.

[27] D. Köster and W. Mauerer, "Claim against measurement: Statistical artefacts in quantum error
mitigation benchmarks," arXiv:2605.29872, 2026.

[28] H. Ohto, Y. Ishimoto, S. Matsumoto, and S. Kusumoto, "On the reproducibility of quantum
software defect datasets: A case study of Bugs4Q," arXiv:2606.27124, 2026 (ICSME 2026, replication
track).

[29] Amazon Web Services, "Spending limits," Amazon Braket Developer Guide, and
`CreateSpendingLimit`, Amazon Braket API Reference, both accessed 27 Sep. 2026 (announced on the AWS
Quantum Technologies Blog, 21 Nov. 2025). The refusal for a simulator device quoted in Section 5.2 is
this project's own observation of the API.
https://docs.aws.amazon.com/braket/latest/developerguide/braket-spending-limits.html
https://docs.aws.amazon.com/braket/latest/APIReference/API_CreateSpendingLimit.html

[30] Amazon Web Services, "Managing your costs with AWS Budgets," documentation, accessed
27 Sep. 2026. https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html

[31] Amazon Web Services, `aws-samples/cost-control-for-amazon-braket`, commit `19eaad6`, accessed
27 Sep. 2026: EventBridge task-state events, Lambda, DynamoDB, CloudWatch and an IAM deny on
`braket:CreateQuantumTask` (announced on the AWS Quantum Technologies Blog, 31 May 2023). The
deployment measured in Section 5.2 is of that commit.
https://github.com/aws-samples/cost-control-for-amazon-braket

[32] C. J. Anders, K. A. Nicoli, B. Wu, N. Elosegui, S. Pedrielli, L. Funcke, K. Jansen, S. Kühn, and
S. Nakajima, "Adaptive observation cost control for variational quantum eigensolvers,"
arXiv:2502.01704, 2025.

[33] J. Szefer, "Exploiting reset operations in cloud-based quantum computers to run quantum
circuits for free," arXiv:2512.14582, 2025.

[34] M.-C. Isfan, L.-I. Caramete, and A. Caramete, "Evaluating state-of-the-art cloud quantum
computers for quantum neural networks in gravitational waves data analysis," arXiv:2601.14036, 2026.

[35] Association for Computing Machinery, "Artifact review and badging — current," v1.1, accessed
27 Sep. 2026. https://www.acm.org/publications/policies/artifact-review-and-badging-current

[36] S. N. Goodman, D. Fanelli, and J. P. A. Ioannidis, "What does research reproducibility mean?"
*Sci. Transl. Med.*, vol. 8, no. 341, 341ps12, 2016.

[37] H. E. Plesser, "Reproducibility vs. replicability: A brief history of a confused terminology,"
*Front. Neuroinform.*, vol. 11, art. 76, 2018.

[38] V. Bergholm et al., "PennyLane: Automatic differentiation of hybrid quantum-classical
computations," arXiv:1811.04968, 2018.

## Appendix A. The CDK footprint

Two stacks, region-pinned to `us-east-1` (Braket has no endpoint in the project's default region,
which a first deployment discovered as a DNS failure before any task was submitted). `QfrBraketStack`:
one S3 bucket (AES-256, public access blocked, SSL enforced, 90-day lifecycle) and one IAM role
scoped to Braket and that bucket. `QfrMonitoringStack`: two Budgets, $50 monthly and $50 Braket-only,
with notifications at 50/80/100 % actual and 100 % forecast. Three account-level preconditions are
documented manual steps because no synthesis or test reveals them: the Braket service-linked role is
not created on first use; the default region has no endpoint; the hardware provider's agreement is a
console click. The retained footprint carries no idle charge. A per-QPU spending limit
(`AWS::Braket::SpendingLimit`) is a one-resource addition and is the "both" configuration of
Section 5.2 as infrastructure; the stacks do not declare one, and the $1.00 limit of Section 5.2 was
created through the API for that observation and left in place, which a reader reproducing the
observation should know: the account it was created on still refuses an IonQ task whose estimate
exceeds $1.00.

### A.1 Provider-side controls (Amazon Braket, as of September 2026)

**Table 14. Three controls that can stop or report a Braket charge, and one client-side limit that
cannot. The first three are the "cap" of Section 5.2 in its available forms; the fourth bounds only
how long the client waits, and is listed because it is easy to mistake for a bound on the task.**

| Kind | Control | Scope | When it acts | Observed here |
|---|---|---|---|---|
| service, per device | spending limit (`CreateSpendingLimit`, Nov. 2025): rejects a task whose list-price estimate exceeds the remaining cap | on-demand QPU tasks only; the API refuses to create one for a simulator | at task creation | created ($1.00 on IonQ Forte-1); the simulator refusal observed; the QPU rejection pending device availability (Sec. 5.2) |
| account, detective | AWS Budgets: notification at a threshold; optional action attaching an IAM deny | whole account | data lag of hours | deployed as notification only |
| account, detective | AWS Braket cost-control solution (May 2023) [31]: EventBridge task-state events → Lambda → DynamoDB running cost → notification or IAM deny on `braket:CreateQuantumTask` | simulators, QPUs, hybrid jobs | near real time; "a few" tasks may be created before the denial propagates | deployed for Sec. 5.2 and destroyed: as documented it metered nothing (400 tasks passed); fixed, five runs denied 41–92 tasks (median 47) past the limit, 6–56 (median 10) past the alarm |
| client, not a charge control | `poll_timeout_seconds` of the PennyLane–Braket device. In `AwsQuantumTask._wait_for_completion` (SDK 1.125) the timeout ends the polling loop, logs a warning and returns `None`; the task keeps running on the service and keeps accruing its charge. Cancelling is a separate call, `AwsQuantumTask.cancel()` | the caller's process, waiting for one task | while waiting for a result | set to 3,600 s here against the library's 432,000 s (five-day) default; nothing in this project reached it |

### A.2 The cost-control deployment

Upstream commit `19eaad6` of `aws-samples/cost-control-for-amazon-braket`, deployed 2026-09-17 in
`us-east-1` only with a $0.30 monthly and $3 all-time limit and the denial targeted at the project's
IAM user, then destroyed. Three changes, kept as a patch in the repository (`cdk/cost_control_patch/`):
Lambda images built for x86_64 rather than arm64 (the build host has no arm64 emulation);
`NUMBA_CACHE_DIR=/tmp/numba` on the task-logger function, without which the function fails on import;
and, for the repetitions only, the monthly alarm's evaluation window shortened from 24 h to 5 min so
that a cleared running cost lets the alarm return to OK between runs (`datapoints_to_alarm` stays 1,
so the latency to ALARM is unchanged). Between runs the denial was detached by hand — the sample
detaches only on a transition to OK — and the running-cost table emptied.
The stacks create an EventBridge bus and two rules, five Lambda functions, two DynamoDB tables, seven
CloudWatch alarms, a dashboard, an SNS topic and one managed policy (`Deny braket:CreateQuantumTask`).
Destroying them requires detaching that policy from the controlled identities first and deleting the
event-bus stack last.

## Appendix B. The task ledger

`sv1_task_ledger.py` groups the archived SV1 task records (1,299 tasks) into runs by submission
time and counts analytic against sampling tasks. Run 1 (2026-08-09, N = 8 pilot): 227 tasks, 50
analytic and 177 sampling, 177,000 shots. Run 2 (the same after `frozen_angles`): 204 tasks for four
sizes, 200 analytic and 4 sampling. The ledger is the service-side member of the three-way agreement
of Table 6.

## Appendix C. Reproduction commands for this paper

These are the commands the tables came from, with what each took in wall-clock. The first group runs
on a CPU and bills nothing: a reader who wants to re-derive one table runs one of these, and
`es3_reconstruction.py` is the one that plays the reader (Section 5.3 reports what it costs). The two
at the end bill the account they are run in and create a spending limit in it; they are recorded so
that a reader knows what produced the cells Table 7 marks as observed on the service, not so that the
reader repeats them.

```
# ~12 min, CPU
python -m experiments.system_paper.es1_fault_injection
# ~25 min: the same thirteen checks on the five committed clean records
python -m experiments.system_paper.es1_fault_injection --clean-records
# ~2.5 h: column coverage (133 columns) + 300 random mutations
# (both need the per-run artifacts: e1_runs*.npz, about 100 MB, from the Zenodo deposit of [1])
python -m experiments.system_paper.es1_random_faults --faults 300
# ~1.8 h: the held-out draw, after the relations were frozen (last row of Table 5)
python -m experiments.system_paper.es1_random_faults --faults 300 \
    --seed 7 --tag heldout --no-coverage
# ~1.8 h (N = 20 is 94 min of tape construction); --sizes 8 runs in 1 min
python -m experiments.system_paper.es2_guard_stages
# Table 8 from the committed task records of the six runs (seconds, no AWS)
python -m experiments.system_paper.es2_cost_control_aggregate
# ~15 min with a warm pip cache; clones the public package
python -m experiments.system_paper.es3_reconstruction
# ~7 min: the v1.1 block of Table 9 (the tag this paper ships)
python -m experiments.system_paper.es3_reconstruction --tag v1.1 --out-suffix _v11 \
    --only clone venv tests level1 level2 status
python -m pytest -q tests/test_es1_fault_injection.py tests/test_es1_random_faults.py \
    tests/test_es2_guard_stages.py tests/test_es2_cost_control_aggregate.py \
    tests/test_es3_reconstruction.py tests/test_run_status.py

# The two that bill. Each creates tasks in the account whose credentials it finds, and the
# first also creates a $1.00 spending limit on IonQ Forte-1. Nothing above needs either.
# the sampling fallback re-observed on SV1 (~$0.67) and the spending-limit request
python -m experiments.system_paper.es2_observe_cloud --s3-bucket <bucket> --yes-bill-me
# one repetition of the account-level control, after deploying cdk/cost_control_patch/
# (up to --max-tasks tasks, 400 by default, ~$1.50 on SV1)
python -m experiments.system_paper.es2_observe_cost_control --s3-bucket <bucket> --yes-bill-me
```
