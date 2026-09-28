# Cover letter — ACM Transactions on Quantum Computing

**Manuscript.** *Bound Records, Structural Guards and Costed Reconstruction: Designing and
Evaluating an Experiment Platform for Cloud Quantum Optimization Benchmarks*

**Author.** Hiroaki Tobita, Advanced Institute of Industrial Technology, Tokyo, Japan
(tobita-hiroaki@aiit.ac.jp). Single author; no co-authors to disclose.

**Category.** Regular paper. 38 pages in `acmsmall`, 14 tables, 2 figures, 38 references.

---

Dear Editors,

Please consider the manuscript above for publication in ACM Transactions on Quantum Computing.

**What the paper reports.** A benchmark of a variational quantum algorithm on a cloud service has
three properties a laboratory experiment does not: the input is a query the provider revises between
calls, the number of billable tasks is chosen by the framework rather than by the algorithm, and the
record is large enough that a mix-up between its parts leaves the aggregate looking plausible. At
each of the three, an experiment script tends to record an *inference* where a *fact* belongs. The
paper states three design principles that replace the inference with a record — a run record bound to
its inputs and outputs by machine-checkable relations; billing refused by the configuration before
submission; reconstruction at three levels of recorded cost — and evaluates each against a control
that keeps the inference: fault injection against a pipeline with no checks and against a
declarative data-quality control, guard-stage measurement against a provider's dollar cap, and a
clean-room reconstruction of the published package. The platform is the one behind a 2,565-run QAOA
portfolio benchmark; this paper contributes no new optimisation runs, and its evidence is what the
record, the guards and the package do under injected faults, misconfigurations and reconstruction.

**Why TQC.** The subject is software engineering for quantum computing, which the journal's charter
names. None of the problems the platform handles is unique to quantum computing, and Section 1 says
so; what is particular is the form each takes here. The billable unit is decided by the trainability
of a tensor, because the gradient of a circuit is more circuits, so a flag in a client library moves
a bill by two orders of magnitude. The provider's spending cap exists for hardware and not for the
simulator on which this project's real cost fallback occurred. A hardware observation is a waiting
problem: the device this paper needed answered `OFFLINE` to 336 queries over seven days. A result is
a sample, and the distribution it should have come from is an object the simulator computed and the
run stored, so one of the thirteen checks tests the sample against a reference distribution the
record itself holds — the check that caught a defect the others missed. And which columns may
legitimately differ across a reproduction is itself a fact about simulation precision.

**Relationship to other submissions.** The numerical study this platform served is a separate
manuscript, cited as [1], under review at *IEEE Transactions on Quantum Engineering*
(TQE-26-09-REG-0575). The two papers share data and share nothing else: Section 2 ends with an
explicit list of the four subjects that appear in both and states, for each, what [1] reports (a
fact about its runs) and what this paper reports (the mechanism built in response and what that
mechanism was measured to do). No run of [1] is repeated here and no result of [1] is restated as a
result of this paper. A third manuscript, on a different subject (decision materials generated from
records, with a checker for their claims), is in preparation for another venue and shares no text
with either.

**Artifacts.** The platform, the three evaluation harnesses, their tests and all result files are in
the repository named in the availability statement. The package evaluated in the reconstruction
experiment is a public release (`v1.0`, commit `c804a63`) archived at doi:10.5281/zenodo.22743582;
the release this paper ships, `v1.1`, adds its own material and the fixes the evaluation produced,
and the suite, level 1, level 2 and the interruption were run against it as well (Table 9).
Appendix C lists the commands that reproduce every number in this paper, with their run times and
the two that need the deposited per-run artifacts (~100 MB). The availability statement maps the
paper's three reconstruction levels onto ACM's badging terminology; the author intends to apply for
artifact review if the paper is accepted.

**Cloud spend and ethics.** Every cloud task paid for by this paper is itemised in Table 13: $4.69
in total, against the project's $42.59 on the provider to date. Every other dollar figure in the
paper is a list price applied to a measured task count, and the paper says so. The work involves no
human participants and no personal data: every run reported here was executed by the author, and the
question of whether a stranger can follow the package's instructions is named in Section 7 as a
question about people that would require consent and ethics review rather than another harness.

**Conflicts.** The author declares no competing interests. The work used a commercial cloud service
(Amazon Braket) as a paying customer, with no funding, support or other relationship with the
provider or with any hardware vendor named in the paper.

Thank you for your consideration.

Hiroaki Tobita

---

## Optional: suggested reviewers (no prior collaboration or other conflict with the author)

| Reviewer | Affiliation | Why |
|---|---|---|
| Wolfgang Mauerer | OTH Regensburg / Siemens AG | Reproducibility in quantum computing research and reproduction packaging [9, 24, 25, 27] |
| Vlad Stirbu | University of Jyväskylä | Provenance and experiment tracking for quantum software [3, 4, 5] |
| Benjamin Weder | University of Stuttgart | QProv and quantum middleware [2, 6] |
| Andre Luckow | BMW Group / LMU Munich | Application-level quantum benchmarking (QUARK) [7] |

## Notes for the submission form (not part of the letter)

- Abstract: 200 words (ACM's guideline is 150–200).
- CCS concepts and keywords are in the manuscript; four `\ccsdesc` lines.
- The manuscript is in `acmsmall` with `review` enabled (line numbers); a non-review PDF can be
  supplied on request.
- Suggested subject areas: quantum software engineering; benchmarking and evaluation methodology.
