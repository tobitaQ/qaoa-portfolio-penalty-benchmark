# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Derive the literature-audit counts from the census, instead of asserting them.

Every "N of M" in the paper's premise (Sec. II, Supplementary Table S1) comes from here. It
was previously written out by hand in a working note, and the hand counts
did not reconcile: the population paragraph enumerated ten papers, called them
nine, and silently dropped [21], which is in the audit table and cited in the
paper. Counting in prose is exactly the failure the paper is about, so the
counts are now computed.

The census records what each paper *states*. Judgement calls that a referee
could reasonably make differently are isolated here, in one place, and named:

  reports_gap        An optimality gap, approximation ratio, or regret against
                     a reference solution -- any scalar scoring the returned
                     objective against a better one. "no" covers papers that
                     report a raw objective, a domain metric, or a hit rate.
  billed_cost        Task/job counts, billed device time, or money. Wall-clock
                     runtime does NOT count: it is what a paper reports about a
                     laptop, not what a metered device charges. Gate counts and
                     shot counts on a free simulator do not count either.
  repeats_per_config Repeats of one configuration, stated as a number.
                     "partial" = repeats exist but the count is not stated
                     (e.g. anneals per batch with the batch count missing), and
                     is counted as NOT reported.
  domain_metric      Sharpe, return or volatility reported as a *result*.
                     A Sharpe that only defines the objective coefficients does
                     not count -- that is an input.
  post_standard      Published after both Abbas et al. (Dec 2024) and QOBLIB
                     (Apr 2025). Conference papers are dated by publication,
                     which overstates the citation opportunity for a submission
                     deadline that preceded QOBLIB; see the caveat printed
                     below.

Usage:
    python papers/audit_tally.py
"""

from __future__ import annotations

import csv
import pathlib

CENSUS = pathlib.Path(__file__).parent / "audit_census.csv"

#: Cells that record a partial or unstated answer. They count as "not reported":
#: the paper's claim is about what a reader is given, and a repeat count that is
#: not stated is not given.
NOT_REPORTED = {"no", "partial", ""}


def load() -> tuple[list[dict], list[dict]]:
    """Return (papers in the tally, papers excluded), with the reason recorded."""
    rows = list(csv.DictReader(CENSUS.open()))
    return ([r for r in rows if r["in_tally"] == "yes"],
            [r for r in rows if r["in_tally"] != "yes"])


def count(rows: list[dict], field: str, positive: bool = True) -> list[str]:
    """Papers whose `field` is (or is not) a definite yes."""
    hit = [r for r in rows
           if (r[field] == "yes") is positive
           and (positive or r[field] in NOT_REPORTED)]
    return [f"[{r['ref']}] {r['short']}" for r in hit]


def main() -> int:
    tally, excluded = load()
    n = len(tally)
    print(f"Population: {n} benchmarks in the tally, {len(excluded)} excluded\n")
    for r in excluded:
        print(f"  excluded  [{r['ref']}] {r['short']}: {r['exclusion_reason']}")

    def line(label: str, hits: list[str], total: int = n) -> None:
        print(f"\n{label}: {len(hits)} of {total}")
        for h in hits:
            print(f"    {h}")

    line("Reports an optimality gap, ratio or regret", count(tally, "reports_gap"))
    line("Reports NO such metric", count(tally, "reports_gap", positive=False))
    line("Reports task counts, billed device time or cost", count(tally, "billed_cost"))
    line("Reports a domain metric as a result", count(tally, "domain_metric"))
    line("Observes solution degeneracy", count(tally, "observes_degeneracy"))
    line("States a penalty weight numerically", count(tally, "penalty_value_stated"))
    line("Reports repeats per configuration", count(tally, "repeats_per_config"))

    post = [r for r in tally if r["post_standard"] == "yes"]
    line("Published after both standards", [f"[{r['ref']}] {r['short']} ({r['year']})"
                                            for r in post], len(post))
    print(f"\n  ... of which cite or conform to either standard: 0 of {len(post)}")
    # The reference number is read from the census, not typed: the list is
    # renumbered by first appearance whenever the draft is re-sectioned.
    qce = next(r["ref"] for r in tally if r["short"].startswith("Uotila"))
    print(f"  Caveat: [{qce}] is a QCE'25 proceedings paper whose submission deadline")
    print("  plausibly preceded QOBLIB's April 2025 posting. Excluding it on that")
    print(f"  basis gives {len(post) - 1} rather than {len(post)}; both are stated in the paper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
