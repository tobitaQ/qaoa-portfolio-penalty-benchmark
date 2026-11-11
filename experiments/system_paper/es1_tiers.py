# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Re-count the E-S1 detections by what the detecting clause read.

Four nested tiers of *record*, not of check names: the run table with its
reference join (T1); plus the pinned statistics (T2); plus the per-run
artifacts (T3); plus a re-simulation (T4). A check whose clauses read different
files is split by clause: ``identities`` has table, input and artifact clauses,
and ``es1_clause_sources.py`` records which fired on every mutation and fault.
``state-batch`` reads only run-row columns and is in T1, whatever kind of rule
it is. ``seed-distinct`` has an angle clause over the run row and a batch
clause over the artifacts; a mutation that leaves every artifact as it was can
only have fired the angle clause, so the clause is decided by the operator
(inside the run table: angle, T1; outside: batch, T3), and on the injected
faults only the copied batch of shot_seed_noop fires it (batch, T3).

Reads the detection matrix, the mutation files and the clause-source files of
one record; writes ``paper02_es1_tiers{record}.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RESULTS = Path("experiments/system_paper/results")

#: Nested tiers by the record each clause reads. ``identities:<source>`` names a
#: clause family of the identities check.
TIERS = {
    "T1 table and reference": ["schema", "manifest", "penalty", "reference-row", "scorer",
                               "state-batch", "identities:table", "seed-distinct:angle"],
    "T2 + pinned inputs": ["input-digest", "reference-input", "identities:input"],
    "T3 + per-run artifacts": ["row-npz", "npz-internal", "seed-distinct:batch", "identities:artifact"],
    "T4 + recomputation": ["state-recompute"],
}
OUTSIDE = ("npz_perturb", "npz_swap", "input_perturb")
SOURCES = ("table", "input", "artifact")


def _cumulative() -> dict[str, set[str]]:
    out, acc = {}, []
    for name, checks in TIERS.items():
        acc = acc + checks
        out[name] = set(acc)
    return out


def _units(fired: str, clause_row, outside: bool = True) -> set[str]:
    """The detecting units of one mutation or fault: check names, with
    ``identities`` replaced by the clause families that fired and
    ``seed-distinct`` by the clause the operator can have reached."""
    units = {x for x in str(fired).split(";") if x and x != "nan"}
    if "seed-distinct" in units:
        units.discard("seed-distinct")
        units.add("seed-distinct:batch" if outside else "seed-distinct:angle")
    if "identities" in units:
        units.discard("identities")
        hit = False
        for s in SOURCES:
            if clause_row is not None and int(clause_row[f"identities_{s}"]) > 0:
                units.add(f"identities:{s}"); hit = True
        # identities failed in the committed run but no clause fired on the
        # replay: the check raised, because a column it reads was dropped.
        # The missing column is the table's, so the detection is the table's.
        if not hit:
            units.add("identities:table")
    return units


def rows(record: str = "") -> list[dict]:
    cum = _cumulative()
    out = []

    inj = pd.read_csv(RESULTS / f"paper02_es1_detection_matrix{record}.csv")
    inj = inj[inj.fault != "clean"]
    cl = pd.read_csv(RESULTS / f"paper02_es1_clause_sources{record}_faults.csv").set_index("fault")
    checks = [c for c in inj.columns if c in {c2.split(":")[0] for s in cum.values() for c2 in s}]
    for name, allowed in cum.items():
        hit = 0
        for _, r in inj.iterrows():
            fired = ";".join(c for c in checks if r[c] == "FAIL")
            if allowed & _units(fired, cl.loc[r.fault] if r.fault in cl.index else None):
                hit += 1
        out.append({"population": "injected faults", "n": len(inj), "tier": name, "detected": hit})

    for tag, label in (("", "mutations, main draw"), ("_heldout", "mutations, held-out draw")):
        d = pd.read_csv(RESULTS / f"paper02_es1_random_faults{record}{tag}.csv")
        cs = pd.read_csv(RESULTS / f"paper02_es1_clause_sources{record}{tag}.csv").set_index("fault_no")
        assert len(cs) == len(d) and (cs.operator.values == d.operator.values).all(), \
            "the clause-source file is not a replay of this mutation stream"
        outside = d.operator.isin(OUTSIDE).to_numpy()
        units = [_units(f, cs.loc[k], bool(o))
                 for k, f, o in zip(d.fault_no, d.failing_checks.fillna(""), outside)]
        for name, allowed in cum.items():
            hit = sum(1 for u in units if allowed & u)
            hit_out = sum(1 for u, o in zip(units, outside) if o and allowed & u)
            hit_in = hit - hit_out
            out.append({"population": label, "n": len(d), "tier": name, "detected": hit})
            out.append({"population": f"{label}, outside the run table",
                        "n": int(outside.sum()), "tier": name, "detected": hit_out})
            out.append({"population": f"{label}, inside the run table",
                        "n": int((~outside).sum()), "tier": name, "detected": hit_in})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--record", default="", help="a sibling arm's file suffix, e.g. _e5")
    args = p.parse_args(argv)
    frame = pd.DataFrame(rows(args.record))
    wide = frame.pivot(index="population", columns="tier", values="detected")
    n = frame.groupby("population").n.first()
    print(pd.concat([n, wide], axis=1).to_string())
    out = RESULTS / f"paper02_es1_tiers{args.record}.csv"
    frame.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(frame)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
