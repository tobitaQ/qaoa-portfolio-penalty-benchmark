# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""What each firing clause of ``identities`` read, on every committed mutation and fault.

``identities`` is one check name over three kinds of comparison: a column
recomputed from other columns of the same table, a column compared with the
per-run artifact, and a portfolio metric recomputed from the pinned statistics.
The detection matrix and the mutation files record the check's name only, so a
tier count by *what was read* cannot be made from them for this check. This
driver replays the committed mutation streams (seeded, as
``es1_pandera_mutations`` does) and the ten injected faults, runs ``identities``
alone on each, and records how many findings carry each source prefix.

    python -m experiments.system_paper.es1_clause_sources --record _e5
    python -m experiments.system_paper.es1_clause_sources --record _e5 --seed 7 \\
        --skip-coverage --tag _heldout --expect-tag _heldout

The replay refuses to write unless every mutation's operator, target and
magnitude match the committed stream.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.system_paper import es1_fault_injection as es1
from experiments.system_paper import es1_random_faults as es1r

RESULTS = es1.RESULTS_DIR
SOURCES = ("table", "input", "artifact")


def sources(findings: list[str]) -> dict[str, int]:
    """Findings by the source their clause read; a check that raised (a column
    it reads is gone) is counted apart, and the tier count places it with the
    table, since the missing column is the table's."""
    out = {f"identities_{s}": sum(1 for m in findings if m.startswith(f"[{s}]")) for s in SOURCES}
    out["identities_raised"] = sum(1 for m in findings if m.startswith("check raised"))
    return out


def replay(clean, n_faults: int, rng, tmp: Path, log=print) -> list[dict]:
    names = list(es1r.OPERATORS)
    probs = np.array([es1r.WEIGHTS[n] for n in names], dtype=float)
    probs /= probs.sum()
    rows: list[dict] = []
    k = 0
    while len(rows) < n_faults:
        name = str(rng.choice(names, p=probs))
        result = es1r.OPERATORS[name](clean, rng, tmp)
        if result is None:
            continue
        rec, info = result
        k += 1
        t0 = time.perf_counter()
        found = es1.run_check(es1.CHECKS[[c.name for c in es1.CHECKS].index("identities")], rec)
        rows.append({"fault_no": k, "operator": name, "target": info["target"],
                     "magnitude": info["magnitude"], **sources(found),
                     "seconds": round(time.perf_counter() - t0, 2)})
        if k % 50 == 0:
            log(f"  [{k:>3}/{n_faults}] replayed", flush=True)
        if rec.derived_dir != clean.derived_dir:
            shutil.rmtree(rec.derived_dir, ignore_errors=True)
    return rows


def verify(rows: list[dict], expected: Path, log=print) -> bool:
    if not expected.exists():
        log(f"  cannot verify: {expected} is not there")
        return False
    want = pd.read_csv(expected)
    got = pd.DataFrame(rows)
    if len(want) != len(got):
        log(f"  stream length differs: committed {len(want)}, replayed {len(got)}")
        return False
    for col in ("operator", "target", "magnitude"):
        a = want[col].reset_index(drop=True).astype(str)
        b = got[col].reset_index(drop=True).astype(str)
        if col == "magnitude":
            a = pd.to_numeric(want[col], errors="coerce").reset_index(drop=True)
            b = pd.to_numeric(got[col], errors="coerce").reset_index(drop=True)
            same = (a.fillna(-1) == b.fillna(-1))
        else:
            same = a == b
        if not same.all():
            log(f"  {col} differs on {int((~same).sum())} mutations; first at fault_no "
                f"{int(got.fault_no[~same].iloc[0])}")
            return False
    return True


def faults(clean, tmp: Path, log=print) -> list[dict]:
    ident = es1.CHECKS[[c.name for c in es1.CHECKS].index("identities")]
    rows = []
    for fault in es1.FAULTS:
        d = tmp / fault.name
        d.mkdir()
        rec = fault.apply(clean, d)
        found = es1.run_check(ident, rec)
        rows.append({"fault": fault.name, **sources(found)})
        log(f"  {fault.name:<20} {sources(found)}")
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sizes", type=int, nargs="+", default=list(es1.SIZES))
    p.add_argument("--record", default="", help="a sibling arm's file suffix, e.g. _e5")
    p.add_argument("--faults", type=int, default=300)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--skip-coverage", action="store_true", help="the draw was taken with --no-coverage")
    p.add_argument("--tag", default="")
    p.add_argument("--expect-tag", default="")
    p.add_argument("--injected", action="store_true",
                   help="the ten injected faults instead of a mutation stream")
    args = p.parse_args(argv)

    print(f"loading the E1{args.record} record for N in {args.sizes} ...", flush=True)
    clean = es1.load_record(tuple(args.sizes), suffix=args.record)
    tmp = Path(tempfile.mkdtemp(prefix="es1clause_"))
    try:
        if args.injected:
            rows = faults(clean, tmp)
            out = RESULTS / f"paper02_es1_clause_sources{args.record}_faults.csv"
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"wrote {out} ({len(rows)})")
            return 0
        rng = np.random.default_rng(args.seed)
        if not args.skip_coverage:
            print("replaying the column-coverage sweep to advance the generator ...", flush=True)
            es1r.coverage(clean, rng, lambda *a, **k: None)
        rows = replay(clean, args.faults, rng, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    expected = RESULTS / f"paper02_es1_random_faults{args.record}{args.expect_tag}.csv"
    if not verify(rows, expected):
        print("refusing to write: the replay is not the committed stream")
        return 1
    out = RESULTS / f"paper02_es1_clause_sources{args.record}{args.tag}.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    print(f"wrote {out} ({len(frame)} mutations)")
    fired = frame[[f"identities_{s}" for s in SOURCES]] > 0
    print(f"identities fired on {int(fired.any(axis=1).sum())}: table-only "
          f"{int((fired.identities_table & ~fired.identities_input & ~fired.identities_artifact).sum())}, "
          f"any artifact {int(fired.identities_artifact.sum())}, any input {int(fired.identities_input.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
