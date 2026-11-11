# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Replay E-S1's random mutations and record what a pandera schema detects.

``es1_random_faults`` draws a mutation stream from a seeded generator and asks
the thirteen relations and the hand-written declarative control about each one.
This driver replays the *same* stream -- same seed, same operator weights, same
order -- and asks the pandera schema of ``es1_pandera_control`` instead, so the
control reported in the paper is a run of the library.

The replay is only valid if it reproduces the stream, so the driver compares the
operator, target and magnitude of every mutation against the committed file and
refuses to write its own output if any row differs.

    python -m experiments.system_paper.es1_pandera_mutations \\
        --faults 300 --seed 7 --skip-coverage --expect-tag _heldout --tag _heldout

The main draw ran the column-coverage sweep first, which also draws from the
generator, so replaying it needs ``--coverage`` and costs what that sweep costs.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import es1_fault_injection as es1
from . import es1_pandera_control as pc
from . import es1_random_faults as es1r

RESULTS = Path("experiments/system_paper/results")


def _families(messages: list[str]) -> str:
    fams: list[str] = []
    for m in messages:
        head = m.split(":", 1)[0]
        if head not in fams:
            fams.append(head)
    return ";".join(sorted(fams))


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
        found = es1.run_check(pc.PANDERA, rec)
        rows.append({"fault_no": k, "operator": name,
                     "target": info["target"], "magnitude": info["magnitude"],
                     "detected_by_pandera": "yes" if found else "no",
                     "pandera_findings": len(found),
                     "pandera_families": _families(found),
                     "seconds": round(time.perf_counter() - t0, 2)})
        if k % 50 == 0:
            log(f"  [{k:>3}/{n_faults}] replayed")
        if rec.derived_dir != clean.derived_dir:
            shutil.rmtree(rec.derived_dir, ignore_errors=True)
    return rows


def verify(rows: list[dict], expected: Path, log=print) -> bool:
    """The replay reproduced the committed stream, mutation for mutation."""
    if not expected.exists():
        log(f"  cannot verify: {expected} is not there")
        return False
    want = pd.read_csv(expected)
    got = pd.DataFrame(rows)
    if len(want) != len(got):
        log(f"  stream length differs: committed {len(want)}, replayed {len(got)}")
        return False
    bad = []
    for col in ("operator", "target", "magnitude"):
        a = want[col].reset_index(drop=True)
        b = got[col].reset_index(drop=True)
        na = pd.to_numeric(a, errors="coerce")
        nb = pd.to_numeric(b, errors="coerce")
        if na.notna().any() or nb.notna().any():
            # A magnitude is a float and is absent for the operators that take
            # none; "nan" against "None" is a difference in how two writers spell
            # the same absence, not a difference in the stream.
            same = ((na.isna() & nb.isna())
                    | np.isclose(na.fillna(0.0), nb.fillna(0.0), rtol=1e-12, atol=0.0))
        else:
            same = a.astype(str) == b.astype(str)
        diff = int((~same).sum())
        if diff:
            bad.append(f"{col}: {diff} of {len(a)} differ")
    if bad:
        log("  stream does NOT match: " + "; ".join(bad))
        return False
    log(f"  stream matches the committed file on operator, target and magnitude "
        f"for all {len(got)} mutations")
    return True


def summarise(got: pd.DataFrame, committed: pd.DataFrame) -> pd.DataFrame:
    """pandera beside the relations and the hand-written control, same mutations."""
    j = committed.merge(got[["fault_no", "detected_by_pandera", "pandera_families"]],
                        on="fault_no", how="inner")
    # The mutations that touch a file the run table does not contain: a per-run
    # artifact or the pinned statistics.
    outside = j.operator.isin(["npz_perturb", "npz_swap", "input_perturb"])
    moved = j.headline_numbers_moved > 0
    def block(name: str, mask) -> dict:
        g = j[mask]
        return {"group": name, "n": len(g),
                "relations": int((g.detected == "yes").sum()),
                "handwritten": int((g.detected_by_declarative == "yes").sum()),
                "pandera": int((g.detected_by_pandera == "yes").sum())}
    return pd.DataFrame([block("all", slice(None) if False else j.index == j.index),
                         block("headline moved", moved),
                         block("outside the run table", outside),
                         block("inside the run table", ~outside)])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sizes", type=int, nargs="+", default=list(es1.SIZES))
    p.add_argument("--faults", type=int, default=300)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--skip-coverage", action="store_true",
                   help="the draw was taken with --no-coverage")
    p.add_argument("--tag", default="")
    p.add_argument("--expect-tag", default="")
    p.add_argument("--record", default="",
                   help="replay against a sibling arm (file suffix, e.g. _e5)")
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    print(f"loading the record for N in {args.sizes} ...", flush=True)
    clean = es1.load_record(tuple(args.sizes), suffix=args.record)
    print(f"  {len(clean.rows)} rows; pandera on the committed record: "
          f"{len(es1.run_check(pc.PANDERA, clean))} findings", flush=True)

    if not args.skip_coverage:
        print("replaying the column-coverage sweep to advance the generator ...", flush=True)
        es1r.coverage(clean, rng, lambda *a, **k: None)

    tmp = Path(tempfile.mkdtemp(prefix="es1pand_"))
    try:
        rows = replay(clean, args.faults, rng, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    expected = RESULTS / f"paper02_es1_random_faults{args.expect_tag}.csv"
    print("verifying the replay:", flush=True)
    if not verify(rows, expected):
        print("refusing to write: the replay is not the committed stream")
        return 1

    out = RESULTS / f"paper02_es1_pandera_mutations{args.tag}.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    print(f"wrote {out} ({len(frame)} mutations)")

    summary = summarise(frame, pd.read_csv(expected))
    sout = RESULTS / f"paper02_es1_pandera_summary{args.tag}.csv"
    summary.to_csv(sout, index=False)
    print(f"wrote {sout}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
