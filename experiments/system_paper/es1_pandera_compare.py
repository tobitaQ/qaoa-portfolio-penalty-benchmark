# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Run E-S1's ten injected faults past both declarative controls.

``es1_fault_injection``'s control states the declarative vocabulary in
hand-written pandas; ``es1_pandera_control`` states the same expectations as a
pandera schema and validates with the library. This driver puts the committed
record and each of the ten faults past both, so the paper can report what the
library actually detects rather than what a re-implementation of its vocabulary
detects.

It touches none of the committed result files: it reads the record and writes
``paper02_es1_pandera_compare.csv``.

    python -m experiments.system_paper.es1_pandera_compare
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import pandas as pd

from . import es1_fault_injection as es1
from . import es1_pandera_control as pc

OUT = es1.RESULTS_DIR / "paper02_es1_pandera_compare.csv" \
    if hasattr(es1, "RESULTS_DIR") else None


def _families(messages: list[str]) -> str:
    fams: list[str] = []
    for m in messages:
        head = m.split(":", 1)[0]
        if head not in fams:
            fams.append(head)
    return ";".join(sorted(fams))


def run(sizes=es1.SIZES, log=print, record: str = "") -> pd.DataFrame:
    log(f"loading the E1{record} record for N in {list(sizes)} ...")
    clean = es1.load_record(sizes, suffix=record)
    log(f"  {len(clean.rows)} rows, {len(clean.artifacts)} artifacts")

    rows = []
    tmp_root = Path(tempfile.mkdtemp(prefix="es1_pandera_"))
    for fault in [None] + es1.FAULTS:
        if fault is None:
            rec = clean
            desc = "the committed record"
        else:
            tmp = tmp_root / fault.name
            tmp.mkdir()
            rec = fault.apply(clean, tmp)
            desc = fault.description

        t0 = time.perf_counter()
        hand = es1.run_check(es1.DECLARATIVE, rec)
        t_hand = time.perf_counter() - t0
        t0 = time.perf_counter()
        lib = es1.run_check(pc.PANDERA, rec)
        t_lib = time.perf_counter() - t0

        rows.append({
            "fault": rec.label,
            "description": desc,
            "detected_by_handwritten": "yes" if hand else "no",
            "detected_by_pandera": "yes" if lib else "no",
            "agree": "yes" if bool(hand) == bool(lib) else "NO",
            "handwritten_findings": len(hand),
            "pandera_findings": len(lib),
            "handwritten_families": _families(hand),
            "pandera_families": _families(lib),
            "handwritten_seconds": round(t_hand, 2),
            "pandera_seconds": round(t_lib, 2),
        })
        log(f"  {rec.label:<22} hand {len(hand):>3} ({t_hand:>5.1f} s)   "
            f"pandera {len(lib):>3} ({t_lib:>5.1f} s)   "
            f"{'agree' if bool(hand) == bool(lib) else 'DISAGREE'}")

    frame = pd.DataFrame(rows)
    faults_only = frame[frame.fault != "clean"]
    log("")
    log(f"detected of {len(faults_only)}: "
        f"hand-written {int((faults_only.detected_by_handwritten == 'yes').sum())}, "
        f"pandera {int((faults_only.detected_by_pandera == 'yes').sum())}")
    log(f"false positives on the committed record: "
        f"hand-written {int(frame.loc[frame.fault == 'clean', 'handwritten_findings'].iloc[0])}, "
        f"pandera {int(frame.loc[frame.fault == 'clean', 'pandera_findings'].iloc[0])}")
    disagree = frame[frame.agree == "NO"]
    if len(disagree):
        log(f"DISAGREEMENTS: {disagree.fault.tolist()}")
    return frame


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=list(es1.SIZES))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--record", default="",
                   help="compare on a sibling arm (file suffix, e.g. _e5)")
    args = p.parse_args(argv)
    frame = run(tuple(args.sizes), record=args.record)
    out = (Path(f"experiments/system_paper/results/paper02_es1_pandera_compare{args.record}.csv")
           if args.out is None else args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out} ({len(frame)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
