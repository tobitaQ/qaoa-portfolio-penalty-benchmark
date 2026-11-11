# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""A fault the run table cannot show, injected by re-running rather than editing.

E-S1's three operators that reach outside the run table -- a perturbed artifact,
an exchanged pair of artifacts, a perturbed input statistic -- all leave every
row as it was. The headline numbers are a function of the run table alone, so
none of those operators can move one, and the evaluation therefore never
exhibits a fault that the relations catch, the table-only control misses, *and*
a reader would have seen in a published number.

This driver injects that fault the way the project actually met it: the wrong
input produces the rows. It perturbs one cell of the pinned statistics, re-runs
a block of the cross from the perturbed file, and splices the resulting rows and
their reference into the committed record, which is then read beside the
*published* statistics. Inside the table the record is consistent -- each row's
`input_digest` matches its own reference row, because both came from the same
perturbed file -- so an expectation stated over the table has nothing to see.
Only a relation that recomputes the digest from the statistics on disk does.

    python -m experiments.system_paper.es1_input_rerun_fault --delta 1e-6 \\
        --instances 0 1 --seeds 42 43 44 45 46 --penalties Aheur Amargin

Local, no cloud task, no charge. The re-run writes under its own
``--results-tag`` and never touches the committed files.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.finance.data_loader import FinanceDataLoader
from experiments.paper01_qubo_baseline import run_e1_cross as e1
from . import es1_fault_injection as es1f

RESULTS = Path("experiments/system_paper/results")
WINDOW_END = "2026-08-05"


def perturb_statistics(src: Path, dst: Path, delta: float, seed: int,
                       tickers: list[str] | None = None) -> dict:
    """Copy the pinned statistics and move one cell by a relative delta.

    The ticker must be one the re-run instances actually hold. A ticker drawn
    from the whole fifty-name universe reaches a given twelve-name instance
    about a quarter of the time, and when it does not, the re-run reads the same
    numbers as the committed record and the experiment measures nothing. The
    pilot of this driver did exactly that.
    """
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    files = sorted(dst.glob(f"stats_*_{WINDOW_END}_*.csv"))
    if not files:
        raise SystemExit(f"no statistics for the {WINDOW_END} window in {dst}")
    path = files[0]
    frame = pd.read_csv(path, index_col=0, float_precision="round_trip")
    rng = np.random.default_rng(seed)
    # A mean return, not a covariance entry: it is the quantity the provider
    # revises, and it enters every instance that holds the ticker.
    pool = [t for t in (tickers or []) if t in frame.index] or list(frame.index)
    row = str(pool[int(rng.integers(len(pool)))])
    col = str(frame.columns[0])
    before = float(frame.at[row, col])
    after = before * (1.0 + delta) if before != 0.0 else delta
    frame.at[row, col] = after
    frame.to_csv(path, float_format="%.17g")
    return {"file": path.name, "cell": f"{col}[{row}]",
            "before": before, "after": after, "delta": delta}


#: Columns a re-run must reproduce exactly, or the splice is detectable by a
#: uniformity check and the experiment measures the re-run rather than the
#: perturbation. The first attempt at this experiment changed two of them and
#: read the resulting movement as a result.
INVARIANT = ("backend", "precision", "steps", "p_layers", "shots", "optimizer")


def check_injection(clean: es1f.Record, arm: es1f.Record) -> dict:
    """Is the injection the shape it is meant to be? Gate on what can be wrong.

    Two of these abort, because a failure makes the measurement meaningless and
    says so only if someone looks: a changed invariant column, and a block that
    covers an instance only in part. The input and output deltas are reported,
    not gated -- whether a given perturbation moves a result is the thing being
    measured, and a perturbation too small to move one is a finding.
    """
    key = ["n_assets", "instance"]
    bad = []
    for c in INVARIANT:
        if c in arm.rows and c in clean.rows:
            got = set(arm.rows[c].dropna().astype(str))
            want = set(clean.rows[c].dropna().astype(str))
            if not got <= want:
                bad.append(f"{c}: re-run has {sorted(got)}, committed has {sorted(want)}")
    if bad:
        raise SystemExit("the re-run changed something the committed record holds fixed, so the "
                         "splice is visible to a uniformity check:\n  " + "\n  ".join(bad))

    partial = []
    for (n, i), g in arm.rows.groupby(key):
        have = len(clean.rows[(clean.rows.n_assets == n) & (clean.rows.instance == i)])
        if len(g) != have:
            partial.append(f"N={n} inst {i}: re-ran {len(g)} of the instance's {have} rows")
        if g.input_digest.nunique() != 1:
            partial.append(f"N={n} inst {i}: {g.input_digest.nunique()} input digests in one instance")
    if partial:
        raise SystemExit("the block covers an instance only in part, so its remaining rows keep the "
                         "old input digest beside the new reference and the table is inconsistent "
                         "for a reason that is not the fault:\n  " + "\n  ".join(partial))

    j = clean.rows.merge(arm.rows, on="run_id", suffixes=("_c", "_a"))
    inputs_changed = not (j.input_digest_c == j.input_digest_a).all()
    state = [c for c in ("final_expectation", "final_P_F", "final_P_tau_main", "g_off", "g_F",
                         "feasible_shot_fraction") if f"{c}_c" in j]
    out_changed = {c: int((j[f"{c}_c"].astype(float) != j[f"{c}_a"].astype(float)).sum())
                   for c in state}
    return {"rows_rerun": len(arm.rows), "inputs_changed": inputs_changed,
            "rows_with_changed_state": max(out_changed.values()) if out_changed else 0,
            "state_columns_changed": sum(1 for v in out_changed.values() if v)}


def splice(clean: es1f.Record, arm: es1f.Record, label: str) -> es1f.Record:
    """The committed record with the re-run block's rows and reference in it."""
    out = clean.copy(label)
    # Replace in place. Concatenating and re-sorting changes the order the rows
    # are summed in, and a mean over the block then differs from the committed
    # one in the last bit -- a difference this experiment must not manufacture.
    key = ["n_assets", "instance", "penalty_label", "qaoa_seed"]
    idx = out.rows.set_index(key).index
    hit = arm.rows.set_index(key).index
    pos = {k: i for i, k in enumerate(idx)}
    keep = ~idx.isin(hit)
    for k, (_, r) in zip(hit, arm.rows.iterrows()):
        if k in pos:
            out.rows.iloc[pos[k]] = r.reindex(out.rows.columns).values
    rkey = ["n_assets", "instance"]
    ridx = out.reference.set_index(rkey).index
    rpos = {k: i for i, k in enumerate(ridx)}
    for k, (_, r) in zip(arm.reference.set_index(rkey).index, arm.reference.iterrows()):
        if k in rpos:
            out.reference.iloc[rpos[k]] = r.reindex(out.reference.columns).values
    for rid, art in arm.artifacts.items():
        out.artifacts[rid] = art
    return out, int(len(arm.rows)), int(keep.size - keep.sum())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--delta", type=float, default=1e-6)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--instances", type=int, nargs="+", default=[0, 1])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--penalties", nargs="+", default=None,
                   help="default: every setting the spec gives the instance, which is what\nkeeps the spliced block internally consistent -- re-running a subset leaves the\ninstance's other rows carrying the old input digest beside the new reference")
    p.add_argument("--seed", type=int, default=20261109, help="which cell to perturb")
    p.add_argument("--backend", default="lightning_gpu",
                   help="must match the committed record, or the splice is detectable "
                        "by a uniformity check and the experiment measures the backend")
    p.add_argument("--precision", default="single", choices=["single", "double"])
    p.add_argument("--tag", default=None)
    p.add_argument("--skip-rerun", action="store_true",
                   help="reuse an existing tagged arm instead of running it again")
    p.add_argument("--ladder", default="",
                   help="suffix of the result file; a second instance pair gets its own ladder "
                        "(e.g. _pair23) so the first pair's file is not rewritten")
    args = p.parse_args(argv)

    tag = args.tag or f"inputfault{args.delta:g}".replace("-", "m").replace(".", "")
    work = RESULTS / "_input_rerun" / tag
    work.mkdir(parents=True, exist_ok=True)
    clean_dir = FinanceDataLoader.DERIVED_DIR
    perturbed = work / "derived"

    shared = set(es1f.instance_tickers(args.n, args.instances[0]))
    for i in args.instances[1:]:
        shared &= set(es1f.instance_tickers(args.n, i))
    if not shared:
        raise SystemExit("the chosen instances share no ticker; pick instances that do, "
                         "or perturb one statistic per instance")
    info = perturb_statistics(clean_dir, perturbed, args.delta, args.seed, sorted(shared))
    info["shared_tickers"] = len(shared)
    info["n"] = args.n
    info["instances"] = " ".join(str(i) for i in args.instances)
    # The delta that matters is the one the arm was run with, not the one on
    # this command line: with --skip-rerun the two can differ, and re-evaluating
    # a ladder once wrote every row's delta as the last value typed.
    if args.skip_rerun and "--delta" not in sys.argv:
        info["delta"] = None
    print(f"perturbed {info['cell']} in {info['file']}: "
          f"{info['before']!r} -> {info['after']!r} (delta {args.delta:g})", flush=True)

    if not args.skip_rerun:
        # Both directories, as es1_fault_injection._loader_dirs does. The price
        # cache is read before the derived statistics, so pointing only
        # DERIVED_DIR at the perturbed copy leaves the re-run reading the real
        # prices and reproducing the committed inputs exactly -- which is what
        # the first attempt at this experiment did, silently.
        with es1f._loader_dirs(cache_dir=None, derived_dir=perturbed):
            argv2 = ["--n", str(args.n),
                     "--instances", *[str(i) for i in args.instances],
                     "--seeds", *[str(s) for s in args.seeds],
                     *(["--penalties", *args.penalties] if args.penalties else []),
                     "--backend", args.backend, "--precision", args.precision,
                     "--results-tag", tag]
            print("re-running:", " ".join(argv2), flush=True)
            t0 = time.perf_counter()
            rc = e1.main(argv2)
            print(f"driver exit {rc} in {time.perf_counter() - t0:.0f} s", flush=True)
            if rc != 0:
                return rc

    print("loading the committed record and the re-run arm ...", flush=True)
    clean = es1f.load_record((args.n,))
    arm = es1f.load_record((args.n,), suffix=f"_{tag}", label="arm")
    gate = check_injection(clean, arm)
    print(f"  injection: {gate['rows_rerun']} rows re-run; inputs changed "
          f"{gate['inputs_changed']}; state changed on up to "
          f"{gate['rows_with_changed_state']} rows in {gate['state_columns_changed']} columns",
          flush=True)
    if not gate["inputs_changed"]:
        print("  WARNING: the re-run read the same inputs; the perturbation did not reach it",
              flush=True)
    faulted, n_new, n_kept = splice(clean, arm, "input_rerun")
    print(f"  spliced {n_new} rows into {len(clean.rows)} "
          f"({n_kept} of the committed rows replaced)", flush=True)
    return evaluate(clean, faulted, info, tag, gate, ladder=args.ladder)


def evaluate(clean: es1f.Record, faulted: es1f.Record, info: dict, tag: str,
             gate: dict | None = None, ladder: str = "") -> int:
    from . import es1_pandera_control as pc
    base = es1f.headline(clean)
    head = es1f.headline(faulted)
    # "Moved" is bitwise inequality, so a last-bit difference counts. How far it
    # moved is a separate question and the paper reports both: three numbers
    # that moved by 10^-5 are not three numbers that changed a conclusion.
    moved = {k: (base[k], head.get(k)) for k in base
             if not (pd.isna(base[k]) and pd.isna(head.get(k))) and base[k] != head.get(k)}
    rels = [abs(b - a) / abs(a) for a, b in moved.values() if a]
    largest = max(rels) if rels else 0.0

    rel = {c.name: es1f.run_check(c, faulted) for c in es1f.CHECKS}
    firing = [n for n, f in rel.items() if f]
    decl = es1f.run_check(es1f.DECLARATIVE, faulted)
    pand = es1f.run_check(pc.PANDERA, faulted)
    # The routine aggregation gate on the same spliced record, reading the
    # *published* statistics (the loader's default derived directory): what a
    # reader who only aggregates would have been told.
    from src import record_checks
    t0 = time.perf_counter()
    agg = record_checks.check_record(faulted.rows, faulted.reference)
    gate_seconds = time.perf_counter() - t0
    agg_warn = record_checks.check_record_warnings(faulted.rows, faulted.reference)
    print(f"aggregation gate ({len(record_checks.RECORD_CHECKS)} routines): "
          f"{sum(len(v) for v in agg.values())} finding(s) in {gate_seconds:.2f} s; "
          f"{sum(len(v) for v in agg_warn.values())} warning(s)")

    print()
    print(f"headline numbers moved: {len(moved)} of {len(base)}"
          f"; largest relative change {largest:.2e}")
    for k, (a, b) in list(moved.items())[:6]:
        print(f"    {k}: {a!r} -> {b!r}")
    print(f"relations firing ({len(firing)}): {firing}")
    print(f"hand-written declarative control: {len(decl)} finding(s) {decl[:2]}")
    print(f"pandera schema:                   {len(pand)} finding(s) {pand[:2]}")

    # Only a run that passed both gates reaches here, so the file holds the
    # ladder and not the attempts: six earlier ones measured a backend change,
    # a cache that was read instead of the perturbed statistics, or a block that
    # covered an instance in part. Those belong in the operational log.
    out_path = RESULTS / f"paper02_es1_input_rerun{ladder}.csv"
    delta = info["delta"]
    if delta is None:
        prev = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
        hit = prev[prev.tag == tag] if "tag" in prev else prev
        if not len(hit):
            raise SystemExit(f"--skip-rerun on {tag}, which has no recorded delta to reuse")
        delta = float(hit.delta.iloc[0])
    row = {"tag": tag, "delta": delta, "perturbed_cell": info["cell"],
           "n": info.get("n", ""), "instances": info.get("instances", ""),
           **{k: v for k, v in (gate or {}).items()},
           "rows_replaced": int(len(faulted.rows[faulted.rows.run_id.isin(
               set(faulted.rows.run_id) - set(clean.rows.run_id))])),
           "headline_total": len(base), "headline_moved": len(moved),
           "largest_relative_change": round(largest, 12),
           "relations_firing": ";".join(firing),
           "detected_by_relations": "yes" if firing else "no",
           "detected_by_declarative": "yes" if decl else "no",
           "detected_by_pandera": "yes" if pand else "no",
           "gate_routines": len(record_checks.RECORD_CHECKS),
           "gate_findings": sum(len(v) for v in agg.values()),
           "gate_input_digest_findings": len(agg.get("input-digest", [])),
           "gate_refuses": "yes" if agg else "no",
           "gate_seconds": round(gate_seconds, 2),
           "gate_warnings": sum(len(v) for v in agg_warn.values())}
    out = out_path
    frame = pd.DataFrame([row])
    if out.exists():
        old = pd.read_csv(out)
        frame = pd.concat([old[old.tag != tag], frame], ignore_index=True)
    frame = frame.sort_values("delta").reset_index(drop=True)
    frame.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
