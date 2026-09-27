# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Check every data-backed table of the system paper against the file it came from.

The paper says a number in a manuscript should be bound to a record, and the
companion package has carried such a checker for its own tables since before
either paper was written (``papers/check_tables.py``, 2,578 cells). This is the
same instrument turned on this paper: it parses Tables 3-11 and 13 out of the
Markdown and compares each cell with the CSV the harness wrote.

Three tables are prose (Table 1 the injected faults, Table 2 the
misconfigurations, Table 12 the seven defects); they are listed as unbound
rather than silently skipped, and so is any cell whose source is not a committed
file (the project's billing-console total in Table 13).

Usage:
    python papers/system_paper/check_tables.py     # exit 1 if any cell disagrees
"""

from __future__ import annotations

import pathlib
import re

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "experiments" / "system_paper" / "results"
SRC = (ROOT / "papers" / "system_paper" / "system_paper_draft.md").read_text(encoding="utf-8")

OK: list[str] = []
BAD: list[str] = []
UNBOUND: list[str] = []


def table(n) -> list[dict]:
    m = re.search(rf"\*\*Table {n}\. ", SRC)
    assert m, f"Table {n} not found"
    rows, started = [], False
    for line in SRC[m.end():].split("\n"):
        if line.startswith("|"):
            rows.append(line)
            started = True
        elif started:
            break
    cells = lambda l: [x.strip().replace("\x00", "|") for x in l.replace("\\|", "\x00").strip("|").split("|")]
    hdr = cells(rows[0])
    out = []
    for line in rows[1:]:
        if re.match(r"^\|[\s:|-]+\|$", line):
            continue
        c = cells(line)
        if len(c) == len(hdr):
            out.append(dict(zip(hdr, c)))
    return out


def num(x):
    """The first number in a cell, with the draft's markup and separators removed."""
    if x is None:
        return None
    x = re.sub(r"\*\*|[,$%×]|\s|†|✓|✗", "", str(x)).replace("−", "-").replace("⁻", "-")
    m = re.match(r"^-?\d+\.?\d*", x)
    return float(m.group(0)) if m else None


SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def sci(cell):
    """Every number in a cell, reading the draft's 1.1·10⁻⁴ as 1.1e-4."""
    text = str(cell).replace("**", "").replace(",", " ").translate(SUP)
    out = []
    for m in re.finditer(r"(-?\d+\.?\d*)(?:\s*(?:·|x|×)\s*10(-?\d+))?", text):
        if not m.group(0).strip():
            continue
        mant = float(m.group(1))
        out.append(mant * 10.0 ** int(m.group(2)) if m.group(2) is not None else mant)
    return out


def chk(t, label, paper, csv, tol=0.0):
    good = paper is not None and csv is not None and abs(paper - csv) <= tol
    (OK if good else BAD).append(f"Table {t} {label}: paper={paper} file={csv}")


def chks(t, label, paper, csv):
    good = str(paper).strip() == str(csv).strip()
    (OK if good else BAD).append(f"Table {t} {label}: paper={paper!r} file={csv!r}")


def unbound(t, why):
    UNBOUND.append(f"Table {t}: {why}")


# ---------------------------------------------------------------- Table 3
d = pd.read_csv(R / "paper02_es1_detection_matrix.csv")
CHECK_NAMES = ["schema", "manifest", "penalty", "reference-row", "input-digest", "reference-input",
               "scorer", "row-npz", "npz-internal", "identities", "state-batch", "seed-distinct",
               "state-recompute"]
for r in table(3):
    row = d[d.fault == r["Fault"]]
    assert len(row) == 1, r["Fault"]
    row = row.iloc[0]
    fails = [c for c in CHECK_NAMES if row[c] == "FAIL"]
    first = row.first_failing_check if isinstance(row.first_failing_check, str) else ""
    chks(3, f"{r['Fault']} first", r["First failing check (stage)"].split(" (")[0],
         first or "—")
    also = [x.strip() for x in r["Also failing"].split(",") if x.strip() and x.strip() != "—"]
    chks(3, f"{r['Fault']} also", sorted(also), sorted(n for n in fails if n != first))
    naive = "yes" if "yes" in r["Naive"] else ("—" if r["Fault"] == "clean" else "no")
    chks(3, f"{r['Fault']} naive", naive,
         "—" if r["Fault"] == "clean" else row.detected_by_naive)
    fam = row.declarative_families if isinstance(row.declarative_families, str) and row.declarative_families else \
        ("—" if r["Fault"] == "clean" else "no")
    chks(3, f"{r['Fault']} declarative", r["Declarative"], fam)
    chk(3, f"{r['Fault']} moved", num(r["Moved"]), float(row.headline_numbers_moved))

# ---------------------------------------------------------------- Table 4
COV4 = ("paper02_es1_column_coverage_before.csv", "paper02_es1_column_coverage_pass2.csv",
        "paper02_es1_column_coverage.csv")
scored = lambda f: int((f.covered != "n/a (column empty)").sum())
for r, name in zip(table(4), COV4):
    frame = pd.read_csv(R / name)
    hit, of = [num(x) for x in r["Covered"].split("/")]
    chk(4, f"pass {r['Pass']} covered", hit, float((frame.covered == "yes").sum()))
    chk(4, f"pass {r['Pass']} of", of, float(scored(frame)))
    alone = ((frame.covered == "yes") & (frame.failing_checks == "identities")).sum()
    chk(4, f"pass {r['Pass']} identities alone", num(r["Covered by `identities` alone"]), float(alone))

# ---------------------------------------------------------------- Table 5
s5 = pd.read_csv(R / "paper02_es1_random_summary.csv")
op = s5[s5.group == "operator"].set_index("value")
KEY5 = {"cell perturbed": "cell_perturb", "cell replaced by another row's": "cell_from_other",
        "row's metrics copied onto another row": "row_metrics_copy", "row deleted": "row_delete",
        "penalty labels of a block permuted": "label_shuffle",
        "reference cell perturbed": "reference_perturb", "artifact element perturbed": "npz_perturb",
        "two artifacts exchanged": "npz_swap", "derived statistic perturbed": "input_perturb",
        "column dropped": "column_drop"}
allr = s5[s5.group == "all"].iloc[0]
held = pd.read_csv(R / "paper02_es1_random_summary_heldout.csv")
heldall = held[held.group == "all"].iloc[0]
for r in table(5):
    label = r["Operator"].replace("**", "")
    if label.startswith("all, held-out"):
        g = heldall
    elif label == "all":
        g = allr
    else:
        key = next((v for k, v in KEY5.items() if label.startswith(k)), None)
        assert key, label
        g = op.loc[key]
    chk(5, f"{label} n", num(r["n"]), float(g.n))
    chk(5, f"{label} detected", num(r["Detected"]), float(g.detected))
    chk(5, f"{label} declarative", num(r["Declarative"]), float(g.detected_declarative))
    chk(5, f"{label} moved", num(r["Moved a headline"]), float(g.moved_headline))
    chk(5, f"{label} moved and missed", num(r["Moved and missed"]), float(g.moved_and_undetected))

# ---------------------------------------------------------------- Table 6
t6 = pd.read_csv(R / "paper02_es2_task_counts.csv")
for r in table(6):
    n = int(num(r["N"]))
    mock = t6[(t6.n == n) & (t6.device == "mock")].iloc[0]
    chk(6, f"N={n} stand-in", num(r["stand-in (`default.qubit`, no derivatives)"]), float(mock.tapes_per_step))
    chk(6, f"N={n} formula", num(r["formula"]), float(mock.expected_per_step))
    tr = t6[(t6.n == n) & (t6.device == "param_shift transform")]
    if len(tr):
        chk(6, f"N={n} transform", num(r["transform"]), float(tr.iloc[0].tapes_total))
    bl = t6[(t6.n == n) & (t6.device == "braket_local")]
    if len(bl) and num(r["`braket.local.qubit`"]) is not None:
        chk(6, f"N={n} braket.local", num(r["`braket.local.qubit`"]), float(bl.iloc[0].tapes_per_step))

# ---------------------------------------------------------------- Table 7
t7 = pd.read_csv(R / "paper02_es2_guard_stages.csv")
ORDER7 = [("qpu_optimizer", 8), ("qpu_optimizer", 20), ("trainable_sampling_sv1", 8),
          ("trainable_sampling_sv1", 20), ("trainable_sampling_qpu", 8), ("trainable_sampling_qpu", 20),
          ("qpu_sampling_1000_shots", 8), ("untagged_cloud_run", 8)]
COLS7 = {"none": "none", "cap only": "cap", "structural": "structural", "both": "both"}
for r, (mis, n) in zip(table(7), ORDER7):
    for col, cfg in COLS7.items():
        g = t7[(t7.misconfiguration == mis) & (t7.n == n) & (t7.configuration == cfg)]
        if not len(g):
            continue
        g = g.iloc[0]
        cell = r[col].replace("**", "")
        # The provenance mark of the caption: a dagger where the cap acts inside
        # the service (never observed, the device stayed offline), a double
        # dagger where the cloud observation covers the cell.
        mark, cell = (cell[0], cell[1:].strip()) if cell.strip()[:1] in "†‡" else ("", cell)
        observed = set(pd.read_csv(R / "paper02_es2_cloud_observation.csv").step)
        want = "†" if g.stage == "service" else (
            "‡" if (mis == "trainable_sampling_sv1" and n == 8
                         and {"sv1-sampling-trainable", "sv1-sampling-frozen"} <= observed) else "")
        chks(7, f"{mis} N={n} {cfg} provenance", mark or "(none)", want or "(none)")
        head = cell.split(";")[0].strip()
        if mis == "untagged_cloud_run":
            # The cell says what happened to the baseline, which is the point of the row.
            kept = "intact" in cell
            chks(7, f"{mis} N={n} {cfg} baseline", "kept" if kept else "overwritten",
                 "overwritten" if g.baseline_overwritten == "yes" else "kept")
        else:
            chks(7, f"{mis} N={n} {cfg} stage", head, g.stage)
        m = re.search(r"(\d[\d,]*)\s*(?:task|tasks)?\s*(?:;|$)", cell.split(";", 1)[1] if ";" in cell else "")
        if m:
            chk(7, f"{mis} N={n} {cfg} tasks", float(m.group(1).replace(",", "")), float(g.tasks_billed))
        d7 = re.search(r"\$([\d,]+(?:\.\d+)?)", cell)
        if d7:
            chk(7, f"{mis} N={n} {cfg} usd", float(d7.group(1).replace(",", "")), float(g.usd_billed), 0.6)

# ---------------------------------------------------------------- Table 8
rep = pd.read_csv(R / "paper02_es2_cost_control_repetitions.csv").set_index("run")
sm8 = pd.read_csv(R / "paper02_es2_cost_control_repetitions_summary.csv").set_index("quantity")
F8 = {"Tasks created": "tasks_created",
      "Created past the limit": "tasks_created_after_limit_crossed",
      "Limit → alarm": "seconds_limit_to_alarm",
      "Created past the alarm": "tasks_created_after_alarm",
      "Alarm → first denial": "seconds_alarm_to_denial",
      "Created past the first denial": "tasks_created_after_first_denial"}
for r in table(8):                      # one row per quantity, one column per repetition
    field = F8[r["Quantity"]]
    for run in (1, 2, 3, 4, 5):
        chk(8, f"run {run} {field}", num(r[str(run)]), float(rep.loc[run, field]))
    chk(8, f"median {field}", num(r["median"]), float(sm8.loc[field, "median"]))

# ---------------------------------------------------------------- Table 9
t9 = pd.read_csv(R / "paper02_es3_steps.csv").set_index("step")
t9b = pd.read_csv(R / "paper02_es3_steps_v11.csv").set_index("step")
# The v1.0 rows come from the full pass, the "(v1.1)" block from the pass that
# clones the release this paper ships with.
ORDER9 = [(t9, ["clone"]), (t9, ["venv", "pip"]), (t9, ["tests"]), (t9, ["level1-figures"]),
          (t9, ["level1-tables"]), (t9, ["level2"]), (t9, ["level3-block"]),
          (t9, ["interrupt-kill"]), (t9, ["interrupt-resume"]),
          (t9, ["offline-no-derived"]), (t9, ["missing-npz"]),
          (t9b, ["clone", "venv", "pip"]), (t9b, ["tests"]),
          (t9b, ["level1-figures", "level1-tables"]), (t9b, ["level2"]),
          (t9b, ["status-after-kill"]), (t9b, ["status-after-resume"])]
for r, (frame, steps) in zip(table(9), ORDER9):
    cell = r["Wall-clock"]
    got = [float(x) for x in re.findall(r"(\d+\.?\d*)\s*s", cell)]
    for value, step in zip(got, steps):
        chk(9, f"{r['Level']} {step} seconds", value, round(float(frame.loc[step, "seconds"])), 0.51)

# ---------------------------------------------------------------- Table 10
t10 = pd.read_csv(R / "paper02_es3_column_agreement.csv")
NAME10 = {"final_expectation": ["final_expectation"],
          "expectation_at_final_angles": ["expectation_at_final_angles"],
          "final_P_F": ["final_P_F"], "final_P_opt, final_P_τ": ["final_P_opt", "final_P_tau_main"],
          "initial_P_F": ["initial_P_F"], "final_Q_τ(1000)": ["final_Q_tau_main_1000"],
          "feasible_shots": ["feasible_shots"], "shots_optimal": ["shots_optimal"],
          "initial_angles": ["initial_angles"], "final_angles": ["final_angles"],
          "best_all_E, g_F, g_off": ["best_all_E", "g_F", "g_off"]}
for r in table(10):
    cols = NAME10.get(r["Column"])
    if cols is None:
        continue
    diffs = sci(r["Max abs. difference"]) if r["Max abs. difference"] != "—" else []
    for i, col in enumerate(cols):
        g = t10[t10.column == col].iloc[0]
        hit, of = [num(x) for x in r["Identical rows"].split("/")]
        chk(10, f"{col} identical", hit, float(g.identical_rows))
        chk(10, f"{col} rows", of, float(g.rows))
        if pd.notna(g.max_abs_diff) and len(diffs) > (i if len(diffs) > 1 else 0):
            paper_diff = diffs[i if len(diffs) > 1 else 0]
            chk(10, f"{col} max diff", paper_diff, float(g.max_abs_diff),
                abs(float(g.max_abs_diff)) * 0.06 + 1e-12)

# ---------------------------------------------------------------- Table 11
t11 = pd.read_csv(R / "paper02_level1_machines.csv")
for r, (_, g) in zip(table(11), t11.iterrows()):
    p, s_ = re.match(r"(\d+) passed, (\d+) skipped", r["Test suite"]).groups()
    chk(11, f"{g.machine} passed", float(p), float(g.passed))
    chk(11, f"{g.machine} skipped", float(s_), float(g.skipped))
    chk(11, f"{g.machine} cells", num(r["`check_tables.py`"]), float(g.cells_agree))
    chks(11, f"{g.machine} python", r["Python"], str(g.python))

# ---------------------------------------------------------------- Table 13
SV1_TASK_USD = 3.0 / 60 * 0.075
obs = pd.read_csv(R / "paper02_es2_cloud_observation.csv")
sv1_tasks = sum(int(m.group(1)) for m in
                (re.match(r"(\d+) task\(s\) created", str(o)) for o in obs.outcome) if m)
cc = sum(int((pd.read_csv(R / f"paper02_es2_cost_control_tasks_{t}.csv").outcome == "created").sum())
         for t in ("as_deployed", "run1", "run2", "run3", "run4", "run5"))
rows13 = table(13)
for r in rows13:
    what = r["What"].replace("**", "")
    if what.startswith("the sampling-fallback"):
        chk(13, "fallback tasks", num(r["Tasks"]), float(sv1_tasks))
        chk(13, "fallback usd", num(r["Cost"]), round(sv1_tasks * SV1_TASK_USD, 2), 0.005)
    elif what.startswith("the six account-level"):
        chk(13, "account-level tasks", num(r["Tasks"]), float(cc))
        chk(13, "account-level usd", num(r["Cost"]), round(cc * SV1_TASK_USD, 2), 0.005)
    elif what.startswith("this paper"):
        chk(13, "total tasks", num(r["Tasks"]), float(sv1_tasks + cc))
        chk(13, "total usd", num(r["Cost"]),
            round(sv1_tasks * SV1_TASK_USD, 2) + round(cc * SV1_TASK_USD, 2), 0.005)
unbound(13, "the project's $42.59 and the numerical study's $37.90 are the provider's billing console")

for n, why in ((1, "the injected faults, described in prose"),
               (2, "the misconfigurations, described in prose"),
               (12, "the seven defects and the relation that now holds each")):
    unbound(n, why)

# ------------------------------------------------- the figures, from the same files
import hashlib, os, shutil, subprocess, sys, tempfile  # noqa: E402

FIGS = ["fig_record_relations.pdf", "fig_record_relations.png",
        "fig_guard_stages.pdf", "fig_guard_stages.png"]
digest = lambda p_: hashlib.sha256(p_.read_bytes()).hexdigest()
tmp = pathlib.Path(tempfile.mkdtemp(prefix="sysfig_"))
try:
    env = dict(os.environ, SYSTEM_PAPER_FIGDIR=str(tmp))
    r = subprocess.run([sys.executable, str(ROOT / "papers" / "system_paper" / "make_figures.py")],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        BAD.append(f"figures: make_figures.py exited {r.returncode}: {r.stderr.strip()[-200:]}")
    else:
        for name in FIGS:
            here = ROOT / "papers" / "system_paper" / "figures" / name
            there = tmp / name
            if not there.exists():
                BAD.append(f"figure {name}: not regenerated")
            elif digest(here) == digest(there):
                OK.append(f"figure {name}: byte-identical")
            else:
                BAD.append(f"figure {name}: regenerates to a different file")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# A [[placeholder]] left in the draft is an identifier nobody has filled in yet.
for m in re.finditer(r"\[\[([^\]]+)\]\]", SRC):
    BAD.append(f"placeholder still in the draft: [[{m.group(1)}]]")

print(f"{len(OK)} cell(s) agree with the result files, {len(BAD)} disagree, "
      f"{len(UNBOUND)} table(s) or cell(s) not bound to one")
for b in BAD:
    print("  x", b)
for u in UNBOUND:
    print("  -", u)
raise SystemExit(1 if BAD else 0)
