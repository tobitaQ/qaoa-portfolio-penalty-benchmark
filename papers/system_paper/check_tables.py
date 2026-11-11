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

import csv
import pathlib
import re
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "experiments" / "system_paper" / "results"
SRC = (ROOT / "papers" / "system_paper" / "system_paper_draft.md").read_text(encoding="utf-8")
#: Where ``--csv`` writes when no path follows it; ``--csv <path>`` writes elsewhere.
DATASHEET = ROOT / "papers" / "system_paper" / "table_data.csv"

OK: list[str] = []
BAD: list[str] = []
UNBOUND: list[str] = []
#: One row per comparison, for ``--csv``: the datasheet of every number in the
#: paper that is bound to a file, and of every table and cell that is not.
ROWS: list[dict] = []

#: The result files each table is read from. ``chk``/``chks`` name them in the
#: datasheet so that a reader can go from a cell to the file without reading
#: this script. Kept beside the checks rather than derived from them, because a
#: check may read a file only to index another.
SOURCES = {
    3: ("paper02_es1_detection_matrix.csv", "paper02_es1_pandera_compare.csv"),
    6: ("paper02_es1_input_rerun.csv", "paper02_es1_input_rerun_pair23.csv"),
    "S2": ("paper02_es1_column_coverage_before.csv", "paper02_es1_column_coverage_pass2.csv",
        "paper02_es1_column_coverage.csv"),
    4: ("paper02_es1_random_summary.csv", "paper02_es1_random_summary_heldout.csv",
        "paper02_es1_pandera_mutations.csv", "paper02_es1_pandera_mutations_heldout.csv"),
    7: ("paper02_es2_task_counts.csv",),
    8: ("paper02_es2_guard_stages.csv", "paper02_es2_cloud_observation.csv"),
    9: ("paper02_es2_cost_control_repetitions.csv",
        "paper02_es2_cost_control_repetitions_summary.csv"),
    10: ("paper02_es3_steps.csv", "paper02_es3_steps_v12.csv"),
    11: ("paper02_es3_column_agreement.csv",),
    12: ("paper02_level1_machines.csv",),
    "S4": ("paper02_es1_random_faults.csv", "paper02_es1_random_faults_heldout.csv"),
    "S3": ("paper02_es2_cloud_observation.csv", "paper02_es2_cost_control_tasks_as_deployed.csv",
         "paper02_es2_cost_control_tasks_run1.csv", "paper02_es2_cost_control_tasks_run2.csv",
         "paper02_es2_cost_control_tasks_run3.csv", "paper02_es2_cost_control_tasks_run4.csv",
         "paper02_es2_cost_control_tasks_run5.csv"),
    "5.1": ("paper02_es1_finding_counts.csv",),
}


SUPTEXT = (ROOT / "papers" / "system_paper" / "system_paper_supplement.md").read_text(encoding="utf-8")


def stable(n) -> list[dict]:
    """A table of the supplement, by its number there (S1, S2, ...)."""
    m = re.search(rf"\*\*Table {n}\. ", SUPTEXT)
    assert m, f"supplement Table {n} not found"
    rows, started = [], False
    for line in SUPTEXT[m.end():].split("\n"):
        if line.startswith("|"):
            rows.append(line)
            started = True
        elif started:
            break
    head = [h.strip() for h in rows[0].strip("|").split("|")]
    return [dict(zip(head, [x.strip() for x in r.strip("|").split("|")])) for r in rows[2:]]


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


def _row(t, label, paper, csv, kind, tol, good):
    ROWS.append({"table": t, "item": label, "paper": paper, "file": csv,
                 "kind": kind, "tolerance": tol, "agrees": "yes" if good else "no",
                 "source": "; ".join(SOURCES.get(t, ()))})


def chk(t, label, paper, csv, tol=0.0):
    good = paper is not None and csv is not None and abs(paper - csv) <= tol
    (OK if good else BAD).append(f"Table {t} {label}: paper={paper} file={csv}")
    _row(t, label, paper, csv, "numeric", tol, good)


def chks(t, label, paper, csv):
    good = str(paper).strip() == str(csv).strip()
    (OK if good else BAD).append(f"Table {t} {label}: paper={paper!r} file={csv!r}")
    _row(t, label, paper, csv, "text", "", good)


def unbound(t, why):
    UNBOUND.append(f"Table {t}: {why}")
    _row(t, why, "", "", "unbound", "", True)


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

# The declarative control is stated twice -- written out in pandas and built as a
# pandera schema -- and the captions of Tables 3 and 5 claim the two agree. Bind
# that claim instead of printing a second column of identical values; a reviewer
# asked whether the written-out control was a fair stand-in for the library, and
# this is where the answer is checked.
pand = pd.read_csv(R / "paper02_es1_pandera_compare.csv")
for _, pr in pand.iterrows():
    chks(3, f"{pr.fault} pandera verdict", pr.detected_by_handwritten, pr.detected_by_pandera)
    fam = lambda v: v if isinstance(v, str) and v else "—"
    chks(3, f"{pr.fault} pandera families", fam(pr.handwritten_families), fam(pr.pandera_families))
chk(3, "rows where the two statements of the control agree", 11.0,
    float((pand.agree == "yes").sum()))

# ---------------------------------------------------------------- Table 4
COV4 = ("paper02_es1_column_coverage_before.csv", "paper02_es1_column_coverage_pass2.csv",
        "paper02_es1_column_coverage.csv")
scored = lambda f: int((f.covered != "n/a (column empty)").sum())
for r, name in zip(stable(2), COV4):
    frame = pd.read_csv(R / name)
    hit, of = [num(x) for x in r["Covered"].split("/")]
    chk("S2", f"pass {r['Pass']} covered", hit, float((frame.covered == "yes").sum()))
    chk("S2", f"pass {r['Pass']} of", of, float(scored(frame)))
    alone = ((frame.covered == "yes") & (frame.failing_checks == "identities")).sum()
    chk("S2", f"pass {r['Pass']} identities alone", num(r["Covered by `identities` alone"]), float(alone))

# The main text keeps the final coverage and what the identity check alone binds;
# the three passes are the supplement's Table S2.
_cov = pd.read_csv(R / "paper02_es1_column_coverage.csv")
_FLAT5 = " ".join(SRC.split())
_m = re.search(r"The relations bind (\d+) of the record's (\d+) value-carrying columns", _FLAT5)
chks(4, "final column coverage", [int(x) for x in _m.groups()] if _m else "(absent)",
     [int((_cov.covered == "yes").sum()), int(scored(_cov))])
_m = re.search(r"pass, binds (\d+) that no other relation does", _FLAT5)
chk(4, "columns the identity check alone binds", float(_m.group(1)) if _m else -1.0,
    float(((_cov.covered == "yes") & (_cov.failing_checks == "identities")).sum()))

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
for r in table(4):
    label = r["Operator"].replace("**", "")
    if label.startswith("all, held-out"):
        g = heldall
    elif label == "all":
        g = allr
    else:
        key = next((v for k, v in KEY5.items() if label.startswith(k)), None)
        assert key, label
        g = op.loc[key]
    chk(4, f"{label} n", num(r["n"]), float(g.n))
    chk(4, f"{label} detected", num(r["Detected"]), float(g.detected))
    chk(4, f"{label} declarative", num(r["Declarative"]), float(g.detected_declarative))
    chk(4, f"{label} moved", num(r["Moved"]), float(g.moved_headline))
    chk(4, f"{label} moved and missed", num(r["Missed"]), float(g.moved_and_undetected))

# The caption's two pandera totals, and the per-operator agreement behind them.
for tag, want_total in (("", 81.0), ("_heldout", 86.0)):
    mut = pd.read_csv(R / f"paper02_es1_pandera_mutations{tag}.csv")
    com = pd.read_csv(R / f"paper02_es1_random_faults{tag}.csv")
    j = com.merge(mut[["fault_no", "detected_by_pandera"]], on="fault_no")
    assert len(j) == len(com), tag
    chk(4, f"pandera total{tag or '_main'}", want_total,
        float((j.detected_by_pandera == "yes").sum()))
    for op, g in j.groupby("operator"):
        chk(4, f"{op} pandera vs written out{tag or '_main'}",
            float((g.detected_by_pandera == "yes").sum()),
            float((g.detected_by_declarative == "yes").sum()))

# ---------------------------------------------------------------- Table 7
t7 = pd.read_csv(R / "paper02_es2_task_counts.csv")
for r in table(7):
    n = int(num(r["N"]))
    mock = t7[(t7.n == n) & (t7.device == "mock")].iloc[0]
    chk(7, f"N={n} stand-in", num(r["stand-in (`default.qubit`, no derivatives)"]), float(mock.tapes_per_step))
    chk(7, f"N={n} formula", num(r["formula"]), float(mock.expected_per_step))
    tr = t7[(t7.n == n) & (t7.device == "param_shift transform")]
    if len(tr):
        chk(7, f"N={n} transform", num(r["transform"]), float(tr.iloc[0].tapes_total))
    bl = t7[(t7.n == n) & (t7.device == "braket_local")]
    if len(bl) and num(r["`braket.local.qubit`"]) is not None:
        chk(7, f"N={n} braket.local", num(r["`braket.local.qubit`"]), float(bl.iloc[0].tapes_per_step))

# ----------------------------------------- the tier re-count, Section 5.1
# Twenty counts, every one a re-count of detections already in the files. The
# prose carries them, so a number changed in one place and not the other fails
# here rather than reaching a reader.
tiers = pd.read_csv(R / "paper02_es1_tiers.csv")
FLAT = " ".join(SRC.split())
WORDS = {"three": 3, "six": 6, "seven": 7, "ten": 10, "none": 0, "fifty-one": 51}
_num = lambda x: WORDS[x] if x in WORDS else int(x)


def _tier(pop):
    return tiers[tiers.population == pop].sort_values("tier").detected.tolist()


def _tier_both(pop):
    return [a + b for a, b in zip(_tier(f"mutations, main draw, {pop}"), _tier(f"mutations, held-out draw, {pop}"))]
_in, _outt = _tier_both("inside the run table"), _tier_both("outside the run table")
_fault_t = _tier("injected faults")
_both = pd.concat([pd.read_csv(R / "paper02_es1_random_faults.csv"),
                   pd.read_csv(R / "paper02_es1_random_faults_heldout.csv")])
_out = _both[_both.operator.isin(("npz_perturb", "npz_swap", "input_perturb"))]
_inn = _both[~_both.operator.isin(("npz_perturb", "npz_swap", "input_perturb"))]
_mtx = pd.read_csv(R / "paper02_es1_detection_matrix.csv"); _mtx = _mtx[_mtx.fault != "clean"]
_ctrl = [int((_mtx.detected_by_declarative == "yes").sum()),
         int((_inn.detected_by_declarative == "yes").sum()), int((_out.detected_by_declarative == "yes").sum())]
# ---------------------------------------------------------------- Table 6
# Every cell of the tier table against the clause-level tier file; the control
# row against the mutation files and the detection matrix.
rows6 = table(5)
assert len(rows6) == 5, "Table 6 has four tiers and the control row"
head6 = list(rows6[0].keys())
chk(5, "mutations inside the table", float(re.search(r"\((\d+)\)", head6[3]).group(1)), float(len(_inn)))
chk(5, "mutations outside the table", float(re.search(r"\((\d+)\)", head6[4]).group(1)), float(len(_out)))
tier_rows = [r for r in rows6 if r["Tier"].startswith("T")]
ctrl_row = [r for r in rows6 if r["Tier"].startswith("control")][0]
for i, r in enumerate(tier_rows):
    chk(5, f"{r['Tier'][:2]} faults", num(r[head6[2]]), float(_fault_t[i]))
    chk(5, f"{r['Tier'][:2]} inside", num(r[head6[3]]), float(_in[i]))
    chk(5, f"{r['Tier'][:2]} outside", num(r[head6[4]]), float(_outt[i]))
chks(5, "control row", [int(num(ctrl_row[head6[2]])), int(num(ctrl_row[head6[3]])), int(num(ctrl_row[head6[4]]))], _ctrl)
# The prose that reads Table 6: the lead paragraph, the "what each kind of record buys" paragraph.
m = re.search(r"of (\d+) random mutations, (\d+) move one, and the relations catch (\d+) of them against the "
              r"declarative control's (\d+)", FLAT)
_mv = _both[_both.headline_numbers_moved > 0]
chks(4, "lead: headline-moving mutations, both draws", [int(x) for x in m.groups()] if m else "(absent)",
     [len(_both), len(_mv), int((_mv.detected == "yes").sum()), int((_mv.detected_by_declarative == "yes").sum())])
m = re.search(r"of the (\d+) mutations that edit the run or reference table, the relations catch (\d+) and the "
              r"control (\d+), and Table \d+ says where the difference comes from — (\d+) of it from clauses "
              r".*? (\d+) from recomputing portfolio metrics .*? and (\d+) from comparing", FLAT)
chks(5, "lead: inside-the-table decomposition", [int(x) for x in m.groups()] if m else "(absent)",
     [len(_inn), _in[3], _ctrl[1], _in[0] - _ctrl[1], _in[1] - _in[0], _in[2] - _in[1]])
m = re.search(r"Of the (\d+) mutations confined to a file the table does not hold, the control catches none and "
              r"the relations (\d+), (\d+) of them through the digest", FLAT)
chks(5, "lead: outside the table", [int(x) for x in m.groups()] if m else "(absent)", [len(_out), _outt[3], _outt[1]])
m = re.search(r"adds one injected fault and (\d+) of the (\d+) mutations outside the table; reading the artifacts "
              r"adds the last fault and (\d+) more", FLAT)
chks(5, "buys: digest and artifacts outside the table", [int(x) for x in m.groups()] if m else "(absent)",
     [_outt[1], len(_out), _outt[2] - _outt[1]])
chks(5, "buys: tiers add one fault then the last", [_fault_t[1] - _fault_t[0], _fault_t[2] - _fault_t[1]], [1, 1])
m = re.search(r"T1 alone — the same two files the control reads — detects (\d+) where the control detects (\d+)", FLAT)
chks(5, "buys: T1 against the control inside the table", [int(x) for x in m.groups()] if m else "(absent)", [_in[0], _ctrl[1]])
chk(5, "T4 adds nothing to T3", float(_in[3] + _outt[3] + _fault_t[3]), float(_in[2] + _outt[2] + _fault_t[2]))
m = re.search(r"they catch (\d+) where the relations catch (\d+), and most of that difference", FLAT)
chks(7, "conclusion: inside the table", [int(x) for x in m.groups()] if m else "(absent)", [_ctrl[1], _in[3]])
m = re.search(r"where the relations catch (\d+), and none of two instance pairs", FLAT)
chks(7, "conclusion: outside the table", [int(x) for x in m.groups()] if m else "(absent)", [_outt[3]])
chk(5, "abstract: about twice as many inside the table", 2.0, _in[3] / _ctrl[1], 0.15)
chk(4, "lead: control catches none outside the table", 0.0, float((_out.detected_by_declarative == "yes").sum()))

# Supplement Table S4: detections by magnitude, both draws, every perturbing operator.
for _tag, _col in (("", "Main draw: n"), ("_heldout", "Held-out: n")):
    _d = pd.read_csv(R / f"paper02_es1_random_faults{_tag}.csv"); _d = _d[_d.magnitude.notna()]
    _g = _d.groupby("magnitude").agg(n=("detected", "size"), det=("detected", lambda x: (x == "yes").sum()))
    _lab = {"±1 (a count)": 1.0, "10⁻⁹": 1e-9, "10⁻⁶": 1e-6, "10⁻³": 1e-3, "10⁻¹": 1e-1}
    _rows = stable(4)
    _cols = list(_rows[0].keys()); _i = _cols.index(_col)
    for r in _rows:
        m = _lab[r["Magnitude"]]
        chk("S4", f"{_tag or 'main'} {r['Magnitude']} n", num(r[_cols[_i]]), float(_g.loc[m, "n"]))
        chk("S4", f"{_tag or 'main'} {r['Magnitude']} detected", num(r[_cols[_i + 1]]), float(_g.loc[m, "det"]))
        chk("S4", f"{_tag or 'main'} {r['Magnitude']} rate", float(re.search(r"\((\d+) %\)", r[_cols[_i + 1]]).group(1)),
            round(100 * float(_g.loc[m, "det"]) / float(_g.loc[m, "n"])), 0.5)

_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
# Every re-run row re-simulated (not the sampled relation): the ladder prose
# quotes, per pair, the rows off at 1e-3 and 1e-1 and the largest gaps at 1e-3.
_rs = pd.read_csv(R / "paper02_es1_input_rerun_recompute.csv")
def _rs_at(ladder, delta): return _rs[(_rs.ladder == ladder) & ((_rs.delta - delta).abs() < 1e-15)].iloc[0]
_m = re.search(r"finds the perturbation from 1e-3 in both pairs — (\d+) of (\d+) and (\d+) of (\d+) rows off by up to "
               r"(\d\.\d)·10⁻([⁰¹²³⁴⁵⁶⁷⁸⁹]) and (\d\.\d)·10⁻([⁰¹²³⁴⁵⁶⁷⁸⁹]) against the relation's 10⁻⁵ tolerance — and on "
               r"(\d+) of (\d+) and (\d+) of (\d+) rows at 1e-1", FLAT)
if _m:
    g = _m.groups()
    chks(6, "full re-simulation at 1e-3 and 1e-1, both pairs",
         [int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[8]), int(g[9]), int(g[10]), int(g[11])],
         [int(_rs_at("_pair01", 1e-3).rows_off), int(_rs_at("_pair01", 1e-3).rows_resimulated),
          int(_rs_at("_pair23", 1e-3).rows_off), int(_rs_at("_pair23", 1e-3).rows_resimulated),
          int(_rs_at("_pair01", 1e-1).rows_off), int(_rs_at("_pair01", 1e-1).rows_resimulated),
          int(_rs_at("_pair23", 1e-1).rows_off), int(_rs_at("_pair23", 1e-1).rows_resimulated)])
    chk(6, "full re-simulation: largest gap at 1e-3, first pair", float(g[4]) * 10 ** -int(g[5].translate(_SUP)),
        float(_rs_at("_pair01", 1e-3).max_gap), 0.02 * float(_rs_at("_pair01", 1e-3).max_gap))
    chk(6, "full re-simulation: largest gap at 1e-3, second pair", float(g[6]) * 10 ** -int(g[7].translate(_SUP)),
        float(_rs_at("_pair23", 1e-3).max_gap), 0.02 * float(_rs_at("_pair23", 1e-3).max_gap))
    chk(6, "full re-simulation silent below 1e-3", 0.0, float(_rs[_rs.delta < 1e-3].rows_off.sum()))
else:
    chks(6, "full re-simulation sentence", "(absent from prose)", ["3/60", "6/45"])

# The two sibling records the faults and mutations were repeated on. Every
# number of that paragraph is read out of the tagged result files.
def _sib(rec):
    out = {}
    mtx = pd.read_csv(R / f"paper02_es1_detection_matrix{rec}.csv")
    mtx = mtx[mtx.fault != "clean"]
    out["faults"] = int((mtx.detected_by_relations == "yes").sum())
    out["decl"] = int((mtx.detected_by_declarative == "yes").sum())
    main = pd.read_csv(R / "paper02_es1_detection_matrix.csv")
    out["same_first"] = bool((main.first_failing_check.fillna("") ==
                              pd.read_csv(R / f"paper02_es1_detection_matrix{rec}.csv")
                              .first_failing_check.fillna("")).all())
    cmp_ = pd.read_csv(R / f"paper02_es1_pandera_compare{rec}.csv")
    out["pandera_rows"] = bool((cmp_.agree == "yes").all())
    draws = [pd.read_csv(R / f"paper02_es1_random_faults{rec}{t}.csv") for t in ("", "_heldout")]
    out["pandera_mut"] = all((pd.read_csv(R / f"paper02_es1_pandera_summary{rec}{t}.csv")
                              .pipe(lambda f: (f.handwritten == f.pandera).all()))
                             for t in ("", "_heldout"))
    out["rates"] = [round(100 * (d.detected == "yes").mean()) for d in draws]
    both = pd.concat(draws)
    out["n"] = len(both)
    out["detected"] = int((both.detected == "yes").sum())
    mv = both[both.headline_numbers_moved > 0]
    out["moved"] = [len(mv), int((mv.detected == "yes").sum()),
                    int((mv.detected_by_declarative == "yes").sum())]
    o = both[both.operator.isin(("npz_perturb", "npz_swap", "input_perturb"))]
    out["outside"] = [len(o), int((o.detected == "yes").sum()),
                      int((o.detected_by_declarative == "yes").sum())]
    tr = pd.read_csv(R / f"paper02_es1_tiers{rec}.csv")
    def tier(pop): return tr[tr.population == pop].sort_values("tier").detected.tolist()
    out["tiers_faults"] = tier("injected faults")
    out["tiers_outside"] = [a + b for a, b in zip(tier("mutations, main draw, outside the run table"),
                                                  tier("mutations, held-out draw, outside the run table"))]
    ip = both[(both.operator == "input_perturb") & (both.detected == "no")]
    out["input_missed"] = len(ip)
    return out
_e5, _lb = _sib("_e5"), _sib("_e3_lbfgs")
chks(3, "siblings: ten of ten, same first relation, same eight by the control",
     [10, 10, True, True, 8, 8] if re.search(r"the ten faults are detected ten of ten, each by the "
                                            r"same first relation as on the main record; the declarative "
                                            r"control catches the same eight", FLAT) else "(absent)",
     [_e5["faults"], _lb["faults"], _e5["same_first"], _lb["same_first"], _e5["decl"], _lb["decl"]])
chks(3, "siblings: pandera agrees on every row and mutation",
     [True] * 4 if "agrees with the hand-written control on every row of the matrix and on every mutation" in FLAT else "(absent)",
     [_e5["pandera_rows"], _lb["pandera_rows"], _e5["pandera_mut"], _lb["pandera_mut"]])
m = re.search(r"Of E5's (\d+) mutations (\d+) are detected \((\d+) % and (\d+) % in the two draws\), "
              r"of the L-BFGS-B arm's (\d+) \((\d+) % and (\d+) %\)", FLAT)
chks(4, "siblings: detected counts and rates",
     [int(x) for x in m.groups()] if m else "(absent)",
     [_e5["n"], _e5["detected"], *_e5["rates"], _lb["detected"], *_lb["rates"]])
m = re.search(r"Of the (\d+) E5 mutations that move a headline number the relations catch (\d+) and "
              r"the control (\d+); of the arm's (\d+), (\d+) and (\d+)", FLAT)
chks(4, "siblings: headline-moving mutations",
     [int(x) for x in m.groups()] if m else "(absent)", _e5["moved"] + _lb["moved"])
m = re.search(r"Of the (\d+) that reach outside the table, in each record, the relations catch (\d+) "
              r"and (\d+) and the control none", FLAT)
chks(4, "siblings: outside the table",
     [int(x) for x in m.groups()] if m else "(absent)",
     [_e5["outside"][0], _e5["outside"][1], _lb["outside"][1]])
chks(4, "siblings: control catches none outside, both arms", [0, 0],
     [_e5["outside"][2], _lb["outside"][2]])
chk(4, "siblings: outside counts equal across arms", float(_e5["outside"][0]), float(_lb["outside"][0]))
m = re.search(r"the tiers of Table \d+ detect (\w+), (\w+), (\w+) and (\w+) of the faults on both; "
              r"inside the run table (\d+), (\d+), (\d+) and (\d+) of (\d+) on E5 and (\d+), (\d+), (\d+) and (\d+) of (\d+) on the arm; "
              r"outside it (\w+), (\d+), (\d+) and (\d+) on E5 and (\w+), (\d+), (\d+) and (\d+) on the arm", FLAT)
def _tier_sum(rec, pop):
    tr = pd.read_csv(R / f"paper02_es1_tiers{rec}.csv")
    a = tr[tr.population == f"mutations, main draw, {pop}"].sort_values("tier")
    b = tr[tr.population == f"mutations, held-out draw, {pop}"].sort_values("tier")
    return [int(x + y) for x, y in zip(a.detected, b.detected)], int(a.n.iloc[0] + b.n.iloc[0])
_e5_in, _e5_in_n = _tier_sum("_e5", "inside the run table"); _lb_in, _lb_in_n = _tier_sum("_e3_lbfgs", "inside the run table")
_e5_out, _ = _tier_sum("_e5", "outside the run table"); _lb_out, _ = _tier_sum("_e3_lbfgs", "outside the run table")
chks(4, "siblings: tier counts",
     [_num(x) for x in m.groups()] if m else "(absent)",
     _e5["tiers_faults"] + _e5_in + [_e5_in_n] + _lb_in + [_lb_in_n] + _e5_out + _lb_out)
chks(4, "siblings: tier counts on the faults equal across arms", _e5["tiers_faults"], _lb["tiers_faults"])
m = re.search(r"(\w+) perturbed input statistics per record are not detected", FLAT)
chks(4, "siblings: undetected input perturbations, a cell no run reads",
     [WORDS.get(m.group(1).lower(), m.group(1))] * 2 if m else "(absent)", [_e5["input_missed"], _lb["input_missed"]])

# ---------------------------------------------------------------- Table 6
# The fault the stored-record operators cannot express: the wrong input produced
# the rows. Every cell of the ladder is bound, and so is the claim the paragraph
# above it rests on -- that at each magnitude the relations fire and neither
# statement of the declarative control does.
# Two instance pairs, one file each; the table lists the first pair's five rungs
# and then the second's.
t6a = pd.read_csv(R / "paper02_es1_input_rerun.csv").sort_values("delta").reset_index(drop=True)
t6b = pd.read_csv(R / "paper02_es1_input_rerun_pair23.csv").sort_values("delta").reset_index(drop=True)
t6 = pd.concat([t6a, t6b], ignore_index=True)
rows6 = table(6)
chk(6, "rows in the table", float(len(rows6)), float(len(t6)))
for r, (_, q) in zip(rows6, t6.iterrows()):
    lab = r["\u03b4"] + " " + r["Instances"]
    chks(6, f"{lab} instances", r["Instances"].replace("\u2013", " "),
         str(q.instances) if isinstance(q.instances, str) else "0 1")
    chk(6, f"{lab} rows re-run", num(r["Rows re-run"]), float(q.rows_rerun))
    chk(6, f"{lab} state rows", num(r["State"]), float(q.rows_with_changed_state))
    chk(6, f"{lab} headline moved", num(r["Headline"].split("/")[0]), float(q.headline_moved))
    chk(6, f"{lab} headline total", num(r["Headline"].split("/")[1]), float(q.headline_total))
    cell = r["Largest relative change"]
    if cell.strip() != "—":
        chk(6, f"{lab} largest relative change", sci(cell)[0],
            float(q.largest_relative_change), float(q.largest_relative_change) * 0.02)
    else:
        chk(6, f"{lab} no headline moved", 0.0, float(q.headline_moved))
    chk(6, f"{lab} declarative silent (no column: caption)", 0.0,
        0.0 if q.detected_by_declarative == "no" and q.detected_by_pandera == "no" else 1.0)
    fired = [] if not isinstance(q.relations_firing, str) else q.relations_firing.split(";")
    named = re.findall(r"`([a-z-]+)`", r["Relations that fire"])
    chks(6, f"{lab} relations", sorted(named), sorted(fired))
chk(6, "magnitudes where neither control fires", float(len(t6)),
    float(((t6.detected_by_declarative == "no") & (t6.detected_by_pandera == "no")).sum()))
chk(6, "magnitudes where the relations fire", 8.0,
    float((t6.detected_by_relations == "yes").sum()))

# The routine gate on the spliced records of both ladders: refuses every
# perturbed rung, nothing at delta = 0, no warning anywhere; the findings and
# the seconds the prose quotes are the files'.
_g = pd.concat([t6a, t6b])
_pert, _ctrl = _g[_g.delta > 0], _g[_g.delta == 0]
chk(6, "gate refuses every perturbed rung", 1.0, 1.0 if (_pert.gate_refuses == "yes").all() else 0.0)
chk(6, "gate silent at delta = 0", 0.0, float(_ctrl.gate_findings.sum()))
chk(6, "gate warnings", 0.0, float(_g.gate_warnings.sum()))
_m = re.search(r"refuses every perturbed rung — (\d+) findings on the first pair and (\d+) on the second, "
               r"every re-run row and both reference rows, in about (\d\.\d) s", FLAT)
if _m:
    chks(6, "gate findings per pair", [int(_m.group(1)), int(_m.group(2))],
         [int(t6a[t6a.delta > 0].gate_findings.iloc[0]), int(t6b[t6b.delta > 0].gate_findings.iloc[0])])
    chk(6, "gate seconds", float(_m.group(3)), float(_pert.gate_seconds.mean()), 0.15)
    chk(6, "gate findings = rows re-run + 2 reference rows (first pair)",
        float(t6a[t6a.delta > 0].gate_findings.iloc[0]), float(t6a[t6a.delta > 0].rows_rerun.iloc[0] + 2))
else:
    chks(6, "gate sentence", "(absent from prose)", ["62", "47"])
_m = re.search(r"as first released \(six routines, `v1\.2`\)", FLAT)
import sys as _sys
_sys.path.insert(0, str(ROOT))
from src.record_checks import RECORD_CHECKS as _GATE
chk(6, "the released gate's routine count is one fewer than now", 6.0, float(len(_GATE) - 1))

# ---------------------------------------------------------------- Table 8
t7 = pd.read_csv(R / "paper02_es2_guard_stages.csv")
ORDER7 = [("qpu_optimizer", 8), ("qpu_optimizer", 20), ("trainable_sampling_sv1", 8),
          ("trainable_sampling_sv1", 20), ("trainable_sampling_qpu", 8), ("trainable_sampling_qpu", 20),
          ("qpu_sampling_1000_shots", 8), ("untagged_cloud_run", 8)]
COLS7 = {"none": "none", "cap only": "cap", "structural": "structural", "both": "both"}
for r, (mis, n) in zip(table(8), ORDER7):
    for col, cfg in COLS7.items():
        g = t7[(t7.misconfiguration == mis) & (t7.n == n) & (t7.configuration == cfg)]
        if not len(g):
            continue
        g = g.iloc[0]
        cell = r[col].replace("**", "")
        # The provenance mark of the caption: a dagger where the cap acts inside
        # the service (never observed, the device stayed offline), a double
        # dagger where the cloud observation covers the cell.
        mark, cell = (cell[0], cell[1:].strip()) if cell.strip()[:1] in "†‡§" else ("", cell)
        observed = set(pd.read_csv(R / "paper02_es2_cloud_observation.csv").step)
        # A task count obtained by counting circuits: the transform's tapes for
        # one sampling call, or one step's tapes multiplied out over fifty.
        counted_circuits = g.stage == "submitted" and mis in (
            "qpu_optimizer", "trainable_sampling_sv1", "trainable_sampling_qpu")
        on_service = (mis == "trainable_sampling_sv1" and n == 8
                      and {"sv1-sampling-trainable", "sv1-sampling-frozen"} <= observed)
        want = ("‡" if on_service
                else "§" if counted_circuits
                else "†" if g.stage == "service"
                else "")
        chks(8, f"{mis} N={n} {cfg} provenance", mark or "(none)", want or "(none)")
        head = cell.split(";")[0].strip()
        if mis == "untagged_cloud_run":
            # The cell says what happened to the baseline, which is the point of the row.
            kept = "intact" in cell
            chks(8, f"{mis} N={n} {cfg} baseline", "kept" if kept else "overwritten",
                 "overwritten" if g.baseline_overwritten == "yes" else "kept")
        else:
            chks(8, f"{mis} N={n} {cfg} stage", head, g.stage)
        m = re.search(r"(\d[\d,]*)\s*(?:task|tasks)?\s*(?:;|$)", cell.split(";", 1)[1] if ";" in cell else "")
        if m:
            chk(8, f"{mis} N={n} {cfg} tasks", float(m.group(1).replace(",", "")), float(g.tasks_billed))
        d7 = re.search(r"\$([\d,]+(?:\.\d+)?)", cell)
        if d7:
            chk(8, f"{mis} N={n} {cfg} usd", float(d7.group(1).replace(",", "")), float(g.usd_billed), 0.6)

# ---------------------------------------------------------------- Table 9
rep = pd.read_csv(R / "paper02_es2_cost_control_repetitions.csv").set_index("run")
sm8 = pd.read_csv(R / "paper02_es2_cost_control_repetitions_summary.csv").set_index("quantity")
F8 = {"Tasks created": "tasks_created",
      "Created past the limit": "tasks_created_after_limit_crossed",
      "Limit → alarm": "seconds_limit_to_alarm",
      "Created past the alarm": "tasks_created_after_alarm",
      "Alarm → first denial": "seconds_alarm_to_denial",
      "Created past the first denial": "tasks_created_after_first_denial"}
for r in table(9):                      # one row per quantity, one column per repetition
    field = F8[r["Quantity"]]
    for run in (1, 2, 3, 4, 5):
        chk(9, f"run {run} {field}", num(r[str(run)]), float(rep.loc[run, field]))
    chk(9, f"median {field}", num(r["median"]), float(sm8.loc[field, "median"]))

# ------------------------------ the prose of Section 5's opener and Section 8
# The conclusion and the results opener quote six measured numbers in prose,
# where the table bindings above cannot see them. A reviewer found one of them
# rounded before it was summed (27 + 2.5 written as 30, for 26.7 + 2.5 = 29.2),
# so each is read from the prose and compared with the file it comes from.
FLAT8 = " ".join(SRC.split())
_s12 = pd.read_csv(R / "paper02_es3_steps_v12.csv").set_index("step")
_l3 = pd.read_csv(R / "paper02_es3_steps_v12level3.csv").set_index("step")
_cc = pd.read_csv(R / "paper02_es2_cost_control_repetitions_summary.csv").set_index("quantity")
for pat, label, want, tol in (
    (r"rebuilt (?:at three levels|its aggregates) in (\d+) s", "level 1 sum, shipped release",
     float(_s12.loc["level1-figures", "seconds"] + _s12.loc["level1-tables", "seconds"]), 0.51),
    (r"(?:in 29 s, |one instance in )(\d+) s", "level 2, shipped release",
     float(_s12.loc["level2", "seconds"]), 0.51),
    (r"(\d+) s for a 24-run block|17 s\s*and (\d+) s|a 24-run block in (\d+) s", "level 3, archived release",
     float(_l3.loc["level3-block", "seconds"]), 0.51),
    (r"a median of (\d+) tasks past", "cost-control median past the limit",
     float(_cc.loc["tasks_created_after_limit_crossed", "median"]), 0.0),
    (r"admitted (\d+) to (\d+) tasks past", "cost-control range past the limit",
     (float(_cc.loc["tasks_created_after_limit_crossed", "min"]),
      float(_cc.loc["tasks_created_after_limit_crossed", "max"])), 0.0),
):
    hits = [tuple(int(x) for x in m.groups() if x) for m in re.finditer(pat, FLAT8)]
    if not hits:
        chks(8, label, "(absent from prose)", want)
        continue
    for h in set(hits):
        if isinstance(want, tuple):
            chks(8, label, list(h), [int(w) for w in want])
        else:
            chk(8, label, float(h[0]), want, tol)

# The fallback's wall-clock factor (Section 5.2): the pilot run of the SV1 ledger
# (227 tasks, trainable sampling) against the corrected N = 8 run and its two
# later replicates, all from the committed result files of the numerical study.
_E1R = ROOT / "experiments" / "paper01_qubo_baseline" / "results"
_led = pd.read_csv(_E1R / "paper01_sv1_task_ledger.csv")
_pilot = (pd.to_datetime(_led.last_task_utc) - pd.to_datetime(_led.first_task_utc)).dt.total_seconds().iloc[0]
def _sv1_n8(name):
    d = pd.read_csv(_E1R / name)
    return float(d[(d.n_assets == 8) & (d.solver == "braket_sv1")].runtime_s.iloc[0])
_fixed = _sv1_n8("paper01_results_sv1.csv")
_reps = [_sv1_n8("paper01_results_sv1rep2.csv"), _sv1_n8("paper01_results_sv1rep3.csv")]
m = re.search(r"(\d\.\d)× the wall-clock of the corrected run: (\d+) s between the pilot's first and last "
              r"task against (\d+) s for the N = 8 run that replaced it, or (\d\.\d)× against", FLAT8)
if m:
    chk(8, "fallback factor: pilot span", float(m.group(2)), _pilot, 0.51)
    chk(8, "fallback factor: corrected N = 8 run", float(m.group(3)), _fixed, 0.51)
    chk(8, "fallback factor", float(m.group(1)), _pilot / _fixed, 0.05)
    chk(8, "fallback factor against the replicates", float(m.group(4)),
        _pilot / (sum(_reps) / 2), 0.05)
else:
    chks(8, "fallback factor sentence", "(absent from prose)", [round(_pilot / _fixed, 1)])

# The v1.2 pass repeated level 3 (Table 10) and the manuscript says its column
# agreement reproduces Table 11 row for row; the two agreement files must then
# be identical, and the conclusion's largest state difference is Table 11's.
_ag0 = pd.read_csv(R / "paper02_es3_column_agreement.csv")
_ag2 = pd.read_csv(R / "paper02_es3_column_agreement_v12level3.csv")
chk(11, "v1.2 level-3 agreement reproduces v1.0's row for row", 1.0,
    1.0 if _ag0.equals(_ag2) else 0.0)
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_m = re.search(r"differing from the committed GPU single-precision rows by up to (\d\.\d)·10⁻([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", FLAT8)
if _m:
    chk(11, "conclusion: largest state difference",
        float(_m.group(1)) * 10 ** -int(_m.group(2).translate(_SUP)),
        float(_ag0[_ag0.group == "state"].max_abs_diff.max()), 0.02 * float(_ag0[_ag0.group == "state"].max_abs_diff.max()))
else:
    chks(11, "conclusion: largest state difference", "(absent from prose)", [float(_ag0[_ag0.group == "state"].max_abs_diff.max())])

# ---------------------------------------------------------------- Table 10
t9 = pd.read_csv(R / "paper02_es3_steps.csv").set_index("step")
t9b = pd.read_csv(R / "paper02_es3_steps_v12.csv").set_index("step")
# Level 3 on the shipped release was run on its own, after the rest of that pass.
t9c = pd.read_csv(R / "paper02_es3_steps_v12level3.csv").set_index("step")
# The v1.0 rows come from the full pass, the "(v1.2)" block from the pass that
# clones the release this paper ships with.
ORDER9 = [(t9, ["clone"]), (t9, ["venv", "pip"]), (t9, ["tests"]), (t9, ["level1-figures"]),
          (t9, ["level1-tables"]), (t9, ["level2"]), (t9, ["level3-block"]),
          (t9, ["interrupt-kill"]), (t9, ["interrupt-resume"]),
          (t9, ["offline-no-derived"]), (t9, ["missing-npz"]),
          (t9b, ["clone", "venv", "pip"]), (t9b, ["tests"]),
          (t9b, ["level1-figures", "level1-tables"]), (t9b, ["level2"]),
          (t9c, ["level3-block"]),
          (t9b, ["status-after-kill"]), (t9b, ["status-after-resume"])]
for r, (frame, steps) in zip(table(10), ORDER9):
    cell = r["Wall-clock"]
    got = [float(x) for x in re.findall(r"(\d+\.?\d*)\s*s", cell)]
    for value, step in zip(got, steps):
        chk(10, f"{r['Level']} {step} seconds", value, round(float(frame.loc[step, "seconds"])), 0.51)

# The commit the paper names as the final implementation must be the commit the
# shipped-release pass actually cloned, in Table 9's row, in Section 5.3 and in the
# availability statement. Re-exporting the package without re-running that pass
# leaves the paper pointing at an artefact its evidence is not about.
cloned = re.search(r"commit ([0-9a-f]{7,40})", str(t9b.loc["clone", "metric"])).group(1)
for row in table(10):
    if row["Level"] == "0 (v1.2)" and "commit" in row["Outcome"]:
        chks(10, "v1.2 clone commit", re.search(r"commit ([0-9a-f]{7,40})", row["Outcome"]).group(1),
             cloned)
for label, pat in (("Section 5.3", r"a tagged commit \(`([0-9a-f]{7,40})`\)"),
                   ("availability", r"`v1\.2` \(commit `([0-9a-f]{7,40})`")):
    m = re.search(pat, SRC)
    chks(10, f"{label} names the cloned commit", m.group(1) if m else "(absent)", cloned)

# ---------------------------------------------------------------- Table 11
t10 = pd.read_csv(R / "paper02_es3_column_agreement.csv")
NAME10 = {"final_expectation": ["final_expectation"],
          "expectation_at_final_angles": ["expectation_at_final_angles"],
          "final_P_F": ["final_P_F"], "final_P_opt, final_P_τ": ["final_P_opt", "final_P_tau_main"],
          "initial_P_F": ["initial_P_F"], "final_Q_τ(1000)": ["final_Q_tau_main_1000"],
          "feasible_shots": ["feasible_shots"], "shots_optimal": ["shots_optimal"],
          "initial_angles": ["initial_angles"], "final_angles": ["final_angles"],
          "best_all_E, g_F, g_off": ["best_all_E", "g_F", "g_off"]}
for r in table(11):
    cols = NAME10.get(r["Column"])
    if cols is None:
        continue
    diffs = sci(r["Max abs. difference"]) if r["Max abs. difference"] != "—" else []
    for i, col in enumerate(cols):
        g = t10[t10.column == col].iloc[0]
        hit, of = [num(x) for x in r["Identical rows"].split("/")]
        chk(11, f"{col} identical", hit, float(g.identical_rows))
        chk(11, f"{col} rows", of, float(g.rows))
        if pd.notna(g.max_abs_diff) and len(diffs) > (i if len(diffs) > 1 else 0):
            paper_diff = diffs[i if len(diffs) > 1 else 0]
            chk(11, f"{col} max diff", paper_diff, float(g.max_abs_diff),
                abs(float(g.max_abs_diff)) * 0.06 + 1e-12)

# ---------------------------------------------------------------- Table 12
t11 = pd.read_csv(R / "paper02_level1_machines.csv")
for r, (_, g) in zip(table(12), t11.iterrows()):
    p, s_ = re.match(r"(\d+) passed, (\d+) skipped", r["Test suite"]).groups()
    chk(12, f"{g.machine} passed", float(p), float(g.passed))
    chk(12, f"{g.machine} skipped", float(s_), float(g.skipped))
    chk(12, f"{g.machine} cells", num(r["`check_tables.py`"]), float(g.cells_agree))
    chks(12, f"{g.machine} python", r["Python"], str(g.python))
# Only one of the four rows compared the regenerated figures with the committed
# ones; the other three record that the command ran. The table carries no figure
# column, and this keeps it that way rather than letting it pick up a "yes" that
# means less than it looks.
assert "byte-identical" in str(t11[t11.machine.str.startswith("GPU host")].figures.iloc[0])
assert (t11[~t11.machine.str.startswith("GPU host")].figures
        == "command ran; bytes not compared").all()
assert not any("figure" in h.lower() for h in table(12)[0]), \
    "Table 11 has a figure column; three of its four rows cannot fill one"

# ---------------------------------------------------------------- Table 14
SV1_TASK_USD = 3.0 / 60 * 0.075
obs = pd.read_csv(R / "paper02_es2_cloud_observation.csv")
sv1_tasks = sum(int(m.group(1)) for m in
                (re.match(r"(\d+) task\(s\) created", str(o)) for o in obs.outcome) if m)
def _cc(*tags):
    return sum(int((pd.read_csv(R / f"paper02_es2_cost_control_tasks_{t}.csv").outcome == "created").sum())
               for t in tags)


cc_deployed = _cc("as_deployed")
cc_fixed = _cc("run1", "run2", "run3", "run4", "run5")
cc = cc_deployed + cc_fixed
rows13 = stable(3)
for r in rows13:
    what = r["What"].replace("**", "")
    if what.startswith("the sampling-fallback"):
        chk("S3", "fallback tasks", num(r["Tasks"]), float(sv1_tasks))
        chk("S3", "fallback usd", num(r["Cost"]), round(sv1_tasks * SV1_TASK_USD, 2), 0.005)
    elif what.startswith("the account-level cost control as documented"):
        chk("S3", "as-deployed tasks", num(r["Tasks"]), float(cc_deployed))
        chk("S3", "as-deployed usd", num(r["Cost"]), round(cc_deployed * SV1_TASK_USD, 2), 0.005)
    elif what.startswith("the same control with its import fixed"):
        chk("S3", "five-repetition tasks", num(r["Tasks"]), float(cc_fixed))
        chk("S3", "five-repetition usd", num(r["Cost"]), round(cc_fixed * SV1_TASK_USD, 2), 0.005)
    elif what.startswith("this paper"):
        chk("S3", "total tasks", num(r["Tasks"]), float(sv1_tasks + cc))
        chk("S3", "total usd", num(r["Cost"]),
            round(sv1_tasks * SV1_TASK_USD, 2)
            + round(cc_deployed * SV1_TASK_USD, 2) + round(cc_fixed * SV1_TASK_USD, 2), 0.005)
unbound("S3", "the project's $42.59 and the numerical study's $37.90 are the provider's billing console")

for n, why in ((1, "the injected faults, described in prose"),
               (2, "the misconfigurations, described in prose"),
               (13, "the seven defects and the relation that now holds each")):
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
                _row("figure", name, digest(here)[:16], digest(there)[:16], "sha256", "", True)
            else:
                BAD.append(f"figure {name}: regenerates to a different file")
                _row("figure", name, digest(here)[:16], digest(there)[:16], "sha256", "", False)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# Section 5.1 quotes what each check costs and how much it finds. Those are
# prose, not cells, and prose is where a number goes stale unnoticed -- "2,299
# rows" stood for three revisions when the file said 2,299 findings on 2,252
# rows. Each is bound here to paper02_es1_finding_counts.csv.
fc = pd.read_csv(R / "paper02_es1_finding_counts.csv")
clean = fc[fc.fault == "clean"].set_index("check")
bf = fc[(fc.fault == "batch_from_initial") & (fc.check == "state-batch")].iloc[0]
pp = fc[fc.fault == "price_perturbation"].set_index("check")
for label, said, got in (
        ("schema on the clean record", 0.009, clean.loc["schema", "seconds"]),
        ("every check on the clean record", 46.6, clean.seconds.sum()),
        ("reference-input on the clean record", 10.5, clean.loc["reference-input", "seconds"]),
        ("identities on the clean record", 10.3, clean.loc["identities", "seconds"]),
        ("state-recompute on the clean record", 13.1, clean.loc["state-recompute", "seconds"]),
        ("input-digest under price_perturbation", 1.2, pp.loc["input-digest", "seconds"]),
        ("reference-input under price_perturbation", 9.7, pp.loc["reference-input", "seconds"]),
        ("input-digest records", 1009, pp.loc["input-digest", "n_findings"]),
        ("state-batch findings", 2299, bf.n_findings)):
    assert re.search(rf"(?<![\d.]){re.escape(f'{said:,}' if said >= 1000 else str(said))}(?![\d])",
                     SRC), f"Section 5.1 no longer quotes {said} for {label}"
    chk("5.1", label, float(said), round(float(got), 1 if said < 100 else 0), 0.06)

# The prose splits state-batch's subjects into rows and blocks; the two must add
# up to what the harness counted.
assert "2,252 of the 2,565 rows" in SRC and "all nineteen (N, penalty) blocks" in SRC, \
    "Section 5.1 no longer splits state-batch's subjects into rows and blocks"
chk("5.1", "state-batch rows plus blocks", 2252.0 + 19.0, float(bf.n_subjects))

# A [[placeholder]] left in the draft is an identifier nobody has filled in yet.
for m in re.finditer(r"\[\[([^\]]+)\]\]", SRC):
    BAD.append(f"placeholder still in the draft: [[{m.group(1)}]]")
    _row("draft", f"placeholder [[{m.group(1)}]]", "", "", "placeholder", "", False)

# ``--csv [path]`` writes the datasheet: one row per comparison, with the value
# the paper states, the value the file holds, the tolerance the comparison
# allowed, and the files the table is read from. It is the same information the
# lines above report, in a form a reader can sort and a reviewer can work
# through, and it lists the tables that are prose rather than omitting them.
if "--csv" in sys.argv:
    i = sys.argv.index("--csv")
    out = pathlib.Path(sys.argv[i + 1]) if len(sys.argv) > i + 1 \
        and not sys.argv[i + 1].startswith("-") else DATASHEET
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["table", "item", "paper", "file", "kind",
                                           "tolerance", "agrees", "source"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} ({len(ROWS)} rows)")

print(f"{len(OK)} cell(s) agree with the result files, {len(BAD)} disagree, "
      f"{len(UNBOUND)} table(s) or cell(s) not bound to one")
for b in BAD:
    print("  x", b)
for u in UNBOUND:
    print("  -", u)
raise SystemExit(1 if BAD else 0)
