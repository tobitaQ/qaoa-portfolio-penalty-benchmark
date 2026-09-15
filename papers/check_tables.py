# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Check every table in the draft against the CSV it was computed from.

Table IV shipped for three revisions carrying the values of a discarded run: the
experiment had been redone on the unified price cache and the table had not.
Fig. 6 regenerates from the CSV, so the figure and the table disagreed on the
same page, and two sentences drew conclusions from cells that no longer existed.
An external reviewer found it, not us.

Nothing about that was hard to detect -- the numbers are in the repository and
the table is in the draft. It was only hard to remember. So this parses each
table out of the Markdown and compares it against the result file, which turns
"did we regenerate the tables after that run?" from a question into a command.

A tolerance is given per check, because the draft rounds and the CSV does not.

Usage:
    python papers/check_tables.py     # exit 1 if any cell disagrees
"""

from __future__ import annotations

import pathlib
import math
import re
from math import comb

import pandas as pd

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.qubo import scoring  # noqa: E402  (the bound formula, shared with the analysis)
ROOT = pathlib.Path(__file__).resolve().parent.parent
R = str(ROOT / "experiments" / "paper01_qubo_baseline" / "results") + "/"
s = (ROOT / "papers" / "paper01_draft.md").read_text()
# The supplement holds the tables the main text summarises (S1, S2, ...); they
# are checked exactly as the main text's are, from the same result files.
s_supp = (ROOT / "papers" / "paper01_supplement.md").read_text()

def table(n):
    src = s_supp if str(n).startswith("S") else s
    m=re.search(rf"\*\*Table {n}\. ", src)
    assert m, f"Table {n} not found"
    seg=src[m.end():]
    rows=[]; started=False
    for l in seg.split("\n"):
        if l.startswith("|"): rows.append(l); started=True
        elif started: break
    # An escaped pipe (\|F\|, \|Δp\|) is text, not a column boundary.
    cells=lambda l: [x.strip().replace("\x00", "|") for x in l.replace("\\|", "\x00").strip("|").split("|")]
    hdr=cells(rows[0])
    out=[]
    for l in rows[1:]:
        if re.match(r"^\|[\s:|-]+\|$", l): continue
        c=cells(l)
        if len(c)==len(hdr): out.append(dict(zip(hdr,c)))
    return out

def num(x):
    if x is None: return None
    x=re.sub(r"\*\*|[×%,]|\s|†|✓|✗", "", str(x)).replace("−","-").lstrip("+")
    m=re.match(r"^-?\d+\.?\d*", x); return float(m.group(0)) if m else None

OK=[];BAD=[];PENDING=[]
def chk(t,l,a,b,tol=0):
    good = a is not None and b is not None and abs(a-b)<=tol
    (OK if good else BAD).append(f"{t} {l}: 論文={a} CSV={b}")

def pending(t, l, cell) -> bool:
    """A ``[[...]]`` placeholder awaiting a re-run is listed, not scored."""
    if "[[" in str(cell):
        PENDING.append(f"{t} {l}: {cell}")
        return True
    return False

# v3.1 numbering (2026-09-12, review round 3): main-text Tables I-VIII (IV = D_cond, new), supplement S1-S38 (S37-S38 added in review round 4; the lean variant dropped the fallback-evidence table). The single-seed seed sweep and penalty sweep
# tables of v1.x are retired: E1 contains both as sub-blocks (checked below
# against the cross itself).

piv=pd.read_csv(R+"paper01_depth_sweep.csv").pivot_table(index="n_assets",columns="p_layers",values="gap_pct")
for r in table("S20"):
    n=int(num(r["N"]))
    for p in [1,2,3,4]: chk("S20",f"N={n} p{p}",num(r[f"p = {p}"]),round(piv.loc[n,p],4),5e-5)

# Table S3: the census. C(N,K) and the 0.1 % band from the degeneracy file,
# the deflation from the instance census (instance 0 is the published subset).
d=pd.read_csv(R+"paper01_degeneracy.csv"); band=d[(d.band_kind=="reported_gap")&(d.band_pct==0.1)]
inst=pd.read_csv(R+"paper01_instances.csv"); pub=inst[inst.instance==0]
for r in table("S3"):
    n=int(num(r["N"])); g=band[band.n_assets==n]
    if not len(g): continue
    chk("S3",f"N={n} C(N,K)",num(r["C(N,K)"]),float(g.feasible_set_size.iloc[0]),0)
    chk("S3",f"N={n} in band",num(r["In 0.1 % band"]),float(g.n_within.iloc[0]),0)
    pp=pub[pub.n_assets==n]
    if len(pp):
        chk("S3",f"N={n} deflation",num(r["Deflation"]),round(float(pp.deflation_factor.iloc[0]),0),1.5)
        chk("S3",f"N={n} spread",num(r["Spread"]),round(float(pp.objective_range.iloc[0]),3),6e-4)

d=pd.read_csv(R+"paper01_penalty.csv"); d["ov"]=d.penalty_used/d.penalty_min
for r in table("S7"):
    n=int(num(r["N"])); g=d[d.n_assets==n]
    if len(g):
        chk("S7",f"N={n} A used",num(r["A used"]),round(g.penalty_used.iloc[0],0),0.5)
        chk("S7",f"N={n} overshoot",num(r["Overshoot"]),round(g.ov.iloc[0],0),1.5)

q=pd.read_csv(R+"paper01_instances_qaoa.csv")
for r in table("S12"):
    n=int(num(r["N"])); g=q[q.n_assets==n]
    m=re.match(r"(\d+)/(\d+)", r["Exact optimum"].replace("**",""))
    if m: chk("S12",f"N={n} exact",float(m.group(1)),float((g.quantum_rank==1).sum()),0)

h=pd.read_csv(R+"paper01_hardware_ionq.csv")
for r in table("S32"):
    n=int(num(r["N"])); sd=int(num(r["seed"])); g=h[(h.n_assets==n)&(h.qaoa_seed==sd)]
    if len(g):
        chk("S32",f"N={n}/s{sd} gap",num(r["Gap"]),round(g.gap_pct.iloc[0],4),5e-4)
        cell = r["Feasible shots [95 % CI]"]
        k = round(g.feasible_fraction.iloc[0] * 100)
        chk("S32",f"N={n}/s{sd} feas",num(cell),float(k),0.5)
        # The interval is arithmetic on that count, so it is recomputed here
        # rather than trusted: a Wilson interval typed by hand is a number no
        # other check would catch.
        lo,hi=[num(x) for x in cell.split("[")[1].rstrip("]").split(",")]
        z=1.959963984540054; d=1+z*z/100; ph=k/100
        c=(ph+z*z/200)/d; hw=z*math.sqrt(ph*(1-ph)/100+z*z/40000)/d
        chk("S32",f"N={n}/s{sd} CI low",lo,round((c-hw)*100,1),0.06)
        chk("S32",f"N={n}/s{sd} CI high",hi,round((c+hw)*100,1),0.06)
        chk("S32",f"N={n}/s{sd} distinct",num(r["Distinct strings"]),float(g.distinct_bitstrings.iloc[0]),0)

d=pd.read_csv(R+"paper01_sv1_replicates.csv"); K={8:2,12:2,16:3,20:4}
for r in table("S28"):
    n=int(num(r["N"])); sv=d[(d.n_assets==n)&(d.backend=="braket_sv1")]
    chk("S28",f"N={n} C(N,K)",num(r["C(N,K)"]),float(comb(n,K[n])),0)
    chk("S28",f"N={n} distinct",num(r["Distinct answers in 3 SV1 runs"]),float(sv.energy.nunique()),0)

# Table S5: the band as a function of the reported-gap threshold
d = pd.read_csv(R + "paper01_degeneracy.csv")
band = d[d.band_kind == "reported_gap"]
for r in table("S5"):
    thr = num(r["Threshold"])
    for col in [c for c in r if c.startswith("N = ")]:
        n = int(num(col.replace("N = ", "")))
        g = band[(band.n_assets == n) & (band.band_pct == thr)]
        if len(g):
            chk("S5", f"{thr}% N={n}", num(r[col]),
                round(float(g.frac_within.iloc[0]) * 100, 1), 0.06)

# Table S6: the census under two normalizations
d = pd.read_csv(R + "paper01_normalization_sweep.csv")
for r in table("S6"):
    n = int(num(r["N"]))
    for col, norm in [("Deflation, per-instance", "per-instance"),
                      ("Deflation, global", "global")]:
        g = d[(d.n_assets == n) & (d.normalization == norm)]
        if len(g):
            chk("S6", f"N={n} {norm} deflation", num(r[col]),
                round(float(g.deflation_factor.iloc[0]), 0), 1.5)
    for col, norm in [("Band, per-instance", "per-instance"),
                      ("Band, global", "global")]:
        g = d[(d.n_assets == n) & (d.normalization == norm)]
        if len(g):
            chk("S6", f"N={n} {norm} band", num(r[col]),
                round(float(g.frac_within_band.iloc[0]) * 100, 1), 0.06)

# Table S26: matched-precision control
g32 = pd.read_csv(R + "paper01_results.csv")
g64 = pd.read_csv(R + "paper01_results_gpu64.csv")
sv1 = pd.read_csv(R + "paper01_results_sv1.csv")
cl = g32[g32.solver == "classical_auto"]
for r in table("S26"):
    n = int(num(r["N"]))
    opt = float(cl[cl.n_assets == n].energy_no_offset.iloc[0])
    for col, d, solver in [("GPU, single", g32, "lightning_gpu"),
                           ("GPU, double", g64, "lightning_gpu"),
                           ("SV1, double", sv1, "braket_sv1")]:
        sub = d[(d.n_assets == n) & (d.solver == solver)]
        if len(sub):
            gap = abs(float(sub.energy_no_offset.iloc[0]) - opt) / abs(opt) * 100
            chk("S26", f"N={n} {col}", num(r[col]), round(gap, 4), 5e-4)

# Table S27: the ten cross-backend pairs
a = pd.read_csv(R + "paper01_seed_sweep_sv1.csv")
b = pd.read_csv(R + "paper01_seed_sweep.csv")
pairs = a.merge(b, on=["n_assets", "qaoa_seed"], suffixes=("_sv1", "_gpu"))
for r in table("S27"):
    n, sd = int(num(r["N"])), int(num(r["seed"]))
    g = pairs[(pairs.n_assets == n) & (pairs.qaoa_seed == sd)]
    if len(g):
        chk("S27", f"N={n}/s{sd} SV1", num(r["Gap, SV1"]),
            round(float(g.gap_pct_sv1.iloc[0]), 4), 5e-4)
        chk("S27", f"N={n}/s{sd} GPU", num(r["Gap, GPU"]),
            round(float(g.gap_pct_gpu.iloc[0]), 4), 5e-4)

# Table S11: classical annealing on the penalty-encoded QUBO
d = pd.read_csv(R + "paper01_sa_on_qubo.csv")
for r in table("S11"):
    n = int(num(r["N"]))
    g = d[d.n_assets == n]
    if not len(g):
        continue
    m = re.match(r"(\d+) of (\d+)", r["Exact optimum"])
    if m:
        chk("S11", f"N={n} exact", float(m.group(1)),
            float((g.reported_gap_pct == 0).sum()), 0)
    lo, hi = [num(x) for x in r["Reported gap"].split("–")]
    chk("S11", f"N={n} gap lo", lo, round(float(g.reported_gap_pct.min()), 2), 6e-3)
    chk("S11", f"N={n} gap hi", hi, round(float(g.reported_gap_pct.max()), 2), 6e-3)
    chk("S11", f"N={n} worst rank", num(r["Worst rank"]),
        float(g.rank_in_feasible.max()), 0)

# Table S2: the optimized expectation column added in Round 3
conv = pd.read_csv(R + "paper01_convergence.csv")
for r in table("S2"):
    n = int(num(r["N"]))
    g = conv[conv.n_assets == n].sort_values("step")
    v = num(r.get("\u27e8H_norm\u27e9", ""))
    if len(g) and v is not None:
        chk("S2", f"N={n} <H_norm>", v, round(float(g.expectation.iloc[-1]), 2), 6e-3)


# ---- v2.0: the cross and its controls ----
def _pct(x):
    """A '+0.44 [+0.18, +0.75]' style cell → its leading number."""
    return num(str(x).split("[")[0])

# Table II: what ran. E1 rows carry the per-size manifest; E3 and E5 rows their own.
man = pd.read_csv(R + "paper01_e1_manifest.csv")
MAN_TAGS = {"E3, ADAM 0.03": "e3_adam003", "E3, L-BFGS-B": "e3_lbfgs", "E5, second window": "e5"}
for r in table("II"):
    lab = r["Experiment"]
    m = re.match(r"E1, N = (\d+) \(K = (\d+), \|F\| = ([\d,]+)(?:; (\d+) of 30 instances with A_crit = 0)?", lab)
    if m:
        n = int(m.group(1)); g = man[man.n_assets == n]
        nom, exp, done, failed = [float(x) for x in re.findall(r"\d+", r["Runs nominal / expected / done / failed"].replace(",", ""))]
        chk("II", f"N={n} nominal", nom, float(g.runs_nominal.iloc[0]), 0)
        chk("II", f"N={n} expected", exp, float(g.runs_expected.iloc[0]), 0)
        chk("II", f"N={n} done", done, float(g.runs_done.iloc[0]), 0)
        chk("II", f"N={n} failed", failed, float(g.runs_failed.iloc[0]), 0)
        chk("II", f"N={n} no feasible shot", num(r["No feasible shot"]), float(g.runs_without_feasible_shot.iloc[0]), 0)
        chk("II", f"N={n} min-energy infeasible", num(r["Min-energy shot infeasible"]), float(g.runs_best_shot_infeasible.iloc[0]), 0)
        chk("II", f"N={n} K", float(m.group(2)), float(g.n_select.iloc[0]), 0)
        chk("II", f"N={n} |F|", float(m.group(3).replace(",", "")), float(g.feasible_set_size.iloc[0]), 0)
        chk("II", f"N={n} A_crit=0", float(m.group(4) or 0), float(g.instances_a_crit_zero.iloc[0]), 0)
        chk("II", f"N={n} GPU h", num(r["Time"]), round(float(g.gpu_hours_total.iloc[0]), 2), 6e-3)
        continue
    tag = next((t for k, t in MAN_TAGS.items() if lab.startswith(k)), None)
    if tag:
        g = pd.read_csv(R + f"paper01_e1_manifest_{tag}.csv")
        nom, exp, done, failed = [float(x) for x in re.findall(r"\d+", r["Runs nominal / expected / done / failed"])]
        chk("II", f"{tag} expected", exp, float(g.runs_expected.sum()), 0)
        chk("II", f"{tag} done", done, float(g.runs_done.sum()), 0)
        chk("II", f"{tag} failed", failed, float(g.runs_failed.sum()), 0)
        if not pending("II", f"{tag} no feasible shot", r["No feasible shot"]):
            chk("II", f"{tag} no feasible shot", num(r["No feasible shot"]), float(g.runs_without_feasible_shot.sum()), 0)
        if not pending("II", f"{tag} min-energy infeasible", r["Min-energy shot infeasible"]):
            chk("II", f"{tag} min-energy infeasible", num(r["Min-energy shot infeasible"]), float(g.runs_best_shot_infeasible.sum()), 0)
        chk("II", f"{tag} GPU h", num(r["Time"]), round(float(g.gpu_hours_total.sum()), 1), 0.06)
        # The E3 rows of Table II and the six cells of Table VI / S22 count the
        # same runs from two result files; the manifest's no-feasible total has
        # to equal the sum of the per-cell "(no feas.)" counts.
        if tag.startswith("e3_"):
            opt = {"e3_adam003": "adam_0.03", "e3_lbfgs": "lbfgs_100grad"}[tag]
            cells = pd.read_csv(R + "paper01_e3_summary.csv")
            cells = cells[cells.optimizer == opt]
            chk("II", f"{tag} no feasible shot = sum over Table VI cells", float(g.runs_without_feasible_shot.sum()),
                float(cells.no_feasible_shot_runs.sum()), 0)
            chk("II", f"{tag} runs done = sum over Table VI cells", done, float(cells.runs_ok.sum()), 0)

# Table III: E1 main effects (levels from the summary, differences from the contrasts)
summ = pd.read_csv(R + "paper01_e1_summary.csv")
con = pd.read_csv(R + "paper01_e1_contrasts.csv")
FIELD = {"P_F": ("final_P_F", 1.0), "P_τ": ("final_P_tau_main", 1.0),
         "Q_τ(1000)": ("final_Q_tau_main_1000", 1.0), "g_off (%)": ("g_off", 100.0),
         "g_F (%)": ("g_F", 100.0), "Rank of best feasible shot": ("rank_best_feasible", 1.0)}
def _sci(x):
    """'2.6·10⁻³' → 2.6e-3; plain numbers pass through."""
    x = str(x).replace("**", "").strip().lstrip("+")
    m = re.match(r"^([−-]?\d+\.?\d*)·10([⁻⁺]?)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)$", x)
    if not m:
        return num(x)
    mant = float(m.group(1).replace("−", "-"))
    exp = int(m.group(3).translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")))
    return mant * 10 ** (-exp if m.group(2) == "⁻" else exp)
def _half_last_digit(cell, ref, scale):
    """Tolerance from the cell's own precision: ±0.5 of its last printed
    decimal, or 6 % when it is in ·10ⁿ notation (mantissa rounding)."""
    txt = str(cell).replace("**", "").strip().lstrip("+")
    if "·" in txt or "." not in txt:
        return max(abs(ref) * 0.06, 6e-4 * scale) if scale == 100 else max(abs(ref) * 0.06, 1e-3)
    return 0.51 * 10 ** (-len(txt.split(".")[-1]))

for r in table("III"):
    n = int(num(r["N"])); q = r["Quantity"].split(",")[0].strip()
    if q not in FIELD:
        continue
    field, scale = FIELD[q]
    for col, label in [("A_heur level", "Aheur"), ("A_margin level", "Amargin")]:
        g = summ[(summ.n_assets == n) & (summ.penalty_label == label)]
        v = _sci(r[col]); ref = float(g[f"{field}_median"].iloc[0]) * scale
        # Half a unit in the last printed digit. The earlier flat 0.06-point
        # allowance for gap cells let a stale 0.0064 stand against 0.0143 for
        # five revisions (review round 5 follow-up, 2026-09-14).
        chk("III", f"N={n} {q} {label} level", v, ref, _half_last_digit(r[col], ref, scale))
    g = con[(con.n_assets == n) & (con.treatment == "Amargin") & (con.field == field)]
    if len(g):
        d = _sci(r["Paired median diff [95 % CI]"].split("[")[0])
        ref = float(g.instance_median_diff.iloc[0]) * scale
        chk("III", f"N={n} {q} diff", d, ref, max(abs(ref) * 0.06, 6e-3 if scale == 100 else 1e-3))
        better = num(r["Better (tied) of 30"])
        chk("III", f"N={n} {q} better", better, float(g.instances_treatment_better.iloc[0]), 0)

# Table S13: decomposition
dec = pd.read_csv(R + "paper01_e1_decomposition.csv")
for r in table("S13"):
    n = int(num(r["N"])); g = dec[(dec.n_assets == n) & (dec.treatment == "Amargin")]
    med = r["Medians: g_off ratio / solution (IQR) / denominator"].split("/")
    chk("S13", f"N={n} ratio", num(med[0]), round(float(g.g_off_ratio_median.iloc[0]), 1), 0.06)
    chk("S13", f"N={n} solution part", num(med[1]), round(float(g.solution_part_median.iloc[0]), 2), 6e-3)
    chk("S13", f"N={n} denominator part", num(med[-1]), round(float(g.denominator_part_median.iloc[0]), 1), 0.06)
    chk("S13", f"N={n} pairs", num(r["Pairs used (zero gap / no feasible shot)"]), float(g.pairs.iloc[0]), 0)
    # The geometric column is an identity as well as three numbers: it is the
    # summary that multiplies, which the medians beside it do not.
    geo = r["Geometric mean: g_off ratio = solution × denominator"]
    tot, rest = geo.split("=")
    sol, den = rest.split("×")
    chk("S13", f"N={n} geo ratio", num(tot), round(float(g.g_off_ratio_geomean.iloc[0]), 1), 0.06)
    chk("S13", f"N={n} geo solution", num(sol), round(float(g.solution_part_geomean.iloc[0]), 2), 6e-3)
    chk("S13", f"N={n} geo denominator", num(den), round(float(g.denominator_part_geomean.iloc[0]), 1), 0.06)
    chk("S13", f"N={n} geo identity", num(tot), num(sol) * num(den), max(0.05 * num(tot), 0.06))

# Table S14: E2 shot budgets at A_heur
sh = pd.read_csv(R + "paper01_e1_shots.csv")
for r in table("S14"):
    n = int(num(r["N"])); S = int(num(r["S"]))
    g = sh[(sh.n_assets == n) & (sh.penalty_label == "Aheur") & (sh.shots == S)]
    if not len(g):
        continue
    for col, f in [("Optimised QAOA", "Q_tau_final_median"), ("Unoptimised (same initial angles)", "Q_tau_initial_median"),
                   ("Uniform over F", "Q_tau_uniform_F_median"), ("Uniform over 2^N", "Q_tau_uniform_all_median")]:
        ref = float(g[f].iloc[0]); v = num(r[col])
        chk("S14", f"N={n} S={S} {col}", v, ref, max(0.0006, abs(ref) * 0.02))
    beats = num(r["QAOA beats uniform-F, τ = 1 %"].split("/")[0])
    chk("S14", f"N={n} S={S} beats", beats, round(float(g.final_beats_uniform_F_rate.iloc[0]) * float(g.runs.iloc[0])), 0)
    chk("S14", f"N={n} S={S} beats opt", num(r["exact optimum"].split("/")[0]), float(g.final_beats_uniform_F_runs_opt.iloc[0]), 0)
    chk("S14", f"N={n} S={S} beats alt", num(r["τ = 5 %"].split("/")[0]), float(g.final_beats_uniform_F_runs_alt.iloc[0]), 0)

# Table S22: E3 levels and paired differences
e3 = pd.read_csv(R + "paper01_e3_summary.csv")
e3c = pd.read_csv(R + "paper01_e3_contrasts.csv")
OPT = {"ADAM 0.1 × 50": "adam_0.1", "ADAM 0.03 × 100": "adam_0.03", "L-BFGS-B": "lbfgs_100grad"}
for T, r in [(t, row) for t in ("VI", "S22") for row in table(t)]:
    n = int(num(r["N"])); pen = r["A"].replace("_", ""); opt = OPT[r["Setting"]]
    g = e3[(e3.n_assets == n) & (e3.penalty_label == pen) & (e3.optimizer == opt)]
    if "⟨H_norm⟩" in r:   # the compact main-text table drops the levels
        chk(T, f"N={n} {pen} {opt} <H>", num(r["⟨H_norm⟩"]), round(float(g.best_expectation_median.iloc[0]), 3), 6e-4)
        chk(T, f"N={n} {pen} {opt} Q", num(r["Q_τ(1000)"]), round(float(g.final_Q_tau_main_1000_median.iloc[0]), 3), 6e-4)
    if not pending(T, f"N={n} {pen} {opt} opt", r["Optimum hit / 50 (no feas.)"]):
        hits, nofeas = r["Optimum hit / 50 (no feas.)"].replace(")", "").split("(")
        chk(T, f"N={n} {pen} {opt} opt", num(hits.split("/")[0]),
            round(float(g.optimal_rate_unconditional.iloc[0]) * float(g.runs_ok.iloc[0])), 0)
        chk(T, f"N={n} {pen} {opt} runs", num(hits.split("/")[1]), float(g.runs_ok.iloc[0]), 0)
        chk(T, f"N={n} {pen} {opt} no-feasible", num(nofeas), float(g.no_feasible_shot_runs.iloc[0]), 0)
    if "Predicted" in r:
        # What the recorded exact states predict for the two counts under
        # independent draws; the column that exposed the L-BFGS-B handoff.
        phit, pnofeas = r["Predicted"].replace(")", "").split("(")
        chk(T, f"N={n} {pen} {opt} predicted hit", num(phit), round(float(g.expected_optimum_hits.iloc[0]), 1), 0.06)
        chk(T, f"N={n} {pen} {opt} predicted no-feasible", num(pnofeas), round(float(g.expected_no_feasible_shot_runs.iloc[0]), 1), 0.06)
    grad, secs = [num(x) for x in r["Budget (∇, s)"].split(",")]
    chk(T, f"N={n} {pen} {opt} grad", grad, float(g.gradient_evaluations_median.iloc[0]), 0.5)
    chk(T, f"N={n} {pen} {opt} s", secs, round(float(g.optimisation_seconds_median.iloc[0])), 0)
    if opt != "adam_0.1":
        c = e3c[(e3c.n_assets == n) & (e3c.penalty_label == pen) & (e3c.treatment == opt)]
        cH = c[c.field == "best_expectation"]; cP = c[c.field == "final_P_tau_main"]
        dH = _sci(r["Δ⟨H_norm⟩ vs. baseline [CI]; lower in"].split("[")[0].replace("**", ""))
        ref = float(cH.instance_median_diff.iloc[0])
        chk(T, f"N={n} {pen} {opt} dH", dH, ref, max(abs(ref) * 0.06, 5e-5))
        low = num(r["Δ⟨H_norm⟩ vs. baseline [CI]; lower in"].split(";")[-1])
        chk(T, f"N={n} {pen} {opt} lower-in", low, float(cH.instances_treatment_better.iloc[0]), 0)
        if "ΔP_τ [CI]" in r:
            dP = _sci(r["ΔP_τ [CI]"].split("[")[0]); refP = float(cP.instance_median_diff.iloc[0])
            chk(T, f"N={n} {pen} {opt} dP", dP, refP, max(abs(refP) * 0.1, 1e-9))
        # The batch probability is a separate measurement on the same pairs,
        # and the reason this column exists: a difference in P_tau too small to
        # print is worth a tenth of a batch at N = 20, A_margin.
        cQ = c[c.field == "final_Q_tau_main_1000"]
        cell = r["ΔQ_τ(1000) [CI]"]
        dQ = _sci(cell.split("[")[0]); refQ = float(cQ.instance_median_diff.iloc[0])
        lo, hi = [_sci(x) for x in cell.split("[")[1].split("]")[0].split(",")]
        sci = "·10" in cell
        tolQ = (lambda v: max(abs(v) * 0.06, 1e-7)) if sci else (lambda v: 6e-5)
        chk(T, f"N={n} {pen} {opt} dQ", dQ, refQ if sci else round(refQ, 4), tolQ(refQ))
        chk(T, f"N={n} {pen} {opt} dQ lo", lo, float(cQ.ci95_low.iloc[0]) if sci else round(float(cQ.ci95_low.iloc[0]), 4), tolQ(float(cQ.ci95_low.iloc[0])))
        chk(T, f"N={n} {pen} {opt} dQ hi", hi, float(cQ.ci95_high.iloc[0]) if sci else round(float(cQ.ci95_high.iloc[0]), 4), tolQ(float(cQ.ci95_high.iloc[0])))


# Table S23: the dispersion behind Table XIII's signed medians. Same rows, same
# CSV; the point of the table is that these columns are not recoverable from the
# median, so they are checked rather than trusted.
for r in table("S23"):
    n = int(num(r["N"])); pen = r["A"].replace("_", ""); opt = OPT[r["Setting"]]
    c = e3c[(e3c.n_assets == n) & (e3c.penalty_label == pen) & (e3c.treatment == opt)]
    for field, cols, conv in (
            ("final_P_tau_main",
             [("median |ΔP_τ|", "instance_median_abs_diff"),
              ("p95 |ΔP_τ|", "instance_p95_abs_diff")], _sci),
            ("final_Q_tau_main_1000",
             [("median |ΔQ|", "instance_median_abs_diff"),
              ("p95 |ΔQ|", "instance_p95_abs_diff")], num)):
        g = c[c.field == field]
        for col, key in cols:
            ref = float(g[key].iloc[0]); v = conv(r[col])
            chk("S23", f"N={n} {pen} {opt} {col}", v, ref, max(abs(ref) * 0.06, 6e-5))
        col = "Q_τ better/worse" if field.endswith("1000") else "P_τ better/worse"
        split = [num(x) for x in r[col].split("/")]
        for label, key, got in (("better", "instances_treatment_better", split[0]),
                                ("worse", "instances_worse", split[1])):
            chk("S23", f"N={n} {pen} {opt} {field} {label}", got, float(g[key].iloc[0]), 0)

# Tables S16/S17: threshold sensitivity and the feasible-mass split. The point
# of S16 is that the headline 76 % is one cell of it, so every cell is checked.
th = pd.read_csv(R + "paper01_e1_thresholds.csv")
TAUNAME = {"1 %": "tau_main", "5 %": "tau_alt", "exact": "optimum"}
for r in table("S16"):
    n = int(num(r["N"])); pen = r["A"].replace("_", "")
    tname = TAUNAME[r["quality rule"].strip()]
    for col, gt in (("0.01 %", 0.0001), ("0.05 %", 0.0005), ("0.1 %", 0.001), ("0.2 %", 0.002)):
        g = th[(th.n_assets == n) & (th.penalty_label == pen)
               & (th.tau_name == tname) & (th.g_off_threshold == gt)]
        chk("S16", f"N={n} {pen} {tname} {col}", num(r[col]),
            round(float(g.disagreement_rate.iloc[0]), 2), 6e-3)

sp = pd.read_csv(R + "paper01_e1_feasibility_split.csv")
for r in table("S17"):
    n = int(num(r["N"])); pen = r["A"].replace("_", "")
    g = sp[(sp.n_assets == n) & (sp.penalty_label == pen)]
    chk("S17", f"N={n} {pen} P_F", num(r["P_F"]), round(float(g.P_F_median.iloc[0]), 3), 6e-4)
    for col, key, d in (("Pr[g_F ≤ τ | x ∈ F]", "conditional_quality_rate_median", 0.06),
                        ("|F_τ|/|F|", "feasible_set_quality_share", 0.06)):
        ref = float(g[key].iloc[0])
        chk("S17", f"N={n} {pen} {key}", _sci(r[col]), ref, abs(ref) * 0.06)
    med, iqr = r["R_τ (IQR)"].split("(")
    chk("S17", f"N={n} {pen} R", num(med), round(float(g.enrichment_R_tau_median.iloc[0]), 2), 6e-3)
    lo, hi = [num(x) for x in iqr.rstrip(")").split("–")]
    chk("S17", f"N={n} {pen} R p25", lo, round(float(g.enrichment_R_tau_p25.iloc[0]), 2), 6e-3)
    chk("S17", f"N={n} {pen} R p75", hi, round(float(g.enrichment_R_tau_p75.iloc[0]), 2), 6e-3)
    chk("S17", f"N={n} {pen} R>1", num(r["R_τ > 1"].split("/")[0]),
        float(g.runs_with_R_tau_above_1.iloc[0]), 0)

# Table S19: the main contrast under subsets of the design. Every cell is a
# re-run of the same paired contrast, so every cell is checked -- the table's
# point is that one of them has the opposite sign.
sn = pd.read_csv(R + "paper01_e1_sensitivity.csv")
for r in table("S19"):
    subset = r["Subset"].strip()
    for col, n in (("N = 12", 12), ("N = 16", 16), ("N = 20", 20)):
        cell = r[col].strip()
        if cell in ("—", ""):
            continue
        g = sn[(sn.subset == subset) & (sn.n_assets == n)
               & (sn.field == "final_Q_tau_main_1000")]
        assert len(g) == 1, f"S18 {subset} {col}: {len(g)} matching rows"
        head, counts = cell.rsplit(";", 1)
        chk("S19", f"{subset} {col} median", num(head.split("[")[0]),
            round(float(g.instance_median_diff.iloc[0]), 3), 6e-4)
        if "[" in head:
            lo, hi = [num(x) for x in head.split("[")[1].rstrip("]").split(",")]
            chk("S19", f"{subset} {col} lo", lo, round(float(g.ci95_low.iloc[0]), 3), 6e-4)
            chk("S19", f"{subset} {col} hi", hi, round(float(g.ci95_high.iloc[0]), 3), 6e-4)
        better, total = [num(x) for x in counts.split("/")]
        chk("S19", f"{subset} {col} better", better,
            float(g.instances_treatment_better.iloc[0]), 0)
        chk("S19", f"{subset} {col} of", total, float(g.instances.iloc[0]), 0)

# Table S15: the E1 block on two price windows (E5). Six value columns:
# (N, window) pairs; the CSV has one row per (N, quantity) with an E1 and an E5 column.
e5 = pd.read_csv(R + "paper01_e5_period_compare.csv")
def e5v(n, q, w):
    g = e5[(e5.n_assets == n) & (e5.quantity == q)]
    return g[w].iloc[0]
def _nums(x):
    """Every signed number in a cell, left to right (bold stripped)."""
    return [float(v.replace("−", "-")) for v in re.findall(r"[−-]?\d+\.?\d*", re.sub(r"\*\*", "", str(x)))]
E5ROWS = {r["Quantity"]: r for r in table("S15")}
E5COMPACT = {r["Quantity"]: r for r in table("V")}
for q, row in E5COMPACT.items():
    assert q in E5ROWS and row == E5ROWS[q], f"Table V row differs from Table S15: {q}"
for n in (12, 16, 20):
    for w, tag in (("E1", "2023–26"), ("E5", "2020–23")):
        col = f"N = {n}, {tag}"; L = f"N={n} {tag}"
        chk("S15", f"{L} Δ_F", num(E5ROWS["Δ_F, median"][col]), round(float(e5v(n, "Δ_F, median", w)), 3), 6e-4)
        ov, z = _nums(E5ROWS["A_heur / A_crit, median (instances with A_crit = 0)"][col])
        chk("S15", f"{L} overshoot", ov, round(float(e5v(n, "A_heur / A_crit, median", w))), 0.5)
        chk("S15", f"{L} A_crit=0", z, float(e5v(n, "instances with A_crit = 0", w)), 0)
        chk("S15", f"{L} Q", num(E5ROWS["A_heur: Q_τ(1000)"][col]), round(float(e5v(n, "Aheur: Q_τ(1000) median", w)), 3), 6e-4)
        go, gf = _nums(E5ROWS["A_heur: g_off (%) / g_F (%)"][col])
        chk("S15", f"{L} g_off", go, round(float(e5v(n, "Aheur: g_off median (%)", w)), 3), 6e-4)
        chk("S15", f"{L} g_F", gf, round(float(e5v(n, "Aheur: g_F median (%)", w)), 2), 6e-3)
        chk("S15", f"{L} pass/fail", num(E5ROWS["A_heur: pass g_off ≤ 0.1 %, fail g_F ≤ 1 %"][col]),
            round(float(e5v(n, "Aheur: pass g_off ≤ 0.1 % but fail g_F ≤ τ", w)), 2), 6e-3)
        chk("S15", f"{L} no-feasible", num(E5ROWS["A_heur: runs with no feasible shot"][col]),
            float(e5v(n, "Aheur: runs with no feasible shot", w)), 0)
        for label, fld, nd in (("Paired Q_τ(1000) [CI]; better of 10", "final_Q_tau_main_1000", 3),
                               ("Paired g_off (pt) [CI]; better of 10", "g_off", 2),
                               ("Paired g_F (pt) [CI]; better of 10", "g_F", 2)):
            v = _nums(E5ROWS[label][col]); d, lo, hi, better = v[0], v[1], v[2], v[3]
            for name, val, ref in (("diff", d, f"Amargin − Aheur: {fld} median diff"),
                                   ("lo", lo, f"Amargin − Aheur: {fld} CI low"),
                                   ("hi", hi, f"Amargin − Aheur: {fld} CI high")):
                chk("S15", f"{L} {fld} {name}", val, round(float(e5v(n, ref, w)), nd), 0.6 * 10 ** -nd)
            bt = e5v(n, f"Amargin − Aheur: {fld} instances better / tied / of", w).split("/")
            chk("S15", f"{L} {fld} better", better, float(bt[0]), 0)
            if len(v) > 4: chk("S15", f"{L} {fld} tied", v[4], float(bt[1]), 0)
        tot, sol, den, pairs = _nums(E5ROWS["g_off ratio / solution / denominator, medians (pairs)"][col])
        ct, cs, cd = [float(x) for x in e5v(n, "Amargin / Aheur g_off ratio: median total / solution / denominator", w).split("/")]
        chk("S15", f"{L} ratio", tot, ct, 0.06); chk("S15", f"{L} ratio sol", sol, cs, 0.006); chk("S15", f"{L} ratio den", den, cd, 0.06)
        chk("S15", f"{L} ratio pairs", pairs, float(e5v(n, "Amargin / Aheur g_off ratio: pairs", w)), 0)
        gt, gs, gd = _nums(E5ROWS["g_off ratio = solution × denominator, geometric means"][col])
        cgt, cgs, cgd = _nums(e5v(n, "Amargin / Aheur g_off ratio: geometric total = solution × denominator", w))
        chk("S15", f"{L} geo", gt, cgt, 0.06); chk("S15", f"{L} geo sol", gs, cgs, 0.006)
        chk("S15", f"{L} geo den", gd, cgd, 0.06)
        chk("S15", f"{L} geo identity", gt, gs * gd, max(0.05 * gt, 0.06))
        beats = _nums(E5ROWS["E2: A_heur beats uniform-F at S = 1,000"][col])[0]
        chk("S15", f"{L} E2 beats", beats, round(float(e5v(n, "Aheur: E2 optimised beats uniform-F at S = 1000 (rate)", w)) * 50), 0)

# Table S18: the conditional distribution on F, per cell, from the run artifacts' summary.
ct = pd.read_csv(R + "paper01_e1_conditional_tv_summary.csv")
PEN = {"A = 0": "A0", "1.1·A_crit": "1.1xAcrit", "2·A_crit": "2xAcrit", "5·A_crit": "5xAcrit",
       "10·A_crit": "10xAcrit", "A_margin": "Amargin", "A_heur": "Aheur"}
for r in table("S18"):
    n = int(num(r["N"])); pen = PEN[r["A"].strip()]
    g = ct[(ct.n_assets == n) & (ct.penalty_label == pen)]
    runs, undef = _nums(r["Runs (undefined)"])
    chk("S18", f"N={n} {pen} runs", runs, float(g.runs.iloc[0]), 0)
    chk("S18", f"N={n} {pen} undefined", undef, float(g.runs_undefined.iloc[0]), 0)
    chk("S18", f"N={n} {pen} P_F", num(r["P_F"]), round(float(g.final_P_F_median.iloc[0]), 3), 6e-4)
    med, q1, q3 = _nums(r["D_cond median (IQR)"])
    chk("S18", f"N={n} {pen} D median", med, round(float(g.final_D_cond_median.iloc[0]), 3), 6e-4)
    chk("S18", f"N={n} {pen} D p25", q1, round(float(g.final_D_cond_p25.iloc[0]), 3), 6e-4)
    chk("S18", f"N={n} {pen} D p75", q3, round(float(g.final_D_cond_p75.iloc[0]), 3), 6e-4)
    lo, hi = _nums(r["D_cond min – max"])
    chk("S18", f"N={n} {pen} D min", lo, round(float(g.final_D_cond_min.iloc[0]), 3), 6e-4)
    chk("S18", f"N={n} {pen} D max", hi, round(float(g.final_D_cond_max.iloc[0]), 3), 6e-4)
    chk("S18", f"N={n} {pen} initial", num(r["Initial-state D_cond"]), round(float(g.initial_D_cond_median.iloc[0]), 3), 6e-4)
    chk("S18", f"N={n} {pen} uniform 1k", num(r["Uniform, 1,000 shots"]), round(float(g.uniform_F_1000_shot_D_cond.iloc[0]), 3), 6e-4)
    chk("S18", f"N={n} {pen} point mass", num(r["Point mass"]), round(float(g.point_mass_D_cond_median.iloc[0]), 3), 6e-4)

# Table IV (main text): the D_cond summary, from the same file as S18 plus R_τ
# from the feasibility split. Every cell is checked.
for r in table("IV"):
    n = int(num(r["N"]))
    g = ct[(ct.n_assets == n) & (ct.penalty_label == "Aheur")]
    med, mx = _nums(r["D_cond at A_heur, median (max)"])
    chk("IV", f"N={n} Aheur D median", med, round(float(g.final_D_cond_median.iloc[0]), 3), 6e-4)
    chk("IV", f"N={n} Aheur D max", mx, round(float(g.final_D_cond_max.iloc[0]), 3), 6e-4)
    gm = ct[(ct.n_assets == n) & (ct.penalty_label == "Amargin")]
    chk("IV", f"N={n} Amargin D median", num(r["D_cond at A_margin, median"]),
        round(float(gm.final_D_cond_median.iloc[0]), 3), 6e-4)
    chk("IV", f"N={n} |F|", num(r["|F|"]), float(g.feasible_set_size_median.iloc[0]), 0)
    for pen, col, digits in (("Aheur", "R_τ at A_heur", 3), ("Amargin", "R_τ at A_margin", 2)):
        gs = sp[(sp.n_assets == n) & (sp.penalty_label == pen)]
        chk("IV", f"N={n} {pen} R", num(r[col]), round(float(gs.enrichment_R_tau_median.iloc[0]), digits), 6e-4 if digits == 3 else 6e-3)

# Table S34: initial → final of the same run at a fixed A (review round 3, M4).
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
def _signed(cell):
    """Every signed number in a cell, scientific notation (a·10⁻³) included."""
    t = str(cell).replace("−", "-").replace("·10⁻", "e-").replace("·10", "e").translate(SUP)
    return [float(v) for v in re.findall(r"[+-]?\d+\.?\d*(?:e[+-]?\d+)?", t)]
inifin = pd.read_csv(R + "paper01_e1_initial_final.csv")
S34F = {"P_F": "P_F", "P_τ": "P_tau_main", "Q_τ(1000)": "Q_tau_main_1000", "E[V]": "expected_violation",
        "R_τ": "R_tau", "D_cond": "D_cond"}
for r in table("S34"):
    n = int(num(r["N"])); pen = r["A"].replace("_", ""); f = S34F[r["Quantity"].strip()]
    g = inifin[(inifin.n_assets == n) & (inifin.penalty_label == pen) & (inifin.field == f)]
    tol = lambda v: max(abs(v) * 0.06, 1e-5) if f == "P_tau_main" else (0.06 if f == "expected_violation" else 6e-4)
    for col, key in (("Initial", "initial_median"), ("Final", "final_median")):
        ref = float(g[key].iloc[0]); chk("S34", f"N={n} {pen} {f} {col}", _signed(r[col])[0], ref, tol(ref))
    d, lo, hi = _signed(r["Paired diff [95 % CI]"])
    for name, key, v in (("diff", "paired_diff_median", d), ("lo", "paired_diff_ci95_low", lo), ("hi", "paired_diff_ci95_high", hi)):
        ref = float(g[key].iloc[0]); chk("S34", f"N={n} {pen} {f} {name}", v, ref, tol(ref))
    chk("S34", f"N={n} {pen} {f} up", num(r["Increased of 30"]), float(g.instances_increased.iloc[0]), 0)

# Table S35: the objective's scale in the normalised Hamiltonian (review round 3, M6).
cs = pd.read_csv(R + "paper01_coefficient_scale_summary.csv")
PENC = {"A_heur": "Aheur", "A_margin": "Amargin", "1.1·A_crit": "1.1xAcrit"}
for r in table("S35"):
    n = int(num(r["N"])); pen = PENC[r["A"].strip()]
    g = cs[(cs.n_assets == n) & (cs.penalty_label == pen)]
    for col, f in (("QUBO ratio", "qubo_ratio_objective_over_max"), ("Ising ratio", "ising_ratio_objective_over_s_H"), ("ε_F = Δ_F / s_H", "epsilon_F")):
        med, lo, hi = _signed(r[col])
        for name, key, v in (("median", f + "_median", med), ("min", f + "_min", lo), ("max", f + "_max", hi)):
            chk("S35", f"N={n} {pen} {col} {name}", v, round(float(g[key].iloc[0]) * 100, 2), 6e-3)

# Table S36: the precision diagnostic scored on acquisition (review round 3, M7 / A3).
# Each cell is "median (max)" of |single − double| over the 8 pairs of a size, in
# two significant digits; the last column counts the pairs whose batch verdict agrees.
pa = pd.read_csv(R + "paper01_e3_precision_pairs.csv")
S36F = {"|Δ⟨H_norm⟩|": "best_expectation", "|ΔP_F|": "final_P_F", "|ΔP_τ|": "final_P_tau_main",
        "|ΔQ_τ(1000)|": "final_Q_tau_main_1000", "|ΔD_cond|": "final_D_cond"}
def _sig2(v):
    """Round to two significant digits, as the table prints its cells."""
    return float(f"{v:.1e}")
for r in table("S36"):
    n = int(num(r["N"])); g = pa[pa.n_assets == n]
    chk("S36", f"N={n} pairs", num(r["Pairs"]), float(len(g)), 0)
    for col, f in S36F.items():
        med, mx = _signed(r[col])
        d = g[f + "_diff"].abs()
        chk("S36", f"N={n} {f} median", med, _sig2(float(d.median())), abs(med) * 0.06)
        chk("S36", f"N={n} {f} max", mx, _sig2(float(d.max())), abs(mx) * 0.06)
    if not pending("S36", f"N={n} verdict agrees", r["Verdict agrees"]):
        chk("S36", f"N={n} verdict agrees", num(r["Verdict agrees"]), float(g.within_tau_main_agree.sum()), 0)

# Table S24: multistart at equal shots. The gain column is the table's point,
# so its interval and its count are checked cell by cell.
ms = pd.read_csv(R + "paper01_e3_multistart.csv")
for r in table("S24"):
    n = int(num(r["N"])); pen = PEN[r["A"].strip()]
    g = ms[(ms.n_assets == n) & (ms.penalty_label == pen)]
    for col, key in (("1×1k", "single_seed_Q_tau_1000_median"),
                     ("random 1×5k", "random_seed_Q_tau_5000_median"),
                     ("5×1k pooled", "pooled_5x1000_Q_tau_median"),
                     ("unif-F 5k", "uniform_F_Q_tau_5000_median"),
                     ("Opt, random", "random_seed_P_opt_5000_median"),
                     ("Opt, pooled", "pooled_5x1000_P_opt_median")):
        chk("S24", f"N={n} {pen} {key}", num(r[col]), round(float(g[key].iloc[0]), 3), 6e-4)
    gain, lo, hi = _nums(r["Gain [95 % CI]"])
    chk("S24", f"N={n} {pen} gain", gain, round(float(g.pooled_minus_random_Q_tau_median.iloc[0]), 3), 6e-4)
    chk("S24", f"N={n} {pen} gain lo", lo, round(float(g.pooled_minus_random_Q_tau_ci95_low.iloc[0]), 3), 6e-4)
    chk("S24", f"N={n} {pen} gain hi", hi, round(float(g.pooled_minus_random_Q_tau_ci95_high.iloc[0]), 3), 6e-4)
    chk("S24", f"N={n} {pen} gain>0.01", num(r["> 0.01 of 30"]), float(g["instances_pooled_gains_over_0.01_Q_tau"].iloc[0]), 0)
    a, b = _nums(r["Opt. time"])
    chk("S24", f"N={n} {pen} s single", a, round(float(g.single_seed_optimisation_seconds_median.iloc[0])), 0)
    chk("S24", f"N={n} {pen} s five", b, round(float(g.all_seeds_optimisation_seconds_median.iloc[0])), 0)

# E1 reproduces the retired seed sweep bit for bit (instance 0, A_heur)
e1 = pd.read_csv(R + "paper01_e1_cross.csv")
ss = pd.read_csv(R + "paper01_seed_sweep.csv")
blk = e1[(e1.instance == 0) & (e1.penalty_label == "Aheur")]
m = blk.merge(ss[ss.n_assets >= 12], on=["n_assets", "qaoa_seed"])   # the sweep also ran N = 8
chk("E1", "seed-sweep rows matched", float(len(m)), float((ss.n_assets >= 12).sum()), 0)
# Same bitstrings: the two evaluators differ by an ulp (4e-16 relative), not by a solution.
rel = ((m.best_all_E - m.energy_quantum) / m.energy_quantum).abs()
chk("E1", "seed-sweep energies within 1e-12", float((rel < 1e-12).sum()), float(len(m)), 0)

# Table S30: E4-A, numerics (max over circuits) and sampling (ranges per size)
cmp = pd.read_csv(R + "paper01_e4_states_compare.csv")
e4s = pd.read_csv(R + "paper01_e4_batches_summary.csv")
E4ARM = {"CPU double, 1 thread": "cpu_double", "GPU single": "gpu_single",
         "SV1 double (cloud)": "sv1", "GPU double (reference)": "gpu_double"}
def _rng(cell):
    v = [num(x) for x in re.split(r"[–-]", str(cell).replace("**", ""))]
    return (v[0], v[-1])
for T, r in [(t, row) for t in ("VII", "S30") for row in table(t)]:
    arm = E4ARM.get(r["Arm"])
    if arm is None:
        continue
    g = cmp[cmp.arm == arm]
    if arm != "gpu_double":
        if "max |Δp| on F" in r:
            chk(T, f"{arm} max|dp|", _sci(r["max |Δp| on F"]), float(g.max_abs_dp_F.max()), float(g.max_abs_dp_F.max()) * 0.06)
            chk(T, f"{arm} TV", _sci(r["TV on F"]), float(g.tv_F.max()), float(g.tv_F.max()) * 0.06)
        if "Bound at S = 1,000" in r:
            # 1 − (1 − d)^S with S = 1,000: the coupling bound of Sec. V-D, recomputed.
            dmax = float(g.tv_F_perp.max()); bound = scoring.selected_output_tv_bound(dmax, 1000)
            chk(T, f"{arm} bound", _sci(r["Bound at S = 1,000"]), bound, bound * 0.06)
        # The distribution the selection rule sees: feasible outcomes plus one
        # symbol for "no feasible shot". tv_F on its own is a partial sum.
        chk(T, f"{arm} TV F+perp", _sci(r["TV on F ∪ {⊥}"]),
            float(g.tv_F_perp.max()), float(g.tv_F_perp.max()) * 0.06)
        chk(T, f"{arm} dPtau", _sci(r["|ΔP_τ/P_τ|"]), float(g.rel_dP_tau_main.abs().max()), float(g.rel_dP_tau_main.abs().max()) * 0.06)
    b = e4s[e4s.arm == arm]
    for n, col in ((12, "Distinct best, N = 12 (seed 42 / 43)"), (16, "N = 16 (42 / 43)")):
        for seed, cell in zip((42, 43), r[col].split("/")):
            gg = b[(b.n_assets == n) & (b.qaoa_seed == seed)]
            lo, hi = _rng(cell)
            chk(T, f"{arm} N={n} s{seed} distinct lo", lo, float(gg.distinct_best_feasible.min()), 0)
            chk(T, f"{arm} N={n} s{seed} distinct hi", hi, float(gg.distinct_best_feasible.max()), 0)
    gg = b[b.n_assets == 20]
    lo, hi = _rng(r["N = 20"])
    chk(T, f"{arm} N=20 distinct lo", lo, float(gg.distinct_best_feasible.min()), 0)
    chk(T, f"{arm} N=20 distinct hi", hi, float(gg.distinct_best_feasible.max()), 0)
    gg = b[(b.n_assets == 16) & (b.qaoa_seed == 43)]
    lo, hi = _rng(r["No-feasible-shot batches, N = 16 seed 43"])
    chk(T, f"{arm} noF lo", lo, float(gg.no_feasible_shot_batches.min()), 0)
    chk(T, f"{arm} noF hi", hi, float(gg.no_feasible_shot_batches.max()), 0)

# Table S37: exact-optimum share per E1 setting (review round 4, P02).
hits = pd.read_csv(R + "paper01_e1_optimum_hits.csv")
S38 = {"A = 0": "A0", "1.1·A_crit": "1.1xAcrit", "2·A_crit": "2xAcrit", "5·A_crit": "5xAcrit",
       "10·A_crit": "10xAcrit", "A_margin": "Amargin", "A_heur": "Aheur"}
for r in table("S37"):
    n = int(num(r["N"]))
    for col, lab in S38.items():
        g = hits[(hits.n_assets == n) & (hits.penalty_label == lab)]
        if r[col].strip() == "—":
            chk("S37", f"N={n} {lab} absent", 0.0, float(len(g)), 0); continue
        m = re.match(r"(\d+)/(\d+) \((\d+) %\)", r[col].strip())
        h, f, pct = [float(x) for x in m.groups()]
        chk("S37", f"N={n} {lab} hits", h, float(g.optimum_hits.iloc[0]), 0)
        chk("S37", f"N={n} {lab} feasible", f, float(g.runs_with_feasible_shot.iloc[0]), 0)
        # The caption's rule for completed runs per cell.
        chk("S37", f"N={n} {lab} runs", 45.0 if lab == "A0" else (105.0 if n == 12 and "Acrit" in lab else 150.0),
            float(g.runs.iloc[0]), 0)
        chk("S37", f"N={n} {lab} share", pct, round(100 * float(g.share_of_runs_with_feasible_shot.iloc[0])), 0)

# Table S38: the paired P_τ ratio as P_F ratio × R_τ ratio (review round 4, C02).
fac = pd.read_csv(R + "paper01_e1_acquisition_factors.csv")
for r in table("S38"):
    n = int(num(r["N"])); g = fac[fac.n_assets == n]
    chk("S38", f"N={n} pairs", num(r["Pairs"]), float(g.pairs.iloc[0]), 0)
    chk("S38", f"N={n} P_tau geo", num(r["P_τ ratio, geometric mean"]), round(float(g.P_tau_ratio_geomean.iloc[0]), 2), 6e-3)
    a, b = [num(x) for x in r["= P_F ratio × R_τ ratio"].split("×")]
    chk("S38", f"N={n} P_F geo", a, round(float(g.P_F_ratio_geomean.iloc[0]), 2), 6e-3)
    chk("S38", f"N={n} R geo", b, round(float(g.R_tau_ratio_geomean.iloc[0]), 2), 6e-3)
    chk("S38", f"N={n} share", num(r["Share of the log gain in P_F"]), round(100 * float(g.log_share_P_F.iloc[0])), 0)
    m1, m2, m3 = [num(x) for x in r["Medians (P_τ / P_F / R_τ ratio)"].split("/")]
    chk("S38", f"N={n} med P_tau", m1, round(float(g.P_tau_ratio_median.iloc[0]), 2), 6e-3)
    chk("S38", f"N={n} med P_F", m2, round(float(g.P_F_ratio_median.iloc[0]), 2), 6e-3)
    chk("S38", f"N={n} med R", m3, round(float(g.R_tau_ratio_median.iloc[0]), 2), 6e-3)
    chk("S38", f"N={n} R>1", num(r["Pairs with R_τ ratio > 1"]), float(g.pairs_R_tau_ratio_above_1.iloc[0]), 0)


# ---------------------------------------------------------------------------
# Coverage pass (2026-09-14). A per-table count of numeric cells against checks
# showed nine supplement tables with no check at all (S1, S4, S8, S9, S10, S21,
# S25, S31, S33) and five checked only in part (S2, S3, S7, S12, S28): the
# v1.x-era tables, transcribed by hand from the CSVs before this file existed.
# Every cell below was re-derived from its result file before being wired in;
# all agreed. The one column no file can reproduce is S31's "Circuit time /
# T2", whose gate time and T2 are stated nowhere -- it is listed, not scored.

def _pct_cell(x):
    """'0.053 %' → 0.053; '— (infeasible)' → None."""
    return None if "—" in str(x) else num(x)

# Table S2: the single-instance benchmark, from paper01_results.csv (energies,
# gap, runtime, feasibility; ⟨H_norm⟩ is checked above from the convergence).
res = pd.read_csv(R + "paper01_results.csv")
for r in table("S2"):
    n = int(num(r["N"]))
    cl = res[(res.n_assets == n) & (res.solver == "classical_auto")].iloc[0]
    chk("S2", f"N={n} K", num(r["K"]), float(cl.n_select), 0)
    chk("S2", f"N={n} classical E", num(r["Classical E"]), round(float(cl.energy_no_offset), 2), 6e-3)
    qs = res[(res.n_assets == n) & (res.solver == "lightning_gpu")]
    if not len(qs):
        assert "—" in r["QAOA E"] and "infeasible" in r["Offset-normalized gap g_off"], f"S2 N={n}"
        continue
    q = qs.iloc[0]
    chk("S2", f"N={n} QAOA E", num(r["QAOA E"]), round(float(q.energy_no_offset), 2), 6e-3)
    g_off = (q.energy_no_offset - cl.energy_no_offset) / abs(cl.energy_no_offset) * 100
    chk("S2", f"N={n} g_off", _pct_cell(r["Offset-normalized gap g_off"]), round(g_off, 3), 6e-4)
    t = str(r["QAOA runtime"])
    secs = num(t) * (3600 if t.endswith("h") else 60 if t.endswith("min") else 1)
    chk("S2", f"N={n} runtime", secs, float(q.runtime_s), 0.06 * float(q.runtime_s))
    chk("S2", f"N={n} feasible", 1.0, float(bool(q.feasible)), 0)

# Table S3: |E*|, the band fraction and its Sharpe range, from the degeneracy
# files (C(N,K), count, deflation and spread are checked above).
deg = pd.read_csv(R + "paper01_degeneracy.csv")
degq = pd.read_csv(R + "paper01_degeneracy_quantum.csv")
for r in table("S3"):
    n = int(num(r["N"]))
    b = deg[(deg.n_assets == n) & (deg.band_kind == "reported_gap") & (deg.band_pct == 0.10)].iloc[0]
    pq = degq[degq.n_assets == n].iloc[0]
    chk("S3", f"N={n} |E*|", num(r["|E*|"]), round(abs(float(pq.energy_optimum)), 1), 0.06)
    pct = num(re.search(r"\(([\d.]+) %\)", r["In 0.1 % band"]).group(1))
    chk("S3", f"N={n} band %", pct, round(float(b.frac_within) * 100, 1), 0.06)
    lo, hi = [num(x) for x in r["Sharpe there"].split("–")]
    chk("S3", f"N={n} Sharpe min", lo, round(float(b.sharpe_min), 3), 6e-4)
    chk("S3", f"N={n} Sharpe max", hi, round(float(b.sharpe_max), 3), 6e-4)

# Table S4: the published QAOA solutions in the full ranking.
for r in table("S4"):
    n = int(num(r["N"])); g = degq[degq.n_assets == n].iloc[0]
    chk("S4", f"N={n} gap", num(r["Gap as reported"]), round(float(g.reported_gap_pct), 4), 6e-5)
    chk("S4", f"N={n} gap/spread", num(r["Gap vs objective spread"]), round(float(g.gap_of_objective_range_pct), 2), 6e-3)
    rk, of = [num(x) for x in r["Rank"].split("/")]
    chk("S4", f"N={n} rank", rk, float(g["rank"]), 0)
    chk("S4", f"N={n} of", of, float(g.feasible_set_size), 0)
    chk("S4", f"N={n} percentile", num(r["Percentile"]), round(float(g.percentile), 2), 6e-3)
    m = re.match(r"([\d.]+) → ([\d.]+) \(([−+-][\d.]+) %\)", r["Sharpe (optimum → QAOA)"])
    chk("S4", f"N={n} Sharpe opt", float(m.group(1)), round(float(g.sharpe_optimum), 3), 6e-4)
    chk("S4", f"N={n} Sharpe QAOA", float(m.group(2)), round(float(g.sharpe_quantum), 3), 6e-4)
    chk("S4", f"N={n} Sharpe delta", num(m.group(3)), round(float(g.sharpe_delta_pct), 1), 0.06)

# Table S7: the rest of the penalty table (A used and overshoot are above).
pen = pd.read_csv(R + "paper01_penalty.csv")
for r in table("S7"):
    n = int(num(r["N"])); g = pen[pen.n_assets == n].iloc[0]
    chk("S7", f"N={n} K", num(r["K"]), float(g.n_select), 0)
    chk("S7", f"N={n} A_crit", num(r["A_crit (A_safe for N > 20)"]), round(float(g.penalty_min), 4), 6e-5)
    chk("S7", f"N={n} gap at A used", num(r["Gap at A used"]), round(float(g.gap_at_penalty_used_pct), 4), 6e-5)
    chk("S7", f"N={n} gap at A_crit", num(r["Gap at A_crit"]), round(float(g.gap_at_penalty_min_pct), 2), 6e-3)
    chk("S7", f"N={n} gap vs spread", num(r["Gap vs objective spread"]), round(float(g.gap_of_objective_range_pct), 2), 6e-3)

# Table S8: the all-states approximation ratio. The heuristic-A rows are the
# ``published_qaoa`` / ``optimum`` rows of the E0 re-scoring at A_heur. The
# 10× rows are not in that file (its multiples are of A_crit, not A_heur);
# they follow from the same rows because, for a feasible x, AR = 1 − (f_x −
# f*)/(C_max − E*) and C_max − E* = f(x_max) − f* + A(m − K)² grows by
# (A' − A)·(m − K)² when the
# maximising assignment (cardinality m, all-ones here) does not change --
# which the recorded ``c_max_cardinality == N`` supports and a direct
# recomputation on 2026-09-14 confirmed to six decimals.
e0r = pd.read_csv(R + "paper01_e0_rescore.csv")
for r in table("S8"):
    n = int(num(r["N"])); mult = 10.0 if "10×" in r["Penalty A"] else 1.0
    g = e0r[(e0r.n_assets == n) & (e0r.penalty_label == "Aheur")]
    opt = g[g.solution == "optimum"].iloc[0]; pub = g[g.solution == "published_qaoa"].iloc[0]
    assert int(opt.c_max_cardinality) == n, f"S8 N={n}: C_max not at the all-ones assignment"
    chk("S8", f"N={n} ×{mult:g} A", num(r["Penalty A"]), round(float(opt.penalty) * mult, 0), 0.5)
    k = int(opt.n_select); d = float(opt.c_max_all_states - opt.E_A_star)
    d_scaled = d + (mult - 1) * float(opt.penalty) * (n - k) ** 2
    for col, row in (("AR, our QAOA solution", pub), ("AR, worst feasible", g[g.solution == "worst_feasible"].iloc[0])):
        ar = 1 - (float(row.f) - float(row.f_star)) / d_scaled
        if mult == 1:
            assert abs(ar - float(row.r_all)) < 1e-9, f"S8 N={n}: identity check {ar} vs {row.r_all}"
        chk("S8", f"N={n} ×{mult:g} {col}", num(r[col]), round(ar, 6), 6e-7)

# Table S9: the 30-instance census. Overshoot uses the exact A_crit of the E1
# e1 (A_safe of the instance file is only an upper bound; it makes 7, not
# 9, of the N = 12 instances A_crit = 0) for N ≤ 20 and A_safe beyond.
inst = pd.read_csv(R + "paper01_instances.csv")
ac = e1[e1.penalty_label == "Aheur"].groupby(["n_assets", "instance"]).agg(
    a_crit=("a_crit", "first"), a_heur=("a_heur", "first")).reset_index()
for r in table("S9"):
    n = int(num(r["N"])); g = inst[inst.n_assets == n]
    m = re.match(r"(\d+)× \((\d+)–(\d+)\)", r["Deflation med. (range)"])
    chk("S9", f"N={n} deflation median", float(m.group(1)), round(float(g.deflation_factor.median()), 0), 0.5)
    chk("S9", f"N={n} deflation min", float(m.group(2)), round(float(g.deflation_factor.min()), 0), 0.5)
    chk("S9", f"N={n} deflation max", float(m.group(3)), round(float(g.deflation_factor.max()), 0), 0.5)
    chk("S9", f"N={n} ref inst", num(r["Ref. inst."]), round(float(g[g.is_published_instance].deflation_factor.iloc[0]), 0), 0.5)
    chk("S9", f"N={n} band", num(r["In 0.1 % band"]), round(float(g.band_count.median()), 0), 0.5)
    chk("S9", f"N={n} Sharpe spread", num(r["Sharpe spread"]), round(float(g.band_sharpe_spread_pct.median()), 0), 0.5)
    if n <= 20:
        e = ac[ac.n_assets == n]; pos = e[e.a_crit > 0]
        over = (pos.a_heur / pos.a_crit).median(); zeros = int((e.a_crit == 0).sum())
    else:
        pos = g[g.penalty_min > 0]; over = pos.penalty_overshoot.median(); zeros = int((g.penalty_min == 0).sum())
    chk("S9", f"N={n} overshoot", num(r["Overshoot"]), round(float(over), 0), 0.5)
    mm = re.search(r"over the (\d+) with A_crit > 0; (\d+)/30 have A_crit = 0", r["Overshoot"])
    if mm:
        chk("S9", f"N={n} overshoot count", float(mm.group(1)), float(len(pos)), 0)
        chk("S9", f"N={n} A_crit = 0 count", float(mm.group(2)), float(zeros), 0)
    else:
        chk("S9", f"N={n} A_crit = 0 count", 0.0, float(zeros), 0)

# Table S10: annealing and QAOA over the same instances.
ann = pd.read_csv(R + "paper01_instances_annealing.csv")
iq = pd.read_csv(R + "paper01_instances_qaoa.csv")
def table_rows(n):
    """Positional cells of a table whose header repeats a name ("median
    runtime" twice in S10), which ``table``'s dict rows would collapse."""
    m = re.search(rf"\*\*Table {n}\. ", s_supp)
    rows = []
    for l in s_supp[m.end():].split("\n"):
        if l.startswith("|"):
            rows.append([x.strip() for x in l.strip("|").split("|")])
        elif rows:
            break
    return [r for r in rows[1:] if not re.match(r"^[\s:|-]+$", "".join(r))]
for n_, sa, sa_t, qa, qa_t in table_rows("S10"):
    n = int(num(n_)); g = ann[ann.n_assets == n]
    hit, tot = [num(x) for x in sa.split("/")]
    chk("S10", f"N={n} SA optimal", hit, float((g["rank"] == 1).sum()), 0)
    chk("S10", f"N={n} SA runs", tot, float(len(g)), 0)
    chk("S10", f"N={n} SA runtime", num(sa_t), round(float(g.runtime_s.median()), 1), 0.06)
    gq = iq[iq.n_assets == n]
    if "—" in qa:
        assert not len(gq), f"S10 N={n}: QAOA rows exist but the table says none"
        continue
    hit, tot = [num(x) for x in qa.split("/")]
    chk("S10", f"N={n} QAOA optimal", hit, float((gq.quantum_rank == 1).sum()), 0)
    chk("S10", f"N={n} QAOA runs", tot, float(len(gq)), 0)
    chk("S10", f"N={n} QAOA runtime", num(qa_t), round(float(gq.quantum_runtime_s.median()), 1), 0.06)

# Table S12: the rest of the instance-axis QAOA table (exact count is above).
for r in table("S12"):
    n = int(num(r["N"])); g = iq[iq.n_assets == n]
    med, mx = re.match(r"([\d.]+) % \(([\d.]+) %\)", r["Gap med. (max)"]).groups()
    chk("S12", f"N={n} gap median", float(med), round(float(g.reported_gap_pct.median()), 4), 6e-5)
    chk("S12", f"N={n} gap max", float(mx), round(float(g.reported_gap_pct.max()), 4), 6e-5)
    chk("S12", f"N={n} gap/spread", num(r["Gap/spread med."]), round(float(g.gap_of_objective_range_pct.median()), 2), 6e-3)
    med, mx = re.match(r"(\d+) \((\d+)\)", r["Rank med. (max)"]).groups()
    chk("S12", f"N={n} rank median", float(med), round(float(g.quantum_rank.median()), 0), 0.5)
    chk("S12", f"N={n} rank max", float(mx), float(g.quantum_rank.max()), 0)
    med, mn = re.match(r"([−+-][\d.]+) % \(([−+-][\d.]+) %\)", r["ΔSharpe med. (worst)"]).groups()
    chk("S12", f"N={n} dSharpe median", num(med), round(float(g.sharpe_delta_pct.median()), 1), 0.06)
    chk("S12", f"N={n} dSharpe worst", num(mn), round(float(g.sharpe_delta_pct.min()), 1), 0.06)

# Table S21: the last optimised expectation of three grid cells at seed 42.
gconv = pd.read_csv(R + "paper01_grid_convergence.csv")
for r in table("S21"):
    n, p_ = [int(x) for x in re.match(r"N = (\d+), p = (\d+)", r["final ⟨H_norm⟩"]).groups()]
    for col, steps in (("steps = 50", 50), ("100", 100), ("400", 400)):
        x = gconv[(gconv.n_assets == n) & (gconv.p_layers == p_) & (gconv.qaoa_seed == 42) & (gconv.steps == steps)]
        chk("S21", f"N={n} p={p_} steps={steps}", num(r[col]), round(float(x.sort_values("step").expectation.iloc[-1]), 4), 6e-5)

# Table S25: the same code on two backends, at the ten decimals the table prints.
res_rt = pd.read_csv(R + "paper01_results.csv", float_precision="round_trip")
sv1 = pd.read_csv(R + "paper01_results_sv1.csv", float_precision="round_trip")
for r in table("S25"):
    n = int(num(r["N"]))
    gq = res_rt[(res_rt.n_assets == n) & (res_rt.solver == "lightning_gpu")].iloc[0]
    gc = res_rt[(res_rt.n_assets == n) & (res_rt.solver == "classical_auto")].iloc[0]
    sq = sv1[(sv1.n_assets == n) & (sv1.solver == "braket_sv1")].iloc[0]
    chk("S25", f"N={n} E gpu", num(r["E, lightning.gpu"]), round(float(gq.energy_no_offset), 10), 6e-11)
    chk("S25", f"N={n} E sv1", num(r["E, SV1"]), round(float(sq.energy_no_offset), 10), 6e-11)
    chk("S25", f"N={n} bitwise", 1.0 if "✓" in r["Bitwise equal"] else 0.0, float(gq.energy_no_offset == sq.energy_no_offset), 0)
    gap = lambda e: (e - gc.energy_no_offset) / abs(gc.energy_no_offset) * 100
    chk("S25", f"N={n} gap gpu", num(r["Gap, GPU"]), round(float(gap(gq.energy_no_offset)), 3), 6e-4)
    chk("S25", f"N={n} gap sv1", num(r["Gap, SV1"]), round(float(gap(sq.energy_no_offset)), 3), 6e-4)
    chk("S25", f"N={n} Sharpe gpu", num(r["Sharpe, GPU"]), round(float(gq.sharpe_ratio), 3), 6e-4)
    chk("S25", f"N={n} Sharpe sv1", num(r["Sharpe, SV1"]), round(float(sq.sharpe_ratio), 3), 6e-4)
    chk("S25", f"N={n} t gpu", num(r["t, GPU"]), float(gq.runtime_s), 0.06 * float(gq.runtime_s))
    chk("S25", f"N={n} t sv1", num(r["t, SV1"]), float(sq.runtime_s), 0.06 * float(sq.runtime_s))

# Table S28: the Sharpe spread across the three SV1 submissions (C(N,K) and
# the distinct count are above).
rep = pd.read_csv(R + "paper01_sv1_replicates.csv")
for r in table("S28"):
    n = int(num(r["N"])); g = rep[rep.n_assets == n]; s = g[g.backend == "braket_sv1"].sharpe
    cell = r["Sharpe range across runs"]
    if "no spread" in cell:
        chk("S28", f"N={n} Sharpe", num(cell), round(float(s.iloc[0]), 4), 6e-5)
        chk("S28", f"N={n} no spread", 0.0, float(s.max() - s.min()), 1e-12)
    else:
        lo, hi, pct = re.match(r"([\d.]+) – ([\d.]+) \(([\d.]+) %\)", cell).groups()
        chk("S28", f"N={n} Sharpe min", float(lo), round(float(s.min()), 4), 6e-5)
        chk("S28", f"N={n} Sharpe max", float(hi), round(float(s.max()), 4), 6e-5)
        chk("S28", f"N={n} Sharpe spread %", float(pct), round(float((s.max() - s.min()) / s.min() * 100), 1), 0.06)
    chk("S28", f"N={n} GPU Sharpe", num(r["GPU"]), round(float(g[g.backend == "lightning_gpu"].sharpe.iloc[0]), 4), 6e-5)

# Table S31: the product fidelity model from the constants the paragraph
# states (two-qubit 0.9905, SPAM 0.9938 per qubit, 2·C(N,2) gates at p = 2).
# A "Circuit time / T2" column, whose gate time and T2 were stated nowhere,
# was removed on 2026-09-14 rather than sourced.
supp_par = s_supp[s_supp.find("**The fidelity model.**"):]
F2 = float(re.search(r"two-qubit fidelity of ([\d.]+)", supp_par).group(1))
SPAM = float(re.search(r"SPAM at ([\d.]+) per\s+qubit", supp_par).group(1))
for r in table("S31"):
    n = int(num(r["N"])); gates = 2 * comb(n, 2)
    chk("S31", f"N={n} gates", num(r["2-qubit gates"]), float(gates), 0)
    chk("S31", f"N={n} fidelity %", num(r["Predicted fidelity"]), round(F2 ** gates * SPAM ** n * 100, 0), 0.5)

# Table S33: the counts and GPU hours that a result file can reproduce (dollars
# and the wall-clock of the older sweeps are billing records and logs).
S33 = {r["Experiment"]: r for r in table("S33")}
def _s33(prefix):
    return next(v for k, v in S33.items() if k.startswith(prefix))
chk("S33", "E1 runs", num(_s33("E1 cross")["Runs"]), float(len(e1)), 0)
chk("S33", "E1 GPU h", num(_s33("E1 cross")["Time"]), round(float(e1.runtime_s.sum()) / 3600, 1), 0.06)
chk("S33", "E0 rows", num(_s33("E0 re-scoring")["Runs"]), float(len(e0r)), 0)
e3a, e3l = [num(x) for x in _s33("E3 controls")["Runs"].split("+")]
chk("S33", "E3 ADAM runs", e3a, float(len(pd.read_csv(R + "paper01_e1_cross_e3_adam003.csv"))), 0)
chk("S33", "E3 L-BFGS-B runs", e3l, float(len(pd.read_csv(R + "paper01_e1_cross_e3_lbfgs.csv"))), 0)
e5c = pd.read_csv(R + "paper01_e1_cross_e5.csv")
chk("S33", "E5 runs", num(_s33("E5 second window")["Runs"]), float(len(e5c)), 0)
chk("S33", "E5 GPU h", num(_s33("E5 second window")["Time"]), round(float(e5c.runtime_s.sum()) / 3600, 2), 0.06)
cen = _s33("Degeneracy census")["Runs"]
chk("S33", "census annealing runs", num(re.search(r"(\d+) \+ (\d+) annealing", cen).group(1)), float(len(ann)), 0)
chk("S33", "census annealing extra", num(re.search(r"(\d+) \+ (\d+) annealing", cen).group(2)), float(len(pd.read_csv(R + "paper01_sa_on_qubo.csv"))), 0)
chk("S33", "census QAOA runs", num(re.search(r"(\d+) QAOA", cen).group(1)), float(len(iq)), 0)
ledger = pd.read_csv(R + "paper01_sv1_task_ledger.csv")
chk("S33", "SV1 tasks ≈", num(_s33("SV1 cross-backend")["Tasks"].replace("≈", "")), float(ledger.tasks.sum()), 50)
bench = _s33("Single-instance benchmark")["Runs"]
chk("S33", "benchmark QAOA rows", num(bench.split("+")[0]), float((res.solver == "lightning_gpu").sum()), 0)
chk("S33", "benchmark classical rows", num(bench.split("+")[1]), float((res.solver == "classical_auto").sum()), 0)
sw = [num(x) for x in _s33("Seed and depth sweeps")["Runs"].split("+")]
chk("S33", "seed sweep rows", sw[0], float(len(pd.read_csv(R + "paper01_seed_sweep.csv"))), 0)
chk("S33", "depth sweep rows", sw[1], float(len(pd.read_csv(R + "paper01_depth_sweep.csv"))), 0)
chk("S33", "grid rows", sw[2], float(len(pd.read_csv(R + "paper01_steps_depth_grid.csv"))), 0)
ci, cs = [num(x) for x in re.match(r"(\d+) instances × (\d+) sizes", cen).groups()]
chk("S33", "census instances × sizes", ci * cs, float(len(inst)), 0)
chk("S33", "census sizes", cs, float(inst.n_assets.nunique()), 0)
e4b = pd.read_csv(R + "paper01_e4_batches.csv"); e4s = pd.read_csv(R + "paper01_e4_states.csv")
arms, circ, per = [num(x) for x in re.match(r"(\d+) arms × (\d+) circuits × \(1 \+ (\d+)\)", _s33("E4-A fixed angles, local")["Runs"]).groups()]
chk("S33", "E4-A local arms", arms, float(e4s.arm.nunique()), 0)
chk("S33", "E4-A local states", arms * circ, float(len(e4s)), 0)
chk("S33", "E4-A local batches", arms * circ * per, float(len(e4b)), 0)
bill = pd.read_csv(R + "paper01_e4_billing.csv")
pilot_tasks = len(pd.read_csv(R + "paper01_e4_batches_e4a_pilot.csv")) + len(pd.read_csv(R + "paper01_e4_states_e4a_pilot.csv"))
e4sv1 = _s33("E4-A fixed angles, SV1")
chk("S33", "E4-A SV1 tasks", num(e4sv1["Tasks"]), float(len(bill) + pilot_tasks), 0)
chk("S33", "E4-A SV1 pilot tasks", num(re.search(r"\+ (\d+) \(pilot\)", e4sv1["Runs"]).group(1)), float(pilot_tasks), 0)
# The billing file holds the 132 main tasks ($0.495); the two pilot tasks
# (≈ $0.008) are outside it, so the table's $0.50 is matched to a cent.
chk("S33", "E4-A SV1 cost $", num(e4sv1["Cost"].replace("$", "")), float(bill.usd.sum()), 0.011)

# Table S1: the census counts, from the committed audit CSV through the same
# code that prints them (audit_tally.load / count).
sys.path.insert(0, str(ROOT / "papers"))
import audit_tally  # noqa: E402
_tally, _ = audit_tally.load()
S1 = {r["Reported quantity"]: r["Papers reporting it"] for r in table("S1")}
_n = len(_tally)
for label, field in (("An optimality gap, approximation ratio or regret", "reports_gap"),
                     ("Task counts, billed device time, or monetary cost", "billed_cost"),
                     ("Repeats per configuration", "repeats_per_config"),
                     ("A finance/domain metric as a *result*", "domain_metric"),
                     ("Solution degeneracy, observed", "observes_degeneracy")):
    hit, of = [num(x) for x in S1[label].split(" of ")]
    chk("S1", label, hit, float(len(audit_tally.count(_tally, field))), 0)
    chk("S1", f"{label} (population)", of, float(_n), 0)
post = [r for r in _tally if r["post_standard"] == "yes"]
cite = S1["Citation of [19], [20] or [21]"]
mm = re.match(r"(\d+) of the (\d+) published after .* \((\d+) of (\d+) overall\)", cite)
chk("S1", "cite standards: of post-standard", float(mm.group(2)), float(len(post)), 0)
chk("S1", "cite standards: overall population", float(mm.group(4)), float(_n), 0)
chk("S1", "cite standards: count", float(mm.group(1)), 0.0, 0)  # audit_tally prints the 0 as a constant

print(f"{len(OK)} cell(s) agree with the result files, {len(BAD)} disagree, {len(PENDING)} pending a re-run")
for b in BAD:
    print("  x", b)
for p_ in PENDING:
    print("  ?", p_)
raise SystemExit(1 if BAD else 0)
