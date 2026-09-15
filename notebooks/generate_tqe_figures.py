# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Figures for the TQE revision (plan §12.2, Figs. 1-4) from E0/E1/E2/E3 outputs.

Reads the CSVs written by ``run_e0_rescore.py`` and ``run_e1_cross.py`` and
renders four figures (PNG at 300 dpi + vector PDF) into ``notebooks/figures/``
(``notebooks/figures/paper/`` at print size when ``PAPER_FIGURES=1``):

    fig_e0_rescore.*   The same fixed solutions scored under three denominators
                       as only A changes (E0, plan §4.3). No QAOA is re-run.
    fig_e1_penalty.*   Penalty setting vs conventional gap, feasible-range gap,
                       P_F and Q_τ(1000) over instances × seeds (E1, plan §5.7).
    fig_e2_shots.*     Shot budget S vs Q_τ(S) for the optimised state, the
                       unoptimised state and the two uniform controls (E2, §6.4).
    fig_e3_optimizer.* Optimizer settings on the pre-fixed E3 block: Q_τ(1000)
                       per setting, and the paired change in the optimised
                       expectation next to the within-instance seed spread
                       (E3, plan §7.2).
    fig_e4_backends.*  The same angles on four backends: distance of each
                       arm's exact state from the GPU double-precision
                       reference (numerics), and how many distinct best
                       feasible portfolios ten 1,000-shot batches of one state
                       return, against what i.i.d. sampling from that state
                       predicts (E4-A, plan §8.3).

The files carry experiment names rather than numbers because the manuscript
is being re-sectioned (plan §12.1) and the figure numbers will move; the
manifest maps them when the draft is assembled.

Statistics follow plan §10.1: the unit of generalisation is the instance, so
every summary is first a median over the seeds that share an instance, then a
median / IQR over instances. Runs without a feasible shot have no g_F and are
left out of the gap panels only, and the panel says how many rows it drew from.

Reproducible: the only randomness is a fixed-seed jitter; all inputs are
committed experiment outputs.

Usage:
    python -m notebooks.generate_tqe_figures [--results-tag pilot]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.paper01_qubo_baseline.analyze_e3 import (
    BASELINE_LABEL,
    CONTROLS,
    _normalise_baseline,
    e3_block,
)
from experiments.paper01_qubo_baseline.run_e1_cross import SHOT_BUDGETS, TAU_MAIN
from notebooks.generate_figures import (  # noqa: F401  (imports also apply rcParams)
    _panel,
    COLUMN_IN,
    PAPER,
    RESULTS_DIR,
    TEXT_IN,
    _legend,
    _read_csv,
    _save,
    _title,
)

RESCORE_CSV = RESULTS_DIR / "paper01_e0_rescore.csv"

#: Order in which the penalty settings appear on a categorical axis: the
#: exact-threshold multiples in increasing A, then the two literature
#: heuristics. ``A0`` is what the multiples collapse to when A_crit = 0.
PENALTY_ORDER = ["A0", "1.1xAcrit", "2xAcrit", "5xAcrit", "10xAcrit", "Amargin", "Aheur"]
PENALTY_TICK = {"A0": "A = 0", "1.1xAcrit": "1.1", "2xAcrit": "2", "5xAcrit": "5",
                "10xAcrit": "10", "Amargin": "A$_{margin}$", "Aheur": "A$_{heur}$"}

#: Categorical hues in fixed order (validated: adjacent CVD ΔE ≥ 9.2). Series
#: keep their slot across figures: N = 12 / 16 / 20 are always slots 1-3.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
MARKERS = ["o", "s", "^", "D"]
NEUTRAL, INK = "#8a8985", "#0b0b0b"


def _f(v) -> float:
    return float(v) if v not in ("", None) else np.nan


def _legend_below(ax, ncol: int, drop: float, **kw):
    """Hang the legend under the panel in the paper build, in-panel otherwise.

    At column width the three panels of Fig. 1 have no empty corner left: a
    "best" legend landed on the data in (a) and (c), and (b) needed two
    decades of headroom just to hold eight entries -- whose box then ran into
    (c)'s y label. Under the panel (as Fig. 4 already does) the axes keep all
    their area for the data. ``drop`` is the gap below the axes in axes
    fraction, set per panel to clear its tick and axis labels.
    """
    if not PAPER:
        return _legend(ax, **kw)
    kw.pop("loc", None)
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, -drop), ncol=ncol,
                     fontsize=8, frameon=False, handlelength=1.4, columnspacing=1.0,
                     handletextpad=0.5)


def _instance_median(rows: list[dict], field: str) -> np.ndarray:
    """Median over seeds within each instance → one value per instance.

    Rows whose ``field`` is empty (no feasible shot for a gap field) are
    dropped from that instance's median; an instance with nothing left is
    dropped from the array.
    """
    by_inst: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = _f(r.get(field, ""))
        if not np.isnan(v):
            by_inst[r["instance"]].append(v)
    return np.asarray([np.median(v) for v in by_inst.values()], dtype=float)


def _iqr_series(groups: dict[str, list[dict]], labels: list[str], field: str):
    """(median, p25, p75, n_instances) per label, instance-level."""
    med, lo, hi, n = [], [], [], []
    for lab in labels:
        v = _instance_median(groups.get(lab, []), field)
        if v.size:
            med.append(np.median(v)); lo.append(np.percentile(v, 25))
            hi.append(np.percentile(v, 75)); n.append(v.size)
        else:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan); n.append(0)
    return np.asarray(med), np.asarray(lo), np.asarray(hi), n


# --------------------------------------------------------------------------
# Fig. E0: the same solutions, three denominators
# --------------------------------------------------------------------------

def fig_e0_rescore(rows: list[dict], instances: list[dict] | None = None) -> None:
    """Re-scoring fixed solutions as only A changes (plan §4.3).

    Left: at N = 20 (the largest size with an exact all-states enumeration)
    one fixed feasible solution -- the median-rank one -- is scored three
    ways as A grows: the conventional gap g_off = (f−f*)/|f*−AK²|, the
    all-states approximation gap 1−r_all = (E−E*)/(C_max−E*), and the
    feasible-range gap g_F = (f−f*)/Δ_F. Only the last is independent of A;
    the solution never changes, so every movement in the other two is the
    denominator.

    Middle: the deflation D(A) = g_F / g_off = |f*−AK²|/Δ_F for every size,
    i.e. how many times smaller the offset-normalized number is than the
    feasible-range one at that A. The default A_heur is marked. Above N = 20
    the threshold is the bounded A_safe rather than the exact A_crit, so those
    lines are drawn hollow.

    Right (when the instance census is given): the same deflation at A_heur
    over thirty drawn instances per size, box = IQR, whisker = range, the
    reference instance (0) marked -- so the one-instance mechanism on the left and
    its size across instances sit in one figure.
    """
    if not rows:
        print("  skipped fig_e0_rescore: no rows")
        return
    by_n: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_n[int(r["n_assets"])].append(r)
    ns = sorted(by_n)
    n_left = 20 if 20 in by_n else max(n for n in ns if n <= 20)

    if instances:
        fig, (ax_l, ax_r, ax_c) = plt.subplots(
            1, 3, figsize=(TEXT_IN, TEXT_IN * 0.50) if PAPER else (16, 4.6))
    else:
        fig, (ax_l, ax_r) = plt.subplots(
            1, 2, figsize=(TEXT_IN, TEXT_IN * 0.4) if PAPER else (12.5, 4.8))
        ax_c = None

    # --- left: one solution, three denominators ------------------------------
    sol = sorted((r for r in by_n[n_left] if r["solution"] == "median"),
                 key=lambda r: _f(r["penalty"]))
    a_over = np.asarray([_f(r["penalty"]) / _f(r["threshold"]) for r in sol])
    g_off = np.asarray([_f(r["g_off"]) for r in sol]) * 100
    g_all = (1 - np.asarray([_f(r["r_all"]) for r in sol])) * 100
    g_f = np.asarray([_f(r["g_F"]) for r in sol]) * 100
    rank = int(sol[0]["rank"]); size = int(sol[0]["feasible_set_size"])
    ax_l.plot(a_over, g_off, "-", marker=MARKERS[0], color=SERIES[0],
              label="Offset-normalized $g_{off}$" if PAPER else
                    "Offset-normalized  $(f-f^*)/|f^*-AK^2|$")
    ax_l.plot(a_over, g_all, "-", marker=MARKERS[1], color=SERIES[1],
              label="All-states range" if PAPER else
                    "All-states range  $(E-E^*)/(C_{max}-E^*)$")
    ax_l.plot(a_over, g_f, "--", marker=MARKERS[2], color=SERIES[2],
              label="Feasible range $g_F$" if PAPER else
                    "Feasible range  $(f-f^*)/\\Delta_F$")
    ax_l.set_xscale("log"); ax_l.set_yscale("log")
    ax_l.set_xlabel("Penalty weight  $A / A_{crit}$")
    ax_l.set_ylabel("Gap of the same solution (%)")
    heur = [r for r in sol if r["penalty_label"] == "Aheur"]
    if heur:
        x = _f(heur[0]["penalty"]) / _f(heur[0]["threshold"])
        ax_l.axvline(x, color=NEUTRAL, lw=1, ls=":")
        ax_l.annotate("$A_{heur}$", (x, g_f[0]), textcoords="offset points",
                      xytext=(-4, 4), ha="right", fontsize=8, color=NEUTRAL)
    _title(ax_l, f"N = {n_left}: the feasible solution of median rank "
                 f"({rank} / {size:,}),\nscored three ways as only A changes",
           fontsize=10)
    _legend_below(ax_l, ncol=1, drop=0.30, fontsize=8, loc="lower left")
    _panel(ax_l, "a")

    # --- right: deflation per size ------------------------------------------
    cmap = plt.get_cmap("Blues")
    shades = cmap(np.linspace(0.45, 0.95, len(ns)))
    for shade, n in zip(shades, ns):
        opt = sorted((r for r in by_n[n] if r["solution"] == "optimum"),
                     key=lambda r: _f(r["penalty"]))
        x = np.asarray([_f(r["penalty"]) / _f(r["threshold"]) for r in opt])
        d = np.asarray([_f(r["deflation_D"]) for r in opt])
        exact = opt[0]["threshold_kind"].startswith("A_crit")
        ax_r.plot(x, d, "-", marker="o", color=shade,
                  markerfacecolor=shade if exact else "white",
                  label=f"N = {n}" + ("" if exact else " ($A_{safe}$)"))
        h = [i for i, r in enumerate(opt) if r["penalty_label"] == "Aheur"]
        if h:
            ax_r.plot(x[h], d[h], marker="*", ms=11 if not PAPER else 7,
                      color=INK, ls="none", zorder=5)
    ax_r.plot([], [], marker="*", color=INK, ls="none",
              ms=11 if not PAPER else 7, label="at $A_{heur}$ (default)")
    ax_r.set_xscale("log"); ax_r.set_yscale("log")
    ax_r.set_xlabel("Penalty weight  $A / A_{crit}$  ($A / A_{safe}$, N > 20)" if PAPER else
                    "Penalty weight  $A / A_{crit}$  ($A / A_{safe}$ above N = 20)")
    ax_r.set_ylabel("Deflation  $D(A) = g_F / g_{off}$" if PAPER else
                    "Deflation  $D(A) = g_F / g_{off} = |f^*-AK^2| / \\Delta_F$")
    _title(ax_r, "How many times smaller the offset-normalized gap reads\n"
                 "than the feasible-range gap, per size", fontsize=10)
    if PAPER:
        _legend_below(ax_r, ncol=2, drop=0.30)
    else:
        # On screen the upper-left corner (small A, large D) is empty once the
        # axis is given headroom, so the legend is pinned there.
        ax_r.set_ylim(top=ax_r.get_ylim()[1] * 60)
        ax_r.legend(fontsize=8, loc="upper left", ncol=2, frameon=True,
                    framealpha=0.88, borderpad=0.35, handlelength=1.2, columnspacing=0.8)
    _panel(ax_r, "b")

    # --- right: the deflation at A_heur over thirty instances per size ------
    if ax_c is not None:
        by_size: dict[int, list[dict]] = defaultdict(list)
        for r in instances:
            by_size[int(r["n_assets"])].append(r)
        sizes = sorted(by_size)
        xs = np.arange(len(sizes))
        for i, n in zip(xs, sizes):
            vals = sorted(_f(r["deflation_factor"]) for r in by_size[n])
            q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
            ax_c.vlines(i, vals[0], vals[-1], color=NEUTRAL, lw=1.2, zorder=1)
            ax_c.add_patch(plt.Rectangle((i - 0.22, q1), 0.44, q3 - q1, facecolor=SERIES[0],
                                         alpha=0.30, edgecolor=SERIES[0], zorder=2))
            ax_c.hlines(float(np.median(vals)), i - 0.22, i + 0.22, color=SERIES[0], lw=2, zorder=3)
        published = [_f(r["deflation_factor"]) for n in sizes for r in by_size[n]
                     if r["is_published_instance"] == "True"]
        ax_c.plot(xs[:len(published)], published, "D", color=SERIES[1], ms=5 if PAPER else 6,
                  zorder=5, label="reference instance (0)")
        ax_c.set_yscale("log")
        ax_c.set_xticks(xs); ax_c.set_xticklabels([str(n) for n in sizes])
        ax_c.set_xlabel("Number of assets N")
        ax_c.set_ylabel("Deflation $D(A_{heur})$ over instances")
        counts = {len(v) for v in by_size.values()}
        label = f"{min(counts)}" if len(counts) == 1 else f"{min(counts)}–{max(counts)}"
        _title(ax_c, f"The same deflation at $A_{{heur}}$ over {label} instances\n"
                     "per size (box = IQR, whisker = range)", fontsize=10)
        _legend_below(ax_c, ncol=1, drop=0.30, fontsize=8, loc="upper left")
        _panel(ax_c, "c")
    _save(fig, "fig_e0_rescore")


# --------------------------------------------------------------------------
# Fig. E1: penalty × instance × seed
# --------------------------------------------------------------------------

def fig_e1_penalty(rows: list[dict]) -> None:
    """Four quantities against the penalty setting, per size (plan §5.7).

    2 × 2 panels: the conventional gap g_off, the feasible-range gap g_F (both
    of the best feasible shot, in %), the feasible probability P_F of the
    optimised state, and Q_τ(1000) with τ = 1 % of the feasible range. Each
    point is the median over instances of the within-instance median over
    seeds; the bar is the inter-quartile range over instances. Penalty
    settings are categorical on x because A_crit, A_margin and hence every A
    differ per instance.

    Gap panels use a symlog axis so exact hits (gap = 0) stay on the plot;
    the linear region below the threshold is where they sit. They also show
    every run as a faint jittered point (the jitter is seeded, so the file is
    still a function of the CSV) with the share of exact hits under the
    baseline, because the instance-level median hides the tail.
    """
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        print("  skipped fig_e1_penalty: no completed runs")
        return
    ns = sorted({int(r["n_assets"]) for r in ok})
    groups: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in ok:
        groups[int(r["n_assets"])][r["penalty_label"]].append(r)
    labels = [p for p in PENALTY_ORDER if any(p in groups[n] for n in ns)]
    xs = np.arange(len(labels))

    # At column width the descriptive axis labels collide; the caption carries
    # the description there and the symbol alone survives.
    panels = [
        ("g_off", "$g_{off}$ of best feasible shot (%)" if PAPER else
         "Offset-normalized gap $g_{off}$ of best feasible shot (%)", 100.0, "symlog"),
        ("g_F", "$g_F$ of best feasible shot (%)" if PAPER else
         "Feasible-range gap $g_F$ of best feasible shot (%)", 100.0, "symlog"),
        ("final_P_F", "$P_F$ of the optimised state" if PAPER else
         "$P_F$: feasible probability of the optimised state", 1.0, "linear"),
        ("final_Q_tau_main_1000", f"$Q_\\tau(1000)$, $\\tau$ = {TAU_MAIN:.0%}" if PAPER else
         f"$Q_\\tau(1000)$, $\\tau$ = {TAU_MAIN:.0%} of feasible range", 1.0, "linear"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_IN, TEXT_IN * 0.7) if PAPER else (12.5, 8.4))
    width = 0.8 / max(len(ns), 1)
    jitter = np.random.default_rng(0)   # fixed: the figure stays a function of the CSV
    for ax, letter, (field, ylabel, scale, yscale) in zip(axes.ravel(), "abcd", panels):
        _panel(ax, letter)
        for k, n in enumerate(ns):
            med, lo, hi, cnt = _iqr_series(groups[n], labels, field)
            off = (k - (len(ns) - 1) / 2) * width
            if yscale == "symlog":
                # The gap panels also show every run: the instance-level
                # median is 0 wherever most seeds hit the optimum, and the
                # tail -- which is what a penalty setting changes -- would
                # otherwise vanish. The share of runs at exactly 0 is written
                # under the baseline.
                for j, lab in enumerate(labels):
                    v = np.asarray([_f(r.get(field, "")) for r in groups[n].get(lab, [])])
                    v = v[~np.isnan(v)]
                    if not v.size:
                        continue
                    x = xs[j] + off + jitter.uniform(-width / 3, width / 3, v.size)
                    ax.plot(x, v * scale, MARKERS[k], color=SERIES[k], alpha=0.25,
                            ms=3 if not PAPER else 2, mec="none", zorder=1)
                    if not PAPER:
                        # One row per N so the three labels do not overprint.
                        ax.annotate(f"{np.mean(v == 0):.0%}", (xs[j] + off, 0),
                                    textcoords="offset points", xytext=(0, -9 - 8 * k),
                                    ha="center", va="top", fontsize=6.5,
                                    color=SERIES[k])
            ax.errorbar(xs + off, med * scale, yerr=[(med - lo) * scale, (hi - med) * scale],
                        fmt=MARKERS[k], color=SERIES[k], ecolor=SERIES[k],
                        elinewidth=1.2, capsize=2.5, ms=6 if not PAPER else 4,
                        mec="white", mew=0.8, zorder=4, label=f"N = {n}")
        if yscale == "symlog":
            # Percent units: 1e-3 % is below every non-zero gap in the data,
            # so the linear region only carries the exact hits.
            ax.set_yscale("symlog", linthresh=1e-3, linscale=0.4)
            ax.axhline(0, color=NEUTRAL, lw=0.8)
            ax.set_ylim(bottom=-6e-4)   # keeps the "-1e-3" tick off the axis
        else:
            ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(xs)
        ax.set_xticklabels([PENALTY_TICK[p] for p in labels])
        ax.set_xlabel("Penalty setting  ($\\times A_{crit}$; $A_{margin}$; $A_{heur}$)" if PAPER else
                      "Penalty setting  (multiples of $A_{crit}$, then the two reference settings)")
        ax.set_ylabel(ylabel)
        if field == "g_off":
            ax.axhline(0.1, color=NEUTRAL, lw=0.8, ls=":")
            ax.annotate("0.1 %" if PAPER else "0.1 % (the tables' success line)", (xs[0] - 0.4, 0.1),
                        textcoords="offset points", xytext=(0, -3), ha="left",
                        va="top", fontsize=8, color=NEUTRAL)
        if field == "g_F":
            ax.axhline(TAU_MAIN * 100, color=NEUTRAL, lw=0.8, ls=":")
            ax.annotate(f"$\\tau$ = {TAU_MAIN * 100:g} %", (xs[0] - 0.4, TAU_MAIN * 100),
                        textcoords="offset points", xytext=(0, -3), ha="left",
                        va="top", fontsize=8, color=NEUTRAL)
    # The legend and the coverage note go in the P_F panel, whose upper half
    # is empty by construction (P_F ≤ 1 and the data sit well below it).
    n_inst = {n: len({r["instance"] for r in ok if int(r["n_assets"]) == n}) for n in ns}
    n_seed = {n: len({r["qaoa_seed"] for r in ok if int(r["n_assets"]) == n}) for n in ns}
    _legend(axes[1, 0], fontsize=9, loc="upper right")
    if not PAPER:
        axes[1, 0].text(0.02, 0.97, "\n".join(
            f"N = {n}: {n_inst[n]} instances × {n_seed[n]} seeds" for n in ns),
            transform=axes[1, 0].transAxes, va="top", fontsize=8.5, color="#52514e")
    _save(fig, "fig_e1_penalty")


# --------------------------------------------------------------------------
# Fig. E2: shot budget vs success probability
# --------------------------------------------------------------------------

def fig_e2_shots(rows: list[dict], refs: list[dict], penalty: str = "Aheur") -> None:
    """Q_τ(S) against the shot budget S for four samplers (plan §6.4).

    One panel per size, at the penalty setting the published tables used
    (A_heur, overridable). The optimised QAOA state and the same-seed
    unoptimised state come from the run's exact final / initial
    probabilities; the two uniform controls are instance properties from the
    feasible-set census (P_τ under uniform-over-all-bitstrings and under
    uniform-over-F), with Q_τ(S) = 1 − (1 − P_τ)^S for i.i.d. shots. Lines
    are instance-level medians, bands the inter-quartile range over
    instances.
    """
    ok = [r for r in rows if r["status"] == "ok" and r["penalty_label"] == penalty]
    if not ok:
        print(f"  skipped fig_e2_shots: no completed {penalty} runs")
        return
    ref_by = {(r["n_assets"], r["instance"]): r for r in refs}
    ns = sorted({int(r["n_assets"]) for r in ok})
    fig, axes = plt.subplots(1, len(ns), sharey=True,
                             figsize=(TEXT_IN, TEXT_IN * 0.36) if PAPER else (4.3 * len(ns), 4.4),
                             squeeze=False)
    s_grid = np.asarray(SHOT_BUDGETS, dtype=float)
    samplers = [
        ("Optimised QAOA state", SERIES[0], "-", lambda r, s: _f(r[f"final_Q_tau_main_{s}"])),
        ("Unoptimised state (same initial angles)", SERIES[1], "--",
         lambda r, s: _f(r[f"initial_Q_tau_main_{s}"])),
        ("Uniform over feasible set F", SERIES[2], "-.",
         lambda r, s: 1 - (1 - _f(ref_by[(r["n_assets"], r["instance"])]["P_tau_main_uniform_F"])) ** s),
        ("Uniform over all bitstrings", NEUTRAL, ":",
         lambda r, s: 1 - (1 - _f(ref_by[(r["n_assets"], r["instance"])]["P_tau_main_uniform_all"])) ** s),
    ]
    for ax, n in zip(axes[0], ns):
        g = [r for r in ok if int(r["n_assets"]) == n]
        for label, color, ls, value in samplers:
            med, lo, hi = [], [], []
            for s in SHOT_BUDGETS:
                by_inst: dict[str, list[float]] = defaultdict(list)
                for r in g:
                    by_inst[r["instance"]].append(value(r, s))
                v = np.asarray([np.median(x) for x in by_inst.values()])
                med.append(np.median(v)); lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
            ax.plot(s_grid, med, ls, color=color, marker="o", ms=4, label=label)
            ax.fill_between(s_grid, lo, hi, color=color, alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(s_grid)
        ax.set_xticklabels([f"{int(s):,}" for s in s_grid])
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Shots S")
        n_inst = len({r["instance"] for r in g}); n_seed = len({r["qaoa_seed"] for r in g})
        _title(ax, f"N = {n}", fontsize=10)
        if not PAPER:
            ax.text(0.03, 0.97, f"{n_inst} instances × {n_seed} seeds\nA = {PENALTY_TICK[penalty]}",
                    transform=ax.transAxes, va="top", fontsize=8.5, color="#52514e")
    axes[0, 0].set_ylabel(f"$Q_\\tau(S)$ at $A_{{{penalty[1:]}}}$, $\\tau$ = {TAU_MAIN:.0%}" if PAPER else
                          f"$Q_\\tau(S)$: P[best of S shots feasible and within $\\tau$ = {TAU_MAIN:.0%}]")
    for ax, letter in zip(axes[0], "abc"):
        _panel(ax, letter)
    # Four samplers × three panels leave no corner free in every panel, so the
    # legend hangs under the row; savefig's tight bbox keeps it in the file.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4 if not PAPER else 2,
               bbox_to_anchor=(0.5, -0.02 if not PAPER else -0.10),
               fontsize=8, frameon=False)
    _save(fig, "fig_e2_shots")


# --------------------------------------------------------------------------
# Fig. E3: optimizer settings versus initial-angle variation
# --------------------------------------------------------------------------

OPTIMIZER_TICK = {BASELINE_LABEL: "ADAM 0.1\n50 steps", "adam_0.03": "ADAM 0.03\n100 steps",
                  "lbfgs_100grad": "L-BFGS\n$\\leq$100 grad."}
E3_PENALTY_STYLE = {"Aheur": (SERIES[3], "D"), "Amargin": (SERIES[2], "^")}


def fig_e3_optimizer(groups: dict[str, list[dict]]) -> None:
    """Optimizer controls on the E3 block (plan §7.2), one column per size.

    Top row: Q_τ(1000) under each optimizer setting, for the two penalty
    settings of the block; the marker is the median over instances of the
    within-instance median over seeds, the bar the IQR over instances, and
    every run is a faint jittered point.

    Bottom row: the change in the optimised expectation ⟨H_norm⟩ relative to
    the baseline run with the same instance, seed and A (so the two are
    comparable), one point per run, on a symlog axis; the grey band is the
    baseline's own within-instance spread over the five seeds (median over
    instances of max − min), i.e. the scale a different initial angle moves
    the same quantity. The evaluation budget each setting actually used is
    written under its tick. Negative is better.
    """
    opts = [BASELINE_LABEL] + [lab for _, lab in CONTROLS]
    ns = sorted({int(r["n_assets"]) for r in groups[BASELINE_LABEL]})
    if not ns or any(not groups.get(o) for o in opts):
        print("  skipped fig_e3_optimizer: E3 block incomplete")
        return
    base = {(r["n_assets"], r["instance"], r["penalty_label"], r["qaoa_seed"]): r
            for r in groups[BASELINE_LABEL] if r["status"] == "ok"}
    pens = [p for p in ("Aheur", "Amargin")
            if any(r["penalty_label"] == p for r in groups[BASELINE_LABEL])]
    xs = np.arange(len(opts))
    width = 0.8 / max(len(pens), 1)
    jitter = np.random.default_rng(0)
    # No sharex: with shared axes one column's tick labels would overwrite
    # the others', and the budget written under the ticks differs per N.
    fig, axes = plt.subplots(2, len(ns),
                             figsize=(TEXT_IN, TEXT_IN * 0.62) if PAPER else (4.3 * len(ns), 8.0),
                             squeeze=False)
    for col, n in enumerate(ns):
        top, bot = axes[0, col], axes[1, col]
        # Reference scale for the bottom row: the baseline seed spread.
        spread = []
        for pen in pens:
            by_inst: dict[str, list[float]] = defaultdict(list)
            for r in groups[BASELINE_LABEL]:
                if int(r["n_assets"]) == n and r["penalty_label"] == pen and r["status"] == "ok":
                    by_inst[r["instance"]].append(_f(r["best_expectation"]))
            spread += [max(v) - min(v) for v in by_inst.values() if len(v) >= 2]
        seed_spread = float(np.median(spread)) if spread else np.nan
        if not np.isnan(seed_spread):
            bot.axhspan(-seed_spread, seed_spread, color=NEUTRAL, alpha=0.18, lw=0, zorder=0)
            bot.annotate("seed spread\n(same instance)" if PAPER else
                         "baseline spread over 5 seeds\nwithin an instance (median)",
                         (xs[-1] + 0.45, seed_spread), ha="right", va="bottom",
                         fontsize=8, color="#52514e",
                         textcoords="offset points", xytext=(0, 2))
        bot.axhline(0, color=NEUTRAL, lw=0.8)
        for k, pen in enumerate(pens):
            color, marker = E3_PENALTY_STYLE[pen]
            off = (k - (len(pens) - 1) / 2) * width
            med, lo, hi = [], [], []
            for j, opt in enumerate(opts):
                g = [r for r in groups[opt]
                     if int(r["n_assets"]) == n and r["penalty_label"] == pen and r["status"] == "ok"]
                v = _instance_median(g, "final_Q_tau_main_1000")
                med.append(np.median(v) if v.size else np.nan)
                lo.append(np.percentile(v, 25) if v.size else np.nan)
                hi.append(np.percentile(v, 75) if v.size else np.nan)
                runs = np.asarray([_f(r["final_Q_tau_main_1000"]) for r in g])
                x = xs[j] + off + jitter.uniform(-width / 3, width / 3, runs.size)
                top.plot(x, runs, marker, color=color, alpha=0.22, ms=2 if PAPER else 3,
                         mec="none", zorder=1)
                if opt != BASELINE_LABEL:
                    d = []
                    for r in g:
                        b = base.get((r["n_assets"], r["instance"], r["penalty_label"], r["qaoa_seed"]))
                        if b is not None:
                            d.append(_f(r["best_expectation"]) - _f(b["best_expectation"]))
                    d = np.asarray(d)
                    x = xs[j] + off + jitter.uniform(-width / 3, width / 3, d.size)
                    bot.plot(x, d, marker, color=color, alpha=0.45, ms=2.5 if PAPER else 3.5,
                             mec="none", zorder=2)
                    bot.plot([xs[j] + off], [np.median(d)], marker, color=color, ms=6 if not PAPER else 4,
                             mec="white", mew=0.8, zorder=4)
            med, lo, hi = map(np.asarray, (med, lo, hi))
            top.errorbar(xs + off, med, yerr=[med - lo, hi - med], fmt=marker, color=color,
                         ecolor=color, elinewidth=1.2, capsize=2.5, ms=6 if not PAPER else 4,
                         mec="white", mew=0.8, zorder=4,
                         label=f"A = {PENALTY_TICK[pen]}")
        top.set_ylim(-0.02, 1.02)
        bot.set_yscale("symlog", linthresh=1e-3, linscale=0.5)
        bot.set_ylim(-20, 20)
        # Budget actually used, under the tick.
        ticks = []
        for opt in opts:
            g = [r for r in groups[opt] if int(r["n_assets"]) == n and r["status"] == "ok"]
            ge = np.asarray([_f(r["gradient_evaluations"]) for r in g])
            ts = np.asarray([_f(r["optimisation_seconds"]) for r in g])
            ticks.append(f"{OPTIMIZER_TICK[opt]}\n{np.median(ge):.0f} $\\nabla$, {np.median(ts):.0f} s")
        top.set_xticks(xs)
        top.set_xticklabels([])
        bot.set_xticks(xs)
        bot.set_xticklabels(ticks, fontsize=8)
        _title(top, f"N = {n}", fontsize=10)
    axes[0, 0].set_ylabel(f"$Q_\\tau(1000)$, $\\tau$ = {TAU_MAIN:.0%}")
    axes[1, 0].set_ylabel("$\\Delta\\langle H_{norm}\\rangle$ vs. baseline\n(same instance, seed, A)")
    _legend(axes[0, 0], fontsize=8, loc="upper right")
    n_inst = len({r["instance"] for r in groups[BASELINE_LABEL]})
    n_seed = len({r["qaoa_seed"] for r in groups[BASELINE_LABEL]})
    if not PAPER:
        axes[0, -1].text(0.03, 0.97, f"{n_inst} instances × {n_seed} seeds per setting",
                         transform=axes[0, -1].transAxes, va="top", fontsize=8.5, color="#52514e")
    _save(fig, "fig_e3_optimizer")


# --------------------------------------------------------------------------
# Fig. E4: numerics versus sampling across backends
# --------------------------------------------------------------------------

ARM_STYLE = {"cpu_double": (SERIES[0], "o", "CPU, double (1 thread)"),
             "gpu_single": (SERIES[1], "s", "GPU, single"),
             "sv1": (SERIES[3], "D", "SV1, double (cloud)"),
             "gpu_double": (SERIES[2], "^", "GPU, double (reference)")}


def fig_e4_backends(compare: list[dict], summary: list[dict]) -> None:
    """Fixed angles on four backends (plan §8.3), two panels.

    Left: for each of the twelve circuits, the total variation distance on
    F ∪ {⊥} -- the feasible bitstrings plus one symbol for "no feasible shot",
    p_⊥ = 1 − P_F -- between the arm's exact output probabilities and the GPU
    double-precision ones at the same angles (log y). This is the numerics:
    what a different machine, precision or vendor does to the *output
    distribution* (not the quantum state: equal basis probabilities leave the
    phases unconstrained).

    Right: for each circuit and arm, how many distinct best-feasible
    portfolios ten independent 1,000-shot batches of that one state returned
    (marker), with the 95 % interval of the same count under i.i.d. draws from
    the arm's own exact distribution (bar, Monte Carlo). This is the sampling:
    what finite shots do to the *answer* when the state is identical.
    """
    if not compare or not summary:
        print("  skipped fig_e4_backends: no E4 rows")
        return
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(TEXT_IN, TEXT_IN * 0.42) if PAPER else (12.5, 5.0))
    circuits = sorted({(int(r["n_assets"]), int(r["instance"]), int(r["qaoa_seed"]))
                       for r in compare})
    xpos = {c: i for i, c in enumerate(circuits)}
    # One tick per circuit reads "i0 s42"; the size is written once under
    # each group of four rather than repeated on every tick, where twelve
    # "N=12" labels ran into each other at column width.
    labels = [f"i{i}\ns{s}" for n, i, s in circuits]

    def _size_labels(ax):
        for n in sorted({c[0] for c in circuits}):
            xs = [xpos[c] for c in circuits if c[0] == n]
            ax.annotate(f"N = {n}", (float(np.mean(xs)), 0), xycoords=("data", "axes fraction"),
                        textcoords="offset points", xytext=(0, -26 if PAPER else -30),
                        ha="center", va="top", fontsize=8 if PAPER else 9)
            if xs[0] > 0:
                ax.axvline(xs[0] - 0.5, color=NEUTRAL, lw=0.6, ls=":")

    for arm in ("cpu_double", "gpu_single", "sv1"):
        color, marker, label = ARM_STYLE[arm]
        pts = [(xpos[(int(r["n_assets"]), int(r["instance"]), int(r["qaoa_seed"]))], _f(r["tv_F_perp"]))
               for r in compare if r["arm"] == arm and r.get("tv_F_perp", "") != ""]
        if pts:
            x, y = zip(*pts)
            ax_l.plot(x, np.maximum(y, 1e-18), marker, color=color, ms=6 if not PAPER else 4,
                      mec="white", mew=0.6, label=label, ls="none")
    ax_l.set_yscale("log")
    ax_l.set_ylabel("TV on F ∪ {⊥} vs. GPU double precision" if PAPER else
                    "TV on F ∪ {⊥} from the GPU double-precision output")
    _panel(ax_l, "a")
    ax_l.set_xticks(range(len(circuits)))
    ax_l.set_xticklabels(labels, fontsize=8)
    _size_labels(ax_l)
    # The two ε lines are named in the legend, not by text inside the panel:
    # the grey labels sat across the N = 12 points at column width, and the
    # twelve columns leave no run of empty x at either height.
    ax_l.axhline(np.finfo(np.float32).eps, color=NEUTRAL, lw=0.8, ls=":",
                 label="single-precision ε")
    ax_l.axhline(np.finfo(np.float64).eps, color=NEUTRAL, lw=0.8, ls="--",
                 label="double-precision ε")
    # Under the panel, like (b): the in-panel box sat over the N = 16 points.
    lh, ll = ax_l.get_legend_handles_labels()
    ax_l.legend(lh, ll, loc="upper center", ncol=2, columnspacing=1.0, handletextpad=0.5,
                bbox_to_anchor=(0.5, -0.30 if PAPER else -0.16), fontsize=8, frameon=False)
    _title(ax_l, "Same angles, different backend: the output distribution", fontsize=10)

    arms = [a for a in ("gpu_double", "cpu_double", "gpu_single", "sv1")
            if any(r["arm"] == a for r in summary)]
    width = 0.8 / max(len(arms), 1)
    for k, arm in enumerate(arms):
        color, marker, label = ARM_STYLE[arm]
        off = (k - (len(arms) - 1) / 2) * width
        for r in summary:
            if r["arm"] != arm:
                continue
            c = (int(r["n_assets"]), int(r["instance"]), int(r["qaoa_seed"]))
            if c not in xpos:
                continue
            x = xpos[c] + off
            y = _f(r["distinct_best_feasible"])
            lo, hi = _f(r.get("mc_distinct_best_p025", "")), _f(r.get("mc_distinct_best_p975", ""))
            if not np.isnan(lo):
                ax_r.plot([x, x], [lo, hi], "-", color=color, alpha=0.45, lw=3, solid_capstyle="butt")
            ax_r.plot([x], [y], marker, color=color, ms=6 if not PAPER else 4, mec="white", mew=0.6,
                      label=label if r is next(rr for rr in summary if rr["arm"] == arm) else None)
    ax_r.set_ylim(0, 10.8)
    ax_r.set_yticks(range(0, 11, 2))
    ax_r.set_ylabel("Distinct best-feasible portfolios, 10 × 1,000 shots" if PAPER else
                    "Distinct best-feasible portfolios in 10 batches of 1,000 shots")
    _panel(ax_r, "b")
    ax_r.set_xticks(range(len(circuits)))
    ax_r.set_xticklabels(labels, fontsize=8)
    _size_labels(ax_r)
    # The four arms fill the panel at every x (N = 12 sits low, N = 20 at the
    # ceiling), so an in-panel legend covers data wherever it is put — it sat
    # over the N = 16 points in the paper build. It hangs under the panel
    # instead, as Fig. 3's does; savefig's tight bbox keeps it in the file.
    rh, rl = ax_r.get_legend_handles_labels()
    ax_r.legend(rh, rl, loc="upper center", ncol=2,
                bbox_to_anchor=(0.5, -0.30 if PAPER else -0.16), fontsize=8, frameon=False)
    _title(ax_r, "Same output distribution, ten batches: the answer", fontsize=10)
    if not PAPER:
        ax_r.text(0.99, 0.03, "bar: 95 % interval under i.i.d. draws\nfrom the arm's own exact state",
                  transform=ax_r.transAxes, ha="right", va="bottom", fontsize=8, color="#52514e")
    _save(fig, "fig_e4_backends")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-tag", default=None,
                    help="suffix of the E1 CSVs to read (e.g. 'pilot'); default is the main run")
    ap.add_argument("--e2-penalty", default="Aheur", choices=PENALTY_ORDER,
                    help="penalty setting shown in the shot-budget figure")
    args = ap.parse_args(argv)
    suffix = f"_{args.results_tag}" if args.results_tag else ""
    cross = RESULTS_DIR / f"paper01_e1_cross{suffix}.csv"
    refs = RESULTS_DIR / f"paper01_e1_reference{suffix}.csv"

    print("Generating TQE figures ...")
    if RESCORE_CSV.exists():
        inst_csv = RESULTS_DIR / "paper01_instances.csv"
        fig_e0_rescore(_read_csv(RESCORE_CSV),
                       _read_csv(inst_csv) if inst_csv.exists() else None)
    else:
        print(f"  skipped fig_e0_rescore: {RESCORE_CSV.name} not found "
              "(run experiments.paper01_qubo_baseline.run_e0_rescore)")
    if cross.exists() and refs.exists():
        rows, ref_rows = _read_csv(cross), _read_csv(refs)
        fig_e1_penalty(rows)
        fig_e2_shots(rows, ref_rows, args.e2_penalty)
    else:
        print(f"  skipped fig_e1_penalty / fig_e2_shots: {cross.name} not found "
              "(run experiments.paper01_qubo_baseline.run_e1_cross)")
    e3 = {tag: RESULTS_DIR / f"paper01_e1_cross_{tag}.csv" for tag, _ in CONTROLS}
    if not suffix and cross.exists() and all(p.exists() for p in e3.values()):
        groups = {BASELINE_LABEL: e3_block(_normalise_baseline(_read_csv(cross)))}
        for tag, label in CONTROLS:
            groups[label] = e3_block(_read_csv(e3[tag]))
        fig_e3_optimizer(groups)
    else:
        print("  skipped fig_e3_optimizer: E3 CSVs not found (run the E3 commands in run_e1_cross)")
    e4c = RESULTS_DIR / "paper01_e4_states_compare.csv"
    e4s = RESULTS_DIR / "paper01_e4_batches_summary.csv"
    if not suffix and e4c.exists() and e4s.exists():
        fig_e4_backends(_read_csv(e4c), _read_csv(e4s))
    else:
        print("  skipped fig_e4_backends: E4 aggregates not found (run analyze_e4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
