# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Generate publication figures for paper① from the experiment CSVs.

Reads the result files produced by
``experiments/paper01_qubo_baseline/run_experiment.py`` and renders four
figures (PNG at 300 dpi + vector PDF) into ``notebooks/figures/``
(``notebooks/figures/paper/`` at print size when ``PAPER_FIGURES=1``):

    fig1_convergence.*     QAOA expectation vs ADAM step, one curve per N
    fig3_runtime_scaling.* Runtime vs N, classical vs quantum (log y)
    fig2_optimality_gap.*  Quantum optimality gap (%) vs N
    fig4_sharpe.*          Sharpe ratio of the selected portfolio vs N

If the sweep CSVs from ``run_sweep.py`` are also present, three more figures are
rendered; without them the script still produces Fig. 1-4, and each sweep figure
is skipped independently of the others:

    fig5_seed_variance.*   Optimality gap across QAOA initial-angle seeds
    fig6_depth_sensitivity.* Gap and runtime vs QAOA depth p
    fig7_steps_depth_grid.*  Gap vs optimizer budget per depth (--sweep grid)
    fig8_cross_backend.*   SV1 vs lightning.gpu on the same instances
    fig9_degeneracy.*      Sharpe spread inside a 0.1% optimality gap
    fig10_instances.*      the same two quantities over many problem instances
    fig11_gap_vs_sharpe.*  reported gap vs Sharpe loss over the instance sweep

Reproducible: no randomness, all inputs are the committed experiment outputs.

Usage:
    python -m notebooks.generate_figures
"""

from __future__ import annotations

import csv
import os
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "paper01_qubo_baseline" / "results"

#: Set PAPER_FIGURES=1 to render at the size the figures are actually printed
#: at. IEEEtran's column is 3.5 in and its text is 10 pt; a 12.5 in figure
#: scaled into that column shrinks an 11 pt label to 3.1 pt, which is not
#: readable. Rather than scaling a screen figure down, this renders each one at
#: its final width so the fonts are set once and never rescaled. Two-panel
#: figures target the 7.16 in full-width float; single-panel ones the column.
PAPER = os.environ.get("PAPER_FIGURES") == "1"

#: Screen renders go to ``notebooks/figures/``; print-size renders to
#: ``notebooks/figures/paper/``. They used to share one set of file names, so
#: whichever was generated last was what ``papers/md_to_latex.py`` copied next
#: to the .tex -- and one submitted build shipped the 12.5-16 in screen
#: renders scaled into a 7.16 in float, with 8 pt legends printed at 4.5 pt.
#: Separate directories make the build independent of generation order; the
#: converter reads only the ``paper/`` directory and refuses anything wider
#: than the text block.
FIG_DIR = Path(__file__).resolve().parent / "figures" / ("paper" if PAPER else "")

RESULTS_CSV = RESULTS_DIR / "paper01_results.csv"
CONVERGENCE_CSV = RESULTS_DIR / "paper01_convergence.csv"
SEED_SWEEP_CSV = RESULTS_DIR / "paper01_seed_sweep.csv"
DEPTH_SWEEP_CSV = RESULTS_DIR / "paper01_depth_sweep.csv"
GRID_SWEEP_CSV = RESULTS_DIR / "paper01_steps_depth_grid.csv"
GRID_CONVERGENCE_CSV = RESULTS_DIR / "paper01_grid_convergence.csv"
SV1_RESULTS_CSV = RESULTS_DIR / "paper01_results_sv1.csv"
SV1_CONVERGENCE_CSV = RESULTS_DIR / "paper01_convergence_sv1.csv"
DEGENERACY_CSV = RESULTS_DIR / "paper01_degeneracy.csv"
DEGENERACY_QUANTUM_CSV = RESULTS_DIR / "paper01_degeneracy_quantum.csv"
INSTANCES_CSV = RESULTS_DIR / "paper01_instances.csv"
INSTANCES_QAOA_CSV = RESULTS_DIR / "paper01_instances_qaoa.csv"

CLASSICAL = "classical_auto"
QUANTUM = "lightning_gpu"
SV1 = "braket_sv1"

#: Widths in inches: (single-column, full-width) for IEEEtran.
COLUMN_IN, TEXT_IN = 3.5, 7.16

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 8 if PAPER else 11,
    "axes.titlesize": 8 if PAPER else 11,
    "axes.labelsize": 8 if PAPER else 11,
    "xtick.labelsize": 8 if PAPER else 10,
    "ytick.labelsize": 8 if PAPER else 10,
    "legend.fontsize": 8 if PAPER else 9,
    "lines.linewidth": 1.2 if PAPER else 1.8,
    "lines.markersize": 4 if PAPER else 6,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
})


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _save(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        # PDF carries a CreationDate and PNG a timestamp chunk by default, so
        # rerunning this script rewrites every figure with new bytes even when
        # the data is unchanged. Suppressing both keeps the output a pure
        # function of the CSVs, which is what "reproducible figures" has to mean.
        metadata = {"CreationDate": None} if ext == "pdf" else {"Software": None}
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight",
                    metadata=metadata)
    plt.close(fig)
    print(f"  wrote figures/{stem}.png / .pdf")



def _title(ax, text: str, **kw) -> None:
    """Set a panel title, but drop the descriptive ones in the paper build.

    In the paper the caption already carries the description, and the caption
    also says which panel is which ("left: ... right: ..."). Printing the same
    sentence inside the image duplicates it and, at column width, the two
    panels' titles collide. Short titles survive: those identify a panel
    ("N = 16") in a way no caption can.

    The cut is at 14 characters because that is where the two kinds separate:
    every identifier in this file is of the form "N = 16" (6), and the shortest
    description is "Solver runtime scaling" (22). An earlier cut at 28 split the
    descriptions instead, so a two-panel figure kept the title on one panel and
    dropped it on the other -- which reads as leftover text, and was reported as
    such for Figs. 6 and 8.
    """
    if PAPER and (len(text) > 14 or "\n" in text):
        return
    ax.set_title(text.replace("\n", " ") if PAPER else text, **kw)


def _panel(ax, letter: str) -> None:
    """Label a panel "(a)", "(b)", ... in the paper build.

    The caption then says what each panel shows, and the image carries only
    the identifier -- the referee's request for the multi-panel figures.
    """
    if not PAPER:
        return
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold")


def _legend(ax, **kw):
    """Place a legend where it overlaps the data least, and stay readable.

    Hard-coded corners were chosen against the on-screen figure size; at column
    width the data moves and the legend lands on top of it. ``best`` re-solves
    that per figure, and an opaque box keeps the text legible wherever it ends
    up.
    """
    if PAPER:
        kw["loc"] = "best"
        kw["frameon"] = True
        kw.setdefault("framealpha", 0.88)
        # Never below 8 pt at print size: a 7 pt legend was the one thing a
        # referee could still not read once the figures were rendered at
        # their printed width.
        kw["fontsize"] = max(kw.get("fontsize", 8), 8)
        kw.setdefault("borderpad", 0.35)
        kw.setdefault("handlelength", 1.4)
    return ax.legend(**kw)


def fig_convergence(conv: list[dict]) -> None:
    """QAOA expectation of the (normalized) cost Hamiltonian vs optimizer step."""
    by_n: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for r in conv:
        by_n[int(r["n_assets"])].append((int(r["step"]), float(r["expectation"])))

    ns = sorted(by_n)
    colors = cm.viridis(np.linspace(0, 0.9, len(ns)))
    fig, ax = plt.subplots(figsize=(COLUMN_IN, COLUMN_IN * 0.72) if PAPER else (6.4, 4.2))
    for color, n in zip(colors, ns):
        pts = sorted(by_n[n])
        steps = [p[0] for p in pts]
        vals = [p[1] for p in pts]
        ax.plot(steps, vals, color=color, lw=1.6, label=f"N={n}")
    ax.set_xlabel("ADAM optimization step")
    ax.set_ylabel("Expectation ⟨H⟩ (normalized cost Hamiltonian)")
    if not PAPER:
        _title(ax, "QAOA convergence (p = 2, lightning.gpu, single precision)")
    if PAPER:
        # Seven curves leave no empty corner, so make one: headroom above the
        # traces, with the legend laid flat in it rather than over the data.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.42)
        ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center",
                  columnspacing=1.0, handlelength=1.2, handletextpad=0.4)
    else:
        _legend(ax, ncol=2, fontsize=9, frameon=False)
    _save(fig, "fig1_convergence")


def fig_runtime_scaling(results: list[dict]) -> None:
    """Runtime vs N for classical and quantum solvers (log y)."""
    cls = sorted((int(r["n_assets"]), float(r["runtime_s"]))
                 for r in results if r["solver"] == CLASSICAL)
    qnt = sorted((int(r["n_assets"]), float(r["runtime_s"]))
                 for r in results if r["solver"] == QUANTUM)

    fig, ax = plt.subplots(figsize=(COLUMN_IN, COLUMN_IN * 0.72) if PAPER else (6.4, 4.2))
    ax.plot([x for x, _ in cls], [y for _, y in cls], "o-", color="#2a7de1",
            lw=1.8, label="Classical (exact C(N,K) / SA)")
    ax.plot([x for x, _ in qnt], [y for _, y in qnt], "s-", color="#e15a2a",
            lw=1.8, label="QAOA (lightning.gpu)")
    ax.set_yscale("log")
    ax.set_xlabel("Number of assets N")
    ax.set_ylabel("Wall-clock runtime (s, log scale)")
    if not PAPER:
        _title(ax, "Solver runtime scaling")
    # Annotate the quantum ceiling, placed low-left so it clears the title.
    ax.annotate("N=30: 3.14 h\n(VRAM 41.3/49 GB)",
                xy=(30, qnt[-1][1]), xytext=(34, 900) if PAPER else (15, 900),
                fontsize=8.5, color="#e15a2a", ha="left",
                arrowprops=dict(arrowstyle="->", color="#e15a2a", lw=1))
    _legend(ax, fontsize=9, frameon=False, loc="center right")
    _save(fig, "fig3_runtime_scaling")


def fig_optimality_gap(results: list[dict]) -> None:
    """Quantum optimality gap (%) relative to the classical objective."""
    cls = {int(r["n_assets"]): float(r["energy_no_offset"])
           for r in results if r["solver"] == CLASSICAL}
    qnt = {int(r["n_assets"]): float(r["energy_no_offset"])
           for r in results if r["solver"] == QUANTUM}
    ns = sorted(set(cls) & set(qnt))
    gaps = [abs(qnt[n] - cls[n]) / abs(cls[n]) * 100 for n in ns]

    fig, ax = plt.subplots(figsize=(COLUMN_IN, COLUMN_IN * 0.72) if PAPER else (6.4, 4.2))
    bars = ax.bar([str(n) for n in ns], gaps, color="#7a5cff", width=0.6)
    for b, g in zip(bars, gaps):
        ax.text(b.get_x() + b.get_width() / 2, g + 0.001,
                f"{g:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xlabel("Number of assets N")
    ax.set_ylabel("Optimality gap vs classical (%)")
    if not PAPER:
        _title(ax, "QAOA solution quality (0% = classical optimum)")
    ax.set_ylim(0, max(gaps) * 1.25 if gaps else 1)
    _save(fig, "fig2_optimality_gap")


def fig_sharpe(results: list[dict]) -> None:
    """Sharpe ratio of the selected portfolio vs N, classical and quantum."""
    def series(solver: str):
        return sorted((int(r["n_assets"]), float(r["sharpe_ratio"]))
                      for r in results if r["solver"] == solver)
    cls, qnt = series(CLASSICAL), series(QUANTUM)

    fig, ax = plt.subplots(figsize=(COLUMN_IN, COLUMN_IN * 0.72) if PAPER else (6.4, 4.2))
    ax.plot([x for x, _ in cls], [y for _, y in cls], "o-", color="#2a7de1",
            lw=1.8, label="Classical")
    ax.plot([x for x, _ in qnt], [y for _, y in qnt], "s--", color="#e15a2a",
            lw=1.8, label="QAOA")
    ax.set_xlabel("Number of assets N")
    ax.set_ylabel("Sharpe ratio of selected portfolio")
    if not PAPER:
        _title(ax, "Portfolio quality vs problem size")
    _legend(ax, fontsize=9, frameon=False)
    _save(fig, "fig4_sharpe")


def fig_seed_variance(seed_rows: list[dict]) -> None:
    """Spread of the optimality gap across QAOA initial-angle seeds, per N.

    The problem instance is identical within each N, so all spread shown here
    comes from the random initial angles. Sizes are restricted to N <= 20, where
    the classical reference is brute force, so a gap is QAOA-vs-exact-optimum.
    """
    by_n: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for r in seed_rows:
        by_n[int(r["n_assets"])].append(
            (float(r["gap_pct"]), r["feasible"] == "True"))

    ns = sorted(by_n)
    # Infeasible runs appear at only some sizes, so the legend entry has to be
    # attached to the first one that actually occurs, not to a fixed column.
    labelled: set[str] = set()

    def once(key: str) -> str | None:
        if key in labelled:
            return None
        labelled.add(key)
        return key

    fig, ax = plt.subplots(figsize=(COLUMN_IN, COLUMN_IN * 0.75) if PAPER else (6.6, 4.4))
    for x, n in enumerate(ns):
        runs = by_n[n]
        ok = [g for g, feasible in runs if feasible]
        bad = [g for g, feasible in runs if not feasible]
        if ok:
            ax.scatter([x] * len(ok), ok, s=44, color="#7a5cff", zorder=3,
                       label=once("feasible"))
        if bad:
            ax.scatter([x] * len(bad), bad, s=90, marker="X", color="#d62728",
                       zorder=5, label=once("cardinality violated"))
        gaps = sorted(g for g, _ in runs)
        median = gaps[len(gaps) // 2]
        ax.hlines(median, x - 0.22, x + 0.22, color="#e15a2a", lw=2, zorder=4,
                  label=once("median"))

    # One run is three orders of magnitude worse than the rest; a linear axis
    # would collapse every other point onto zero. symlog keeps exact hits (0 %)
    # on the axis while still showing the outlier.
    ax.set_yscale("symlog", linthresh=0.001)
    ax.set_xlim(-0.5, len(ns) - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(
        [f"N={n}\n{len(by_n[n])} seeds\nmax {max(g for g, _ in by_n[n]):.3f} %"
         for n in ns])
    ax.set_ylabel("Optimality gap vs exact optimum (%, symlog)")
    if not PAPER:
        _title(ax, "QAOA seed variance on a fixed instance (p = 2)")
    _legend(ax, fontsize=9, frameon=False, loc="upper left")
    _save(fig, "fig5_seed_variance")


def fig_depth_sensitivity(depth_rows: list[dict]) -> None:
    """Optimality gap and runtime as functions of QAOA depth p.

    Two panels share the x axis (p) so the accuracy gained by a deeper circuit
    can be read against the runtime it costs.
    """
    gap_by_n: dict[int, list[tuple[int, float]]] = defaultdict(list)
    time_by_n: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for r in depth_rows:
        n, p = int(r["n_assets"]), int(r["p_layers"])
        gap_by_n[n].append((p, float(r["gap_pct"])))
        time_by_n[n].append((p, float(r["runtime_s"])))

    ns = sorted(gap_by_n)
    colors = cm.viridis(np.linspace(0, 0.75, len(ns)))
    fig, (ax_gap, ax_time) = plt.subplots(1, 2, figsize=(TEXT_IN, TEXT_IN * 0.34) if PAPER else (9.6, 4.0))
    for color, n in zip(colors, ns):
        gaps = sorted(gap_by_n[n])
        times = sorted(time_by_n[n])
        ax_gap.plot([p for p, _ in gaps], [g for _, g in gaps], "o-",
                    color=color, lw=1.8, label=f"N={n}")
        ax_time.plot([p for p, _ in times], [t for _, t in times], "s-",
                     color=color, lw=1.8, label=f"N={n}")

    ps = sorted({p for pts in gap_by_n.values() for p, _ in pts})
    for ax in (ax_gap, ax_time):
        ax.set_xlabel("QAOA depth p")
        ax.set_xticks(ps)
    ax_gap.set_ylabel("Optimality gap vs exact optimum (%)")
    _title(ax_gap, "Solution quality vs depth")
    ax_time.set_ylabel("Wall-clock runtime (s)")
    _title(ax_time, "Cost of depth (lightning.gpu)")
    _legend(ax_gap, fontsize=9, frameon=False)
    _save(fig, "fig6_depth_sensitivity")


def fig_steps_depth_grid(grid_rows: list[dict],
                         grid_conv_rows: list[dict] | None = None) -> None:
    """Two views of the steps x p grid: what the optimizer minimizes, and what we report.

    This figure exists to answer a question the depth sweep (Supplementary S-VI) raised: whether
    p = 4's regression at a fixed 50-step budget was the ansatz or simply
    under-optimization. Answering it needs both rows, because they are different
    quantities:

    Top row — the final expectation <H_norm>, i.e. the objective ADAM actually
    descends. This is where saturation is legible: a curve that flattens means
    the optimizer has stopped making progress, so any remaining difference
    between depths at that budget is about the ansatz, not the budget.

    Bottom row — the optimality gap of the best of 1000 final samples, i.e. the
    number the paper reports. It is a min over samples, not the optimized
    quantity, so it can be near-zero even where the top row converged poorly.
    Plotting them together is the point: the two rows disagree, and that
    disagreement is the finding.

    The x axis is log-scaled because the step ladder is geometric, and gaps use a
    symlog y axis so exact hits (gap = 0) stay on the plot instead of falling off
    a log axis.
    """
    # Gap per (N, p, steps) across the five seeds, split by feasibility. The band
    # and median summarise the *feasible* runs — the typical optimizer behaviour,
    # the quantity the depth/budget question is about. Infeasible runs (a rare bad
    # seed that leaves the constraint set, cf. Fig. 5) are drawn as separate red
    # outlier markers: never hidden, but never allowed to dominate the axis.
    by_n: dict[int, dict[int, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    infeas: dict[int, dict[int, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list))
    for r in grid_rows:
        g = r["gap_pct"]
        if g in ("", None):
            continue
        n, p, steps = int(r["n_assets"]), int(r["p_layers"]), int(r["steps"])
        if r.get("feasible") == "False":
            infeas[n][p].append((steps, float(g)))
        else:
            by_n[n][p][steps].append(float(g))

    # Final expectation per run (N, p, steps, seed): the last convergence point.
    # Grouped by (N, p, steps) so the top row is also a median + min-max band.
    final_exp: dict[int, dict[int, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    if grid_conv_rows:
        last: dict[tuple[int, int, int, int], tuple[int, float]] = {}
        for r in grid_conv_rows:
            key = (int(r["n_assets"]), int(r["p_layers"]), int(r["steps"]),
                   int(r["qaoa_seed"]))
            point = (int(r["step"]), float(r["expectation"]))
            if key not in last or point[0] > last[key][0]:
                last[key] = point
        for (n, p, steps, _seed), (_, value) in last.items():
            final_exp[n][p][steps].append(value)

    ns = sorted(by_n)
    nrows = 2 if final_exp else 1
    fig, axes = plt.subplots(nrows, len(ns), figsize=((TEXT_IN, TEXT_IN * 0.30 * nrows) if PAPER
                                     else (4.2 * len(ns), 3.6 * nrows)),
                             squeeze=False)
    ps = sorted({p for per_p in by_n.values() for p in per_p}
                | {p for per_p in infeas.values() for p in per_p})
    colors = cm.plasma(np.linspace(0, 0.8, len(ps)))
    # Distinct markers as well as colours: several depths reach gap = 0 and their
    # curves then sit exactly on top of each other on the symlog floor.
    markers = ["o", "s", "^", "D", "v", "P"]

    def _draw(ax, series: dict[int, dict[int, list[float]]]) -> None:
        # series[p][steps] = values across seeds -> median line + min-max band.
        for color, marker, p in zip(colors, markers, ps):
            per_steps = series.get(p, {})
            xs = sorted(per_steps)
            if not xs:
                continue
            med = [statistics.median(per_steps[s]) for s in xs]
            # Interquartile (25-75 %) band, not min-max: with five seeds a single
            # bad-basin run otherwise stretches the band across the whole panel and
            # buries the median trend. The extreme runs are still shown — as the
            # infeasible X markers on the gap row.
            lo = [float(np.percentile(per_steps[s], 25)) for s in xs]
            hi = [float(np.percentile(per_steps[s], 75)) for s in xs]
            ax.plot(xs, med, marker=marker, ls="-", color=color, lw=1.8, ms=5,
                    mfc="none", label=f"p={p}")
            # Band only where a cell actually has spread across seeds.
            if any(h > l for l, h in zip(lo, hi)):
                ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.grid(True, which="major", alpha=0.25)
        steps_seen = sorted({s for per_steps in series.values() for s in per_steps})
        ax.set_xticks(steps_seen)
        ax.set_xticklabels([str(s) for s in steps_seen])
        # The log locator would otherwise label decade subdivisions (6x10^1,
        # 3x10^2) on top of the step counts we just set.
        ax.set_xticks([], minor=True)

    gap_row = axes[-1]
    if final_exp:
        for ax, n in zip(axes[0], ns):
            # Each N converges to a different depth of <H>, so a shared y axis
            # would flatten every panel; saturation is a per-panel shape.
            _draw(ax, final_exp[n])
            _title(ax, f"N = {n}")
        axes[0][0].set_ylabel(r"Final $\langle H_{norm} \rangle$ (median, IQR band)")
        axes[0][0].legend(fontsize=9, frameon=False, title="depth")

    for ax, n in zip(gap_row, ns):
        _draw(ax, by_n[n])
        # Infeasible runs as red X, coloured outside the depth palette so they
        # read as "constraint violated", not as another depth curve.
        labelled = False
        for p, pts in sorted(infeas[n].items()):
            if not pts:
                continue
            ax.scatter([s for s, _ in pts], [g for _, g in pts],
                       marker="x", color="#d62728", s=45, zorder=5,
                       label=None if labelled else "infeasible")
            labelled = True
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel("ADAM steps")
        if not final_exp:
            _title(ax, f"N = {n}")
        # No shared y axis: one N can carry an 11 % infeasible outlier while
        # another sits under 0.6 %, so a shared scale would crush the feasible
        # structure. Each panel autoscales; symlog keeps the sub-0.1 % medians
        # legible alongside any outlier.
    gap_row[0].set_ylabel("Optimality gap, best of 1000 (%)\n(median + feasible IQR)")
    if not final_exp:
        gap_row[0].legend(fontsize=9, frameon=False, title="depth", loc="lower left")
    # Outlier-marker legend, kept separate so it never re-lists the depths.
    if any(infeas[n][p] for n in ns for p in infeas[n]):
        from matplotlib.lines import Line2D
        proxy = Line2D([], [], marker="x", color="#d62728", ls="none", ms=7,
                       label="infeasible")
        gap_row[-1].add_artist(gap_row[-1].legend(
            handles=[proxy], fontsize=8, frameon=False, loc="upper right"))
    if not PAPER:
        fig.suptitle("Separating circuit depth from optimizer budget "
                     "(5 seeds: median + IQR band; × = infeasible)", y=0.99)
    _save(fig, "fig7_steps_depth_grid")


def fig_cross_backend(sv1_rows: list[dict], sv1_conv: list[dict],
                      gpu_rows: list[dict], gpu_conv: list[dict]) -> None:
    """Same experiment on SV1 (cloud, double) and lightning.gpu (local, single).

    Left: the optimality gap each backend reports. Right: how far apart the two
    optimizer trajectories actually are, step by step. The right panel is the
    point of the figure — the disagreement is ~1e-6 at every N, so the reason
    the backends return different portfolios at N = 20 and identical ones below
    it is not that the numerical error grows, it is that the solution landscape
    becomes near-degenerate enough for a fixed 1e-6 to change the answer.
    """
    def energies(rows, solver):
        return {int(r["n_assets"]): float(r["energy_no_offset"])
                for r in rows if r["solver"] == solver}

    def traj(rows, backend):
        out: dict[int, dict[int, float]] = defaultdict(dict)
        for r in rows:
            if r["backend"] == backend:
                out[int(r["n_assets"])][int(r["step"])] = float(r["expectation"])
        return out

    sv1_e, gpu_e = energies(sv1_rows, SV1), energies(gpu_rows, QUANTUM)
    # Each run carries its own classical row, so the gap is always computed
    # against the reference the same run produced.
    sv1_c, gpu_c = energies(sv1_rows, CLASSICAL), energies(gpu_rows, CLASSICAL)
    ns = sorted(set(sv1_e) & set(gpu_e))
    if not ns:
        print("  skipped fig8_cross_backend: no overlapping N")
        return

    fig, (ax_gap, ax_diff) = plt.subplots(1, 2, figsize=(TEXT_IN, TEXT_IN * 0.36) if PAPER else (10.4, 4.2))

    width = 0.38
    xs = np.arange(len(ns))
    series = []
    for offset, (e, c, label, color) in enumerate((
        (gpu_e, gpu_c, "lightning.gpu (local, single)", "#e15a2a"),
        (sv1_e, sv1_c, "SV1 (cloud, double)", "#2a7de1"),
    )):
        gaps = [abs(e[n] - c[n]) / abs(c[n]) * 100 for n in ns]
        pos = xs + (offset - 0.5) * width
        ax_gap.bar(pos, gaps, width=width, color=color, label=label)
        series.append((pos, gaps))
    # Label after both series exist. Where the two bars round to the same value
    # -- both exactly optimal at N = 8 and 12 -- their labels overlap into
    # "0.0000.000"; one centred label carries the same information.
    (pa, ga), (pb, gb) = series
    for i, x in enumerate(xs):
        if f"{ga[i]:.3f}" == f"{gb[i]:.3f}":
            ax_gap.text(x, max(ga[i], gb[i]), f"{ga[i]:.3f}",
                        ha="center", va="bottom", fontsize=8)
        else:
            for px, g in ((pa[i], ga[i]), (pb[i], gb[i])):
                ax_gap.text(px, g, f"{g:.3f}", ha="center", va="bottom", fontsize=8)
    ax_gap.set_xticks(xs, [str(n) for n in ns])
    ax_gap.set_xlabel("Number of assets N")
    ax_gap.set_ylabel("Optimality gap vs classical (%)")
    _legend(ax_gap, fontsize=8.5, frameon=False, loc="upper left")

    sv1_t, gpu_t = traj(sv1_conv, SV1), traj(gpu_conv, QUANTUM)
    colors = cm.viridis(np.linspace(0, 0.9, len(ns)))
    for color, n in zip(colors, ns):
        steps = sorted(set(sv1_t.get(n, {})) & set(gpu_t.get(n, {})))
        if not steps:
            continue
        diff = [abs(sv1_t[n][s] - gpu_t[n][s]) for s in steps]
        style = "-" if sv1_e[n] == gpu_e[n] else "--"
        marker = "" if sv1_e[n] == gpu_e[n] else "o"
        ax_diff.plot(steps, diff, style, marker=marker, ms=3, color=color, lw=1.6,
                     label=f"N={n}" + ("" if sv1_e[n] == gpu_e[n] else "  (differing solution)"))
    ax_diff.set_yscale("log")
    ax_diff.set_xlabel("ADAM optimization step")
    ax_diff.set_ylabel("|⟨H⟩$_{SV1}$ − ⟨H⟩$_{GPU}$| (normalized)")
    _title(ax_diff, "Backend disagreement per step")
    _legend(ax_diff, fontsize=8.5, frameon=False)

    if not PAPER:
        fig.suptitle("Cross-backend reproducibility: bit-identical to N = 16, "
                     "different basin at N = 20", y=1.0)
    _save(fig, "fig8_cross_backend")


def fig_degeneracy(band_rows: list[dict], quantum_rows: list[dict]) -> None:
    """What an optimality gap of "0.1%" actually contains.

    Left: the Sharpe ratios spanned by every feasible solution within 0.1% of
    the optimum under the gap definition this paper (and the literature) quotes.
    Right: why that band is so wide. For a feasible x the cardinality penalty
    contributes the constant -A*K^2 to x'Qx, so the gap's denominator |E*| is
    approximately A*K^2 -- a quantity fixed by the formulation, not by the data
    or the solution. It grows as O(N^3) while the portfolio objective stays
    O(1), so the same solution error is quoted as a smaller number at larger N.
    """
    band = 0.1
    rows = sorted(
        (r for r in band_rows
         if r["band_kind"] == "reported_gap" and float(r["band_pct"]) == band),
        key=lambda r: int(r["n_assets"]),
    )
    if not rows:
        print("  skipped fig9_degeneracy: no reported_gap band rows")
        return
    ns = [int(r["n_assets"]) for r in rows]
    xs = np.arange(len(ns))
    quantum = {int(r["n_assets"]): r for r in quantum_rows}

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(TEXT_IN, TEXT_IN * 0.38) if PAPER else (12.5, 4.6))

    # --- left: Sharpe spread inside the band --------------------------------
    for i, r in zip(xs, rows):
        lo, hi = float(r["sharpe_min"]), float(r["sharpe_max"])
        q1, q3 = float(r["sharpe_p25"]), float(r["sharpe_p75"])
        ax_l.vlines(i, lo, hi, color="#8899aa", lw=1.4, zorder=1)
        ax_l.add_patch(plt.Rectangle((i - 0.22, q1), 0.44, q3 - q1,
                                     facecolor="#7a5cff", alpha=0.35,
                                     edgecolor="#7a5cff", zorder=2))
        ax_l.hlines(float(r["sharpe_median"]), i - 0.22, i + 0.22,
                    color="#4b2fd6", lw=2, zorder=3)
    ax_l.plot(xs, [float(r["sharpe_optimum"]) for r in rows], "D",
              color="#111111", ms=6, zorder=5, label="Exact optimum")
    qx = [i for i, n in zip(xs, ns) if n in quantum]
    if qx:
        ax_l.plot(qx, [float(quantum[ns[i]]["sharpe_quantum"]) for i in qx], "X",
                  color="#e04b2f", ms=9, zorder=6, label="QAOA solution")
    for i, r in zip(xs, rows):
        cnt = int(r["n_within"])
        lbl = f"{cnt/1000:.0f}k" if PAPER and cnt >= 10_000 else f"{cnt:,}"
        ax_l.annotate(lbl, (i, float(r["sharpe_max"])),
                      textcoords="offset points", xytext=(0, 7),
                      ha="center", fontsize=8, color="#555555")
    ax_l.set_xticks(xs)
    ax_l.set_xticklabels([str(n) for n in ns])
    ax_l.set_xlabel("Number of assets N")
    ax_l.set_ylabel("Sharpe ratio")
    _title(ax_l, f"Sharpe ratios within a {band}% optimality gap\n"
                   "(box = IQR, whisker = full range, label = solutions in band)",
                   fontsize=10)
    _legend(ax_l, fontsize=9, loc="lower left")

    # --- right: the gap denominator -----------------------------------------
    # Sizes where QAOA hit the optimum exactly have a gap of 0 under either
    # denominator, which a log axis cannot show; they are named instead.
    exact_hits = [n for n in sorted(quantum)
                  if float(quantum[n]["reported_gap_pct"]) == 0.0]
    qn = [n for n in sorted(quantum) if float(quantum[n]["reported_gap_pct"]) > 0.0]
    if qn:
        reported = [float(quantum[n]["reported_gap_pct"]) for n in qn]
        honest = [float(quantum[n]["gap_of_objective_range_pct"]) for n in qn]
        ax_r.plot(qn, reported, "o-", color="#7a5cff", lw=2,
                  label="Gap as reported,  (E−E*)/|E*|")
        ax_r.plot(qn, honest, "s-", color="#e04b2f", lw=2,
                  label="Gap vs objective spread,  (E−E*)/(E$_{max}$−E*)")
        ax_r.set_yscale("log")
        for n, a, b in zip(qn, reported, honest):
            ax_r.annotate(f"{b / a:.0f}×", (n, (a * b) ** 0.5),
                          textcoords="offset points", xytext=(6, -3),
                          fontsize=8.5, color="#555555")
        # Headroom so the legend clears the upper series.
        ax_r.set_ylim(min(reported) / 3, max(honest) * 12)
        ax_r.set_xticks(qn)
        ax_r.set_xlabel("Number of assets N")
        ax_r.set_ylabel("Optimality gap (%, log)")
        _title(ax_r, "The same QAOA solutions under two denominators\n"
                       "(annotation = understatement factor)", fontsize=10)
        _legend(ax_r, fontsize=9, loc="upper left")
        if exact_hits and not PAPER:
            ax_r.text(
                0.5, 0.02,
                "N = " + ", ".join(str(n) for n in exact_hits)
                + " reached the exact optimum (gap 0, off a log axis)",
                transform=ax_r.transAxes, ha="center", fontsize=8.5,
                color="#555555")
    _save(fig, "fig9_degeneracy")


def fig_instances(rows: list[dict]) -> None:
    """The single-instance findings, repeated over many instances.

    Every other figure in this paper is one problem instance per size, because
    the loader always returned the same subset. Both panels redraw the two
    quantities Supplementary S-III rests on over independently drawn subsets of the same
    cached universe, with the reference instance (0) marked so a reader can see
    whether the headline numbers were lucky.
    """
    by_n: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_n[int(r["n_assets"])].append(r)
    ns = sorted(by_n)
    if not ns:
        print("  skipped fig10_instances: no rows")
        return
    xs = np.arange(len(ns))

    def panel(ax, field, ylabel, title, log):
        for i, n in zip(xs, ns):
            vals = sorted(float(r[field]) for r in by_n[n])
            lo, hi = vals[0], vals[-1]
            q1 = float(np.percentile(vals, 25))
            q3 = float(np.percentile(vals, 75))
            ax.vlines(i, lo, hi, color="#8899aa", lw=1.4, zorder=1)
            ax.add_patch(plt.Rectangle((i - 0.22, q1), 0.44, q3 - q1,
                                       facecolor="#2a7de1", alpha=0.30,
                                       edgecolor="#2a7de1", zorder=2))
            ax.hlines(float(np.median(vals)), i - 0.22, i + 0.22,
                      color="#1a4f91", lw=2, zorder=3)
        published = [
            float(r[field]) for n in ns for r in by_n[n]
            if r["is_published_instance"] == "True"
        ]
        ax.plot(xs[: len(published)], published, "D", color="#e04b2f", ms=6,
                zorder=5, label="Reference instance (0)")
        if log:
            ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("Number of assets N")
        ax.set_ylabel(ylabel)
        _title(ax, title, fontsize=10)
        _legend(ax, fontsize=9, loc="upper left")

    counts = {len(v) for v in by_n.values()}
    label = f"{min(counts)}" if len(counts) == 1 else f"{min(counts)}–{max(counts)}"
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(TEXT_IN, TEXT_IN * 0.38) if PAPER else (12.5, 4.6))
    panel(ax_l, "deflation_factor", "|E*| / objective spread  (log)",
          f"How far the gap denominator overstates the objective\n"
          f"({label} instances per size; box = IQR, whisker = range)", True)
    panel(ax_r, "band_sharpe_spread_pct", "Sharpe spread in the band (% of optimum)",
          "Sharpe ratios spanned by a 0.1 % optimality gap\n"
          "(same instances)", False)
    _save(fig, "fig10_instances")


def fig_gap_vs_sharpe(rows: list[dict]) -> None:
    """What the reported gap tells you about the portfolio you actually get.

    Left: every sub-optimal QAOA solution across the instance sweep, plotted as
    reported gap against Sharpe loss. If the gap were a proxy for portfolio
    quality these would form a trend; they do not. Right: how often QAOA reached
    the exact optimum, which is the single-instance claim of Table I ("exact
    agreement up to N = 16") restated as a rate.
    """
    feasible = [r for r in rows if r.get("quantum_feasible") == "True"]
    if not feasible:
        print("  skipped fig11_gap_vs_sharpe: no QAOA rows")
        return
    ns = sorted({int(r["n_assets"]) for r in feasible})
    colors = dict(zip(ns, cm.viridis(np.linspace(0.1, 0.8, len(ns)))))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(TEXT_IN, TEXT_IN * 0.38) if PAPER else (12.5, 4.6))

    missed = [r for r in feasible if int(r["quantum_rank"]) > 1]
    for n in ns:
        pts = [r for r in missed if int(r["n_assets"]) == n]
        if not pts:
            continue
        ax_l.scatter([float(r["reported_gap_pct"]) for r in pts],
                     [float(r["sharpe_delta_pct"]) for r in pts],
                     s=42, color=colors[n], edgecolor="white", lw=0.6,
                     label=f"N={n} ({len(pts)})", zorder=3)
    ax_l.axhline(0, color="#888888", lw=1, zorder=1)
    ax_l.set_xscale("log")
    ax_l.set_xlabel("Reported optimality gap (%, log)")
    ax_l.set_ylabel("Sharpe ratio vs the exact optimum (%)")
    _title(ax_l, "Every sub-optimal QAOA solution in the instance sweep\n"
                   "(the gap does not predict the Sharpe loss)", fontsize=10)
    _legend(ax_l, fontsize=9, loc="lower left", title="assets (count)",
                title_fontsize=9)

    by_n: dict[int, list[dict]] = defaultdict(list)
    for r in feasible:
        by_n[int(r["n_assets"])].append(r)
    rates = [
        100.0 * sum(1 for r in by_n[n] if int(r["quantum_rank"]) == 1) / len(by_n[n])
        for n in ns
    ]
    bars = ax_r.bar([str(n) for n in ns], rates, width=0.55,
                    color=[colors[n] for n in ns])
    for bar, rate, n in zip(bars, rates, ns):
        hits = sum(1 for r in by_n[n] if int(r["quantum_rank"]) == 1)
        ax_r.text(bar.get_x() + bar.get_width() / 2, rate + 1.5,
                  f"{hits}/{len(by_n[n])}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 105)
    ax_r.set_xlabel("Number of assets N")
    ax_r.set_ylabel("Instances reaching the exact optimum (%)")
    _title(ax_r, "Table I reports exact agreement up to N = 16;\n"
                   "across instances that holds just over half the time",
                   fontsize=10)
    _save(fig, "fig11_gap_vs_sharpe")


def main() -> None:
    if not RESULTS_CSV.exists() or not CONVERGENCE_CSV.exists():
        raise SystemExit(
            f"Missing result CSVs under {RESULTS_DIR}. Run the experiment first:\n"
            "  python -m experiments.paper01_qubo_baseline.run_experiment "
            "--backend lightning_gpu --precision single "
            "--quantum-n 8 12 16 20 24 28 30"
        )
    results = _read_csv(RESULTS_CSV)
    conv = _read_csv(CONVERGENCE_CSV)
    print(f"Generating figures into {FIG_DIR} ...")
    fig_convergence(conv)
    fig_runtime_scaling(results)
    fig_optimality_gap(results)
    fig_sharpe(results)

    # The sweeps are a separate, optional run; skip rather than fail without them.
    # Fig. 7 needs the grid's convergence trace as well as its summary rows; it
    # degrades to the gap row alone if the convergence file is missing.
    def _render_grid(rows: list[dict]) -> None:
        conv = _read_csv(GRID_CONVERGENCE_CSV) if GRID_CONVERGENCE_CSV.exists() else None
        fig_steps_depth_grid(rows, conv)

    for path, render, name in (
        (SEED_SWEEP_CSV, fig_seed_variance, "fig5_seed_variance"),
        (DEPTH_SWEEP_CSV, fig_depth_sensitivity, "fig6_depth_sensitivity"),
        (GRID_SWEEP_CSV, _render_grid, "fig7_steps_depth_grid"),
    ):
        if path.exists():
            render(_read_csv(path))
        else:
            print(f"  skipped {name}: {path.name} not found "
                  "(run experiments.paper01_qubo_baseline.run_sweep)")

    # The SV1 comparison is a separate, billed cloud run; skip without it.
    if SV1_RESULTS_CSV.exists() and SV1_CONVERGENCE_CSV.exists():
        fig_cross_backend(_read_csv(SV1_RESULTS_CSV), _read_csv(SV1_CONVERGENCE_CSV),
                          results, conv)
    else:
        print("  skipped fig8_cross_backend: "
              f"{SV1_RESULTS_CSV.name} not found (run_experiment "
              "--backend braket_sv1 --results-tag sv1)")

    # The degeneracy census is its own free run; skip without it.
    if DEGENERACY_CSV.exists() and DEGENERACY_QUANTUM_CSV.exists():
        fig_degeneracy(_read_csv(DEGENERACY_CSV), _read_csv(DEGENERACY_QUANTUM_CSV))
    else:
        print(f"  skipped fig9_degeneracy: {DEGENERACY_CSV.name} not found "
              "(run experiments.paper01_qubo_baseline.run_degeneracy)")

    # The instance sweep is a separate, free run; skip without it.
    if INSTANCES_CSV.exists():
        fig_instances(_read_csv(INSTANCES_CSV))
    else:
        print(f"  skipped fig10_instances: {INSTANCES_CSV.name} not found "
              "(run experiments.paper01_qubo_baseline.run_instances)")

    # The QAOA arm of the instance sweep needs a GPU; skip without it.
    if INSTANCES_QAOA_CSV.exists():
        fig_gap_vs_sharpe(_read_csv(INSTANCES_QAOA_CSV))
    else:
        print(f"  skipped fig11_gap_vs_sharpe: {INSTANCES_QAOA_CSV.name} not found "
              "(run run_instances --backend lightning_gpu --results-tag qaoa)")
    print("Done.")


if __name__ == "__main__":
    main()
