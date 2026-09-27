# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT
"""
Figures of the system paper, drawn from the record and the E-S2 result file.

Fig. 1  The record and its thirteen relations: the four files a run is
        described by, and which check binds which pair of them.
Fig. 2  Where each misconfiguration stops under the four guard configurations
        (from ``paper02_es2_guard_stages.csv``): a lane per configuration, a
        marker at the stage reached, sized by nothing and coloured by whether
        the run was billed.

Text is embedded as TrueType (pdf.fonttype 42), as the numerical paper's
figures are; acmsmall's text width is 395.8 pt = 5.5 in.

Run:
    python papers/system_paper/make_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import os

# A figure that regenerates to a different file cannot be checked against its data:
# matplotlib stamps the PDF with the time unless this is pinned (it is read at import).
os.environ.setdefault("SOURCE_DATE_EPOCH", "1600000000")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).parent
FIG = Path(os.environ.get("SYSTEM_PAPER_FIGDIR", "")) if os.environ.get("SYSTEM_PAPER_FIGDIR") \
    else HERE / "figures"
RESULTS = HERE.parents[1] / "experiments" / "system_paper" / "results"
TEXT_IN = 5.4

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "font.size": 7.5, "font.family": "DejaVu Sans",
})


def _box(ax, x, y, w, h, title, lines, fc="#f4f4f4"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec="black", lw=0.8))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=7.2, fontweight="bold")
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.55 - 0.36 * i, line, ha="center", va="top", fontsize=6.0)


def _arrow(ax, p, q, label, offset=(0, 0), style="-|>", color="black", ls="-", lw=0.8, la=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, lw=lw, color=color,
                                 linestyle=ls, shrinkA=2, shrinkB=2))
    mx, my = (p[0] + q[0]) / 2 + offset[0], (p[1] + q[1]) / 2 + offset[1]
    ax.text(mx, my, label, ha="center", va="center", fontsize=6.4, color=color,
            bbox=dict(fc="white", ec="none", pad=0.6), **(la or {}))


def _check_fits(fig, ax, claims) -> None:
    """Every label must lie inside the box or the gap it was placed in."""
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bad = []
    for t, x0, x1, where in claims:
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        a = inv.transform((bb.x0, bb.y0))[0]
        b = inv.transform((bb.x1, bb.y1))[0]
        if a < x0 + 0.05 or b > x1 - 0.05:
            bad.append(f"{where}: {t.get_text()[:40]!r} spans {a:.2f}-{b:.2f}, room is {x0:.2f}-{x1:.2f}")
    if bad:
        raise AssertionError("figure text does not fit:\n  " + "\n  ".join(bad))


def _width(fig, ax, text: str, size: float, bold: bool = False) -> float:
    """Width of a string in data units, measured rather than guessed."""
    t = ax.text(0, 0, text, fontsize=size, fontweight="bold" if bold else "normal")
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    w = inv.transform((bb.x1, bb.y1))[0] - inv.transform((bb.x0, bb.y0))[0]
    t.remove()
    return w


def fig_record_relations() -> None:
    """Top-to-bottom flow of the four files; relations labelled on the arrows
    between them and listed inside each file for the ones that hold within it.

    Every string is measured at draw time and the geometry follows from the
    measurements: body text is wrapped to a width the column can hold, each box
    is as tall as its own number of lines, and the gap between the columns is set
    by the widest arrow label. The assertion at the end fails the build if a
    label still crosses a border, which is how this figure stopped having labels
    on top of the drawing.
    """
    import textwrap

    fig, ax = plt.subplots(figsize=(TEXT_IN, 3.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.6)
    ax.axis("off")
    fig.canvas.draw()
    BODY, TITLE, ARROW, PAD, LINE = 5.2, 6.2, 5.0, 0.28, 0.32

    files = [
        ("Derived statistics", ["μ, Σ of one price window; the pinned input"]),
        ("Reference row  (per N, instance)",
         ["tickers, input digest, |F|, f*, Δ_F, A_crit, A_margin",
          "within: identities (scales from the cardinality minima)"]),
        ("Run row  (per run, 101 columns)",
         ["keys, digest, angles, P_F, P_τ, Q_τ(S), g_F, rank",
          "within: schema, manifest, penalty, scorer, seed-distinct"]),
        ("Per-run artifact  (npz)",
         ["trajectory, P over F (final, initial), shot indices", "within: npz-internal"]),
    ]
    right = [("Naive pipeline  (control)", ["reads the run rows,", "reports the aggregate"]),
             ("Re-simulation", ["the state at the", "recorded final angles"])]
    arrows = ["schema only", "state-recompute"]
    between = ["input-digest, reference-input", "reference-row, input-digest",
               "row-npz, identities, state-batch"]

    # Wrap the left column's body to the width its own longest title needs, so that
    # the column is set by the titles rather than by one long enumeration.
    wl_title = max(_width(fig, ax, t, TITLE, True) for t, _ in files)
    target = max(wl_title, 4.6)
    def wrap(lines, limit):
        out = []
        for line in lines:
            n = max(10, int(len(line) * limit / max(_width(fig, ax, line, BODY), 1e-6)))
            parts = textwrap.wrap(line, n) or [line]
            if len(parts) == 2:        # balance, so that one word is not left alone
                parts = textwrap.wrap(line, max(10, (len(line) + 1) // 2 + 4)) or parts
            out.extend(parts)
        return out
    files = [(t, wrap(b, target)) for t, b in files]
    right = [(t, b) for t, b in right]

    wl = max([wl_title] + [_width(fig, ax, l, BODY) for _, b in files for l in b])
    wr = max([_width(fig, ax, t, TITLE, True) for t, _ in right]
             + [_width(fig, ax, l, BODY) for _, b in right for l in b])
    wg = max(_width(fig, ax, a, ARROW) for a in arrows)
    W, RW, GAP = wl + 2 * PAD, wr + 2 * PAD, wg + 0.5
    slack = 11.4 - (W + RW + GAP)
    assert slack > 0, f"the three regions need {W + RW + GAP:.2f} of 11.4 units"
    W += slack * 0.4
    RW += slack * 0.3
    GAP += slack * 0.3
    X, RX = 0.3, 0.3 + W + GAP

    heights = [0.42 + LINE * len(b) + 0.12 for _, b in files]
    tops, y = [], 8.25
    for h in heights:
        tops.append(y)
        y -= h + 0.55
    claims: list[tuple] = []

    def box(x, w, top, h, title, lines, fc):
        ax.add_patch(FancyBboxPatch((x, top - h), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=fc, ec="black", lw=0.8))
        claims.append((ax.text(x + w / 2, top - 0.14, title, ha="center", va="top",
                               fontsize=TITLE, fontweight="bold"), x, x + w, "box"))
        for i, line in enumerate(lines):
            claims.append((ax.text(x + w / 2, top - 0.46 - i * LINE, line, ha="center", va="top",
                                   fontsize=BODY, style="italic" if line.startswith("within") else "normal"),
                           x, x + w, "box"))

    for (title, lines), top, h in zip(files, tops, heights):
        box(X, W, top, h, title, lines, "#f4f4f4")
    for i, label in enumerate(between):
        bottom, nxt = tops[i] - heights[i], tops[i + 1]
        ax.add_patch(FancyArrowPatch((X + 1.2, bottom), (X + 1.2, nxt), arrowstyle="<|-|>",
                                     mutation_scale=7, lw=0.8, color="black", shrinkA=1, shrinkB=1))
        claims.append((ax.text(X + 1.45, (bottom + nxt) / 2, label, ha="left", va="center",
                               fontsize=5.6), X, X + W, "between the boxes"))

    for j, (title, lines) in enumerate(right):
        i = j + 2
        h = 0.42 + LINE * len(lines) + 0.12
        top = tops[i] - (heights[i] - h) / 2
        box(RX, RW, top, h, title, lines, "white")
        mid = top - h / 2
        ax.add_patch(FancyArrowPatch((X + W, mid), (RX, mid),
                                     arrowstyle="-|>" if j == 0 else "<|-|>", mutation_scale=7,
                                     lw=0.8, color="black",
                                     linestyle=":" if j == 0 else "-", shrinkA=1, shrinkB=1))
        claims.append((ax.text((X + W + RX) / 2, mid + 0.10, arrows[j], ha="center", va="bottom",
                               fontsize=ARROW), X + W, RX, "the gap"))
    ax.set_ylim(min(t - h for t, h in zip(tops, heights)) - 0.25, 8.45)
    _check_fits(fig, ax, claims)
    fig.savefig(FIG / "fig_record_relations.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_record_relations.png", bbox_inches="tight")
    plt.close(fig)


STAGES = ["cli", "constructor", "pre-submit", "service", "submitted"]
CONFIGS = ["none", "cap", "structural", "both"]
LABELS = {
    "qpu_optimizer": "QPU as optimizer\n(50 steps)",
    "trainable_sampling_sv1": "trainable angles\nat sampling, SV1",
    "trainable_sampling_qpu": "trainable angles\nat sampling, IonQ",
    "qpu_sampling_1000_shots": "QPU sampling,\ndefault 1,000 shots",
    "untagged_cloud_run": "untagged run over\nthe baseline",
}


def fig_guard_stages() -> None:
    """Where each misconfiguration stops under the four guard configurations.

    Three channels, none of them colour: the marker's *shape* is the guard
    configuration, its *fill* is the outcome — open for nothing wasted, grey
    for a correct call with a costly default, solid for wasted spend or a lost
    baseline — and a dagger or double dagger beside it says where the cell came
    from. A reader printing this in black and white, or one who does not
    separate red from green, sees exactly what the screen shows.
    """
    with open(RESULTS / "paper02_es2_guard_stages.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["n"] in ("8",)]
    fig, ax = plt.subplots(figsize=(TEXT_IN, 2.7))
    order = list(LABELS)
    FILL = {"none wasted": "white", "costly": "#9e9e9e", "wasted": "#1a1a1a"}
    for yi, m in enumerate(order):
        for ci, conf in enumerate(CONFIGS):
            r = next(r for r in rows if r["misconfiguration"] == m and r["configuration"] == conf)
            x = STAGES.index(r["stage"])
            y = yi + (ci - 1.5) * 0.20
            billed = float(r["usd_billed"])
            overwritten = r["baseline_overwritten"] == "yes"
            intended_one_task = m.startswith("trainable_sampling") and r["tasks_billed"] == "1"
            if (billed == 0 and not overwritten) or intended_one_task:
                fill = FILL["none wasted"]
            elif m == "qpu_sampling_1000_shots" and r["stage"] == "submitted":
                fill = FILL["costly"]
            else:
                fill = FILL["wasted"]
            marker = {"none": "s", "cap": "D", "structural": "o", "both": "^"}[conf]
            ax.plot(x, y, marker=marker, ms=5, mfc=fill, mec="black", mew=0.7, ls="none")
            # Third channel, in text rather than in the marker: where the cell
            # comes from. A dagger is the cap acting inside the service, which
            # the offline device never let us observe; a double dagger is a cell
            # the cloud observation of Section 5.2 covers.
            prov = ("\u2020" if r["stage"] == "service" else
                    "\u2021" if m == "trainable_sampling_sv1" else "")
            if prov:
                ax.text(x - 0.10, y, prov, fontsize=6.0, va="center", ha="right", color="black")
            note = ""
            if overwritten:
                note = f"${billed:.2f}, baseline replaced"
            elif r["stage"] == "submitted" and billed > 0:
                note = f"${billed:,.2f}" if billed < 100 else f"${billed:,.0f}"
                if r["tasks_would_submit"] not in ("1",):
                    note += f", {int(r['tasks_would_submit']):,} t"
            elif r["stage"] == "service":
                note = f"{int(r['tasks_billed'])} t, ${billed:.2f}, no result" if int(r["tasks_billed"]) else "0 t, $0"
            elif r["stage"] == "pre-submit" and billed > 0:
                note = f"${billed:.2f}, the intended task"
            if note:
                ax.text(x + 0.13, y, note, fontsize=5.8, va="center", ha="left", color="black")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS[m] for m in order], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels(["CLI\n(refused)", "constructor\n(refused)", "pre-submit\n(fixed)",
                        "service\n(cap rejects)", "submitted\n(billed)"], fontsize=6.8)
    ax.set_xlim(-0.4, 6.0)
    for x in range(len(STAGES)):
        ax.axvline(x, color="#dddddd", lw=0.6, zorder=0)
    from matplotlib.lines import Line2D
    shapes = [Line2D([], [], marker=mk, color="w", mec="black", mew=0.7, mfc="#dcdcdc", ms=5, label=lab)
              for mk, lab in (("s", "no guard"), ("D", "$50 cap only"),
                              ("o", "structural guards"), ("^", "both"))]
    fills = [Line2D([], [], marker="o", color="w", mec="black", mew=0.7, mfc=FILL[k], ms=5, label=lab)
             for k, lab in (("none wasted", "nothing wasted"),
                            ("costly", "correct call, costly default"),
                            ("wasted", "wasted spend or file lost"))]
    first = ax.legend(handles=shapes, title="shape: which guards are in place", loc="upper left",
                      bbox_to_anchor=(0.0, -0.30), ncol=2, fontsize=6.0, title_fontsize=6.0,
                      frameon=False, alignment="left")
    ax.add_artist(first)
    ax.legend(handles=fills, title="fill: what the run cost", loc="upper right",
              bbox_to_anchor=(1.0, -0.30), ncol=1, fontsize=6.0, title_fontsize=6.0,
              frameon=False, alignment="left")
    fig.savefig(FIG / "fig_guard_stages.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_guard_stages.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIG.mkdir(exist_ok=True)
    fig_record_relations()
    fig_guard_stages()
    print(f"wrote {sorted(p.name for p in FIG.iterdir())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
