# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Scan the draft for claims worded more strongly than the experiments support.

What it catches is language that runs half a step ahead of the evidence:
"single cause" where the experiment is confounded, "the field" where the
measurement covers fourteen papers, "the honest reading" where the paper is
grading itself. The patterns are calibrated on this paper's vocabulary: another
paper would need its own ALLOW lists before the output means anything.

So this looks for the patterns rather than the instances. It is a lint, not a
judge: several matches below are legitimate, which is why each pattern carries
an ALLOW list of contexts that have already been considered. A match that is not
allowed is a sentence to re-read, not necessarily a sentence to change.

Usage:
    python papers/check_claims.py            # scan, exit 1 if anything is new
    python papers/check_claims.py --verbose  # show allowed matches too
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

DRAFT = pathlib.Path(__file__).parent / "paper01_draft.md"
SUPPLEMENT = pathlib.Path(__file__).parent / "paper01_supplement.md"

#: Each pattern is a way a claim can outrun its evidence, with the reason it
#: matters and the phrasings already judged acceptable.
PATTERNS = [
    (
        "evaluative",
        r"\b(honest|striking|remarkable|damning|dramatic|obvious(?:ly)?|clearly)\b",
        "The paper does not get to grade itself; the reader decides. "
        "Replace with what was measured.",
        [],
    ),
    (
        "unscoped-field",
        r"\bthe field(?:'s)?\b",
        "Every count in this paper is over the fourteen benchmarks the protocol "
        "identifies. Attach the claim to that population.",
        ["assuming the field naive"],
    ),
    (
        "causal-uniqueness",
        r"\b(the single cause|the sole|the unique|is caused solely|the only explanation)\b",
        "The cross-backend comparison is confounded by precision and "
        "implementation; ruling out one channel does not make another unique.",
        [],
    ),
    (
        "unbounded-superlative",
        r"\b(the (?:least|most) [a-z]+)\b",
        "Bound it to what was compared: three environments, fourteen papers, "
        "the sizes we ran.",
        ["we tested", "of the three", "in the population", "we ran", "of the four"],
    ),
    (
        "absolute-absence",
        r"\b(no (?:benchmark|study|paper|prior work)|nobody has|none of the literature)\b",
        "We cannot prove complete coverage. Say 'we found no' or name the "
        "population.",
        ["no benchmark postdating", "no benchmark published after",
         "no benchmark in the population", "we found no", "found no study"],
    ),
    (
        "unqualified-QAOA",
        r"\bQAOA (?:fails|cannot|is unable|does not work)\b",
        "We measured one configuration -- p = 2, 50 ADAM steps, random start, "
        "penalty mixer -- not the algorithm.",
        [],
    ),
    (
        "independent-instances",
        r"\b(independent instances|independently drawn instances)\b",
        "The instances are asset subsets of one 50-ticker universe under one "
        "price window. They measure subset sensitivity.",
        [],
    ),
    (
        "withdrawn-round-3",
        r"(\bis uniform\b|\bexactly uniform\b|\bnot converged\b|1,600 batches|"
        r"headline result is negative|Numbered by order of first appearance|"
        r"number of tapes the gradient transform returns|infeasible for any statevector|"
        r"\bwill be wrong\b|percentage points apart|without touching its solver|"
        r"not that the output rule returns one|Bound at S = 1,000)",
        "Phrasings withdrawn as unsupported: near-uniform, not uniform; the "
        "coupling bound is on the selected-output distribution, not a "
        "disagreement count; 'did not improve within the budget', not "
        "'not converged'; 176 gradient tapes + 1 forward = 177 executions.",
        ["is exactly uniform"],   # the penalty-only symmetry statement, which is exact
    ),
    (
        "withdrawn-round-4",
        r"(three- to five-fold|three- to fivefold|\bsorts within F\b|does not add one|"
        r"\bis not enumerable\b|p = 2 throughout|only at the margin|"
        r"Every finding of this section|fallback billed|so the scale never enters|"
        r"independently across instances and sizes|against what the proposals ask for)",
        "Further phrasings withdrawn: Q_τ(1000) levels, not a P_τ multiple; "
        "'enrichment at N ≤ 16, median R_τ near one at N = 20', not 'sorts within F'; "
        "the 177 tasks are an observation consistent with parameter shift, not a "
        "determined cause; N = 50 'was not enumerated here'; p = 2 for E1–E5.",
        [],
    ),
    (
        "unfilled-marker",
        r"\[\[[^\]]*\]\]",
        "A [[...]] marker is a number or identifier still to be filled in "
        "(release tag, DOI, a GPU-host re-run). The paper is not done while one remains.",
        [],
    ),
]


#: IEEE asks for 250 words or fewer; arXiv's abstract field takes 1,920 characters.
ABSTRACT_WORD_LIMIT, ABSTRACT_CHAR_LIMIT = 250, 1920

#: A figure wider than the 7.16 in text block is a screen render scaled down.
FIGURE_WIDTH_LIMIT_IN = 7.5


def abstract(text: str) -> str:
    """The abstract's words, from the body (which starts right after the heading)."""
    return text.split("**Keywords:**", 1)[0].strip()


def figure_widths() -> list[str]:
    """Every PDF the .tex will include must be a print-size render."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "md_to_latex", pathlib.Path(__file__).parent / "md_to_latex.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = []
    for pdf in sorted((pathlib.Path(__file__).parent / "figures").glob("*.pdf")):
        w = mod.figure_width_in(pdf)
        if w > FIGURE_WIDTH_LIMIT_IN:
            out.append(f"{pdf.name}: {w:.1f} in wide (screen render; regenerate with PAPER_FIGURES=1)")
    return out


def body(text: str) -> str:
    """The manuscript proper: no version log, no bibliography.

    The version log records what earlier drafts claimed, including claims since
    withdrawn, so scanning it would report defects that are already fixed.
    """
    after_front = text.split("## Abstract", 1)[1]
    return after_front.split("## References")[0]


def back_matter_markers(text: str) -> list[str]:
    """Unfilled [[...]] markers after the bibliography, with their line numbers.

    The identifiers that are filled in last — the release tag and the archive
    DOI — live in the availability statement, which sits after "## References"
    and so outside what `body()` scans. A marker there once rode through a
    whole verification pass unreported, which is the one place it must not.
    """
    tail = text.split("## References", 1)[-1]
    offset = text.count("\n", 0, text.index("## References")) if "## References" in text else 0
    out = []
    for i, line in enumerate(tail.split("\n"), start=offset + 1):
        for m in re.finditer(r"\[\[[^\]]*\]\]", line):
            out.append(f"{DRAFT.name}:{i}: {m.group(0)} — {line.strip()[:90]}")
    return out


#: British forms, and the American ones IEEE sets. The manuscript held both --
#: 93 of one against 100 of the other -- in sentences that also carry the fixed
#: names "quadratic unconstrained binary optimization" and "Quantum Approximate
#: Optimization Algorithm", so the two spellings met inside single paragraphs.
#: The stem forms cover every ending (optimise/optimised/optimisation/...).
BRITISH = [r"optimis[aei]", r"normalis[aei]", r"minimis[aei]", r"maximis[aei]",
           r"summaris[aei]", r"generalis[aei]", r"parameteris[aei]",
           r"artefact", r"behaviour", r"judgement", r"favour[^i]",
           r"analys(?:ed|ing)", r"\bgrey\b", r"\bper cent\b", r"modelling",
           r"labelled", r"colour", r"\bcentre\b", r"licence", r"defence",
           r"whilst", r"amongst"]

#: The -ise family is open-ended: "initialisation", "penalised", "factorises",
#: "characterisation" and "annualised" all slipped through a list of stems and
#: were found by a later sweep. So the rule is generic, with the words that are
#: spelled -ise in both dialects listed out.
ISE = re.compile(r"\b\w+is(?:e|ed|es|ing|ation|ations)\b", re.I)
NOT_BRITISH = {
    "rise", "rises", "rising", "raise", "raises", "raised", "noise", "noises",
    "otherwise", "bitwise", "likewise", "wise", "precise", "concise", "promise",
    "promises", "premise", "premises", "comprise", "comprises", "arise", "arises",
    "surprise", "surprises", "surprising", "expertise", "exercise", "exercises",
    "exercised", "revise", "revises", "revised", "revising", "advise", "advises",
    "advised", "devise", "devised", "supervised", "compromise", "compromises",
    "improvise", "paradise", "franchise", "enterprise", "merchandise", "demise",
    "disguise", "guise", "cruise", "bruise", "poise", "praise", "treatise",
    "excise", "incise", "anise", "reprise", "chastise", "televise", "mortise",
}


def citation_order() -> list[str]:
    """Are the references numbered in the order they are first cited?

    IEEE numbers them that way, and the revisions had appended new entries to
    the end of the list instead: [9] appeared before [8], and the Dicke and XY
    references stood at 45 and 46 while first appearing in Sec. I. 42 of the 49
    moved when it was put right, which is the kind of edit that wants a guard
    behind it. A range cites every number in it.
    """
    text = DRAFT.read_text(encoding="utf-8")
    body = text.split("## Abstract", 1)[1].split("## References")[0]
    order, seen = [], set()
    for m in re.finditer(r"\[(\d+)\]\s*[-\u2013]\s*\[(\d+)\]|\[(\d+)\]", body):
        nums = ([int(m.group(3))] if m.group(3)
                else list(range(int(m.group(1)), int(m.group(2)) + 1)))
        for n in nums:
            if n not in seen:
                seen.add(n)
                order.append(n)
    return [f"[{n}] is cited {i + 1}{'st' if i == 0 else 'th'} but numbered {n}"
            for i, n in enumerate(order) if n != i + 1][:8]


def float_order() -> list[str]:
    """Figures and tables numbered in the order they are first cited.

    IEEE numbers them that way. Table V and Table II were first cited in the
    contributions of Sec. I, which made the first-citation order V, II, I, III,
    IV; the two forward references now point at the sections that hold them.
    """
    text = DRAFT.read_text(encoding="utf-8")
    body = text.split("## Abstract", 1)[1].split("## References")[0]
    out = []
    figs = list(dict.fromkeys(int(n) for n in re.findall(r"Fig\. (\d+)", body)))
    if figs != sorted(figs):
        out.append(f"figures are first cited in the order {figs}")
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    tabs = list(dict.fromkeys(re.findall(r"Table ([IVX]+)\b", body)))
    if tabs != roman[:len(tabs)]:
        out.append(f"tables are first cited in the order {tabs}")
    return out


def british_spellings() -> list[str]:
    """Where the manuscript slips into British spelling. IEEE sets American.

    A file name or a CSV column keeps whatever spelling it was written with, so
    only prose is scanned: code spans come out first.
    """
    out = []
    for src in (DRAFT, SUPPLEMENT):
        if not src.exists():
            continue
        for i, line in enumerate(src.read_text(encoding="utf-8").split("\n"), 1):
            prose = re.sub(r"`[^`]*`", "", line)
            for pat in BRITISH:
                for m in re.finditer(pat, prose, flags=re.I):
                    out.append(f"{src.name}:{i}: {m.group(0).strip()} — {line.strip()[:80]}")
            for m in ISE.finditer(prose):
                if m.group(0).lower() not in NOT_BRITISH:
                    out.append(f"{src.name}:{i}: {m.group(0)} — {line.strip()[:80]}")
    return out


def scan(text: str) -> list[tuple[str, str, str]]:
    """Return (pattern name, matched text, surrounding sentence) for each hit."""
    found = []
    for name, pattern, _why, allow in PATTERNS:
        for m in re.finditer(pattern, text, re.I):
            window = re.sub(r"\s+", " ", text[max(0, m.start() - 160):m.end() + 160])
            if any(a.lower() in window.lower() for a in allow):
                continue
            found.append((name, m.group(0), window.strip()))
    return found


def arithmetic(text: str) -> list[str]:
    """Every "a = b x c" written in the prose or a table must actually hold.

    Three of these were found at once, all from the same cause: a decomposition
    that is exact pair by pair was summarised by taking the median of each of
    its three factors, and three medians do not multiply. The claim reads as an
    identity, so it is checked as one. The tolerance is 5 %, which is wider than
    any rounding of the displayed digits and narrower than the errors this
    catches (26.3 against 0.74 x 39.5 = 29.2).
    """
    out = []
    for n, line in enumerate(text.split("\n"), 1):
        for m in re.finditer(r"(\d[\d.]*)\s*=\s*(\d[\d.]*)\s*(?:×|\\times)\s*(\d[\d.]*)", line):
            a, b, c = (float(x.rstrip(".")) for x in m.groups())
            if abs(a - b * c) > 0.05 * max(abs(a), 1e-12):
                out.append(f"line {n}: {m.group(0)} — the right side is {b * c:.4g}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    text = body(DRAFT.read_text(encoding="utf-8"))
    problems = 0
    ab = abstract(text)
    # Every whitespace-separated token counts, as a word processor counts:
    # "×" and "≤" are words to the editorial office's counter, and the limit
    # is theirs. A count that excluded them read 247 where a word processor
    # reads 264.
    n_words, n_chars = len(ab.split()), len(ab)
    if n_words > ABSTRACT_WORD_LIMIT or n_chars > ABSTRACT_CHAR_LIMIT:
        print(f"{DRAFT.name}: abstract is {n_words} words / {n_chars} characters "
              f"(limits {ABSTRACT_WORD_LIMIT} / {ABSTRACT_CHAR_LIMIT})\n")
        problems += 1
    wide = figure_widths()
    if wide:
        print(f"{DRAFT.name}: {len(wide)} figure(s) rendered for the screen, not the page\n")
        for line in wide:
            print("   " + line)
        print()
        problems += 1
    floats = float_order()
    if floats:
        print("floats are not numbered in order of first citation\n")
        for line in floats:
            print("   " + line)
        print()
        problems += 1
    misordered = citation_order()
    if misordered:
        print(f"{len(misordered)} reference(s) are not numbered in order of first citation\n")
        for line in misordered:
            print("   " + line)
        print()
        problems += 1
    spellings = british_spellings()
    if spellings:
        print(f"{len(spellings)} British spelling(s); IEEE sets American\n")
        for line in spellings[:12]:
            print("   " + line)
        print()
        problems += 1
    sums = arithmetic(text)
    if sums:
        print(f"{DRAFT.name}: {len(sums)} stated identity/identities that do not hold\n")
        for line in sums:
            print("   " + line)
        print()
    full = DRAFT.read_text(encoding="utf-8")
    markers = back_matter_markers(full)
    if markers:
        print(f"{DRAFT.name}: {len(markers)} unfilled marker(s) in the back matter\n")
        for line in markers:
            print("   " + line)
        print()
        problems += 1
    hits = scan(text)
    why = {name: reason for name, _p, reason, _a in PATTERNS}

    if not hits:
        if sums or problems:
            return 1
        print(f"{DRAFT.name}: no unscoped claims found "
              f"({len(PATTERNS)} patterns over {len(text.split())} words), "
              f"every stated identity holds, the abstract is {n_words} words / "
              f"{n_chars} characters, and every figure is a print render.")
        return 0

    print(f"{DRAFT.name}: {len(hits)} sentence(s) to re-read\n")
    for name in dict.fromkeys(h[0] for h in hits):
        group = [h for h in hits if h[0] == name]
        print(f"── {name}: {len(group)}")
        print(f"   {why[name]}")
        for _n, word, window in group[:6] if not args.verbose else group:
            print(f"   [{word}] …{window[:150]}…")
        if not args.verbose and len(group) > 6:
            print(f"   … and {len(group) - 6} more")
        print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
