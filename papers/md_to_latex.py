# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Convert papers/paper01_draft.md to an IEEEtran LaTeX source.

Why a bespoke converter rather than pandoc: the draft uses a small, known set of
Markdown constructs (h2/h3 headings, pipe tables, a handful of block-quoted
display formulas, inline bold/italic/code) and a specific set of Unicode symbols
that need deliberate LaTeX spellings rather than a generic mapping. A converter
we control is auditable, and its output is stable across runs — which matters
because this repository's whole argument is that artifacts should regenerate
deterministically.

Three things are read from the draft rather than kept here, so the .tex cannot
lag the prose: the title (the first h1), the figure list (the "Figure manifest"
section, which becomes float environments and is not typeset as text) and the
availability statement (typeset as an unnumbered section before the references).

`validate_latex.py` runs the structural checks a compiler would catch first;
the GPU host also has pdflatex, so the build is checked for real:

    python papers/md_to_latex.py && python papers/validate_latex.py
    (cd papers && pdflatex paper01 && pdflatex paper01)
"""

from __future__ import annotations

import pathlib
import re
import sys

SRC = pathlib.Path(__file__).parent / "paper01_draft.md"
OUT = pathlib.Path(__file__).parent / "paper01.tex"

#: The supplement is converted by the same code with `--supplement`. What
#: differs is which cross-references are local: the main text owns the Roman
#: table numbers and the Arabic figure numbers, the supplement owns the
#: S-prefixed ones, and a reference to the other document is set as text
#: rather than as a \ref that could never resolve.
SUPPLEMENT = False
#: ``--tqe`` sets the main text in the journal's own class (``ieeeaccess.cls``,
#: the TQE template of 2020) instead of generic IEEEtran: the same body, a
#: different front matter, the class's ``\Figure`` macro for floats and its
#: ``\EOD`` mark. The IEEEtran build stays the preprint; this one is what goes
#: to the Author Portal. The class and the three images it loads by name live
#: in ``papers/tqe_class/`` and are put on TEXINPUTS at build time.
TQE = False
LOCAL_TAB = r"[IVXL]+"          # table numerals this document defines
LOCAL_FIG = r"\d+"              # figure numbers this document defines
OTHER_TAB = r"S\d+"
OTHER_FIG = r"S\d+"


def set_mode(supplement: bool, tqe: bool = False) -> None:
    global SUPPLEMENT, TQE, SRC, OUT, LOCAL_TAB, LOCAL_FIG, OTHER_TAB, OTHER_FIG
    SUPPLEMENT = supplement
    TQE = tqe and not supplement
    OUT = pathlib.Path(__file__).parent / ("paper01_tqe.tex" if TQE else "paper01.tex")
    if supplement:
        SRC = pathlib.Path(__file__).parent / "paper01_supplement.md"
        OUT = pathlib.Path(__file__).parent / "paper01_supplement.tex"
        LOCAL_TAB, OTHER_TAB = r"S\d+", r"[IVXL]+"
        LOCAL_FIG, OTHER_FIG = r"S\d+", r"\d+"

#: Unicode the draft uses, and how it should be set in LaTeX. Spelled out rather
#: than passed to a generic Unicode package so the typeset result is a decision
#: rather than a default: en dashes stay dashes, minus signs become math minus,
#: and the bra-ket brackets get the macros they deserve.
UNICODE = {
    "—": "---", "–": "--", "−": "$-$", "×": r"$\times$", "·": r"$\cdot$",
    "†": r"$\dagger$", "Θ": r"$\Theta$",
    "≤": r"$\leq$", "≥": r"$\geq$", "≈": r"$\approx$", "→": r"$\rightarrow$",
    "∈": r"$\in$", "²": "$^2$", "³": "$^3$", "⟨": r"$\langle$", "⟩": r"$\rangle$",
    "Σ": r"$\Sigma$", "λ": r"$\lambda$", "μ": r"$\mu$", "α": r"$\alpha$",
    "β": r"$\beta$", "γ": r"$\gamma$", "ε": r"$\epsilon$", "δ": r"$\delta$",
    "ρ": r"$\rho$", "σ": r"$\sigma$", "φ": r"$\phi$", "ψ": r"$\psi$",
    "ᵀ": r"$^{\mathsf{T}}$", "ᵢ": "$_i$", "…": r"\ldots{}", "§": r"Sec.~",
    "τ": r"$\tau$", "±": r"$\pm$", "∇": r"$\nabla$", "∂": r"$\partial$", "⊥": r"$\perp$", "∪": r"$\cup$", "∏": r"$\prod$", "ℓ": r"$\ell$", "⊗": r"$\otimes$", "∞": r"$\infty$", "₁": "$_1$", "₂": "$_2$",
    "✓": r"\checkmark", "✗": r"$\times$", "⌊": r"$\lfloor$", "⌋": r"$\rfloor$",
    # U+0302 is handled in inline() before this map runs: the hat marks the
    # normalized quantities of Sec. III and is not decoration.
    "≫": r"$\gg$", "≠": r"$\neq$", "ℝ": r"$\mathbb{R}$", "Δ": r"$\Delta$",
    "π": r"$\pi$", "①": "(1)", "②": "(2)", "③": "(3)",
    "½": r"$\tfrac{1}{2}$", "′": r"$'$",
    "ú": r"\'{u}", "ó": r"\'{o}", "é": r"\'{e}", "á": r"\'{a}", "í": r"\'{i}",
    "ü": r'\"{u}', "ö": r'\"{o}', "ä": r'\"{a}', "ñ": r"\~{n}", "ç": r"\c{c}",
    "'": "'", "'": "'", "\u201c": "``", "\u201d": "''",
}

#: Characters LaTeX reserves. Applied only to running text, never inside math or
#: verbatim, both of which are extracted first and restored afterwards.
ESCAPES = {"%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_", "$": r"\$"}

PREAMBLE = r"""\documentclass[journal]{IEEEtran}

% ToUnicode maps for every glyph, so the PDF's text layer carries the angle
% brackets, ell, epsilon and "not equal" instead of what the font's encoding
% slot happens to be called. Without this, pdfTeX 1.40.20 (TeX Live 2019)
% extracted the bra-ket of H as "hHi" and not-equal as "6="; 1.40.26 did so
% on its own. With it the
% two builds' text layers are identical, and the PDF can be built anywhere.
\input{glyphtounicode}
\pdfgentounicode=1

% T1 rather than the OT1 default: OT1 has no > or < glyph and sets them as
% inverted question marks, which is how "A > x" reached the page wrong.
% It also lets TeX hyphenate the accented names in the bibliography.
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\usepackage{amssymb}

% The draft's tables are wide; IEEEtran's two-column measure needs the small
% face and the ability to break a long table across columns.
\usepackage{array}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}

\begin{document}

\title{@TITLE@}

\author{Hiroaki~Tobita%
\thanks{H. Tobita is with the Advanced Institute of Industrial Technology,
Tokyo, Japan (e-mail: tobita-hiroaki@aiit.ac.jp).}}

\maketitle

"""

POSTAMBLE = r"""
\end{document}
"""

#: The TQE template's front matter, in the template's order: the history and
#: DOI lines are the journal's placeholders (filled in by IEEE at production),
#: the abstract and keywords come *before* \maketitle, which the keyword
#: block therefore emits. hyperref stays (the class loads it only in its
#: JTEHM branch); ``cite`` is what the template loads for citations.
PREAMBLE_TQE = r"""\documentclass{ieeeaccess}

\input{glyphtounicode}
\pdfgentounicode=1

\usepackage[T1]{fontenc}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{url}
\usepackage[hidelinks]{hyperref}

\usepackage{array}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}

% The class sets a page-wide figure's caption in a one-column tabular, which
% never breaks lines: a caption longer than one line ran 1,157 pt off the page
% and was cut at the margin (measured on Fig. 1). This is the class's
% \@makecaption with that one branch set as a paragraph of the figure's width;
% the table branch and the column-width figure branch are the class's own.
\makeatletter
\long\def\@makecaption#1#2{%
\ifx\@captype\@IEEEtablestring%
\begin{flushleft}
\vspace*{5pt}
{\vss\color{accessblue}\tablecapheadfont #1. \ }{\raggedright\tablecapfont#2\vss}%
\end{flushleft}
\@IEEEtablecaptionsepspace%
\else
\@IEEEfigurecaptionsepspace%
\setbox\@tempboxa\hbox{{\color{accessblue}\figcapheadfont #1. \ }}%
\ifdim \xfigwd >\columnwidth%
\parbox[t]{\linewidth}{\raggedright\noindent\unhbox\@tempboxa\figcapfont#2}%
\else%
{\vss\raggedright\noindent\unhbox\@tempboxa\figcapfont#2\vss}%
\fi\fi\vskip 1pt plus 1pt minus 1pt}
\makeatother

\begin{document}
\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}
\doi{10.1109/TQE.2020.DOI}

\title{@TITLE@}
\author{\uppercase{Hiroaki Tobita}\authorrefmark{1}}
\address[1]{Advanced Institute of Industrial Technology, Tokyo 140-0011, Japan (e-mail: tobita-hiroaki@aiit.ac.jp)}

\markboth
{Tobita: @SHORTTITLE@}
{Tobita: @SHORTTITLE@}

\corresp{Corresponding author: Hiroaki Tobita (email: tobita-hiroaki@aiit.ac.jp).}

"""

POSTAMBLE_TQE = r"""
\EOD

\end{document}
"""


def protect(text: str) -> tuple[str, list[str]]:
    """Pull inline code out of the text so escaping cannot touch it."""
    held: list[str] = []

    def take(m: re.Match) -> str:
        held.append(m.group(1))
        return f"\x00{len(held) - 1}\x00"

    return re.sub(r"`([^`]+)`", take, text), held


def restore(text: str, held: list[str]) -> str:
    """Put the inline code back, as \\texttt with its own escaping."""
    def give(m: re.Match) -> str:
        body = held[int(m.group(1))]
        for ch, rep in ESCAPES.items():
            body = body.replace(ch, rep)
        body = body.replace("^", r"\^{}").replace("~", r"\~{}")
        # A URL set in \texttt cannot break and runs off the column; \url
        # breaks it at the slashes.
        if body.startswith(("http://", "https://")):
            return r"\url{" + body + "}"
        # Inline code is protected from the Unicode pass, so map here too:
        # inputenc refuses these inside \texttt exactly as it does in text.
        for ch, rep in UNICODE.items():
            body = body.replace(ch, rep)
        return r"\texttt{" + body + "}"

    return re.sub(r"\x00(\d+)\x00", give, text)


#: Greek the draft writes as Unicode but that must be a macro inside math.
#: U+E002 is "f" with a combining macron (U+0304): the feasible-set mean of
#: Sec. III-B, which pairs with a subscript and so has to be one math atom.
HATTED = {"\ue000": "\\hat{\\Sigma}", "\ue001": "\\hat{\\mu}", "\ue002": "\\bar{f}"}

GREEK_MATH = {"\ue000": "\\hat{\\Sigma}", "\ue001": "\\hat{\\mu}", "\ue002": "\\bar{f}",
              "Σ": "\\Sigma", "μ": "\\mu", "λ": "\\lambda",
              "γ": "\\gamma", "β": "\\beta", "α": "\\alpha",
              "σ": "\\sigma", "ρ": "\\rho", "φ": "\\phi",
              "ψ": "\\psi", "ε": "\\epsilon", "Δ": "\\Delta",
              "τ": "\\tau"}

#: Superscript digits and signs the draft writes as Unicode ("10⁻¹⁴"). A run is
#: one exponent, so it is mapped as a run: per-character mapping would set
#: "10⁻¹⁴" as three stacked superscripts.
SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
SUPERSCRIPT_RE = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+")

#: Subscript indices that are words rather than symbols, and want upright type.
WORD_SUBSCRIPTS = {"min", "max", "norm", "opt", "crit", "safe", "margin", "heur", "off", "obj"}

SUBSCRIPT_RE = re.compile(
    r"(?<![A-Za-z0-9_$])([A-Za-z" + "".join(GREEK_MATH) + r"])_"
    r"(\d+\.\d+|[A-Za-z0-9τ]{1,6})(?![\w]|\.\w)")


def _subscript(m):
    base = GREEK_MATH.get(m.group(1), m.group(1))
    idx = GREEK_MATH.get(m.group(2), m.group(2))
    # Upright, and always defined: "\\norm" is not a LaTeX command, and emitting
    # it produced "Undefined control sequence" four times in Sec. VI-B.
    idx = "\\mathrm{" + idx + "}" if idx in WORD_SUBSCRIPTS else idx
    return "\x05" + base + "\x06{" + idx + "}\x05"


#: Inline expressions the character-level passes cannot set: held aside as
#: raw LaTeX before anything else runs, keyed by their exact Markdown.
INLINE_MATH = {
    "|⟨H⟩_SV1 − ⟨H⟩_GPU|": r"$|\langle H\rangle_{\mathrm{SV1}} - \langle H\rangle_{\mathrm{GPU}}|$",
}


def inline(text: str) -> str:
    """Convert one line of running Markdown to LaTeX."""
    text, held = protect(text)
    raw: list[str] = []
    for src, tex in INLINE_MATH.items():
        if src in text:
            raw.append(tex)
            text = text.replace(src, f"\x04{len(raw) - 1}\x04")
    # Undo Markdown's own escaping before LaTeX's begins. \* and \| are the two
    # the draft uses; left in place they become a literal backslash in the output
    # and, in the case of \*, an unbalanced emphasis run.
    # Markdown escapes come off here, but "\\*" has to stay distinguishable from
    # a real emphasis marker until the emphasis pass is over: "obj\\*(K) - obj\\*(m)"
    # is one expression, and unescaping it early let the two asterisks pair into
    # "obj\\textit{(K) - obj}(m)".
    text = text.replace(r"\*", "\x07")
    text = re.sub(r"\\([|_])", r"\1", text)
    # Currency before anything else. The draft writes prices as $1.47, and both
    # the escape pass and the Unicode pass would otherwise leave a bare $ that
    # opens math mode and swallows the rest of the paragraph. Held aside as a
    # sentinel and restored as \$ at the end.
    text = re.sub(r"\$(?=[0-9])", "\x01", text)
    # A bare | is a vertical bar, most often an absolute value. Held as a
    # sentinel too: emitting "$|$" here would put real dollars into the string
    # that the escape pass below then turns into "\$|\$".
    # Subscripts, before the escape pass turns "_" into "\\_". "Q_ii" was set as
    # "Q ii": the underscore escaped and the letters left in text mode, so the
    # matrix entries, the Ising coefficients and A_min each read as two words
    # rather than one symbol. Held behind \x05 so the escape pass below does
    # not turn the dollars this produces into literal dollar signs.
    text = text.replace("Σ\u0302", "\ue000").replace("μ\u0302", "\ue001")
    text = text.replace("f\u0304", "\ue002")
    # Straight quotes are set as two identical marks; TeX wants ``...''. This
    # has to run before the Unicode map, which turns "ü" into \\"{u} and so
    # introduces a quote character of its own: pairing after that shifted every
    # quote in the Brandhofer entry by one and reversed all of them, and left
    # the umlaut as a backtick.
    text = re.sub(r'"([^"]*)"', lambda m: "``" + m.group(1) + "''", text)
    text = SUBSCRIPT_RE.sub(_subscript, text)
    for tok, mac in HATTED.items():
        text = text.replace(tok, "\x05" + mac + "\x05")
    text = text.replace("|", "\x02")
    for ch, rep in ESCAPES.items():
        text = text.replace(ch, rep)
    # "2^N" is a superscript, not a circumflex accent. Escaping it as \^{}
    # typeset "2ˆN" throughout, including in the abstract's first sentence.
    # Anything that looks like base^exponent goes to math; a lone ^ keeps the
    # accent escape.
    # Held behind a sentinel: the blanket escape below would otherwise turn the
    # caret this just produced back into an accent.
    def _sup(m):
        # Braced exponents can hold anything -- "N x N" -- so take the whole
        # group and set its operators in math rather than letting the Unicode
        # pass drop $...$ inside an exponent.
        body = m.group(1) if m.group(1) is not None else m.group(2)
        body = body.replace("\u00d7", r"\times ").replace("\u2212", "-")
        return "$\x03{" + body + "}$"
    text = re.sub(r"(?<![\\$])\^(?:\{([^{}]*)\}|([A-Za-z0-9.+-]+))", _sup, text)
    text = text.replace("^", r"\^{}").replace("~", r"\~{}")
    text = text.replace("\x03", "^").replace("\x05", "$").replace("\x06", "_")
    # A bare > or < in text mode is set as an inverted question mark by the
    # default font encoding: "A > x" printed as "A ¿ x".
    text = re.sub(r"(?<![$\\])([<>])(?![$])", r"$\1$", text)
    # Bare braces are LaTeX grouping and disappear silently: "{0,1}^N" printed
    # as "0,1^N". The set-builder braces the draft writes have to be escaped.
    text = re.sub(r"(?<![\\$])\{([01],[01])\}", r"$\\{\1\\}$", text)
    text = SUPERSCRIPT_RE.sub(lambda m: "$^{" + m.group(0).translate(SUPERSCRIPTS) + "}$", text)
    for ch, rep in UNICODE.items():
        text = text.replace(ch, rep)
    # IEEE closes up the percent sign, and sets computer exponentials as powers
    # of ten. Both are house style rather than correctness, but both are what a
    # copy editor would otherwise mark on every page.
    text = re.sub(r"(\d)\s+\\%", r"\1\\%", text)
    text = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)e([+-]?\d+)(?![\w])",
                  lambda m: "$" + m.group(1) + r"\times 10^{" + str(int(m.group(2))) + "}$",
                  text)
    # A backslash-escaped asterisk is a literal one -- "obj\\*(K)" is the optimal
    # objective at cardinality K, not the start of an emphasis span. Left
    # unprotected, the two in that expression paired and set half the formula in
    # italics: "obj\\textit{(K) - obj}(m)".
    # Bold before italic: ** is a superset of *.
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    # An asterisk touching a word character on the inside is part of that word,
    # not an emphasis delimiter: "E*", "f*" and "obj*(K)" are optimal values,
    # and letting them pair set whole formulae in italics --
    # "\\textit{$|$), which is the same $|$E}".
    text = re.sub(r"(?<![\w*])\*(?=\S)([^*]+?)(?<=\S)\*(?![\w*])",
                  r"\\textit{\1}", text)
    text = text.replace("\x07", "*")
    # Bracketed citation numbers become \cite; the draft's numbering is its own.
    # The supplement has no reference list of its own: a bracketed number
    # there points into the main text's, and stays as text.
    if not SUPPLEMENT:
        text = re.sub(r"\[(\d+)\]", r"\\cite{ref\1}", text)
    # Table and figure references become \ref. The draft numbers its own floats
    # and the running text uses those numbers, while LaTeX assigns its own in
    # float order; left as literal text the two silently disagree, and "see
    # Table V" lands on a different table. Ranges reference both endpoints.
    text = re.sub(rf"Tables ({LOCAL_TAB})(?:--|---|-)({LOCAL_TAB})\b",
                  r"Tables~\\ref{tab:\1}--\\ref{tab:\2}", text)
    text = re.sub(rf"Table ({LOCAL_TAB})\b", r"Table~\\ref{tab:\1}", text)
    text = re.sub(rf"Figs\. ({LOCAL_FIG})(?:--|---|-)({LOCAL_FIG})\b",
                  r"Figs.~\\ref{fig:\1}--\\ref{fig:\2}", text)
    text = re.sub(rf"Fig\. ({LOCAL_FIG})\b", r"Fig.~\\ref{fig:\1}", text)
    # References into the other document are text with a tie, not \ref.
    text = re.sub(rf"(Tables?|Figs?\.) ({OTHER_TAB})\b", r"\1~\2", text)
    text = text.replace("\x01", r"\$").replace("\x02", "$|$")
    text = re.sub(r"\x04(\d+)\x04", lambda m: raw[int(m.group(1))], text)
    return restore(text, held)


def convert_table(rows: list[str], caption: str, label: str) -> list[str]:
    """Render a Markdown pipe table as an IEEEtran table environment.

    The alignment row is dropped and its alignments honoured; every column is
    left-aligned except those Markdown marks right-aligned with a trailing colon,
    which is how the draft marks its numeric columns.
    """
    # Split on unescaped pipes only: the draft writes \|E*\| inside cells to mean
    # an absolute value, and splitting on those silently shreds the row.
    cells = [[c.strip() for c in re.split(r"(?<!\\)\|", r.strip().strip("|"))]
             for r in rows]
    if len(cells) < 2 or not re.fullmatch(r":?-{2,}:?", cells[1][0].replace(" ", "")):
        # No alignment row: a stray pipe line, not a table. Emit it as text so
        # nothing is silently dropped.
        return [inline(" ".join(" ".join(r) for r in cells)), ""]
    spec = "".join("r" if a.endswith(":") else "l" for a in cells[1])
    ncols = len(spec)          # the column *count*, fixed here; `spec` changes
    body = [cells[0]] + cells[2:]
    # IEEEtran's column is about 252 pt. Measured on the first compile, anything
    # past four columns overflows it -- the worst by 305 pt -- so wide tables get
    # the two-column float instead. `table*` can only be placed at the top of a
    # page, which is why the narrow ones stay single-column: they place better.
    # Column count is not the whole story: a two-column table whose cells hold
    # sentences overflows just as badly as an eight-column one, because `l`
    # columns do not wrap. Measured on the first compile, the two worst offenders
    # were 2-column. So decide on the widest row, and give long-celled tables
    # p{} columns that can break.
    widest = max(sum(len(c) for c in row) + 3 * ncols for row in body)
    long_cells = max(max((len(c) for c in row), default=0) for row in body) > 28
    wide = ncols >= 5 or widest > 95
    env = "table*" if wide else "table"
    if long_cells:
        share = [max(max(len(r[j]) for r in body if j < len(r)), 6)
                 for j in range(ncols)]
        unit = "textwidth" if wide else "columnwidth"
        # The p{} columns get what the natural-width columns leave. Sharing
        # the whole measure among the p{} columns alone, as the first version
        # did, ignored the l/r columns and the column separations, and eight
        # tables ran 10--116 pt past the margin.
        measure = 516.0 if wide else 252.0            # pt, IEEEtran journal
        char = 3.6 if wide else 4.0                   # pt per character, \scriptsize / \footnotesize
        # A header such as "Min-energy shot infeasible" over a numeric column
        # is the other way a table overflows, so in the two-column float a
        # shorter cell already earns a wrapping column.
        cut = 14 if wide else 20
        fixed = sum(share[j] * char for j in range(ncols) if share[j] <= cut) + 12.0 * ncols
        free = max(measure * 0.98 - fixed, 0.35 * measure)
        p_total = sum(share[j] for j in range(ncols) if share[j] > cut)
        spec = "".join(
            f"p{{{free * share[j] / p_total / measure:.3f}\\{unit}}}" if share[j] > cut else spec[j]
            for j in range(ncols))
    out = [
        rf"\begin{{{env}}}[!t]", r"\centering",
        r"\scriptsize" if wide else r"\footnotesize",
        # The caption arrives already converted: it is cut from a paragraph
        # that went through inline(), and the fallbacks below are LaTeX. A
        # second inline() pass escaped the math it had produced ("\$A\_{...}\$")
        # and the build died on the first caption with a subscript.
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
    ]
    out.append(" & ".join(inline(c) for c in body[0]) + r" \\")
    out.append(r"\midrule")
    for row in body[1:]:
        row = (row + [""] * ncols)[:ncols]
        out.append(" & ".join(inline(c) for c in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}", ""]
    return out


#: Figures, in the order the paper discusses them: (file stem, LaTeX caption),
#: filled from the draft's own figure manifest by `read_manifest()` so the
#: floats cannot lag the prose. The list below is the v1.x manifest and is what
#: the converter falls back to when the draft carries no manifest section.
FIGURES_V1 = [
    ("fig1_convergence", "QAOA expectation vs. ADAM step, one curve per $N$."),
    ("fig3_runtime_scaling", "Runtime vs. $N$, classical against QAOA (log $y$)."),
    ("fig2_optimality_gap", "QAOA optimality gap vs. $N$."),
    ("fig4_sharpe", "Sharpe ratio of the selected portfolio vs. $N$."),
    ("fig5_seed_variance", "Optimality gap across initial-angle seeds (symlog); "
                           "the infeasible run is marked separately."),
    ("fig6_depth_sensitivity", "Gap and runtime against QAOA depth $p$."),
    ("fig7_steps_depth_grid", "Steps $\\times$ $p$ grid: optimized "
                              "$\\langle H\\rangle$ (top) and reported "
                              "best-of-1000 gap (bottom). The rows disagree by "
                              "design."),
    ("fig8_cross_backend", "SV1 against lightning.gpu: gap per $N$ (left) and "
                           "the per-step distance between the two optimizer "
                           "trajectories (right, log $y$)."),
    ("fig9_degeneracy", "Sharpe ratios spanned by a 0.1\\% optimality gap "
                        "(left), and the same solutions under both denominators "
                        "(right). $N = 8$, 12 and 16 reached the exact optimum, "
                        "so their gap is zero under either denominator and they "
                        "do not appear on the right panel's log axis."),
    ("fig10_instances", "The two quantities of Fig.~9 over 30 drawn instances "
                        "per size, with the reference instance (0) marked."),
    ("fig11_gap_vs_sharpe", "Every sub-optimal QAOA solution in the instance "
                            "sweep (left), and the fraction of instances "
                            "reaching the exact optimum (right)."),
]


FIGURES: list[tuple[str, str]] = list(FIGURES_V1)

MANIFEST_RE = re.compile(
    r"^- \*\*Fig\. (S?\d+)\*\* — `notebooks/figures/([\w-]+)\.pdf` — (.*)$", re.S)


def read_manifest(md: str) -> tuple[str, list[tuple[str, str]], str]:
    """Split the back matter off the draft.

    Returns the draft without its "Figure manifest" and "Data/artifact
    availability" sections, the main-text figure list read from the manifest
    (stem, caption already converted to LaTeX) and the availability text. The
    manifest's supplementary figures (S1–S5) are not floats and stay out.
    """
    m = re.search(r"\n### Figure manifest\n(.*?)(?=\n### |\Z)", md, re.S)
    if not m:
        return md, [], ""
    figs: list[tuple[str, str]] = []
    numbers: list[str] = []
    for item in re.split(r"\n(?=- \*\*Fig\. )", m.group(1)):
        item = " ".join(item.strip().split("\n"))
        mm = MANIFEST_RE.match(item)
        if mm:
            figs.append((mm.group(2), inline(mm.group(3).strip())))
            numbers.append(mm.group(1))
    prefix = "S" if SUPPLEMENT else ""
    if numbers != [f"{prefix}{k}" for k in range(1, len(numbers) + 1)]:
        raise SystemExit(f"figure manifest is not numbered {prefix}1..n in order: {numbers}")
    md = md[:m.start()] + md[m.end():]
    a = re.search(r"\n### Data/artifact availability\n(.*?)(?=\n### |\Z)", md, re.S)
    avail = ""
    if a:
        avail = a.group(1).strip()
        md = md[:a.start()] + md[a.end():]
    return md, figs, avail


#: Figures rendered at the full text width rather than the column width. Read
#: off the PDFs rather than hard-coded, so regenerating at a different size
#: cannot leave this list stale.
#: Anything wider than this was rendered for the screen, not the page; scaling
#: it into the 7.16 in text block shrinks its labels below legibility.
MAX_FIGURE_WIDTH_IN = 7.5


def figure_width_in(path: pathlib.Path) -> float:
    """Page width of a PDF in inches, read off its MediaBox."""
    # The page width comes off the PDF's own MediaBox rather than out of
    # ``pdfinfo``: poppler is installed on the GPU machine and not on the Mac,
    # and a converter that raises on one of the two machines is a converter the
    # paper cannot be rebuilt from. Matplotlib writes the box uncompressed, and
    # width-in-points/72 is the same number pdfinfo prints.
    box = re.search(rb"/MediaBox\s*\[\s*([\d.+-]+)\s+[\d.+-]+\s+([\d.+-]+)\s",
                    path.read_bytes())
    if not box:
        raise SystemExit(f"no /MediaBox in {path}: cannot size the float")
    return (float(box.group(2)) - float(box.group(1))) / 72


def _is_wide(stem: str) -> bool:
    path = pathlib.Path(__file__).parent / "figures" / f"{stem}.pdf"
    if not path.exists():
        return False
    return figure_width_in(path) > 5.0


def figure_float(n: int) -> list[str]:
    """One figure float, spanning both columns when the artwork is wide.

    A 7.16 in figure scaled into a 3.5 in column shrinks its labels past
    legibility -- measured at 3.1 pt before the figures were re-rendered at
    their printed size. Wide artwork therefore goes in ``figure*``.
    """
    stem, caption = FIGURES[n - 1]
    path = pathlib.Path(__file__).parent / "figures" / f"{stem}.pdf"
    if path.exists() and figure_width_in(path) > MAX_FIGURE_WIDTH_IN:
        raise SystemExit(
            f"figures/{stem}.pdf is {figure_width_in(path):.1f} in wide: a screen render, "
            "not a print one. Regenerate with PAPER_FIGURES=1 (output goes to "
            "notebooks/figures/paper/) so its labels are not scaled down.")
    wide = _is_wide(stem)
    env = "figure*" if wide else "figure"
    width = r"\textwidth" if wide else r"\columnwidth"
    if TQE:
        # The class's caption code reads the width the \Figure macro records;
        # a plain figure environment leaves it undefined and the caption dies.
        # The label rides inside the caption argument so \ref still resolves.
        return [
            rf"\Figure[!t](topskip=0pt, botskip=0pt, midskip=0pt)[width={width}]"
            rf"{{figures/{stem}.pdf}}{{{caption}\label{{fig:{n}}}}}", "",
        ]
    return [
        rf"\begin{{{env}}}[!t]", r"\centering",
        rf"\includegraphics[width={width}]{{figures/{stem}.pdf}}",
        rf"\caption{{{caption}}}", rf"\label{{fig:{'S' if SUPPLEMENT else ''}{n}}}",
        rf"\end{{{env}}}", "",
    ]


def place_figures(out: list[str]) -> list[str]:
    """Insert each figure just after the paragraph that first references it.

    Emitting all eleven together before the bibliography made LaTeX flush them
    past the references, which is where a reader will not look for them. A float
    placed next to its first \ref competes only with its neighbours, so the
    placement algorithm has somewhere sensible to put it. Any figure never
    referenced falls back to the end, and says so by being there.
    """
    # Skip references made before the Results section. The contribution list in
    # the introduction forward-references almost every figure, and placing them
    # there drags five floats onto pages 2-3, pages ahead of the text that
    # explains them. Placement follows the *discussion*, not the announcement.
    body_starts = next(
        (k for k, line in enumerate(out) if r"\section{Results}" in line), 0)
    if SUPPLEMENT:
        body_starts = 0
    placed: set[int] = set()
    result: list[str] = []
    for k, line in enumerate(out):
        result.append(line)
        if k < body_starts:
            continue
        for n in range(1, len(FIGURES) + 1):
            if n in placed:
                continue
            if rf"\ref{{fig:{'S' if SUPPLEMENT else ''}{n}}}" in line:
                # LaTeX numbers floats in the order they appear, so a figure
                # whose first reference comes later than a higher-numbered
                # one would be numbered out of step with the manifest and
                # every "Fig. N" in the text would point one off. Any lower
                # unplaced figure therefore goes in first.
                for m in range(1, n + 1):
                    if m not in placed:
                        result += [""] + figure_float(m)
                        placed.add(m)
    for n in range(1, len(FIGURES) + 1):
        if n not in placed:
            result += figure_float(n)
    return result


#: The block quotes of Sec. III are display formulas. Setting them through the
#: inline pass gives text-mode fragments with math sprinkled in ("V(x) = ($\Sigma$$_i$
#: x$_i$ $-$ K)$^2$"), which is legible but not typeset. Each is hand-set here,
#: keyed by its Markdown so a rewritten formula loses its rendering rather than
#: keeping a stale one; a quote not in the table falls back to the inline pass.
FORMULAS = {
    "H_C = (1/s_H) · ( Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j ),   H_M = −Σ_i X_i,":
        r"H_C = \frac{1}{s_H}\Bigl(\sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j\Bigr),"
        r"\qquad H_M = -\sum_i X_i,",
    "|ψ(γ, β)⟩ = ∏_{ℓ=p..1} e^{−i β_ℓ H_M} e^{−i γ_ℓ H_C} · H^{⊗N}|0⟩^{⊗N},":
        r"|\psi(\gamma,\beta)\rangle = \prod_{\ell=p}^{1}"
        r" e^{-i\beta_\ell H_M} e^{-i\gamma_\ell H_C}\,"
        r" H^{\otimes N}|0\rangle^{\otimes N},",
    "minimize  λ·xᵀΣx − (1−λ)·μᵀx   subject to  Σᵢ xᵢ = K":
        r"\min\; \lambda\, x^{\mathsf{T}}\Sigma x - (1-\lambda)\, \mu^{\mathsf{T}}x"
        r"\quad \text{subject to}\quad \sum_i x_i = K",
    "f(x) = λ·xᵀΣ̂x − (1−λ)·μ̂ᵀx  (the portfolio objective),":
        r"f(x) = \lambda\, x^{\mathsf{T}}\hat{\Sigma} x - (1-\lambda)\, \hat{\mu}^{\mathsf{T}}x"
        r"\quad\text{(the portfolio objective),}",
    "V(x) = (Σᵢ xᵢ − K)²  (the constraint violation),":
        r"V(x) = \Bigl(\sum_i x_i - K\Bigr)^2\quad\text{(the constraint violation),}",
    "H_A(x) = f(x) + A·V(x)  (the penalised objective the QUBO encodes),":
        r"\begin{aligned} H_A(x) &= f(x) + A\, V(x) \\"
        r" &\qquad\text{(the penalised objective the QUBO encodes),}\end{aligned}",
    "E_A(x) = H_A(x) − A·K²  (what xᵀQx returns once the expansion's constant is dropped).":
        r"\begin{aligned} E_A(x) &= H_A(x) - A K^2 \\"
        r" &\qquad\text{(what $x^{\mathsf{T}}Qx$ returns once the expansion's}\\"
        r" &\qquad\text{constant is dropped).}\end{aligned}",
    "A_heur = 2·s_obj·N + 1,  s_obj = max(λ·max|Σ̂|, (1−λ)·max|μ̂|);":
        r"\begin{aligned} A_{\mathrm{heur}} &= 2 s_{\mathrm{obj}} N + 1,\\"
        r" s_{\mathrm{obj}} &= \max\bigl(\lambda \max|\hat{\Sigma}|,\, (1-\lambda)\max|\hat{\mu}|\bigr);\end{aligned}",
    "A_crit = max(0, max_{m≠K} (f\\* − a_m)/(m − K)²).":
        r"A_{\mathrm{crit}} = \max\Bigl(0,\; \max_{m \neq K} \frac{f^{*} - a_m}{(m-K)^2}\Bigr).",
    "T = (f\\* + f̄_F)/2,  A_margin = max(0, max_{m≠K} (T − a_m)/(m − K)²) + ε_A,":
        r"T = \frac{f^{*} + \bar{f}_F}{2},\quad "
        r"A_{\mathrm{margin}} = \max\Bigl(0,\; \max_{m \neq K} \frac{T - a_m}{(m-K)^2}\Bigr) + \epsilon_A,",
}


FORMULAS["D_cond = ½ Σ_{x∈F} | p(x)/P_F − 1/|F| |,"] = (
    r"D_{\mathrm{cond}} = \tfrac{1}{2}\sum_{x \in F}\Bigl|\frac{p(x)}{P_F} - \frac{1}{|F|}\Bigr|,")
FORMULAS["TV(selected output under P, selected output under P′) ≤ 1 − (1 − d)^S ≤ S·d."] = (
    r"\begin{aligned} &\mathrm{TV}\bigl(\text{selected output under } P,\ \text{selected output under } P'\bigr)\\"
    r" &\qquad \le 1 - (1-d)^S \le S\,d.\end{aligned}")


def _is_table_row(line: str) -> bool:
    """A Markdown table row: starts with a pipe and ends with one."""
    stripped = line.rstrip()
    return stripped.startswith("|") and stripped.endswith("|")


def _bold_caption(line: str) -> tuple[str, str] | None:
    """Split "\\textbf{Table N. Caption} prose" into (caption, prose).

    Returns None when the line does not open with a bold "Table N." run. The
    closing brace is found by counting, so braces inside the caption's math
    do not end it early.
    """
    m = re.match(r"\s*\\textbf\{(?=Table(?:~\\ref\{tab:S?[IVXL0-9]+\}|[\s~]S?[IVXL0-9]+)\.)", line)
    if not m:
        return None
    depth, k = 1, m.end()
    while k < len(line) and depth:
        depth += {"{": 1, "}": -1}.get(line[k], 0)
        k += 1
    if depth:
        return None
    return line[m.end():k - 1], line[k:]


def convert(md: str) -> str:
    """Convert the whole draft."""
    md, figs, avail = read_manifest(md)
    if figs:
        FIGURES[:] = figs
    paper_title = next((l[2:].strip() for l in md.split("\n") if l.startswith("# ")), "")
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    in_list: str | None = None
    table_no = 0

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append(rf"\end{{{in_list}}}")
            out.append("")
            in_list = None

    while i < len(lines):
        line = lines[i]

        # --- front matter, changelog and the reference list are handled apart ---
        if line.startswith("# "):
            i += 1
            continue
        if line.startswith("**Draft v"):
            # The version changelog is working-document metadata, not paper text.
            while i < len(lines) and lines[i].strip() != "---":
                i += 1
            i += 1
            continue

        # --- tables -------------------------------------------------------
        # A table row starts *and ends* with a pipe. A prose line that merely
        # begins with an absolute value ("|f* − A₁K²|/…") is not one; three
        # such lines were swallowed as one-row tables and printed without
        # their bars (review round 4, §5).
        if _is_table_row(line):
            close_list()
            block = []
            while i < len(lines) and _is_table_row(lines[i]):
                block.append(lines[i])
                i += 1
            # A caption paragraph is "**Table N. Title.** followed by prose".
            # Take only the bold run as the caption and leave the prose in the
            # body: dropping the whole paragraph loses text, and keeping it all
            # produces a caption that runs past its own closing brace.
            caption = ""
            table_key = ""
            for back in range(len(out) - 1, max(-1, len(out) - 4), -1):
                # By the time the caption paragraph reaches here it has been
                # through inline(), so its own "Table V." has already become a
                # \ref. Match either spelling: the raw one for safety, and the
                # converted one because that is what actually arrives.
                # The bold run is found by brace matching, not by a regex up to
                # the first "}": captions carry math like $A_{\mathrm{heur}}$,
                # and the first closing brace is inside it. Two captions
                # (Tables XV and XVI) shipped cut in half that way.
                m = _bold_caption(out[back])
                if m:
                    # Drop the draft's own "Table N." prefix: IEEEtran prints its
                    # own number, so keeping it yields "TABLE V / Table IV. ...".
                    # The draft's numbering survives in the running text, which
                    # is why the two can disagree and must not both appear.
                    head = m[0].strip()
                    numeral = re.match(
                        r"Table(?:~\\ref\{tab:(S?[IVXL0-9]+)\}|[\s~](S?[IVXL0-9]+))\.", head)
                    if numeral:
                        table_key = numeral.group(1) or numeral.group(2)
                    caption = re.sub(
                        r"^Table(?:~\\ref\{tab:S?[IVXL0-9]+\}|[\s~]S?[IVXL0-9]+)\.\s*", "", head)
                    out[back] = m[1].strip()
                    break
            table_no += 1
            if not caption:
                # Three tables in the draft are unnumbered supporting displays.
                # Give them descriptive captions keyed on their header row rather
                # than a placeholder, so no float ships as "Table 7".
                header = block[0].lower()
                if "h\\_norm" in header or "steps = 50" in header:
                    caption = ("Optimized expectation at each optimizer budget, "
                               "for the N = 16 cells of the steps x p grid.")
                elif "reproducible?" in header or "execution environment" in header:
                    caption = ("Run-to-run reproducibility of the three execution "
                               "environments.")
                elif header.startswith("| quantity | definition | role |"):
                    caption = "The evaluation quantities of Sec. III-C."
                elif "2-qubit gates" in header:
                    caption = ("Predicted circuit fidelity on IonQ Forte-1 at "
                               "p = 2, from its measured calibration.")
                else:
                    caption = f"Supporting data {table_no}."
            out += convert_table(block, caption,
                                 f"tab:{table_key or table_no}")
            continue

        # --- headings -----------------------------------------------------
        if line.startswith("### "):
            close_list()
            # IEEEtran letters subsections itself; the draft also writes the
            # letter, which printed as "A. A. Seed variance".
            sub_title = re.sub(r"^[A-Z]\. ", "", line[4:])
            out += [rf"\subsection{{{inline(sub_title)}}}", ""]
            i += 1
            continue
        if line.startswith("## "):
            close_list()
            title = line[3:]
            if title.startswith("References"):
                # Two rules here, both learned from the compiler. No blank line
                # after \begin{thebibliography}, and no paragraph between it and
                # the first \bibitem: the draft puts a note about its citation
                # style under this heading, and inside the environment it has
                # nothing to attach to.
                i += 1
                if avail:
                    out += [r"\section*{Data and Code Availability}", ""]
                    for par in avail.split("\n\n"):
                        out += [inline(" ".join(par.split("\n"))), ""]
                note: list[str] = []
                while i < len(lines) and not lines[i].startswith("["):
                    if lines[i].strip():
                        note.append(lines[i].strip())
                    i += 1
                if note:
                    out += [inline(" ".join(note)), ""]
                out += [r"\begin{thebibliography}{99}"]
                continue
            heading = inline(re.sub(r"^(S-)?[IVX]+\. ", "", title))
            if heading == "Abstract":
                i += 1
                para: list[str] = []
                while i < len(lines) and not lines[i].startswith("**Keywords:"):
                    para.append(lines[i])
                    i += 1
                text = " ".join(x.strip() for x in para if x.strip())
                out += [r"\begin{abstract}", inline(text), r"\end{abstract}", ""]
                if i < len(lines):
                    kw = lines[i]
                    while i + 1 < len(lines) and lines[i + 1].strip():
                        i += 1
                        kw += " " + lines[i].strip()
                    kw = re.sub(r"\*\*Keywords:\*\*\s*", "", kw).strip().rstrip(".")
                    env = "keywords" if TQE else "IEEEkeywords"
                    out += [rf"\begin{{{env}}}", inline(kw), rf"\end{{{env}}}", ""]
                    if TQE:
                        out += [r"\titlepgskip=-15pt", "", r"\maketitle", ""]
                    i += 1
                continue
            # IEEE runs the acknowledgment unnumbered, between the
            # conclusion and the bibliography.
            star = "*" if heading.startswith("Acknowledg") else ""
            out += [rf"\section{star}{{{heading}}}", ""]
            i += 1
            continue

        # --- bibliography entries ----------------------------------------
        m = re.match(r"^\[(\d+)\] (.*)", line)
        if m and r"\begin{thebibliography}" in "\n".join(out[-400:]):
            number = int(m.group(1))
            body = [m.group(2)]
            i += 1
            # A continuation line may *start* with a bracketed number: the
            # entries' notes cite each other ("(The implementation of\n[46]
            # called by BraketSolver.)"), and a wrap can put that citation
            # first on the line. Treating it as the next entry emitted a second
            # \bibitem{ref46}; LaTeX takes the last definition, so every
            # \cite{ref46} in the body printed as [48] -- a number with no
            # entry. The list is consecutive, so only the successor starts an
            # entry.
            while i < len(lines) and lines[i].strip():
                nxt = re.match(r"^\[(\d+)\] ", lines[i])
                if nxt and int(nxt.group(1)) == number + 1:
                    break
                body.append(lines[i].strip())
                i += 1
            text = inline(" ".join(body))
            # A cross-reference inside an entry's note stays a printed number:
            # \cite inside thebibliography is what LaTeX will not resolve, but
            # deleting it left "used by ." in the reference list.
            text = re.sub(r"\\cite\{ref(\d+)\}", r"[\1]", text)
            out.append(rf"\bibitem{{ref{number}}} {text}")
            out.append("")
            continue

        # --- block quote ---------------------------------------------------
        if line.startswith("> "):
            close_list()
            body = []
            while i < len(lines) and lines[i].startswith(">"):
                body.append(lines[i].lstrip("> ").rstrip())
                i += 1
            key = " ".join(body).strip()
            if key in FORMULAS:
                out += [r"\begin{equation*}", FORMULAS[key], r"\end{equation*}", ""]
            else:
                out += [r"\begin{quote}", inline(" ".join(body)), r"\end{quote}", ""]
            continue

        # --- lists ----------------------------------------------------------
        m = re.match(r"^(\d+)\. (.*)", line)
        if m:
            if in_list != "enumerate":
                close_list()
                out.append(r"\begin{enumerate}")
                in_list = "enumerate"
            body = [m.group(2)]
            i += 1
            while i < len(lines) and lines[i].startswith("   "):
                body.append(lines[i].strip())
                i += 1
            out.append(rf"\item {inline(' '.join(body))}")
            continue
        if line.startswith("- "):
            if in_list != "itemize":
                close_list()
                out.append(r"\begin{itemize}")
                in_list = "itemize"
            body = [line[2:]]
            i += 1
            while i < len(lines) and lines[i].startswith("  ") and not lines[i].startswith("- "):
                body.append(lines[i].strip())
                i += 1
            out.append(rf"\item {inline(' '.join(body))}")
            continue

        # --- rules and blanks ------------------------------------------------
        if line.strip() == "---":
            close_list()
            i += 1
            continue
        if not line.strip():
            close_list()
            out.append("")
            i += 1
            continue

        # --- paragraph --------------------------------------------------------
        close_list()
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(#|\||>|- |\d+\. |---$)", lines[i]):
            para.append(lines[i])
            i += 1
        out += [inline(" ".join(para)), ""]

    close_list()
    out = place_figures(out)
    body = "\n".join(out)
    if r"\begin{thebibliography}" in body:
        body += "\n" + r"\end{thebibliography}"
    if TQE:
        pre = (PREAMBLE_TQE.replace("@TITLE@", inline(paper_title))
               .replace("@SHORTTITLE@", inline(paper_title)))
        return pre + body + POSTAMBLE_TQE
    pre = PREAMBLE.replace("@TITLE@", inline(paper_title))
    if SUPPLEMENT:
        pre = pre.replace(r"\begin{document}", "\n".join([
            r"\renewcommand{\thesection}{S-\Roman{section}}",
            r"\renewcommand{\thetable}{S\arabic{table}}",
            r"\renewcommand{\thefigure}{S\arabic{figure}}",
            r"\begin{document}"]))
    return pre + body + POSTAMBLE


def sync_figures() -> int:
    """Copy the generated figures next to the .tex, and report how many moved.

    The print-size figures are generated into ``notebooks/figures/paper``
    (``PAPER_FIGURES=1``); the .tex refers to ``figures/`` relative to itself.
    Keeping the second copy by hand meant the paper could silently ship a stale
    figure -- it did, for one build -- which is exactly the provenance failure
    this paper is about. The copy happens on every conversion so the two cannot
    drift. It reads only the ``paper/`` directory: the screen renders, which
    once shared these file names, are what shipped one build with 4.5 pt
    legends, and ``figure_float`` refuses them by width in any case.
    """
    src = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "figures" / "paper"
    dst = pathlib.Path(__file__).resolve().parent / "figures"
    if not src.is_dir():
        raise SystemExit(f"{src} does not exist: run PAPER_FIGURES=1 python -m "
                         "notebooks.generate_figures and notebooks.generate_tqe_figures first")
    dst.mkdir(exist_ok=True)
    moved = 0
    for f in sorted(src.glob("*.pdf")):
        if figure_width_in(f) > MAX_FIGURE_WIDTH_IN:
            raise SystemExit(f"{f} is {figure_width_in(f):.1f} in wide: not a print render")
        target = dst / f.name
        if not target.exists() or target.read_bytes() != f.read_bytes():
            target.write_bytes(f.read_bytes())
            moved += 1
    return moved


def main() -> int:
    set_mode("--supplement" in sys.argv[1:], "--tqe" in sys.argv[1:])
    moved = sync_figures()
    if moved:
        print(f"synced {moved} figure(s) into papers/figures/")
    tex = convert(SRC.read_text())
    # Anything non-ASCII left here is a character inputenc will refuse. Better to
    # fail now, naming it, than to hand over a file whose first compile dies.
    import collections
    leftover = collections.Counter(c for c in tex if ord(c) > 127)
    if leftover:
        # To stderr, and loudly. This refusal is correct -- guessing at an
        # unmapped character would put the wrong glyph in the paper -- but it
        # leaves paper01.tex on disk holding the *previous* draft, so a caller
        # that discards stdout sees a stale .tex and a stale PDF and no reason
        # why. That happened twice: once on U+00B5 MICRO SIGN in the draft, once
        # on a U+00BF this file's own preamble comment introduced.
        print(f"REFUSED to write: unmapped characters. {OUT.name} is UNCHANGED.",
              file=sys.stderr)
        for ch, n in leftover.most_common():
            print(f"  U+{ord(ch):04X} {ch!r} x{n} — add it to UNICODE or fix the draft",
                  file=sys.stderr)
        return 1
    OUT.write_text(tex)
    print(f"wrote {OUT} ({len(OUT.read_text().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
