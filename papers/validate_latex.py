# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Structural checks on generated LaTeX, standing in for a compiler.

There is no TeX installation on this machine, so `paper01.tex` cannot be built
here. This runs the checks a compiler would fail on first, so that what goes to
Overleaf is wrong for interesting reasons rather than for trivial ones:

    * unbalanced braces, and \\begin/\\end environments that do not pair
    * reserved characters left unescaped in running text
    * table rows whose ampersand count disagrees with the column specification
    * \\cite keys with no matching \\bibitem, and unused \\bibitem entries
    * math mode left open at the end of a line

What it cannot check: whether the result typesets *well* — column widths,
overfull boxes, float placement, whether IEEEtran accepts the class options.
Those need a real build. Treat a clean run here as necessary, not sufficient.

Usage:
    python papers/validate_latex.py papers/paper01.tex
"""

from __future__ import annotations

import pathlib
import re
import sys

#: Reserved in LaTeX text mode. Backslash and braces are excluded because they
#: are the markup itself; the converter is responsible for those.
RESERVED = set("%&#$_")


def strip_math_and_verbatim(line: str) -> str:
    """Remove $...$ spans and \\texttt{...} bodies before checking escapes.

    Escaped dollars go first. The draft quotes prices, so a line can hold
    ``\\$1.47`` and ``$\\times$`` together; pairing math before removing the
    escaped ones makes the currency look like an opening delimiter and reports a
    span that is not there.
    """
    line = re.sub(r"\\[%&#$_]", "", line)
    # A % that is not escaped starts a comment; nothing after it is typeset.
    line = re.split(r"(?<!\\)%", line)[0]
    line = re.sub(r"\$[^$]*\$", "", line)
    line = re.sub(r"\\texttt\{[^{}]*\}", "", line)
    line = re.sub(r"\\url\{[^{}]*\}", "", line)
    # File paths and labels are not typeset, so an underscore in them is fine.
    line = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^{}]*\}", "", line)
    # The TQE class's \Figure macro takes the file as its first brace argument.
    line = re.sub(r"\\Figure(\[[^\]]*\])?(\([^)]*\))?(\[[^\]]*\])?\{[^{}]*\}", "", line)
    line = re.sub(r"\\(label|ref|cite|bibitem)\{[^{}]*\}", "", line)
    return line


def check(path: pathlib.Path) -> list[str]:
    """Return a list of problems, empty if the file passes every check."""
    text = path.read_text()
    lines = text.split("\n")
    problems: list[str] = []

    # --- braces -----------------------------------------------------------
    depth = 0
    for n, line in enumerate(lines, 1):
        bare = re.sub(r"\\[{}]", "", line)
        depth += bare.count("{") - bare.count("}")
        if depth < 0:
            problems.append(f"{n}: closing brace with nothing open")
            depth = 0
    if depth:
        problems.append(f"EOF: {depth} unclosed brace(s)")

    # --- environments -----------------------------------------------------
    stack: list[tuple[str, int]] = []
    for n, line in enumerate(lines, 1):
        for name in re.findall(r"\\begin\{(\w+\*?)\}", line):
            stack.append((name, n))
        for name in re.findall(r"\\end\{(\w+\*?)\}", line):
            if not stack:
                problems.append(rf"{n}: \end{{{name}}} with nothing open")
            elif stack[-1][0] != name:
                problems.append(
                    rf"{n}: \end{{{name}}} closes \begin{{{stack[-1][0]}}} from line {stack[-1][1]}")
                stack.pop()
            else:
                stack.pop()
    for name, n in stack:
        problems.append(rf"{n}: \begin{{{name}}} never closed")

    # --- unescaped reserved characters ------------------------------------
    # '&' is the column separator inside tabular, and a line beginning with '%'
    # is a comment; neither is an error. Everything else must be escaped.
    # A display formula is math from \begin to \end, so its underscores and
    # carets are not text-mode specials either.
    # Macro definitions between \makeatletter and \makeatother are code, not
    # text: their #1/#2 are parameters. Nothing else in the preamble uses them.
    in_tabular = in_display = in_macro = False
    for n, line in enumerate(lines, 1):
        if r"\makeatletter" in line:
            in_macro = True
        if r"\makeatother" in line:
            in_macro = False
            continue
        if in_macro:
            continue
        if r"\begin{tabular}" in line:
            in_tabular = True
        if r"\end{tabular}" in line:
            in_tabular = False
            continue
        if re.match(r"\\begin\{(equation|align|gather)\*?\}", line.strip()):
            in_display = True
            continue
        if re.match(r"\\end\{(equation|align|gather)\*?\}", line.strip()):
            in_display = False
            continue
        if in_display:
            continue
        bare = strip_math_and_verbatim(line)
        reserved = RESERVED - ({"&"} if in_tabular else set())
        for ch in sorted(reserved):
            if ch in bare:
                problems.append(f"{n}: unescaped {ch!r} in text: {line.strip()[:70]}")
                break

    # --- math mode left open ----------------------------------------------
    for n, line in enumerate(lines, 1):
        if (len(re.findall(r"(?<!\\)\$", re.sub(r"\\\$", "", line))) % 2) != 0:
            problems.append(f"{n}: odd number of $ — math left open")

    # --- table column counts ----------------------------------------------
    spec: str | None = None
    spec_line = 0
    for n, line in enumerate(lines, 1):
        m = re.search(r"\\begin\{tabular\}\{(.*)\}\s*$", line)
        if m:
            # Count columns, not characters: p{...} is one column whose braces
            # contain a length expression full of letters that are not columns.
            body_spec = re.sub(r"p\{[^{}]*(\{[^{}]*\}[^{}]*)*\}", "p", m.group(1))
            spec = re.sub(r"[^lcrp]", "", body_spec)
            spec_line = n
            continue
        if r"\end{tabular}" in line:
            spec = None
            continue
        if spec and line.rstrip().endswith("\\\\"):
            body = line.rstrip()[:-2]
            cols = len(re.split(r"(?<!\\)&", body))
            if cols != len(spec):
                problems.append(
                    f"{n}: row has {cols} column(s), spec at line {spec_line} "
                    f"declares {len(spec)}: {line.strip()[:70]}")

    # --- citations vs bibliography ----------------------------------------
    cited = set(re.findall(r"\\cite\{(\w+)\}", text))
    # A cited range "\cite{ref8}--\cite{ref10}" covers ref9 without naming it.
    for lo, hi in re.findall(r"\\cite\{ref(\d+)\}\s*-{2,}\s*\\cite\{ref(\d+)\}", text):
        cited |= {f"ref{i}" for i in range(int(lo), int(hi) + 1)}
    order = re.findall(r"\\bibitem\{(\w+)\}", text)
    defined = set(order)
    for key in sorted(cited - defined):
        problems.append(rf"\cite{{{key}}} has no \bibitem")
    for key in sorted(defined - cited):
        problems.append(rf"\bibitem{{{key}}} is never cited")

    # A key defined twice is not a duplicate entry that a reader skips past:
    # LaTeX keeps the *last* definition, so every \cite to that key prints the
    # position of the second one. A wrapped "[46]" inside another entry's note
    # was picked up as an entry of its own, and the body's L-BFGS-B citation
    # printed as [48] -- a number the reference list does not contain. Comparing
    # sets hid this, because a set has no duplicates.
    for key in sorted({k for k in order if order.count(k) > 1}):
        problems.append(rf"\bibitem{{{key}}} is defined {order.count(key)} times; "
                        r"\cite prints the position of the last one")

    # The draft numbers its own references and the running text cites those
    # numbers, so entry k must be refk. Any drift silently renumbers citations.
    for pos, key in enumerate(order, 1):
        if key != f"ref{pos}":
            problems.append(f"entry {pos} in the list is {key}: the printed "
                            f"number and the draft's number disagree")
            break

    # \cite inside thebibliography is not resolved by LaTeX; entries that
    # cross-reference each other must print the number as text.
    bib = re.search(r"\\begin\{thebibliography\}(.*?)\\end\{thebibliography\}",
                    text, re.S)
    if bib and re.search(r"\\cite\{", bib.group(1)):
        problems.append(r"\cite inside thebibliography: print the number as text")

    return problems


def check_aux(tex: pathlib.Path) -> list[str]:
    """Compare the numbers LaTeX assigned with the numbers the draft uses.

    ``\bibcite{refN}{M}`` with N != M means the body says [N] and the list
    prints [M]. This is the ground truth for the numbering, and it is written
    even when the build reports no error at all.
    """
    aux = tex.with_suffix(".aux")
    if not aux.exists():
        return []
    out = []
    for key, num in re.findall(r"\\bibcite\{ref(\d+)\}\{(\d+)\}", aux.read_text()):
        if key != num:
            out.append(f"reference [{key}] is typeset as [{num}]")
    return out


def check_build_log(tex: pathlib.Path) -> list[str]:
    """Read the compiler's own errors, which a structural check cannot see.

    pdflatex in nonstopmode writes a PDF even when it has failed: four
    "Undefined control sequence" errors still produced 25 pages that looked
    right until the missing subscripts were noticed. Counting overfull boxes and
    undefined references, as this script did, misses exactly that class.
    """
    log = tex.with_suffix(".log")
    if not log.exists():
        return []
    text = log.read_text(errors="ignore")
    out = [l.strip() for l in text.splitlines() if l.startswith("! ")]
    # Undefined references and citations are warnings, not errors: pdflatex
    # exits 0, prints "??", and the paper goes out with a dangling pointer.
    out += [m.strip() for m in re.findall(
        r"LaTeX Warning: (?:Reference|Citation) `[^']*' (?:on page [^ ]+ )?undefined[^.]*\.",
        text)]
    return out


def main(argv: list[str]) -> int:
    path = pathlib.Path(argv[1] if len(argv) > 1
                        else pathlib.Path(__file__).parent / "paper01.tex")
    problems = check(path)
    problems += check_aux(path)
    problems += [f"compiler error: {e}" for e in check_build_log(path)]
    if not problems:
        log = path.with_suffix(".log")
        state = ("and the local build reports no errors"
                 if log.exists() else "compilation unverified — no build log here")
        print(f"{path.name}: structural checks pass "
              f"({len(path.read_text().splitlines())} lines), {state}.")
        return 0
    print(f"{path.name}: {len(problems)} problem(s)")
    for p in problems[:60]:
        print("  " + p)
    if len(problems) > 60:
        print(f"  … and {len(problems) - 60} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
