# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""The Access PDF without the author's name, the AI disclosure or the biography.

Three removals, each of which has to fire or the script stops rather than
quietly shipping a PDF that still names someone:

* the author block -- name, affiliation, e-mail, corresponding-author line and
  the running head that repeats the surname on every page;
* the Acknowledgment, which is the AI assistance disclosure, and the reference
  it cites (the remaining references renumber themselves);
* the biography and its photograph.

Everything else -- text, tables, figures, the availability statement -- is the
submitted build, byte for byte. The script then reads the PDF it produced and
fails if any of the removed names still appears in it.

    python papers/make_anonymous.py

Note what it does NOT remove: the availability statement's repository URL
carries the author's account name. It is reported, not deleted, because taking
it out changes what the paper offers a reader.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "paper01_access.tex"
OUT = ROOT / "paper01_access_anon.tex"


def cut(text: str, pattern: str, repl: str, what: str) -> str:
    """Substitute, and refuse to continue if the pattern was not there."""
    new, n = re.subn(pattern, repl, text, flags=re.S)
    if n == 0:
        raise SystemExit(f"{SRC.name}: nothing matched for {what} -- the "
                         f"template changed, and this PDF would still carry it")
    return new


def anonymise(tex: str) -> tuple[str, list[str]]:
    """The anonymous source, and the names it must no longer contain."""
    author = re.search(r"\\author\{\\uppercase\{([^}]*)\}", tex)
    if not author:
        raise SystemExit(f"{SRC.name}: no \\author{{\\uppercase{{...}}}} to remove")
    name = author.group(1).strip()
    surname = name.split()[-1]

    tex = cut(tex, r"\\author\{[^\n]*\}", r"\\author{\\uppercase{Anonymous}}", "the author")
    tex = cut(tex, r"\\address\[\d+\]\{[^\n]*\}\n", "", "the affiliation")
    tex = cut(tex, r"\\corresp\{[^\n]*\}\n", "", "the corresponding author")
    tex = cut(tex, rf"\{{{re.escape(surname)}: ", "{", "the running head")
    # The Acknowledgment is the AI disclosure; it ends where the next section
    # starts. The citation goes with it, and \thebibliography renumbers.
    tex = cut(tex, r"\\section\*\{Acknowledgment\}.*?(?=\\section\*\{)", "",
              "the AI assistance disclosure")
    # By content, not by number: the bibliography is in order of first
    # appearance, so the tool's entry moves whenever the citations do.
    tex = cut(tex, r"\\bibitem\{ref\d+\} Anthropic[^\n]*\n\n", "", "the Claude Code reference")
    # \EOD goes with the last biography, and the class stops the build without
    # it ("You have not used the command \EOD"), so it stays behind.
    tex = cut(tex, r"\\begin\{IEEEbiography\}.*?\\end\{IEEEbiography\}\n",
              "\\\\EOD\\n", "the biography")
    # hyperref writes whatever the class left in the document information.
    tex = tex.replace(r"\begin{document}",
                      "\\hypersetup{pdfauthor={},pdfsubject={},pdfcreator={}}\n"
                      r"\begin{document}", 1)
    return tex, [name, surname, name.split()[0]]


def main(argv: list[str]) -> int:
    if not SRC.exists():
        raise SystemExit(f"{SRC.name} is not built; run papers/build_access.sh first")
    if not shutil.which("pdflatex"):
        raise SystemExit("pdflatex is not installed")
    classdir = ROOT / "access_class"
    if not (classdir / "ieeeaccess.cls").exists():
        raise SystemExit("papers/access_class/ieeeaccess.cls is missing; see its README")

    tex, names = anonymise(SRC.read_text(encoding="utf-8"))
    OUT.write_text(tex, encoding="utf-8")
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", OUT.stem],
                       cwd=ROOT, capture_output=True,
                       env={**__import__("os").environ, "TEXINPUTS": "./access_class:"})
    log = (ROOT / f"{OUT.stem}.log").read_text(encoding="utf-8", errors="replace")
    errors = [ln for ln in log.split("\n") if ln.startswith("!")]
    undef = [ln for ln in log.split("\n")
             if re.search(r"LaTeX Warning: (Reference|Citation).*undefined", ln)]
    pdf = ROOT / f"{OUT.stem}.pdf"
    if errors or undef or not pdf.exists():
        print(f"{len(errors)} error(s), {len(undef)} undefined reference(s); "
              f"see papers/{OUT.stem}.log")
        for ln in (errors + undef)[:5]:
            print("  " + ln)
        return 1

    # The check that matters: read the PDF, not the source.
    text = subprocess.run(["pdftotext", "-nopgbrk", str(pdf), "-"],
                          capture_output=True, text=True, check=True).stdout
    forbidden = {n: text.count(n) for n in
                 set(names) | {"Claude", "Anthropic", "AI assistance", "@"}
                 if n in text}
    if forbidden:
        print(f"{pdf.name} still carries: " +
              ", ".join(f"{k} ({v}x)" for k, v in forbidden.items()))
        return 1
    pages = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    pages = next(ln.split()[1] for ln in pages.split("\n") if ln.startswith("Pages:"))
    print(f"wrote {pdf} ({pages} pages, no name, no disclosure, no biography)")
    left = [m for m in re.findall(r"https?://\S+|github\.com/\S+", text)
            if "github" in m or "zenodo" in m]
    if left:
        print("  still identifying, by design: " + ", ".join(sorted(set(left))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
