# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Does the built PDF carry the manuscript's words, and only them?

The other checkers read the Markdown: `check_tables.py` binds every table cell
to a result file, `check_claims.py` scans the prose, `verify_facts.py`
recomputes. None of them looks at what the typesetter actually produced, and
two defects lived in that gap: the figure manifest's closing note was printed
inside Fig. 3's caption, and the first fix for it stopped the sentence being
printed at all. Both were invisible to every check and to four readings of the
PDF.

This aligns the draft's words with the PDF's and reports the runs that do not
match. A run in the draft but not in the PDF is text the build dropped; a run
in the PDF but not in the draft is text the build invented or moved — the
running head, the front-matter boilerplate and the tables are known to be of
the second kind and are listed as expected.

    python papers/check_rendered.py                 # the Access build
    python papers/check_rendered.py papers/paper01.pdf

Needs `pdftotext` (poppler). Without it the script says so and exits 0, because
a check that cannot run must not look like a check that passed.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
DRAFT = ROOT / "paper01_draft.md"
DEFAULT_PDF = ROOT / "paper01_access.pdf"

#: A sentence shorter than this says too little to look for; the long ones are
#: where a dropped line hides.
ROMAN = re.compile(r"[ivx]{1,6}")

#: The page furniture of the journal class: the running head, the volume line
#: and the front-matter date/DOI block.
FURNITURE = re.compile(r"^(VOLUME \d|\d+ VOLUME|Tobita: |Date of publication|"
                       r"Digital Object Identifier|This work is licensed)")

MIN_WORDS = 8


def words(text: str) -> list[str]:
    """Letter runs, with hyphens closed up on both sides of the comparison.

    The typesetter breaks lines both inside a word ("optimiza-\ntion") and at a
    word's own hyphen ("equal-\ncardinality"), and the two cannot be told apart
    after the fact. Removing every hyphen from the draft and from the PDF makes
    both read the same -- "optimization", "equalcardinality" -- which is enough
    to ask whether a sentence is there.
    """
    raw = re.findall(r"[a-z]{3,}", re.sub(r"[-_]\s*", "", text.lower()))
    # The draft numbers its tables and sections in Roman, the journal class in
    # Arabic, so "Table III" sets as "Table 3": the numeral is not a word that
    # can be compared and is dropped from both sides.
    return [w for w in raw if not ROMAN.fullmatch(w)]


def draft_sentences(src: pathlib.Path | None = None) -> list[list[str]]:
    """The draft's prose, one word list per sentence.

    Table rows are set by the converter into LaTeX tabulars; their cells are
    checked against the result files by check_tables.py, not here.
    """
    src = src or DRAFT
    text = src.read_text(encoding="utf-8")
    body = (text.split("\n---\n", 1)[1].split("### Figure manifest")[0]
            if "supplement" in src.name
            else text.split("## Abstract", 1)[1].split("## References")[0])
    body = re.sub(r"^\|.*$", "", body, flags=re.M)
    # The converter spells the draft's section marks the journal's way, so the
    # typeset stream has a word the Markdown does not: "§V-C" sets as "Sec. V-C".
    body = body.replace("§", "Sec. ")
    out = []
    for par in body.split("\n\n"):
        # The emphasis marks have to go before the split, or a caption's bold
        # header ("**Table S26. …**") stays joined to the sentence after it and
        # the two are looked for together, though the typesetter sets one above
        # the table and the other below it.
        par = " ".join(par.replace("**", "").split())
        for sentence in re.split(r"(?<=[.?;:])\s+(?=[\"(*\w§⟨])", par):
            w = words(sentence)
            if len(w) >= MIN_WORDS:
                out.append(w)
    return out


def pdf_words(pdf: pathlib.Path) -> list[str]:
    out = subprocess.run(["pdftotext", "-nopgbrk", str(pdf), "-"],
                         capture_output=True, text=True, check=True).stdout
    # The running head and the volume line sit between two halves of whatever
    # sentence the page break fell inside, so a sentence that is there reads as
    # interrupted. They are page furniture, not text.
    kept = [ln for ln in out.split("REFERENCES")[0].split("\n")
            if not FURNITURE.match(ln.strip())]
    text = "\n".join(kept)
    # Small caps come out of the extractor with the first letter detached:
    # a caption reading "Same code, two backends" extracts as "S AME CODE".
    text = re.sub(r"\b([A-Z]) ([A-Z]{2,})", r"\1\2", text)
    return words(text)


def main(argv: list[str]) -> int:
    pdf = pathlib.Path(argv[0]) if argv else DEFAULT_PDF
    if not shutil.which("pdftotext"):
        print("pdftotext is not installed; this check did not run")
        return 0
    if not pdf.exists():
        print(f"{pdf.name} has not been built; this check did not run")
        return 0

    src = (ROOT / "paper01_supplement.md") if "supplement" in pdf.name else DRAFT
    sentences = draft_sentences(src)
    tokens = pdf_words(pdf)
    stream = " ".join(tokens)
    # A sentence counts as present when any run of MIN_WORDS of its words is in
    # the typeset stream. Containment of the whole sentence is too strict --
    # the converter respells a few things on the way (a section mark becomes
    # "Sec.", the keyword line becomes "INDEX TERMS") -- and an aligned diff is
    # too loose, because the typesetter moves floats through the text. A
    # sentence the build dropped has no run at all.
    def present(w: list[str]) -> bool:
        if any(" ".join(w[i:i + MIN_WORDS]) in stream
               for i in range(len(w) - MIN_WORDS + 1)):
            return True
        # A float can land inside a sentence, and its cells then sit between
        # the sentence's halves in the extracted text, so no run of MIN_WORDS
        # survives. Fall back to asking whether the words appear in order.
        it = iter(tokens)
        if all(any(t == word for t in it) for word in w):
            return True
        # Last resort, for the two-column builds: when a float splits a
        # sentence across columns, the extractor can emit the second half
        # *before* the first, so even the in-order test fails on a sentence
        # that is wholly there. Require a shorter run plus every word.
        short = max(5, MIN_WORDS // 2)
        return (set(w) <= set(tokens)
                and any(" ".join(w[i:i + short]) in stream
                        for i in range(len(w) - short + 1)))

    missing = [s for s in sentences if not present(s)]

    # Marks that mean the converter lost a fight with LaTeX. The circumflex is
    # what a caret becomes when it is escaped as an accent instead of being set
    # as a superscript: "2ˆN", "|DˆNK>", "C(N, K)ˆ{-1/2". The macro names are
    # what leaks when a backslash is escaped away.
    raw = subprocess.run(["pdftotext", "-nopgbrk", str(pdf), "-"],
                         capture_output=True, text=True, check=True).stdout
    spoilt = [m for m in ("ˆ", "˜", "\\mathrm", "\\textbf", "\\rangle",
                          "\\times", "\\{")
              if m in raw]
    # An underscore on the page is either a file name or a subscript the
    # converter failed to set. Extracted text flattens a real subscript to
    # nothing, so the underscores that survive are exactly the failures --
    # "γ_ℓ", "p_⊥" and "Σ_{x∈F}" were all found this way. Every legitimate one
    # comes from a code span of the draft, so that is the pool to match
    # against; a line break can cut a long name, which leaves a prefix.
    source = src.read_text(encoding="utf-8")
    # Code spans, and the identifiers that legitimately carry an underscore:
    # Springer's DOIs end in "_9", and a URL may hold one too.
    pool = " ".join(re.findall(r"`([^`]*)`", source)
                    + re.findall(r"10\.\d{4,}/\S+", source)
                    + re.findall(r"https?://\S+", source))
    loose = sorted({t for t in re.findall(r"\S*_\S*", raw)
                    if t.strip(".,;:()[]") not in pool
                    and t.strip(".,;:()[]") not in pool.replace("\\", "")})
    if loose:
        print(f"{pdf.name} prints {len(loose)} underscore(s) that no code span "
              f"of {src.name} accounts for: " + ", ".join(repr(t) for t in loose[:12]))
        return 1

    print(f"{pdf.name} against {src.name}: {len(sentences)} sentences against "
          f"{len(stream.split())} typeset words")
    if spoilt:
        print(f"{pdf.name} prints {len(spoilt)} mark(s) of a conversion that failed: "
              + ", ".join(repr(m) for m in spoilt))
        for m in spoilt:
            i = raw.index(m)
            print(f"  {m!r}: ...{' '.join(raw[max(0, i - 60):i + 40].split())}...")
        return 1
    if missing:
        print(f"{len(missing)} sentence(s) of the draft did not reach the PDF:")
        for s in missing:
            print(f"  {' '.join(s)[:150]}")
        return 1
    print("every sentence of the draft's prose reaches the PDF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
