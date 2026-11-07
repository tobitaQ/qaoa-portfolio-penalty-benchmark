# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Line-level invariants of the manuscripts that a re-wrap can break silently.

Both checks here come from defects a proofreading pass introduced. A script
that re-flowed paragraphs joined the lines of Table I and Table II into one
line each: the converter then read them as a single prose line, `check_tables`
counted 2,592 cells instead of 2,614, and nothing *disagreed* -- the missing
cells were simply no longer findable. The same script's `textwrap` default
breaks a line at a hyphen, so "conditional-enrichment" wrapped as
"conditional-" + "enrichment" and re-joining it later produced
"conditional- enrichment" in the PDF.

Neither defect is visible in a diff that is mostly re-wrapped text, so the
invariants are pinned here: a table row's cell count matches its header, and
no line ends with a hyphen.
"""

from __future__ import annotations

import pathlib
import re

import pytest

PAPERS = pathlib.Path(__file__).resolve().parent.parent / "papers"
DOCS = [PAPERS / "paper01_draft.md",
        PAPERS / "paper01_supplement.md",
        PAPERS / "system_paper" / "system_paper_draft.md"]


def _lines(path: pathlib.Path) -> list[str]:
    return path.read_text(encoding="utf-8").split("\n")


#: A pipe that delimits a cell: the ones written "\\|" are absolute-value bars.
_CELL = re.compile(r"(?<!\\)\|")


def _cells(line: str) -> list[str]:
    return _CELL.split(line.rstrip())[1:-1]


def _is_row(line: str) -> bool:
    """The converter's own rule: a row opens with a pipe and closes with one.

    A prose line that opens with an absolute value, such as "|E_A*| therefore
    carries that constant.", is not a row, and must not be read as one.
    """
    s = line.rstrip()
    return s.startswith("|") and s.endswith("|")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_table_row_has_its_header_s_cell_count(doc: pathlib.Path) -> None:
    header: list[str] | None = None
    for i, line in enumerate(_lines(doc), 1):
        if not _is_row(line):
            header = None
            continue
        cells = _cells(line)
        if header is None:
            header = cells
            continue
        assert len(cells) == len(header), (
            f"{doc.name}:{i} has {len(cells)} cells under a header of "
            f"{len(header)}; a flattened table reads as one long row:\n"
            f"  {line[:120]}")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_a_separator_row_carries_nothing_but_separators(doc: pathlib.Path) -> None:
    """A row holding both separators and data is a table collapsed into a line."""
    sep = re.compile(r"^:?-{2,}:?$")
    for i, line in enumerate(_lines(doc), 1):
        if not _is_row(line):
            continue
        cells = [c.strip() for c in _cells(line)]
        if any(sep.match(c) for c in cells):
            assert all(sep.match(c) for c in cells), (
                f"{doc.name}:{i} mixes a separator with data:\n  {line[:120]}")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_no_line_ends_with_a_hyphen(doc: pathlib.Path) -> None:
    """textwrap breaks on hyphens by default; re-joining inserts a space."""
    bad = [(i, l) for i, l in enumerate(_lines(doc), 1)
           if l.rstrip().endswith("-") and l.strip() != "---"]
    assert not bad, ("re-joining these lines would put a space inside a "
                     f"hyphenated word: {[f'{doc.name}:{i}' for i, _ in bad]}")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_no_list_item_swallowed_another(doc: pathlib.Path) -> None:
    """A second "- " inside a list item is two items a re-wrap joined.

    Round 4's re-wrap guarded tables but not lists, so the QUBO's two entries
    became one line: "- diagonal: … , - off-diagonal: …". LaTeX then set the
    second marker as a hyphen in the middle of the first item, which is how it
    reached the PDF.
    """
    for i, line in enumerate(_lines(doc), 1):
        if line.startswith("- "):
            assert not re.search(r"\S\s+-\s+\w", line[2:]), f"{doc.name}:{i}: {line[:120]}"
        # The same re-wrap joined Sec. IV-B's four solvers into one item, and
        # the markers after the first set as hyphens mid-paragraph.
        assert " - **" not in line, f"{doc.name}:{i}: a list item lost its line: {line[:120]}"


def test_the_manuscript_is_in_one_dialect():
    """IEEE sets American spelling, and the manuscript held both.

    "optimised" and "optimization" met inside single paragraphs, because the
    fixed names -- quadratic unconstrained binary optimization, Quantum
    Approximate Optimization Algorithm -- force the American form while the
    prose around them had been written British: 93 of one against 100 of the
    other in the main text alone.
    """
    import importlib.util
    papers = pathlib.Path(__file__).resolve().parent.parent / "papers"
    spec = importlib.util.spec_from_file_location("check_claims", papers / "check_claims.py")
    check_claims = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_claims)

    assert check_claims.british_spellings() == []
    # and the check is not vacuous
    assert any(re.search(p, "the optimised state was normalised", flags=re.I)
               for p in check_claims.BRITISH)


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_no_prose_line_starts_a_table(doc: pathlib.Path) -> None:
    """A wrapped line that begins with "|" is read as a table row.

    `check_tables.table()` takes the first |-leading line after a caption as the
    header and stops at the next line that is not one. A re-wrap put
    "max_{i<j} |J_ij|)" at the start of a line inside Table III's caption, so the
    checker found a header of prose, no rows, and silently checked 288 cells
    fewer — 2,353 instead of 2,641, with nothing reported as wrong. The
    converter has its own guard for this; the count is what caught it.
    """
    bad = []
    for par in doc.read_text(encoding="utf-8").split("\n\n"):
        lines = par.split("\n")
        if lines[0].startswith("|"):
            continue  # the table itself
        bad += [l[:60] for l in lines if l.startswith("|")]
    assert not bad, f"{doc.name}: prose lines that begin a table row: {bad}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_a_heading_is_one_line(doc: pathlib.Path) -> None:
    """The title is read from the first line, so a re-wrap must not break it.

    A re-wrap that tidied long lines split the 121-character title after
    "Contributes at"; the converter put the rest in the body as a paragraph of
    its own, and the page count went from 20 to 21 before anything else was
    suspected.
    """
    text = doc.read_text(encoding="utf-8")
    # A "#" inside a fenced block is a shell comment, not a heading.
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    bad = []
    for par in text.split("\n\n"):
        lines = par.split("\n")
        if lines[0].startswith("#") and len(lines) > 1 and not lines[1].startswith(("-", "|")):
            bad.append(lines[0][:60])
    assert not bad, f"{doc.name}: headings continued onto a second line: {bad}"
