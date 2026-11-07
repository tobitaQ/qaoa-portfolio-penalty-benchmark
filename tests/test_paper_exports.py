# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""The two exports of the manuscript: the Word extract and the blind PDF.

Both are generated from the submitted sources, and both fail closed. The
anonymiser in particular must stop rather than produce a PDF that still names
the author, so the removals are asserted here on the real `.tex` -- the file
whose template is what would change under it.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

PAPERS = pathlib.Path(__file__).resolve().parent.parent / "papers"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PAPERS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_word_extract_takes_its_sections_from_the_draft():
    make_docx = _load("make_docx")
    md = (PAPERS / "paper01_draft.md").read_text(encoding="utf-8")
    blocks = make_docx.sections(md, make_docx.SECTIONS)
    assert [h for h, _ in blocks] == ["Abstract", "I. Introduction", "VIII. Conclusion"]
    assert all(len(make_docx.paragraphs(b)) >= 2 for _, b in blocks)

    with pytest.raises(SystemExit):
        make_docx.sections(md, ["## No Such Section"])


def test_the_word_extract_sets_indices_rather_than_printing_them():
    """"ε_F" with an underscore is what the LaTeX side spent a month removing."""
    make_docx = _load("make_docx")
    out = make_docx.inline("ε_F is 0.016, P_τ = P_F · q_τ, and 2^N states; `run_e1_cross.py`")
    assert "ε<sub>F</sub>" in out and "P<sub>τ</sub>" in out
    assert "2<sup>N</sup>" in out
    assert "<code>run_e1_cross.py</code>" in out  # a file name keeps its underscores
    assert "_" not in out.replace("run_e1_cross.py", "")

    assert make_docx.inline("**C1 — the contribution**").startswith("<strong>")


def test_the_blind_build_removes_the_name_the_disclosure_and_the_biography():
    make_anonymous = _load("make_anonymous")
    src = PAPERS / "paper01_access.tex"
    if not src.exists():
        pytest.skip("the Access source is not built here")
    tex, names = make_anonymous.anonymise(src.read_text(encoding="utf-8"))
    assert names, "the author's name was not recovered, so nothing would be checked"
    for name in names:
        assert name not in tex, f"{name} survives the anonymiser"
    for mark in ("\\corresp", "\\address[", "IEEEbiography", "Acknowledgment",
                 "Anthropic", "Claude", "e-mail"):
        assert mark not in tex, f"{mark} survives the anonymiser"
    # The class refuses to build without it, and the biography carried it.
    assert "\\EOD" in tex
    # Everything else is the submitted build.
    assert "\\section*{Data and Code Availability}" in tex
    assert tex.count("\\bibitem") == src.read_text(encoding="utf-8").count("\\bibitem") - 1


def test_the_anonymiser_stops_when_the_template_moves():
    """A silent no-op here ships a PDF with the author's name on page 1."""
    make_anonymous = _load("make_anonymous")
    with pytest.raises(SystemExit):
        make_anonymous.anonymise("\\author{\\uppercase{A Name}}\n\\maketitle\n")


def test_the_reference_checker_reads_the_bibliography_and_folds_titles():
    """The parsing half of `check_references.py`, without the network.

    The entries wrap over several lines and the registries disagree with the
    manuscript about dashes, case, accents and — in Phys. Rev. A's "XY mixers"
    — MathML, so the comparison is on letters and digits alone. Two entries
    were found to cite a title the record does not carry; this pins the parts
    that found them.
    """
    check_references = _load("check_references")
    draft = (PAPERS / "paper01_draft.md").read_text(encoding="utf-8")
    refs = check_references.entries(draft)
    assert len(refs) >= 45
    assert [n for n, _ in refs] == list(range(1, len(refs) + 1))  # numbered without gaps

    assert check_references.letters("Mean–variance Sélection!") == "meanvarianceselection"
    assert check_references.letters("<mml:math><mml:mi>XY</mml:mi></mml:math> mixers") == "xymixers"
    entry = 'D. Goldberg, "What every computer scientist should know," *ACM Comput. Surv.*, 1991.'
    assert check_references.quoted_title(entry) == "What every computer scientist should know"
    assert check_references.surname(entry) == "Goldberg"
