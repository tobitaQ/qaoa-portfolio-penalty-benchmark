# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Regression tests for the Markdown -> LaTeX converter and its validator.

Both defects fixed here reached a built PDF and were found by a human reader,
not by the pipeline: the reference list printed "used by ." and the body's
L-BFGS-B citation printed [48], a number the list does not contain. The paper
goes through repeated review rounds, so the checks that catch this class run
here rather than depending on someone re-reading 47 reference entries.
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


md_to_latex = _load("md_to_latex")
validate_latex = _load("validate_latex")


BIB_HEAD = "# T\n\n## Abstract\n\nA.\n\n**Keywords:** k\n\n## References\n\n"


def _bibliography(entries: str) -> str:
    return md_to_latex.convert(BIB_HEAD + entries)


def test_a_wrapped_citation_does_not_start_a_new_entry():
    """"[46]" first on a continuation line is a cross-reference, not an entry.

    Taking it as an entry emitted \\bibitem{ref46} twice; LaTeX keeps the last
    definition, so the body's \\cite{ref46} printed the position of the stray
    one.
    """
    tex = _bibliography(
        "[46] Byrd et al., \"L-BFGS-B.\" (The method used as control B.)\n"
        "[47] Virtanen et al., \"SciPy 1.0.\" (The implementation of\n"
        "[46] called by `BraketSolver`.)\n"
    )
    assert tex.count(r"\bibitem{ref46}") == 1
    assert tex.count(r"\bibitem{ref47}") == 1
    assert "called by" in tex


def test_a_cross_reference_inside_an_entry_keeps_its_number():
    """Dropping the \\cite left "used by ." in the printed reference list."""
    tex = _bibliography("[44] Wang et al., \"XY mixers.\" (Used by [33].)\n")
    assert "Used by [33].)" in tex
    assert r"\cite" not in tex.split(r"\begin{thebibliography}")[1]


def test_consecutive_entries_still_split():
    tex = _bibliography("[1] A, \"One.\"\n[2] B, \"Two.\"\n[3] C, \"Three.\"\n")
    assert [tex.count(rf"\bibitem{{ref{k}}}") for k in (1, 2, 3)] == [1, 1, 1]


def test_figure_width_is_read_without_an_external_tool():
    """The Mac has no poppler; a converter that runs on one host is no use."""
    assert md_to_latex._is_wide("fig_e1_penalty.pdf".removesuffix(".pdf")) is True
    assert md_to_latex._is_wide("fig1_convergence") is False
    assert md_to_latex._is_wide("no_such_figure") is False


# --- the checks that would have caught the defect ------------------------


def test_duplicate_bibitem_is_reported(tmp_path):
    tex = tmp_path / "d.tex"
    tex.write_text(
        r"\cite{ref1}\cite{ref2}" "\n"
        r"\begin{thebibliography}{99}" "\n"
        r"\bibitem{ref1} A." "\n"
        r"\bibitem{ref2} B." "\n"
        r"\bibitem{ref1} stray." "\n"
        r"\end{thebibliography}" "\n"
    )
    problems = validate_latex.check(tex)
    assert any("defined 2 times" in p for p in problems)


def test_printed_number_disagreeing_with_the_draft_is_reported(tmp_path):
    tex = tmp_path / "d.tex"
    tex.write_text(
        r"\cite{ref1}\cite{ref2}" "\n"
        r"\begin{thebibliography}{99}" "\n"
        r"\bibitem{ref2} B." "\n"
        r"\bibitem{ref1} A." "\n"
        r"\end{thebibliography}" "\n"
    )
    assert any("disagree" in p for p in validate_latex.check(tex))


def test_aux_is_the_ground_truth_for_numbering(tmp_path):
    tex = tmp_path / "d.tex"
    tex.write_text("x\n")
    (tmp_path / "d.aux").write_text(
        r"\bibcite{ref45}{45}" "\n" r"\bibcite{ref46}{48}" "\n")
    assert validate_latex.check_aux(tex) == ["reference [46] is typeset as [48]"]


def test_cite_inside_the_bibliography_is_reported(tmp_path):
    tex = tmp_path / "d.tex"
    tex.write_text(
        r"\cite{ref1}" "\n"
        r"\begin{thebibliography}{99}" "\n"
        r"\bibitem{ref1} A, used by \cite{ref1}." "\n"
        r"\end{thebibliography}" "\n"
    )
    assert any("inside thebibliography" in p for p in validate_latex.check(tex))


def test_undefined_reference_warnings_are_errors(tmp_path):
    tex = tmp_path / "d.tex"
    tex.write_text("x\n")
    (tmp_path / "d.log").write_text(
        "LaTeX Warning: Reference `tab:XX' on page 3 undefined on input line 9.\n"
        "LaTeX Warning: Citation `ref99' on page 4 undefined on input line 11.\n")
    found = validate_latex.check_build_log(tex)
    assert len(found) == 2


def test_tqe_mode_sets_the_journal_class_front_matter():
    """--tqe: the TQE template's class, keyword environment and figure macro.

    The template puts the abstract and keywords before \\maketitle, sets figures
    through its own \\Figure macro (a plain figure environment leaves the width
    its caption code reads undefined), and closes with \\EOD. The IEEEtran build
    must be untouched by the flag.
    """
    md = ("# T\n\n## Abstract\n\nA.\n\n**Keywords:** k\n\n## I. Results\n\n"
          "See Fig. 1.\n\n## References\n\n")
    try:
        md_to_latex.set_mode(False, tqe=True)
        assert md_to_latex.OUT.name == "paper01_tqe.tex"
        tex = md_to_latex.convert(md)
    finally:
        md_to_latex.set_mode(False)
    assert tex.startswith("\\documentclass{ieeeaccess}")
    assert tex.index("\\end{keywords}") < tex.index("\\maketitle")
    assert "\\begin{IEEEkeywords}" not in tex
    assert "\\Figure[!t]" in tex and "\\begin{figure" not in tex
    assert "\\label{fig:1}" in tex
    assert tex.rstrip().endswith("\\EOD\n\n\\end{document}")
    assert md_to_latex.OUT.name == "paper01.tex"
    plain = md_to_latex.convert(md)
    assert plain.startswith("\\documentclass[journal]{IEEEtran}")
    assert "\\begin{figure" in plain and "\\Figure" not in plain


def test_the_shipped_documents_pass(tmp_path):
    for name in ("paper01.tex", "paper01_supplement.tex", "paper01_tqe.tex"):
        path = PAPERS / name
        if not path.exists():
            pytest.skip(f"{name} not generated")
        assert validate_latex.check(path) == []
        assert validate_latex.check_aux(path) == []
