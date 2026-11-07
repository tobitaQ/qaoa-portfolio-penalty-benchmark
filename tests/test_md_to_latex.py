# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Regression tests for the Markdown -> LaTeX converter and its validator.

Both defects fixed here reached a built PDF and were found by a human reader,
not by the pipeline: the reference list printed "used by ." and the body's
L-BFGS-B citation printed [48], a number the list does not contain. The paper
goes through repeated revisions, so the checks that catch this class run
here rather than depending on someone re-reading 47 reference entries.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil

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
        r"\end{thebibliography}" "\n", encoding="utf-8")
    problems = validate_latex.check(tex)
    assert any("defined 2 times" in p for p in problems)


def test_printed_number_disagreeing_with_the_draft_is_reported(tmp_path):
    tex = tmp_path / "d.tex"
    tex.write_text(
        r"\cite{ref1}\cite{ref2}" "\n"
        r"\begin{thebibliography}{99}" "\n"
        r"\bibitem{ref2} B." "\n"
        r"\bibitem{ref1} A." "\n"
        r"\end{thebibliography}" "\n", encoding="utf-8")
    assert any("disagree" in p for p in validate_latex.check(tex))


def test_aux_is_the_ground_truth_for_numbering(tmp_path):
    tex = tmp_path / "d.tex"
    tex.write_text("x\n", encoding="utf-8")
    (tmp_path / "d.aux").write_text(
        r"\bibcite{ref45}{45}" "\n" r"\bibcite{ref46}{48}" "\n", encoding="utf-8")
    assert validate_latex.check_aux(tex) == ["reference [46] is typeset as [48]"]


def test_cite_inside_the_bibliography_is_reported(tmp_path):
    tex = tmp_path / "d.tex"
    tex.write_text(
        r"\cite{ref1}" "\n"
        r"\begin{thebibliography}{99}" "\n"
        r"\bibitem{ref1} A, used by \cite{ref1}." "\n"
        r"\end{thebibliography}" "\n", encoding="utf-8")
    assert any("inside thebibliography" in p for p in validate_latex.check(tex))


def test_undefined_reference_warnings_are_errors(tmp_path):
    tex = tmp_path / "d.tex"
    tex.write_text("x\n", encoding="utf-8")
    (tmp_path / "d.log").write_text(
        "LaTeX Warning: Reference `tab:XX' on page 3 undefined on input line 9.\n"
        "LaTeX Warning: Citation `ref99' on page 4 undefined on input line 11.\n", encoding="utf-8")
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
    # \EOD goes at the end of the last paragraph: the biography's last line
    # when there is a biography, the last reference when there is not. After
    # \end{thebibliography} on its own it is a paragraph, which on a full page
    # costs an otherwise empty extra page -- 21 instead of 20, measured on the
    # Access build, where 20 is what the journal asks for.
    tail = tex.split("\\end{thebibliography}")[-1]
    if "\\begin{IEEEbiography" in tex:
        assert tail.count("\\EOD") == 1
        assert tail.index("\\EOD") < tail.index("\\end{IEEEbiography")
    else:
        assert "\\EOD" in tex and "\\EOD" not in tail
    assert md_to_latex.OUT.name == "paper01.tex"
    plain = md_to_latex.convert(md)
    assert plain.startswith("\\documentclass[journal]{IEEEtran}")
    assert "\\begin{figure" in plain and "\\Figure" not in plain


def test_the_shipped_documents_pass(tmp_path):
    for name in ("paper01.tex", "paper01_supplement.tex", "paper01_tqe.tex",
                 "paper01_access.tex"):
        path = PAPERS / name
        if not path.exists():
            pytest.skip(f"{name} not generated")
        assert validate_latex.check(path) == []
        assert validate_latex.check_aux(path) == []


def test_the_biography_carries_the_eod_mark_and_its_photograph_is_checked(tmp_path,
                                                                          monkeypatch):
    """The journal builds include papers/author.tex; the preprint does not.

    The class asks for \\EOD "in the end of the last para", which in an Access
    article is the author biography, and it is the biography that has to carry
    the mark once one exists — on the last reference it would sit in the middle
    of the document. A photograph LaTeX cannot find is reported before the .tex
    is written, because the compile that fails on it happens much later.
    """
    md = ("# T\n\n## Abstract\n\nA.\n\n**Keywords:** k\n\n## I. Results\n\nText.\n\n"
          "## References\n\n[1] A. Author, \"A title,\" 2026.\n\n")
    author = tmp_path / "author.tex"
    author.write_text("\\begin{IEEEbiography}[{\\includegraphics{face.jpg}}]{A Name}\n"
                      "received a degree.\n\n"
                      "He is a member.\n"
                      "\\end{IEEEbiography}\n", encoding="utf-8")
    monkeypatch.setattr(md_to_latex, "AUTHOR", author)
    assert md_to_latex.missing_photographs() == ["face.jpg"]
    (tmp_path / "face.jpg").write_bytes(b"not really a jpeg")
    assert md_to_latex.missing_photographs() == []

    try:
        md_to_latex.set_mode(False, tqe=True)
        with_bio = md_to_latex.convert(md)
        monkeypatch.setattr(md_to_latex, "AUTHOR", tmp_path / "nothing.tex")
        without = md_to_latex.convert(md)
    finally:
        md_to_latex.set_mode(False)

    bio = with_bio.split("\\end{thebibliography}")[-1]
    assert "\\begin{IEEEbiography}" in bio and "He is a member.\\EOD" in bio
    assert "\\bibitem" in with_bio and "\\EOD" not in with_bio.split("\\begin{IEEEbiography}")[0]
    # Without a biography the mark stays on the last reference.
    assert "\\EOD" in without and "\\EOD" not in without.split("\\end{thebibliography}")[-1]
    # The IEEEtran preprint carries neither the biography nor the mark.
    plain = md_to_latex.convert(md)
    assert "IEEEbiography" not in plain and "\\EOD" not in plain


def test_prose_after_a_figure_item_is_refused_not_folded_into_the_caption(tmp_path,
                                                                          monkeypatch):
    """The manifest's items end at a blank line; what follows is not a caption.

    The manifest closed with a note about the supplementary figures, and
    read_manifest joined each chunk of its split wholesale, so the note rode
    into Fig. 3's caption and was printed there: nineteen words describing the
    Supplementary Material under a figure about re-scoring. Prose after an item
    is now refused, with the line to move, because either outcome of keeping it
    — printed in the wrong caption, or silently dropped with the manifest — is
    worse than a failed build.
    """
    md = ("body\n\n### Figure manifest\n\n"
          "- **Fig. 1** — `notebooks/figures/fig_e2_shots.pdf` — a caption.\n\n"
          "A closing note that belongs to no figure.\n")
    with pytest.raises(SystemExit) as exc:
        md_to_latex.read_manifest(md)
    assert "closing note" in str(exc.value)

    ok = ("body\n\n### Figure manifest\n\n"
          "- **Fig. 1** — `notebooks/figures/fig_e2_shots.pdf` — a caption\n"
          "  that wraps onto a second line.\n")
    _rest, figs, _avail = md_to_latex.read_manifest(ok)
    assert figs == [("fig_e2_shots", "a caption that wraps onto a second line.")]


def test_the_built_pdf_carries_every_sentence_of_the_draft():
    """papers/check_rendered.py on the Access build, which is what is submitted.

    The checks that read the Markdown cannot see what the typesetter did with
    it. Two defects lived in that gap: the figure manifest's closing note was
    printed inside Fig. 3's caption, and the first fix for it stopped the
    sentence being printed at all. Both were invisible to every other check.
    """
    check_rendered = _load("check_rendered")
    pdf = PAPERS / "paper01_access.pdf"
    if not pdf.exists() or not shutil.which("pdftotext"):
        pytest.skip("the Access PDF or pdftotext is not available here")
    assert check_rendered.main([str(pdf)]) == 0


def test_every_display_formula_of_the_draft_is_hand_set():
    """A block quote in the draft is a display equation, and has a rendering.

    The keys of FORMULAS were written with the runs of spaces the draft used to
    align its formulas. A re-wrap collapsed them, and eight of the ten
    equations fell back to the inline converter, which sets "Σ_{i<j}" as
    literal text in a quote block. The lookup ignores spacing now, and this
    fails if a formula is edited without its rendering following.
    """
    draft = (PAPERS / "paper01_draft.md").read_text(encoding="utf-8")
    quotes = [" ".join(line[2:].split())
              for line in draft.split("\n") if line.startswith("> ")]
    assert quotes, "the draft has no display formulas; the check is looking in the wrong place"
    missing = [q for q in quotes if q not in md_to_latex.FORMULAS]
    assert not missing, ("these display formulas have no hand-set rendering and would be "
                         f"set by the inline fallback: {missing}")


def test_operators_and_indices_the_page_images_caught():
    """Four more markup losses, all of them visible in the PDF's text layer.

    A subscript the generic rule did not reach printed its underscore ("γ_ℓ",
    "p_⊥"); a summation with a braced index printed the braces too ("Σ_{x∈F}",
    and the same sum was set \\sum in the display equation above it); "~" meaning
    "distributed as" was set as a tilde accent; and an unbraced exponent took
    the sentence's full stop with it, so "anywhere in 2^N." ended without one.
    """
    out = md_to_latex.inline("γ_ℓ, β_ℓ ~ U(0, π) and p_⊥ = 1 − P_F")
    assert r"\gamma_{\ell}" in out and r"\beta_{\ell}" in out
    assert r"p_{\perp}" in out
    assert r"\sim" in out and r"\~{}" not in out

    out = md_to_latex.inline("½ Σ_{x∈F} p(x) and C(N, K)^{−1/2} Σ_{|x| = K} and ∏_k a_k")
    assert r"$\sum_{x\in F}$" in out
    assert r"$\sum_{|x| = K}$" in out
    assert r"$\prod_{k}$" in out
    assert r"\_" not in out  # an escaped underscore is one that printed

    out = md_to_latex.inline("amplitude anywhere in 2^N. E6 reverses that choice")
    assert "$^{N}$." in out, out


def test_a_bare_url_is_set_as_a_link_that_breaks_without_a_hyphen():
    """Reference [40]'s URL broke at one of its own hyphens and died there.

    Set as ordinary text, LaTeX broke
    "…/artifact-review-and-badging-current" after "badging-", so the printed
    line ends in a hyphen a reader cannot tell from the URL's own, and every
    extractor drops it: the copied link resolves to nothing. It was not
    clickable either, since nothing made it a link.
    """
    out = md_to_latex.inline("Available: https://www.acm.org/policies/artifact-review-and-badging-current (accessed Aug. 8, 2026).")
    assert r"\url{https://www.acm.org/policies/artifact-review-and-badging-current}" in out
    assert r"\urlstyle{rm}" in out          # the reference list's own font
    assert "(accessed Aug. 8, 2026)." in out  # the trailing note stays outside
    # a URL inside a code span keeps the typewriter form it had
    assert r"\url{https://example.org/x}" in md_to_latex.inline("`https://example.org/x`")
