# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Regression tests for the Markdown -> acmart converter of the system paper.

The converter had no tests, and that is how it broke. It shares ``inline()``
with the numerical paper's converter, and when that pass started holding the
draft's own braces behind sentinels -- so that a set literal such as
``N in {12, 16, 20}`` survives -- it also escaped the braces of macros this
converter had already inserted. ``cite()`` ran first, so every bibliography
key came out as a literal ``\\cite\\{ref1\\}``. The reference guard refused to
write the LaTeX, which is the right failure, but it refused for two releases:
the shipped release carries a draft its own converter cannot rebuild. These
tests pin the pipeline order instead of the symptom.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def acm():
    """The system paper's converter, loaded by path like the paper's own tools."""
    spec = importlib.util.spec_from_file_location(
        "md_to_acm", ROOT / "papers" / "system_paper" / "md_to_acm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_citations_keep_their_macro_braces(acm) -> None:
    """A bracketed number becomes a \\cite whose braces LaTeX still reads."""
    out = acm.prose("The platform of [1] and the tracker of [2], [3].")
    assert r"\cite{ref1}" in out
    assert r"\cite{ref2}" in out and r"\cite{ref3}" in out
    assert r"\cite\{" not in out


def test_citation_with_a_locator_keeps_its_optional_argument(acm) -> None:
    """The locator reaches the optional argument as written, braces intact."""
    out = acm.prose("as [4, Sec. 3.2] reports")
    assert r"\cite[Sec. 3.2]{ref4}" in out
    assert r"\cite[" in out and r"\cite\[" not in out


def test_the_drafts_own_braces_are_escaped_exactly_once(acm) -> None:
    """Set literals reach the PDF as braces, not as a line break plus a group."""
    out = acm.prose("sizes N in {12, 16, 20} were run")
    assert r"\{12, 16, 20\}" in out
    assert r"\\{" not in out and r"\\}" not in out


def test_the_whole_draft_converts_and_every_entry_is_cited(acm) -> None:
    """The guard that caught the escaped keys, run over the real manuscript."""
    md = acm.SRC.read_text(encoding="utf-8")
    assert acm.check_references(md, acm.convert(md)) == []
