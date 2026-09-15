# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
The audit's citation numbers must agree with the paper's reference list.

`audit_census.csv` carries a `ref` column -- the number the paper gives each
audited work -- and `audit_tally.py` and the review page print it. Nothing tied
it to the bibliography, so when the references were renumbered the census kept
the old numbers: all fifteen cited rows pointed somewhere else, [37] naming
Hegade while the paper's [37] is Kirkpatrick. Table I survived because it
reports counts rather than numbers, but the review page spent that whole time
showing the author the wrong citation beside each verdict -- during the check
the page exists to support.

The mapping is derived here rather than trusted, so a renumbering breaks the
suite instead of the reader's trail.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "papers" / "paper01_draft.md"
CENSUS = ROOT / "papers" / "audit_census.csv"

#: A distinctive fragment of each audited paper's bibliography entry. Author
#: surnames alone are ambiguous -- two Lozano entries, two with Mugel, two with
#: Wu -- so each pattern names the work, not just the person.
SIGNATURE = {
    "Stopfer & Wagner": r"Stopfer.*extensive benchmark",
    "Brandhofer et al.": r"Brandhofer",
    "Lozano (operational audit)": r"Lozano.*operational decomposition",
    "Baker & Radha": r"Baker.*Wasserstein",
    "Rosenberg et al.": r"Rosenberg.*optimal trading",
    "Venturelli & Kondratyev": r"Venturelli and A\. Kondratyev",
    "Mugel et al.": r"Mugel \*et al\.\*.*Dynamic portfolio",
    "Lozano": r"Lozano.*penalty-free pipeline",
    "Mancilla et al.": r"Mancilla",
    "Hegade et al.": r"Hegade",
    "Oralkhan & Zhaxalykov": r"Oralkhan",
    "Wu & Wang": r"Wu and L\. Wang",
    "Innan et al.": r"Innan",
    "Uotila et al.": r"Uotila",
    "Hodson et al.": r"Hodson",
}

pytestmark = pytest.mark.skipif(not DRAFT.exists(), reason="draft not present")


def bibliography() -> dict[int, str]:
    """Reference number -> entry text, from the draft's reference list."""
    refs = DRAFT.read_text().split("## References", 1)[1]
    return {int(m.group(1)): " ".join(m.group(2).split())
            for m in re.finditer(r"^\[(\d+)\]\s*(.+?)(?=\n\[\d+\]|\Z)", refs,
                                 re.S | re.M)}


def rows() -> list[dict]:
    with CENSUS.open() as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("row", rows(), ids=lambda r: r["short"])
def test_census_ref_points_at_the_right_paper(row: dict) -> None:
    """Each row's `ref` must resolve to that row's paper in the bibliography."""
    short, ref = row["short"], row["ref"].strip()
    if short not in SIGNATURE:
        assert ref == "-", (
            f"{short} is not cited by the paper, so its ref should be '-', not {ref!r}")
        return

    assert ref.isdigit(), f"{short} should carry a reference number, found {ref!r}"
    entry = bibliography().get(int(ref))
    assert entry is not None, f"[{ref}] ({short}) is not in the reference list"
    assert re.search(SIGNATURE[short], entry), (
        f"[{ref}] should be {short}, but the reference list has: {entry[:80]}")


def test_every_audited_paper_has_a_distinct_number() -> None:
    """Two rows sharing a number would mean one of them is miscited."""
    cited = [r["ref"] for r in rows() if r["ref"].strip() != "-"]
    assert len(cited) == len(set(cited)), f"duplicate reference numbers: {cited}"


def test_signatures_cover_every_cited_row() -> None:
    """A row added without a signature would skip the check above silently."""
    uncovered = [r["short"] for r in rows()
                 if r["ref"].strip() != "-" and r["short"] not in SIGNATURE]
    assert not uncovered, f"add a bibliography signature for: {uncovered}"
