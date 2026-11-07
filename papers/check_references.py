# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Does every reference that carries an identifier resolve to what it claims?

`validate_latex.py` checks that the citations and the bibliography agree with
each other; it cannot tell whether an entry describes a paper that exists. The
first 29 entries were checked by hand against publisher and arXiv records in
August 2026, and the list has since grown to 49. This asks the registries.

For a DOI it reads Crossref and compares the title, the first author's surname
and the year; for an arXiv identifier it reads the arXiv API and compares the
title. Agreement is on letters and digits only, because "Mean–variance" and
"Mean-variance" are the same title and the registries disagree about dashes,
case and LaTeX accents.

Entries with neither identifier (web pages, a developer guide, an ACM policy,
four conference papers given by volume and pages) are listed as unchecked
rather than silently passed.

    python papers/check_references.py
    python papers/check_references.py --verbose

Needs network access. Without it the script says so and exits 0, because a
check that cannot run must not look like a check that passed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
DRAFT = ROOT / "paper01_draft.md"
MAILTO = "tobita-hiroaki@aiit.ac.jp"      # Crossref asks callers to identify themselves
TIMEOUT = 20


def entries(text: str) -> list[tuple[int, str]]:
    """(number, full text) for every bibliography entry, continuation lines joined."""
    out: list[tuple[int, str]] = []
    cur: list | None = None
    for line in text.split("## References", 1)[1].split("\n"):
        m = re.match(r"^\[(\d+)\] (.*)", line)
        if m:
            if cur:
                out.append((cur[0], cur[1]))
            cur = [int(m.group(1)), m.group(2)]
        elif cur and line.strip() and not line.startswith(("#", "-")):
            cur[1] += " " + line.strip()
        elif cur and not line.strip():
            out.append((cur[0], cur[1]))
            cur = None
    if cur:
        out.append((cur[0], cur[1]))
    return out


def letters(s: str) -> str:
    """Letters and digits, lowercased, accents folded: how two titles are compared.

    Crossref returns some titles with MathML in them -- Phys. Rev. A's "XY
    mixers" arrives wrapped in <mml:math> -- so the markup comes out first.
    """
    s = re.sub(r"<[^>]+>", "", s)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s.lower() if c.isalnum())


def quoted_title(entry: str) -> str | None:
    m = re.search(r"[\"“]([^\"”]+)[\"”]", entry)
    return m.group(1).rstrip(",. ") if m else None


def surname(entry: str) -> str | None:
    """The first author's surname: entries read "A. B. Surname, C. D. Other, ..."."""
    m = re.match(r"\s*(?:[A-Z]\.\s*)+([A-ZÄÖÜÅ][\w'’\-]+)", entry)
    return m.group(1) if m else None


def get(url: str) -> dict | str | None:
    req = urllib.request.Request(url, headers={"User-Agent": f"check_references (mailto:{MAILTO})"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            body = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    return json.loads(body) if url.startswith("https://api.crossref.org") else body


def check_doi(doi: str, entry: str) -> list[str]:
    data = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    if data is None:
        return ["unreachable"]
    msg = data.get("message", {})
    bad = []
    title = (msg.get("title") or [""])[0]
    want = quoted_title(entry)
    if want and letters(want) not in letters(title) and letters(title) not in letters(want):
        bad.append(f'title is "{title}"')
    fam = [a.get("family", "") for a in msg.get("author", [])]
    want_fam = surname(entry)
    if want_fam and fam and letters(want_fam) != letters(fam[0]):
        bad.append(f"first author is {fam[0]}")
    year = (msg.get("issued", {}).get("date-parts") or [[None]])[0][0]
    m = re.search(r"\b(19|20|21)\d{2}\b", entry)
    if year and m and abs(int(m.group(0)) - int(year)) > 1:
        bad.append(f"year is {year}")
    return bad


def check_arxiv(aid: str, entry: str) -> list[str]:
    body = get("http://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(aid))
    if body is None:
        return ["unreachable"]
    m = re.search(r"<entry>.*?<title>(.*?)</title>", body, re.S)
    if not m:
        return ["no such arXiv identifier"]
    title = " ".join(m.group(1).split())
    want = quoted_title(entry)
    if want and letters(want) not in letters(title) and letters(title) not in letters(want):
        return [f'title is "{title}"']
    return []


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    refs = entries(DRAFT.read_text(encoding="utf-8"))
    if get("https://api.crossref.org/works/10.1145/103162.103163") is None:
        print("no network access to the registries; this check did not run")
        return 0

    checked, unchecked, bad = 0, [], []
    for n, entry in refs:
        doi = re.search(r"doi:\s*(10\.\S+?)[.,]?$|doi:\s*(10\.\S+?)[.,]\s", entry + " ")
        axv = re.search(r"arXiv:(\d{4}\.\d{4,5})", entry)
        if doi:
            ident = (doi.group(1) or doi.group(2)).rstrip(".,")
            problems = check_doi(ident, entry)
        elif axv:
            ident = axv.group(1)
            problems = check_arxiv(ident, entry)
        else:
            unchecked.append(f"[{n}] {entry[:70]}")
            continue
        checked += 1
        if problems:
            bad.append(f"[{n}] {ident}: " + "; ".join(problems) + f"\n     entry: {entry[:100]}")
        elif args.verbose:
            print(f"  [{n}] {ident} ok")
        time.sleep(0.2)

    print(f"{checked} of {len(refs)} references resolve to the registries, {len(bad)} disagree")
    for b in bad:
        print("  x " + b)
    print(f"{len(unchecked)} carry no identifier and are not machine-checkable:")
    for u in unchecked:
        print("    " + u)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
