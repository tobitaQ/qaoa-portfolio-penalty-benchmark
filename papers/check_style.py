# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Count the habits that make a passage read as machine-written.

The editor who declined the first submission wrote that the text "reads very
mechanical in English and will be easily identified by readers as having been
primarily composed by genAI". That judgement is a reader's, but the habits
behind it are countable, and counting them is the part a tool can do: how long
the sentences are and how little they vary, how often a gloss is set off by a
pair of em dashes, how often a claim is made by denying its opposite, how often
three things are listed in parallel.

The counts are not a target to optimise. A passage with none of these can still
be unreadable, and one with several can be fine. They exist so that a revision
can be compared with what it replaced, and so that "this reads mechanical" has
something underneath it.

Sentences split on full stops only. An earlier count split on semicolons too
and reported an eighty-nine-word sentence as three.

Usage:
    python papers/check_style.py FILE [FILE ...]
    python papers/check_style.py --section papers/paper01_draft.md   # per heading
"""

from __future__ import annotations

import pathlib
import re
import sys

DASH = re.compile(r"—[^—]{4,100}—")          # — gloss —
NOT_X = re.compile(r",\s+(?:not|rather than)\s+", re.I)      # X, not Y
TRIAD = re.compile(r"\b\w+,\s+\w+,?\s+and\s+\w+\b")          # a, b and c
HEDGE = re.compile(r"\b(?:it is worth|notably|importantly|crucially|"
                   r"that said|in other words)\b", re.I)
META = re.compile(r"\b(?:three|four|five|six|seven)\s+contributions?\b", re.I)


def sentences(text: str) -> list[str]:
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"(\d)\.(\d)", r"\1․\2", t)              # keep 2.6 whole
    out = [s.strip().replace("․", ".") for s in re.split(r"(?<=[.!?])\s+(?=[A-Z“])", t)]
    return [s for s in out if len(s.split()) > 3]


def report(name: str, text: str) -> dict:
    s = sentences(text)
    if not s:
        return {}
    n = sorted(len(x.split()) for x in s)
    return {"name": name, "words": sum(n), "sents": len(s),
            "median": n[len(n) // 2], "longest": n[-1],
            "over40": sum(1 for v in n if v > 40), "under15": sum(1 for v in n if v < 15),
            "dash": len(DASH.findall(text)), "not_x": len(NOT_X.findall(text)),
            "triad": len(TRIAD.findall(text)), "hedge": len(HEDGE.findall(text)),
            "meta": len(META.findall(text))}


def show(rows: list[dict]) -> None:
    head = ("name", "words", "sents", "median", "longest", "over40", "under15",
            "dash", "not_x", "triad", "hedge", "meta")
    w = [max(len(h), *(len(str(r[h])) for r in rows)) for h in head]
    print("  ".join(h.rjust(c) for h, c in zip(head, w)))
    for r in rows:
        print("  ".join(str(r[h]).rjust(c) for h, c in zip(head, w)))
    print("\nlongest sentence per passage:")
    for r in rows:
        print(f"  {r['name']}: {r['longest']} words")


def main(argv: list[str]) -> int:
    per_section = "--section" in argv
    paths = [pathlib.Path(a) for a in argv if not a.startswith("-")]
    if not paths:
        print(__doc__.strip().split("Usage:")[1].strip(), file=sys.stderr)
        return 2
    rows = []
    for p in paths:
        text = p.read_text(encoding="utf-8")
        if per_section:
            parts = re.split(r"^##\s+(.+)$", text, flags=re.M)
            for title, body in zip(parts[1::2], parts[2::2]):
                r = report(title.strip()[:28], body)
                if r:
                    rows.append(r)
        else:
            r = report(p.stem[:28], text)
            if r:
                rows.append(r)
    show(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
