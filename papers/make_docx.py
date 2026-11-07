# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Abstract, Introduction and Conclusion of the draft as one Word file.

The sections are taken from `paper01_draft.md`, so the file cannot drift from
the manuscript: it is generated, never edited by hand. The Markdown is set as
HTML -- subscripts and superscripts as real `<sub>`/`<sup>` runs, because
"epsilon_F" written with an underscore is what the LaTeX conversion spent this
month removing -- and LibreOffice writes the .docx.

    python papers/make_docx.py
    python papers/make_docx.py --sections "## Abstract" "## VIII. Conclusion"

Needs `soffice` (LibreOffice). Without it the HTML is still written and the
script says what to run.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
DRAFT = ROOT / "paper01_draft.md"
OUT = ROOT / "paper01_abstract_intro_conclusion"

#: Default sections, in the order the Word file presents them.
SECTIONS = ["## Abstract", "## I. Introduction", "## VIII. Conclusion"]

#: Greek the draft writes as Unicode and that can carry an index.
GREEK = "ΣμλγβασρφψεΔτ"

CODE, STAR = "\x01", "\x02"


def sections(md: str, wanted: list[str]) -> list[tuple[str, str]]:
    """The body of each requested heading, in the order requested."""
    blocks: dict[str, str] = {}
    head, body = None, []
    for line in md.split("\n"):
        if line.startswith("## "):
            if head:
                blocks[head] = "\n".join(body)
            head, body = line.rstrip(), []
        elif head:
            body.append(line)
    if head:
        blocks[head] = "\n".join(body)
    missing = [w for w in wanted if w not in blocks]
    if missing:
        raise SystemExit(f"{DRAFT.name} has no section {missing}; "
                         f"it has {sorted(blocks)[:12]}")
    return [(w[3:], blocks[w]) for w in wanted]


def inline(text: str) -> str:
    """One paragraph of Markdown as HTML."""
    held: list[str] = []
    text = text.replace(r"\*", STAR)

    def _hold(m):
        held.append(m.group(1))
        return f"{CODE}{len(held) - 1}{CODE}"

    text = re.sub(r"`([^`]*)`", _hold, text)
    text = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    # "P_F", "A_crit", "x_A₂": a real subscript, as the PDF sets it.
    text = re.sub(rf"(?<![A-Za-z0-9_])([A-Za-z{GREEK}])_(\d+\.\d+|[A-Za-z0-9τ₀-₉⊥∇ℓ]{{1,9}})"
                  r"(?![\w]|\.\w)", r"\1<sub>\2</sub>", text)
    # "2^N", "10^{-9}", "C(N, K)^{−1/2}".
    text = re.sub(r"\^\{([^}]*)\}|\^([A-Za-z0-9.+-]*[A-Za-z0-9])", 
                  lambda m: "<sup>" + (m.group(1) or m.group(2)) + "</sup>", text)
    text = text.replace(STAR, "*")
    for i, code in enumerate(held):
        text = text.replace(f"{CODE}{i}{CODE}", f"<code>{code}</code>")
    return text


def paragraphs(body: str) -> list[str]:
    """Blank-line separated blocks, each re-joined into one line."""
    out, buf = [], []
    for line in body.split("\n"):
        if line.strip() in ("", "---"):
            if buf:
                out.append(" ".join(" ".join(buf).split()))
                buf = []
        else:
            buf.append(line.strip())
    if buf:
        out.append(" ".join(" ".join(buf).split()))
    return out


def html(title: str, blocks: list[tuple[str, str]]) -> str:
    parts = ["<!DOCTYPE html>", '<html><head><meta charset="utf-8">',
             f"<title>{title}</title>",
             "<style>body{font-family:'Times New Roman',serif;font-size:11pt;"
             "line-height:1.5} h1{font-size:15pt} h2{font-size:12pt}"
             "code{font-family:'Courier New',monospace;font-size:10pt}</style>",
             "</head><body>", f"<h1>{title}</h1>"]
    for head, body in blocks:
        parts.append(f"<h2>{head}</h2>")
        for para in paragraphs(body):
            if para.startswith("- "):
                parts.append("<p>" + inline(para[2:]) + "</p>")
            else:
                parts.append("<p>" + inline(para) + "</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sections", nargs="+", default=SECTIONS)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    md = DRAFT.read_text(encoding="utf-8")
    title = next(ln[2:].strip() for ln in md.split("\n") if ln.startswith("# "))
    out = pathlib.Path(args.out)
    page = html(title, sections(md, args.sections))
    src = out.with_suffix(".html")
    src.write_text(page, encoding="utf-8")

    if not shutil.which("soffice"):
        print(f"wrote {src}; LibreOffice is not installed, so no .docx was made")
        return 0
    profile = out.parent / ".soffice-profile"
    subprocess.run(["soffice", "--headless", f"-env:UserInstallation=file://{profile}",
                    "--convert-to", "docx:MS Word 2007 XML",
                    "--outdir", str(out.parent), str(src)],
                   check=True, capture_output=True)
    shutil.rmtree(profile, ignore_errors=True)
    src.unlink()
    docx = out.with_suffix(".docx")
    if not docx.exists():
        raise SystemExit("LibreOffice reported success but wrote no .docx")
    words = sum(len(p.split()) for _, b in sections(md, args.sections)
                for p in paragraphs(b))
    print(f"wrote {docx} ({', '.join(s[3:] for s in args.sections)}; "
          f"{words} words, {docx.stat().st_size // 1024} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
