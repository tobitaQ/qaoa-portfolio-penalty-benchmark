# The IEEE Access class and its header logo

This directory is what `papers/build_access.sh` puts on `TEXINPUTS` when it
builds `paper01_access.pdf`, the submitted version.

| File | What it is |
|---|---|
| `ieeeaccess.cls` | the journal class, byte-identical to `../tqe_class/ieeeaccess.cls` — both templates ship the same file |
| `Logo.png` | the IEEE Access masthead with its tagline, from the Access template |
| `notaglineLogo.png` | the same logo without the tagline |
| `bullet.png` | the end-of-document mark `\EOD` draws |

**The names are case-sensitive.** The class loads `Logo.png` and
`notaglineLogo.png` by name; on Linux a file called `logo.png` is simply not
found, and pdflatex then writes no PDF at all.

`../tqe_class/` holds the same four files with the *TQE* masthead, which is what
the TQE submission was built with. A PDF built against that directory prints
"IEEE Transactions on Quantum Engineering" across every page while the DOI line
reads `10.1109/ACCESS.2017.DOI` — the mismatch a reviewer of the Access draft
caught. `build_access.sh` prefers this directory, falls back to `../tqe_class/`
and prints which masthead the PDF it just wrote carries.
