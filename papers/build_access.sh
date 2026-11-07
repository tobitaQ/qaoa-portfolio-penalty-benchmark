#!/bin/bash
# Build the IEEE Access PDF of paper ① — the submitted build.
#
# Why a script: the class and the three images it loads by name live in
# papers/tqe_class/, so a plain `pdflatex paper01_access` dies on
# "File `ieeeaccess.cls' not found", writes no PDF, and leaves the previous one
# on disk. A page count read after that is the old file's, which is exactly what
# happened: two "20 pages" checks were reported off a PDF built hours earlier.
# This script sets TEXINPUTS, fails on a LaTeX error or an undefined reference,
# and prints the page count of the file it just wrote.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
"$PY" papers/md_to_latex.py --access
cd papers
# The class is the same file in both templates; the header logo is not. Prefer
# papers/access_class/ (the Access logo) and say which one the PDF carries, so
# a submitted Access PDF cannot quietly print the TQE masthead.
if [ -f access_class/ieeeaccess.cls ] && [ -f access_class/Logo.png ]; then
  CLASSDIR=./access_class; HEADER="IEEE Access"
else
  CLASSDIR=./tqe_class; HEADER="IEEE Transactions on Quantum Engineering (WRONG for an Access submission: see papers/access_class/README.md)"
fi
for _ in 1 2; do
  TEXINPUTS="$CLASSDIR:" pdflatex -interaction=nonstopmode paper01_access >/dev/null 2>&1 || true
done
errors=$(grep -c '^!' paper01_access.log || true)
undef=$(grep -c 'LaTeX Warning: \(Reference\|Citation\).*undefined' paper01_access.log || true)
[ -f paper01_access.pdf ] || { echo "no PDF was produced; see papers/paper01_access.log"; exit 1; }
if [ "$errors" != 0 ] || [ "$undef" != 0 ]; then
  echo "$errors error(s), $undef undefined reference(s) — see papers/paper01_access.log"
  grep -m5 '^!' paper01_access.log || true
  exit 1
fi
# Only text-level overfull boxes are reported. The class's page furniture logs
# ~70 "while \output is active" boxes of 505pt in every build, including the
# official template's, so counting those hides the few that are ours.
text_overfull=$(grep '^Overfull' paper01_access.log | grep -c 'in paragraph' || true)
echo "paper01_access.pdf: $(pdfinfo paper01_access.pdf | awk '/^Pages:/ {print $2}') pages, \
$text_overfull overfull box(es) in text, \
biography: $(grep -c 'begin{IEEEbiography' paper01_access.tex || true)"
echo "header logo: $HEADER"
