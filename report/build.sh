#!/usr/bin/env bash
# Build report.pdf from report.md (needs pandoc and a TeX distribution with xelatex).
set -euo pipefail
cd "$(dirname "$0")"
pandoc report.md -o report.pdf --pdf-engine=xelatex --toc \
  -V geometry:margin=2cm -V fontsize=11pt \
  -V mainfont="Times New Roman" -V monofont="Menlo" -V monofontoptions="Scale=0.78" \
  -V colorlinks=true -V linkcolor=blue -H header.tex
echo "-> report.pdf"
