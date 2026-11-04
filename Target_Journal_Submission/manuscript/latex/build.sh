#!/usr/bin/env bash
# Build main manuscript and supplement (cross-referenced with xr-hyper), the Word version, and run layout checks.
#   bash build.sh            # PDFs + checks
#   bash build.sh --word     # also the Word version (pandoc + post-processing)
set -euo pipefail
cd "$(dirname "$0")"
python3 make_bibtex.py >/dev/null
python3 make_deviations.py >/dev/null
run() { pdflatex -interaction=nonstopmode -halt-on-error "$1" >/dev/null || { tail -40 "$1.log"; exit 1; }; }
run supplement; bibtex supplement >/dev/null || true
run main; bibtex main >/dev/null || true
for i in 1 2; do run supplement; run main; done

report=build_report.txt
{
  echo "Build report $(date -u +%Y-%m-%dT%H:%MZ)"
  for d in main supplement; do
    pages=$(pdfinfo $d.pdf | awk '/^Pages/{print $2}')
    overfull=$(grep -c "^Overfull" $d.log || true)
    underfull=$(grep -c "^Underfull" $d.log || true)
    undef_ref=$(grep -c "Reference .* undefined" $d.log || true)
    undef_cit=$(grep -c "Citation .* undefined" $d.log || true)
    multiply=$(grep -c "multiply defined" $d.log || true)
    leak=$(pdftotext $d.pdf - | grep -cE '\\[A-Za-z]{3,}|\?\?' || true)
    echo "$d: pages=$pages overfull=$overfull underfull=$underfull undefined_refs=$undef_ref undefined_cites=$undef_cit multiply_defined=$multiply macro_or_??_leaks=$leak"
  done
  bibwarn=$(grep -c "^Warning" main.blg || true)
  echo "bibtex warnings (main): $bibwarn"
} | tee $report

if [[ "${1:-}" == "--word" ]]; then
  python3 make_word.py
fi
