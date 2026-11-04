#!/usr/bin/env bash
# Assemble submission/LOOM_Submission.zip and test that the LaTeX source compiles from the zip alone.
set -euo pipefail
cd "$(dirname "$0")"
SUB=../../submission; NAME=LOOM; STAGE=$(mktemp -d); SRC=$STAGE/latex_source
mkdir -p $SUB $SRC/figures $SRC/tables
for f in main.tex supplement.tex sn-jnl.cls sn-basic.bst references.bib deviations_table.tex; do cp $f $SRC/; done
cp ../../code/tables/macros.tex ../../code/tables/tab_*.tex $SRC/tables/
for f in $(grep -ohE 'includegraphics\[[^]]*\]\{[^}]*\}' main.tex supplement.tex | sed -E 's/.*\{([^}]*)\}/\1/'); do cp ../../code/figures/$f $SRC/figures/; done
sed -i 's#\.\./\.\./code/figures/#figures/#; s#\.\./\.\./code/tables/#tables/#g' $SRC/main.tex $SRC/supplement.tex
(cd $SRC && zip -qr ../5_LaTeX_Source.zip .)
# test: compile from a fresh extraction of the zip
T=$(mktemp -d); (cd $T && unzip -q $STAGE/5_LaTeX_Source.zip && for i in 1; do pdflatex -interaction=nonstopmode supplement >/dev/null; bibtex supplement >/dev/null; pdflatex -interaction=nonstopmode main >/dev/null; bibtex main >/dev/null; pdflatex -interaction=nonstopmode supplement >/dev/null; pdflatex -interaction=nonstopmode main >/dev/null; pdflatex -interaction=nonstopmode main >/dev/null; done
  echo "zip test: main $(pdfinfo main.pdf | awk '/Pages/{print $2}') pages, supplement $(pdfinfo supplement.pdf | awk '/Pages/{print $2}') pages, undefined refs: $(grep -c 'undefined' main.log || true)")
cp main.pdf $STAGE/1_Main_Manuscript.pdf; cp supplement.pdf $STAGE/2_Supplementary_Material.pdf
cp cover_letter.docx $STAGE/3_Cover_Letter.docx; cp cover_letter.pdf $STAGE/3_Cover_Letter.pdf
cp main_word.docx $STAGE/4_Main_Manuscript_Word_version.docx
cp title_page.pdf $STAGE/6_Title_Page.pdf; cp title_page.docx $STAGE/6_Title_Page.docx
cp submission_README.txt $STAGE/README.txt
rm -rf $SRC
(cd $STAGE && zip -qr - .) > $SUB/${NAME}_Submission.zip
unzip -l $SUB/${NAME}_Submission.zip
