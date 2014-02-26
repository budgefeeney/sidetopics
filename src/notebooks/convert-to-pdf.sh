#!/bin/bash

IPYTHON=ipython3-3.3
PDFLATEX=/usr/texbin/pdflatex
SED=sed

inFile=$1
outFileTex="${inFile%.ipynb}.tex"

# Convert to Tex
$IPYTHON nbconvert --to=latex --template=article_nocode.tplx $inFile

# Fix the Tex Conversion Errors

$SED -i -e 's/Unknown Author/Bryan Feeney/g' $outFileTex
$SED -i -e 's/max size={\\textwidth}{\\textheight}/max height=13cm,max width=\\textwidth/g'  $outFileTex

$PDFLATEX $outFileTex



