#!/usr/bin/env bash
OUTNAME="$1"
echo $OUTNAME
RO_SOURCE_DIR="/files/ro-source"
SOURCE_DIR="/files/source"
OUTPUT_DIR="/files/htmls"

cp -r "$RO_SOURCE_DIR" "$SOURCE_DIR"
find "$SOURCE_DIR" -iname '*.tex' -print0 | xargs -0 sed -i \
  -e 's/\\begin{document}/\\usepackage{verbatim}\0/g' \
  -e 's/\\begin\(\[[^]]*\]\)\?{tikzpicture}/\\begin{comment}/g' \
  -e 's/\\end{tikzpicture}/\\end{comment}/g'

if [ -f "$SOURCE_DIR/ms.tex" ]
then
  MAINTEX="$SOURCE_DIR/ms.tex"
elif [ -f "$SOURCE_DIR/main.tex" ]
then
  MAINTEX="$SOURCE_DIR/main.tex"
elif [ -f "$SOURCE_DIR/00_main.tex" ]
then
  MAINTEX="$SOURCE_DIR/00_main.tex"
else
  MAINTEX=$(find "$SOURCE_DIR" -maxdepth 1 -type f -iname "*.tex" -print0 | xargs -0 grep -l documentclass | head -1)
fi
timeout -s KILL 300 engrafo "$MAINTEX" /files/output

cp /files/output/index.html "$OUTPUT_DIR/$OUTNAME"
