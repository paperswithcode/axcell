#!/usr/bin/env bash
OUTNAME="$1"
echo $OUTNAME
SOURCE_DIR="/files/ro-source"
OUTPUT_DIR="/files/htmls"

cd "$SOURCE_DIR"

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
  MAINTEX=$(find $SOURCE_DIR -maxdepth 1 -type f -iname "*.tex" -print0 | xargs -0 grep -l documentclass | head -1)
fi
timeout -s KILL 300 engrafo "$MAINTEX" /files/output

cp /files/output/index.html "$OUTPUT_DIR/$OUTNAME"
