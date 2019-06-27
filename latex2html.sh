#!/usr/bin/env bash
OUTNAME="$1"
echo $OUTNAME
RO_SOURCE_DIR="/files/ro-source"
SOURCE_DIR="/files/source"
OUTPUT_DIR="/files/htmls"

cp -r "$RO_SOURCE_DIR" "$SOURCE_DIR"
cd "$SOURCE_DIR"
MAINTEX=$(find . -type f -iname "*.tex" -print0 | xargs -0 grep -l documentclass | head -1)
echo $MAINTEX
timeout -s KILL 60 htlatex "$MAINTEX" '' '' '' '-interaction=nonstopmode'

FILENAME=$(basename $MAINTEX)
FILENAME="${FILENAME%.tex}.html"
cp "$SOURCE_DIR/$FILENAME" "$OUTPUT_DIR/$OUTNAME"
