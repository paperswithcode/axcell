#!/usr/bin/env bash
OUTNAME="$1"
echo $OUTNAME
RO_SOURCE_DIR="/files/ro-source"
OUTPUT_DIR="/files/htmls"

timeout -s KILL 120 engrafo "$RO_SOURCE_DIR" /files/output

cp /files/output/index.html "$OUTPUT_DIR/$OUTNAME"
