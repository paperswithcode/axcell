#!/usr/bin/env bash
OUTPUT_DIR=$(realpath "$1")
ARCHIVE=$(realpath "$2")
FILENAME=$(basename "$ARCHIVE")

docker run --rm --stop-timeout 60 -v $PWD/latex2html.sh:/files/latex2html.sh:ro -v "$ARCHIVE":/files/ro-source:ro -v "$OUTPUT_DIR":/files/htmls niccokunzmann/ci-latex /files/latex2html.sh "$FILENAME"
