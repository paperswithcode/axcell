#!/usr/bin/env bash
ARCHIVES_DIR="$1"
SOURCES_DIR="$2"

mkdir -p "$SOURCES_DIR"

for archive in "$ARCHIVES_DIR"/????.*
do
  mime_type=$(file --brief --mime-type "$archive")
  if [ "$mime_type" != "application/gzip" ]
  then
    echo "File '$archive' of type '$mime_type' is not a gzip archive. Skipping" 1>&2
  else
    filename=$(basename -- "$archive")
    outdir="$SOURCES_DIR/src-$filename"
    mkdir -p "$outdir"
    tar -xf "$archive" -C "$outdir"
  fi
done
