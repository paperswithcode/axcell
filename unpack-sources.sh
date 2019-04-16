#!/usr/bin/env bash
ARCHIVES_DIR="$1"
SOURCES_DIR="$2"

mkdir -p "$SOURCES_DIR"

for archive in "$ARCHIVES_DIR"/????.*
do
  mime_type=$(file --brief --uncompress --mime-type "$archive")
  case "$mime_type" in
    'application/x-tar')
      filename=$(basename -- "$archive")
      outdir="$SOURCES_DIR/$filename"
      mkdir -p "$outdir"
      tar -xf "$archive" -C "$outdir"
      ;;
    'text/x-tex')
      filename=$(basename -- "$archive")
      outdir="$SOURCES_DIR/$filename"
      mkdir -p "$outdir"
      gunzip -c "$archive" > "$outdir/main.tex"
      ;;
    'application/pdf')
      echo "File '$archive' is a PDF file, not a LaTeX source code. Skipping" 1>&2
      ;;
    *)
      echo "File '$archive' is of unknown type '$mime_type'. Skipping" 1>&2
      ;;
  esac
done
