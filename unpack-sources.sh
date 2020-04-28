#!/usr/bin/env bash

#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

archive="$1"
outdir="$2"

mime_type=$(file --brief --uncompress --mime-type "$archive")
case "$mime_type" in
  'application/x-tar')
    mkdir -p "$outdir"
    tar -xf "$archive" -C "$outdir"
    ;;
  'text/x-tex')
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
