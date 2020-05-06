#!/usr/bin/env bash

#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

OUTNAME="$1"
echo $OUTNAME
RO_SOURCE_DIR="${2:-/files/ro-source}"
SOURCE_DIR="/files/source"
OUTPUT_DIR="${3:-/files/htmls}"

mkdir -p /files
cp -r "$RO_SOURCE_DIR" "$SOURCE_DIR"

# turn tikzpciture instances into comments
find "$SOURCE_DIR" -iname '*.tex' -print0 | xargs -0 sed -i \
  -e 's/\\begin{document}/\\usepackage{verbatim}\0/g' \
  -e 's/\\begin\(\[[^]]*\]\)\?{tikzpicture}/\\begin{comment}/g' \
  -e 's/\\end{tikzpicture}/\\end{comment}/g'

# temporary fixes
# https://github.com/brucemiller/LaTeXML/pull/1171
# https://github.com/brucemiller/LaTeXML/pull/1173
# https://github.com/brucemiller/LaTeXML/pull/1177
for patch in /files/patches/*
do
  patch -i $patch -p 3 -d /usr/local/share/perl/5.28.1/LaTeXML
done

MAINTEX=$(python3 /files/guess_main.py "$SOURCE_DIR")
[ ! -f "$MAINTEX" ] && exit 1

timeout -s KILL 300 engrafo "$MAINTEX" /files/output

[ ! -f /files/output/index.html ] && exit 117
cp /files/output/index.html "$OUTPUT_DIR/$OUTNAME"
