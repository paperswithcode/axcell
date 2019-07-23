#!/usr/bin/env bash
SOURCE_DIR=$(realpath "$1") 	#~/arxiv/unpacked/1701/1701.xyz
[ ! -d "$SOURCE_DIR" ] && echo "Directory $SOURCE_DIR not found." >&2 && exit 1
mkdir -p $(dirname "$2")
OUTPUT=$(realpath "$2")		#~/arxiv/htmls/1701/1701.xyz.html
OUTPUT_DIR=$(dirname "$OUTPUT")	#~/arxiv/htmls/1701
FILENAME=$(basename "$OUTPUT") #1701.xyz.html

docker run --rm -v $PWD/latex2html.sh:/files/latex2html.sh:ro -v "$SOURCE_DIR":/files/ro-source:ro -v "$OUTPUT_DIR":/files/htmls arxivvanity/engrafo /files/latex2html.sh "$FILENAME"
