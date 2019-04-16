#!/usr/bin/env bash

filename="$1"
outdir="test-tables/$(basename $filename)"
mkdir -p "$outdir"
./extract-tables.py $filename "$outdir"
