#!/usr/bin/env bash
SOURCE=$(realpath "$1")
[ ! -f "$SOURCE" ] && echo "File $SOURCE not found." >&2 && exit 1
mkdir -p $(dirname "$2")
OUTPUT=$(realpath "$2")

docker run --rm -v "$SOURCE":/files/index.html:ro --entrypoint '' zenika/alpine-chrome:73 chromium-browser --headless --disable-gpu --disable-software-rasterizer --no-sandbox --timeout=30000 --dump-dom /files/index.html > "$OUTPUT"
