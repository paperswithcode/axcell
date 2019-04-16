#!/usr/bin/env bash
jq '.. | select(.sota?) | .sota.rows[] | .paper_url' "$1" | grep arxiv | sed -e 's#"##g' -e 's#http:#https:#' | sort -u
