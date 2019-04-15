#!/usr/bin/env bash
jq '.. | select(.sota?) | .sota.rows[] | .paper_url' "$1" | sort -u | grep arxiv | sed -e 's#"##g' -e 's#http:#https:#'
