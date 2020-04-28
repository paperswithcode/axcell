#!/usr/bin/env bash

#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

jq '.. | select(.sota?) | .sota.rows[] | .paper_url' "$1" | grep arxiv | sed -e 's#"##g' -e 's#http:#https:#' | sort -u
