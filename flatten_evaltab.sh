#!/usr/bin/env bash
jq -c '.. | select(.datasets?).datasets | .[] | .dataset as $dataset | .sota.rows[] | {paper_url, paper_title, model_name} as $paper | .metrics | . as $metrics | keys[] | {dataset: $dataset, metric_name: ., metric_value: $metrics[.], paper_url: $paper.paper_url, paper_title: $paper.paper_title, model_name: $paper.model_name }' "$1" | grep arxiv\.org | jq -s '.'
