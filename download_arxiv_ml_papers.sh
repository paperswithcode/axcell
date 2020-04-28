#!/bin/bash

#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

index_dir="index"
papers_dir="papers"
src_dir="src"
mkdir -p "${index_dir}" "${papers_dir}" "${src_dir}"

jq -r '.[] | select(.arxiv_id) | "/"+.arxiv_id+"."' pwc/papers-with-abstracts.json | sort -u > wildcards.txt
aws s3 cp --request-payer requester s3://arxiv/src/arXiv_src_manifest.xml .
xmllint --xpath '//filename/text()' arXiv_src_manifest.xml > tars.txt

process_file () {
    path="$1"
    archive_name=$(basename "${path}")
    file="${src_dir}/${archive_name}"
    echo "Processing ${file}..."
    [ -e "${file}" ] && echo "Already exists, skipping..." && return
    aws s3 cp --request-payer requester "s3://arxiv/${path}" "${src_dir}"
    tar -tvf "${file}" > "${index_dir}/${archive_name}.ls"
    tar -tf "${file}" > "${index_dir}/${archive_name}.txt"
    fgrep -f wildcards.txt "${index_dir}/${archive_name}.txt" > to_extract.txt && xargs -a to_extract.txt -- tar xf "${file}" -C "${papers_dir}"
}

while read file
do
    process_file "${file}"
done <tars.txt
