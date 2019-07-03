#!/usr/bin/env python

import fire
from sota_extractor2.data.elastic import Paper
from pathlib import Path

def extract_text(source, target):
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(exist_ok=True, parents=True)

    arxiv_id = source.stem
    doc = Paper.parse_paper(source)
    with open(target, 'wt') as f:
        f.write(doc.to_json())

if __name__ == "__main__": fire.Fire(extract_text)
