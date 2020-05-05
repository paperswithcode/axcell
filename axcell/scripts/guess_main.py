#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pathlib import Path
import re
import sys
import codecs

doccls = re.compile(r"\s*\\documentclass")
docbeg = re.compile(r"\s*\\begin\s*\{\s*document\s*\}")
title  = re.compile(r"\s*\\(icml)?title\s*\{(?P<title>[^%}]*)")

aux = re.compile(r"(rebuttal\s+|instructions\s+(for\s+\\confname|.*proceedings)|(supplementary|supplemental)\s+materials?|appendix|author\s+guidelines|ieeetran\.cls|formatting\s+instructions)")

def aux_title(t):
    t = t.strip().lower()
    return bool(aux.search(t))


def calc_priority(path):
    priority = 0
    if path.name.lower() == "ms.tex":
        return 30
    with codecs.open(path, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            if doccls.match(line):
                priority += 10
                break
        for line in f:
            m = title.match(line)
            if m:
                priority += 5
                t = m["title"]
                if aux_title(t):
                    priority = 5
                break
    return priority


def guess_main(path):
    path = Path(path)
    files = sorted(path.glob("*.tex"), key=lambda p: p.stem.lower())
    if len(files) > 1:
        with_priority = [(f, calc_priority(f)) for f in files]
        with_priority = sorted(with_priority, key=lambda fp: fp[1], reverse=True)
        files = [fp[0] for fp in with_priority]

    return files[0] if len(files) else None

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage:\n\t{sys.argv[0]} DIR", file=sys.stderr)
        exit(1)
    main = guess_main(sys.argv[1])
    if not main:
        print("Unable to find any suitable tex file", file=sys.stderr)
        exit(1)
    else:
        print(main)
