#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pytest
from pathlib import Path
from axcell.helpers.paper_extractor import PaperExtractor
from axcell.data.paper_collection import PaperCollection
from shutil import copyfileobj
import gzip


def test_extraction(tmpdir):
    # pack main.tex to an archive
    tmpdir = Path(tmpdir)
    source = Path(__file__).resolve().parent / "data" / "main.tex"
    paper_id = "1234.56789"
    archive = tmpdir / "sources" / paper_id
    archive.parent.mkdir()
    with source.open("rb") as src, gzip.open(archive, "wb") as dst:
        copyfileobj(src, dst)

    extract = PaperExtractor(tmpdir)
    status = extract(archive)
    assert status == "success"

    pc = PaperCollection.from_files(tmpdir / "papers")
    extracted = len(pc)
    assert extracted == 1, f"Expected to extract exactly one paper, found {extracted}"

    paper = pc[0]
    assert paper.paper_id == paper_id
    assert paper.text.title == "DILBERT: Distilling Inner Latent BERT variables"
    assert len(paper.tables) == 2

    assert paper.tables[0].caption == "Table 1: A table."
    assert paper.tables[1].caption == "Table 2: A table."

    assert paper.tables[0].shape == (5, 3)
    assert paper.tables[1].shape == (4, 3)
