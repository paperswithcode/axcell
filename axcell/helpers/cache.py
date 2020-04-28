#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pandas as pd
import json
from collections import defaultdict
from pathlib import Path


# these functions are used to cache various results
# of corresponding pipeline steps, to make it faster to
# rerun the pipeline or run in on batch of papers with various
# steps on different machines. The exchange formats are ad hoc and
# can be changed.


def _load_json(path):
    with Path(path).open('rt') as f:
        return json.load(f)


def _save_json(obj, path):
    with Path(path).open('wt') as f:
        json.dump(obj, f)


def load_references(path):
    return _load_json(path)


def save_references(references, path):
    _save_json(references, path)


def load_tags(path):
    return _load_json(path)


def save_tags(tags, path):
    _save_json(tags, path)


def load_structure(path):
    return _load_json(path)


def save_structure(structure, path):
    _save_json(structure, path)


def load_proposals(path):
    dtypes = defaultdict(lambda: str)
    dtypes['confidence'] = float
    dtypes['parsed'] = float

    na_values = {'confidence': '', 'parsed': ''}
    proposals = pd.read_csv(path, index_col=0, dtype=dtypes, na_values=na_values, keep_default_na=False)
    return proposals


def save_proposals(proposals, path):
    proposals.to_csv(path)
