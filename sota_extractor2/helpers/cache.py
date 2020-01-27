import pandas as pd
import json


# these functions are used to cache various results
# of corresponding pipeline steps, to make it faster to
# rerun the pipeline or run in on batch of papers with various
# steps on different machines. The exchange formats are ad hoc and
# can be changed.


def load_tags(path):
    with open(path, 'rt') as f:
        tags = json.load(f)
    return tags


def save_tags(tags, path):
    with open(path, 'wt') as f:
        json.dump(tags, f)


def load_structure(path):
    with open(path, 'rt') as f:
        structure = json.load(f)
    return structure


def save_structure(structure, path):
    with open(path, 'wt') as f:
        json.dump(structure, f)


def load_proposals(path):
    proposals = pd.read_csv(path, index_col=0)
    return proposals


def save_proposals(proposals, path):
    proposals.to_csv(path)
