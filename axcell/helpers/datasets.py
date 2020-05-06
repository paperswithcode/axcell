#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pandas as pd


def read_arxiv_papers(path):
    return pd.read_csv(path)


def read_tables_annotations(path):
    return pd.read_json(path)
