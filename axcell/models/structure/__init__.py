#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re
import numpy as np
import pandas as pd
from ...helpers.training import set_seed
from ... import config
from .type_predictor import TableTypePredictor, TableType
from .structure_predictor import TableStructurePredictor

__all__ = ["TableType", "TableTypePredictor", "TableStructurePredictor"]


def split_by_cell_content(df, seed=42, split_column="cell_content"):
    set_seed(seed, "val_split", quiet=True)
    contents = np.random.permutation(df[split_column].unique())
    val_split = int(len(contents)*0.1)
    val_keys = contents[:val_split]
    split = df[split_column].isin(val_keys)
    valid_df = df[split]
    train_df = df[~split]
    len(train_df), len(valid_df)
    return train_df, valid_df


label_map_4 = {
    "model-paper": 1,
    "model-best": 1,
    "model-competing": 2,
    "dataset": 3,
    "dataset-sub": 3,
    "dataset-task": 3,
}


label_map_3 = {
    "model-paper": 1,
    "model-best": 1,
    "model-competing": 2,
}

label_map_2 = {
    "model-paper": 1,
    "model-best": 1,
    "model-competing": 1,
}


class DataBunch:
    def __init__(self, train_name, test_name, label_map):
        self.label_map = label_map
        self.train_df = pd.read_csv(config.datasets_structure/train_name)
        self.test_df = pd.read_csv(config.datasets_structure/test_name)
        self.transform(self.normalize)
        self.transform(self.label)

    def transform(self, fun):
        self.train_df = fun(self.train_df)
        self.test_df = fun(self.test_df)

    def normalize(self, df):
        df = df.drop_duplicates(["text", "cell_content", "cell_type"]).fillna("")
        df = df.replace(re.compile(r"(xxref|xxanchor)-[\w\d-]*"), "\\1 ")
        df = df.replace(re.compile(r"(^|[ ])\d+\.\d+\b"), " xxnum ")
        df = df.replace(re.compile(r"(^|[ ])\d\b"), " xxnum ")
        df = df.replace(re.compile(r"\bdata set\b"), " dataset ")
        return df

    def label(self, df):
        df["label"] = df["cell_type"].apply(lambda x: self.label_map.get(x, 0))
        df["label"] = pd.Categorical(df["label"])
        return df
