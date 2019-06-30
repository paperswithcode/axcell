import numpy as np
from ...helpers.training import set_seed


def split_by_cell_content(df, seed=42, split_column="cell_content"):
    set_seed(seed, "val_split")
    contents = np.random.permutation(df[split_column].unique())
    val_split = int(len(contents)*0.1)
    val_keys = contents[:val_split]
    split = df[split_column].isin(val_keys)
    valid_df = df[split]
    train_df = df[~split]
    len(train_df), len(valid_df)
    return train_df, valid_df
