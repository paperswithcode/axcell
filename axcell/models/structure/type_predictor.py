#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fastai.text import *
from pathlib import Path
import pandas as pd
from .ulmfit import ULMFiT_SP
from ...pipeline_logger import pipeline_logger
import torch
from enum import Enum


class TableType(Enum):
    SOTA = 0
    ABLATION = 1
    IRRELEVANT = 2


def multipreds2preds(preds, threshold=0.5):
    bs = preds.shape[0]
    return torch.cat([preds, preds.new_full((bs,1), threshold)], dim=-1).argmax(dim=-1)


class TableTypePredictor(ULMFiT_SP):
    step = "type_prediction"

    def __init__(self, path, file, sp_path=None, sp_model="spm.model", sp_vocab="spm.vocab", threshold=0.5):
        super().__init__(path, file, sp_path, sp_model, sp_vocab)
        self.threshold = threshold

    def predict(self, paper, tables):
        pipeline_logger(f"{TableTypePredictor.step}::predict", paper=paper, tables=tables)
        if len(tables) == 0:
            predictions = []
        else:
            column = "caption"
            df = pd.DataFrame({column: [table.caption if table.caption else "Table" for table in tables]})
            inputs = df.iloc[:, df_names_to_idx(column, df)]
            tl = TextList(items=inputs.values[:, 0], path='.', inner_df=df, processor=None)
            self.learner.data.add_test(tl)
            preds, _ = self.learner.get_preds(DatasetType.Test, ordered=True)
            pipeline_logger(f"{TableTypePredictor.step}::multiclass_predicted", paper=paper, tables=tables,
                            threshold=self.threshold, predictions=preds.cpu().numpy())
            predictions = [TableType(x) for x in multipreds2preds(preds, self.threshold).cpu().numpy()]
        pipeline_logger(f"{TableTypePredictor.step}::predicted", paper=paper, tables=tables, predictions=predictions)
        return predictions
