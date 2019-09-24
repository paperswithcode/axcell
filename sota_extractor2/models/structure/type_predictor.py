from fastai.text import *
from pathlib import Path
import pandas as pd
from .ulmfit import ULMFiT_SP

class TableTypePredictor(ULMFiT_SP):
    def __init__(self, path, file, sp_path=None, sp_model="spm.model", sp_vocab="spm.vocab", threshold=0.5):
        super().__init__(path, file, sp_path, sp_model, sp_vocab)
        self.threshold = threshold


    def predict(self, paper, tables):
        if len(tables) == 0:
            return []
        df = pd.DataFrame({"caption": [table.caption if table.caption else "" for table in tables]})
        tl = TextList.from_df(df, cols="caption")
        self.learner.data.add_test(tl)
        preds, _ = self.learner.get_preds(DatasetType.Test, ordered=True)
        preds, _ = (preds.cpu() > self.threshold).max(dim=1)
        return preds.numpy().astype(bool).tolist()
