import pandas as pd
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List
from ..helpers.jupyter import display_table

@dataclass
class Cell:
    value: str
    gold_tags: str = ''
    refs: List[str] = None


class Table:
    def __init__(self, df, caption=None, figure_id=None, annotations=None):
        self.df = df
        self.caption = caption
        self.figure_id = figure_id
        self.df = df.applymap(lambda x: Cell(value=x))
        if annotations is not None:
            self.gold_tags = annotations.gold_tags.strip()
            rows, cols = annotations.matrix_gold_tags.shape
            for r in range(rows):
                for c in range(cols):
                    self.df.iloc[r,c].gold_tags = annotations.matrix_gold_tags.iloc[r,c].strip()
        else:
            self.gold_tags = ''

    @classmethod
    def from_file(cls, path, metadata, annotations=None):
        try:
            df = pd.read_csv(path, header=None, dtype=str).fillna('')
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        return cls(df, metadata.get('caption'), metadata.get('figure_id'), annotations)

    def display(self):

        display_table(self.df.applymap(lambda x: x.value).values, self.df.applymap(lambda x: x.gold_tags).values)

def read_tables(path, annotations):
    path = Path(path)
    with open(path / "metadata.json", "r") as f:
        metadata = json.load(f)
    return [Table.from_file(path / m["filename"], m, annotations.get(path.name)) for m in metadata]
