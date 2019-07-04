import pandas as pd
import json
from pathlib import Path
import re
from dataclasses import dataclass, field
from typing import List
from ..helpers.jupyter import display_table

@dataclass
class Cell:
    value: str
    gold_tags: str = ''
    refs: List[str] = field(default_factory=list)


reference_re = re.compile(r"\[(xxref-[^] ]*)\]")
def extract_references(s):
    parts = reference_re.split(s)
    return ''.join(parts[::2]), parts[1::2]


def str2cell(s):
    value, refs = extract_references(s)
    return Cell(value=value, refs=refs)

class Table:
    def __init__(self, df, caption=None, figure_id=None, annotations=None):
        self.df = df
        self.caption = caption
        self.figure_id = figure_id
        self.df = df.applymap(str2cell)
        if annotations is not None:
            self.gold_tags = annotations.gold_tags.strip()
            tags = annotations.matrix_gold_tags
            if self.df.shape != (0,0):
                for r, row in enumerate(tags):
                    for c, cell in enumerate(row):
                        self.df.iloc[r,c].gold_tags = cell.strip()
        else:
            self.gold_tags = ''

    @classmethod
    def from_file(cls, path, metadata, annotations=None):
        try:
            df = pd.read_csv(path, header=None, dtype=str).fillna('')
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        if annotations is not None:
            table_ann = annotations.table_set.filter(name=metadata['filename']) + [None]
            table_ann = table_ann[0]
        else:
            table_ann = None
        return cls(df, metadata.get('caption'), metadata.get('figure_id'), table_ann)

    def display(self):
        display_table(self.df.applymap(lambda x: x.value).values, self.df.applymap(lambda x: x.gold_tags).values)

def read_tables(path, annotations):
    path = Path(path)
    with open(path / "metadata.json", "r") as f:
        metadata = json.load(f)
    return [Table.from_file(path / m["filename"], m, annotations.get(path.name)) for m in metadata]
