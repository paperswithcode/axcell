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
    layout: str = ''


reference_re = re.compile(r"<ref id='([^']*)'>(.*?)</ref>")
num_re = re.compile(r"^\d+$")

def extract_references(s):
    parts = reference_re.split(s)
    refs = parts[1::3]
    text = []
    for i, x in enumerate(parts):
        if i % 3 == 0:
            text.append(x)
        elif i % 3 == 2:
            s = x.strip()
            if num_re.match(s):
                text.append(s)
            else:
                text.append(f"[{s}]")
    text = ''.join(text)
    return text, refs


def str2cell(s):
    value, refs = extract_references(s)
    return Cell(value=value, refs=refs)

def read_str_csv(filename):
    try:
        df = pd.read_csv(filename, header=None, dtype=str).fillna('')
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    return df




class Table:
    def __init__(self, df, layout, caption=None, figure_id=None, annotations=None, old_name=None):
        self.df = df
        self.caption = caption
        self.figure_id = figure_id
        self.df = df.applymap(str2cell)
        self.old_name = old_name

        if layout is not None:
            self.layout = layout
            for r, row in layout.iterrows():
                for c, cell in enumerate(row):
                    self.df.iloc[r,c].layout = cell

        if annotations is not None:
            self.gold_tags = annotations.gold_tags.strip()
            tags = annotations.matrix_gold_tags
            gt_rows = len(annotations.matrix_gold_tags)
            if gt_rows > 0:
                gt_cols = len(annotations.matrix_gold_tags[0])
                if self.df.shape != (0,0) and self.df.shape == (gt_rows, gt_cols):
                    for r, row in enumerate(tags):
                        for c, cell in enumerate(row):
                            self.df.iloc[r,c].gold_tags = cell.strip()
        else:
            self.gold_tags = ''

    @classmethod
    def from_file(cls, path, metadata, annotations=None, match_name=None):
        path = Path(path)
        filename = path / metadata['filename']
        df = read_str_csv(filename)
        if 'layout' in metadata:
            layout = read_str_csv(path / metadata['layout'])
        else:
            layout = None
        if annotations is not None and match_name is not None:
            table_ann = annotations.table_set.filter(name=match_name) + [None]
            table_ann = table_ann[0]
        else:
            table_ann = None
        return cls(df, layout, metadata.get('caption'), metadata.get('figure_id'), table_ann, match_name)

    def display(self):
        display_table(self.df.applymap(lambda x: x.value).values, self.df.applymap(lambda x: x.gold_tags).values)

#####
# this code is used to migrate table annotations from
# tables parsed by htlatex to tables parsed by
# latexml. After all annotated tables will be successfully
# migrated, we switch back to match-by-name

from unidecode import unidecode
import string
from collections import Counter

punctuation_table = str.maketrans('', '', string.punctuation)
def normalize_string(s):
    if s is None:
        return ""
    return unidecode(s.strip().lower().replace(' ', '')).translate(punctuation_table)

def _remove_almost_empty_values(d):
    return {k:v for k,v in d.items() if len(v) >= 10}

def _keep_unique_values(d):
    c = Counter(d.values())
    unique = [k for k,v in c.items() if v == 1]
    return {k: v for k,v in d.items() if v in unique}

def _match_tables_by_captions(annotations, metadata):
    if annotations is None:
        return {}
    old_captions = {x.name: normalize_string(x.desc) for x in annotations.table_set}
    new_captions = {m['filename']: normalize_string(m['caption']) for m in metadata}
    old_captions = _keep_unique_values(_remove_almost_empty_values(old_captions))
    new_captions = _keep_unique_values(_remove_almost_empty_values(new_captions))
    old_captions_reverse = {v:k for k,v in old_captions.items()}
    return {new_name:old_captions_reverse[caption] for new_name, caption in new_captions.items() if caption in old_captions_reverse}

####

def read_tables(path, annotations):
    path = Path(path)
    with open(path / "metadata.json", "r") as f:
        metadata = json.load(f)
    _match_names = _match_tables_by_captions(annotations, metadata)
    return [Table.from_file(path, m, annotations, match_name=_match_names.get(m["filename"])) for m in metadata]
