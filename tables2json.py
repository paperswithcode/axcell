#!/usr/bin/env python

import fire
from sota_extractor.taskdb import TaskDB
from pathlib import Path
import json
import re
import pandas as pd

from label_tables import get_table, get_metadata

def get_celltags(filename):
    filename = Path(filename)
    if filename.exists():
        celltags = pd.read_csv(filename, header=None, dtype=str).fillna('')
        return celltags
    else:
        return pd.DataFrame()


def get_tables(tables_dir):
    tables_dir = Path(tables_dir)
    all_metadata = {}
    all_tables = {}
    all_celltags = {}
    for metadata_filename in tables_dir.glob("1509*/metadata.json"):
        metadata = get_metadata(metadata_filename)
        basedir = metadata_filename.parent
        arxiv_id = basedir.name
        all_metadata[arxiv_id] = metadata
        all_tables[arxiv_id] = {t:get_table(basedir / t) for t in metadata}
        all_celltags[arxiv_id] = {t:get_celltags(basedir / t.replace("table", "celltags")) for t in metadata}
    return all_metadata, all_tables, all_celltags

def t2j(df):
    rows, cols = df.shape
    return [[df.iloc[r, c] for c in range(cols)] for r in range(rows)]


def tables2json(tables_dir):
    metadata, tables, celltags = get_tables(tables_dir)
    all_data = {}
    for arxiv_id in metadata:
        table = {'metadata': metadata[arxiv_id]}
        tabs = []
        cts = []
        for tab in tables[arxiv_id]:
            tabs.append(t2j(tables[arxiv_id][tab]))
        for ct in celltags[arxiv_id]:
            cts.append(t2j(celltags[arxiv_id][ct]))
        table['tables'] = tabs
        table['celltags'] = cts
        all_data[arxiv_id] = table
    print(json.dumps(all_data))

if __name__ == '__main__': fire.Fire(tables2json)
