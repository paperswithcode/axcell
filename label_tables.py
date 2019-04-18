#!/usr/bin/env python

import fire
from sota_extractor.taskdb import TaskDB
from pathlib import Path
import json
import re
import pandas as pd
import sys
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, InvalidOperation


arxiv_url_re = re.compile(r"^https?://(www.)?arxiv.org/(abs|pdf|e-print)/(?P<arxiv_id>\d{4}\.[^./]*)(\.pdf)?$")

def get_sota_tasks(filename):
    db = TaskDB()
    db.load_tasks(filename)
    return db.tasks_with_sota()


def get_metadata(filename):
    with open(filename, "r") as f:
        return json.load(f)


def get_table(filename):
    try:
        return pd.read_csv(filename, header=None, dtype=str).fillna('')
    except pd.errors.EmptyDataError:
        return []


def get_tables(tables_dir):
    tables_dir = Path(tables_dir)
    all_metadata = {}
    all_tables = {}
    for metadata_filename in tables_dir.glob("*/metadata.json"):
        metadata = get_metadata(metadata_filename)
        basedir = metadata_filename.parent
        arxiv_id = basedir.name
        all_metadata[arxiv_id] = metadata
        all_tables[arxiv_id] = {m['filename']:get_table(basedir / m['filename']) for m in metadata}
    return all_metadata, all_tables


metric_na = ['-','']


# problematic values of metrics found in evaluation-tables.json
# F0.5, 70.14 (measured by Ge et al., 2018)
# Test Time, 0.33s/img
# Accuracy, 77,62%
# Electronics, 85,06
# BLEU-1, 54.60/55.55
# BLEU-4, 26.71/27.78
# MRPC, 78.6/84.4
# MRPC, 76.2/83.1
# STS, 78.9/78.6
# STS, 75.8/75.5
# BLEU score,41.0*
# BLEU score,28.5*
# SemEval 2007,**55.6**
# Senseval 2,**69.0**
# Senseval 3,**66.9**
# MAE, 2.42±0.01

## multiple times
# Number of params, 0.8B
# Number of params, 88M
# Parameters, 580k
# Parameters, 3.1m
# Params, 22M



float_value_re = re.compile(r"([+-]?\s*(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)")
whitespace_re = re.compile(r"\s+")


def normalize_float_value(s):
    match = float_value_re.search(s)
    if match:
        return whitespace_re.sub("", match.group(0))
    return '-'


def test_near(x, precise):
    for rounding in [ROUND_DOWN, ROUND_HALF_UP]:
        try:
            if x == precise.quantize(x, rounding=rounding):
                return True
        except InvalidOperation:
            pass
    return False


def fuzzy_match(metric, metric_value, target_value):
    metric_value = normalize_float_value(str(metric_value))
    if metric_value in metric_na:
        return False
    metric_value = Decimal(metric_value)

    for match in float_value_re.findall(target_value):
        value = whitespace_re.sub("", match[0])
        value = Decimal(value)

        if test_near(metric_value, value):
            return True
        if test_near(metric_value.shift(2), value):
            return True
        if test_near(metric_value, value.shift(2)):
            return True

    return False
#
#    if metric_value in metric_na or target_value in metric_na:
#        return False
#    if metric_value != target_value and metric_value in target_value:
#        print(f"|{metric_value}|{target_value}|")
#    return metric_value in target_value


def match_metric(metric, tables, value):
    matching_tables = []
    for table in tables:
        for col in tables[table]:
            for row in tables[table][col]:
                if fuzzy_match(metric, value, row):
                    matching_tables.append(table)
                    break
            else:
                continue
            break

    return matching_tables


def label_tables(tasksfile, tables_dir):
    tasks = get_sota_tasks(tasksfile)
    metadata, tables = get_tables(tables_dir)

#    for arxiv_id in tables:
#        for t in tables[arxiv_id]:
#            table = tables[arxiv_id][t]
#            for col in table:
#                for row in table[col]:
#                    print(row)
#    return
    for task in tasks:
        for dataset in task.datasets:
            for row in dataset.sota.rows:
                # TODO: some results have more than one url, CoRR + journal / conference
                # check if we have the same results for both

                match = arxiv_url_re.match(row.paper_url)
                if match is not None:
                    arxiv_id = match.group("arxiv_id")
                    if arxiv_id not in tables:
                        print(f"No tables for {arxiv_id}. Skipping", file=sys.stderr)
                        continue

                    for metric in row.metrics:
                        #print(f"{metric}\t{row.metrics[metric]}")
                        #print((task.name, dataset.name, metric, row.model_name, row.metrics[metric], row.paper_url))
                        matching = match_metric(metric, tables[arxiv_id], row.metrics[metric])
                        #if not matching:
                        #    print(f"{metric}, {row.metrics[metric]}, {arxiv_id}")
                        print(f"{metric},{len(matching)}")
                        #if matching:
                        #    print((task.name, dataset.name, metric, row.model_name, row.metrics[metric], row.paper_url))
                        #    print(matching)




if __name__ == "__main__": fire.Fire(label_tables)
