#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re
import pandas as pd

from axcell.data.paper_collection import remove_arxiv_version


def norm_score_str(x):
    x = str(x)
    if re.match('^(\+|-|)(\d+)\.9{5,}$', x):
        x = re.sub('^(\+|-|)(\d+)\.9{5,}$', lambda a: a.group(1)+str(int(a.group(2))+1), x)
    elif x.endswith('9' * 5) and '.' in x:
        x = re.sub(r'([0-8])9+$', lambda a: str(int(a.group(1))+1), x)
    if '.' in x:
        x = re.sub(r'0+$', '', x)
    if x[-1] == '.':
        x = x[:-1]
    if x == '-0':
        x = '0'
    return x


epsilon = 1e-10


def precision(tp, fp):
    pred_positives = tp + fp + epsilon
    return ((1.0 * tp) / pred_positives)


def recall(tp, fn):
    true_positives = tp + fn + epsilon
    return ((1.0 * tp) / true_positives)


def f1(prec, recall):
    norm = prec + recall + epsilon
    return (2 * prec * recall / norm)


def stats(predictions, ground_truth, axis=None):
    gold = pd.DataFrame(ground_truth, columns=["paper", "task", "dataset", "metric", "value"])
    pred = pd.DataFrame(predictions, columns=["paper", "task", "dataset", "metric", "value"])

    if axis == 'tdm':
        columns = ['paper', 'task', 'dataset', 'metric']
    elif axis == 'tdms' or axis is None:
        columns = ['paper', 'task', 'dataset', 'metric', 'value']
    else:
        columns = ['paper', axis]
    gold = gold[columns].drop_duplicates()
    pred = pred[columns].drop_duplicates()

    results = gold.merge(pred, on=columns, how="outer", indicator=True)

    is_correct = results["_merge"] == "both"
    no_pred = results["_merge"] == "left_only"
    no_gold = results["_merge"] == "right_only"

    results["TP"] = is_correct.astype('int8')
    results["FP"] = no_gold.astype('int8')
    results["FN"] = no_pred.astype('int8')

    m = results.groupby(["paper"]).agg({"TP": "sum", "FP": "sum", "FN": "sum"})
    m["precision"] = precision(m.TP, m.FP)
    m["recall"] = recall(m.TP, m.FN)
    m["f1"] = f1(m.precision, m.recall)

    TP_ALL = m.TP.sum()
    FP_ALL = m.FP.sum()
    FN_ALL = m.FN.sum()

    prec, reca = precision(TP_ALL, FP_ALL), recall(TP_ALL, FN_ALL)
    return {
        'Micro Precision': prec,
        'Micro Recall':    reca,
        'Micro F1':        f1(prec, reca),
        'Macro Precision': m.precision.mean(),
        'Macro Recall':    m.recall.mean(),
        'Macro F1':        m.f1.mean()
    }


def evaluate(predictions, ground_truth):
    predictions = predictions.copy()
    ground_truth = ground_truth.copy()
    predictions['value'] = predictions['score' if 'score' in predictions else 'value'].apply(norm_score_str)
    ground_truth['value'] = ground_truth['score' if 'score' in ground_truth else 'value'].apply(norm_score_str)
    predictions['paper'] = predictions['arxiv_id'].apply(remove_arxiv_version)
    ground_truth['paper'] = ground_truth['arxiv_id'].apply(remove_arxiv_version)

    metrics = []
    for axis in [None, "tdm", "task", "dataset", "metric"]:
        s = stats(predictions, ground_truth, axis)
        s['type'] = {'tdms': 'TDMS', 'tdm': 'TDM', 'task': 'Task', 'dataset': 'Dataset', 'metric': 'Metric'}.get(axis)
        metrics.append(s)
    columns = ['Micro Precision', 'Micro Recall', 'Micro F1', 'Macro Precision', 'Macro Recall', 'Macro F1']
    return pd.DataFrame(metrics, columns=columns)
