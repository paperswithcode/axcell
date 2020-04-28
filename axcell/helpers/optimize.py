#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pandas as pd, numpy as np
from dataclasses import dataclass, replace
from axcell.models.linking.metrics import CM
from matplotlib import pyplot as plt
import matplotlib.tri as tri


def annotations(matrix, structure, r, c, type='model'):
    ann = []
    for nc in range(0, c):
        if type in structure[r, nc]:
            ann.append(matrix[r, nc])
    for nr in range(0, r):
        if type in structure[nr, c]:
            ann.append(matrix[nr, c])
    return ' '.join(ann)


def estimate_noises(extracted_values, gold_values, short_forms):
    if not len(extracted_values):
        return {}
    extracted_values = set(extracted_values)
    gold_values = set(gold_values)

    return {gold: 1 - len(extracted_values & set(short_forms.get(gold, set()))) / len(extracted_values) for gold in
            gold_values}


def estimate_context_noise(context, records):
    context = context or ""
    abbrvs = context_search.extract_acronyms(context)
    context = normalize_cell_ws(normalize_dataset(context))
    dss = set(cs.find_datasets(context)) | set(abbrvs.keys())
    mss = set(cs.find_metrics(context))
    dss -= mss
    dss = set([normalize_cell(ds) for ds in dss])
    mss = set([normalize_cell(ms) for ms in mss])

    gold_ds = set(records.dataset.values)
    gold_ms = set(records.metric.values)
    ds_noises = estimate_noises(dss, gold_ds, cs.datasets)
    ms_noises = estimate_noises(mss, gold_ms, cs.metrics)

    return ds_noises, ms_noises


def estimate_paper_context_noise(paper, gold_sota_records):
    records = gold_sota_records[gold_sota_records.paper_id == paper.paper_id]
    datasets = de.from_paper(paper)
    context = " ".join(datasets)
    return estimate_context_noise(context, records)


def estimate_caption_context_noise(paper, table, gold_sota_records):
    table_ext_id = f"{paper.paper_id}/{table.name}/"
    records = gold_sota_records[gold_sota_records.index.str.startswith(table_ext_id)]
    return estimate_context_noise(table.caption, records)


def estimate_cell_context_noise(paper, table, row, col, gold_sota_records):
    cell_ext_id = f"{paper.paper_id}/{table.name}/{row}.{col}"
    records = gold_sota_records[gold_sota_records.index == cell_ext_id]
    value = annotations(table.matrix.values, table.matrix_gold_tags.values, row, col, 'dataset')
    return estimate_context_noise(value, records)


def average_dicts(dicts):
    sums = {}
    for d in dicts:
        for k, v in d.items():
            sums.setdefault(k, []).append(v)
    return {k: np.mean(v) for k, v in sums.items()}


def all_equal(row):
    cols = ["model_type", "dataset", "metric", "task", "parsed"]
    return np.all([row[f"{name}_pred"] == row[f"{name}_gold"] for name in cols])


def merge_gold_records(explainer):
    paper_ids = list(explainer.le.proposals.keys())

    proposals = pd.concat(explainer.le.proposals.values())

    papers = {paper_id: explainer.paper_collection.get_by_id(paper_id) for paper_id in paper_ids}
    missing = [paper_id for paper_id, paper in papers.items() if paper is None]
    if missing:
        print("Missing papers in paper collection:")
        print(", ".join(missing))
    papers = [paper for paper in papers.values() if paper is not None]

    if explainer.gold_sota_records is None:
        print("gold_sota_records is missing")
        return
    else:
        gold_sota_records = explainer.gold_sota_records
        which = gold_sota_records.index.to_series().str.split("/", expand=True)[0] \
            .isin([paper.paper_id for paper in papers])
        gold_sota_records = gold_sota_records[which]

    df = gold_sota_records.merge(proposals, 'outer', left_index=True, right_index=True, suffixes=['_gold', '_pred'])
    df = df.reindex(sorted(df.columns), axis=1)
    df.confidence = df.confidence.fillna(0.0)
    df = df.fillna('not-present')
    df["equal"] = df.apply(all_equal, axis=1)
    df["pred_positive"] = df["model_type_pred"].str.contains("model-best")
    df["gold_positive"] = df["model_type_gold"].str.contains("model-best")
    return df


def find_threshold_intervals(proposals, metrics_info, context="paper"):
    # maximal threshold to have this proposal returned
    proposals["max_threshold"] = proposals.confidence

    proposals["min_threshold"] = 0.0

    ignore = (proposals.model_type_pred != 'model-best') | (proposals.struct_model_type == '') | \
             (proposals.struct_dataset.str.contains('dev')) | (proposals.struct_dataset.str.contains('train'))

    # this proposal won't be ever returned due to structure or model type filters
    proposals.loc[ignore, "min_threshold"] = 1.0
    proposals.loc[ignore, "max_threshold"] = 0.0

    all_proposals = proposals
    proposals = proposals[~ignore]

    if context == "paper":
        context_column = proposals.index.to_series().str.split('/', expand=False).apply(lambda x: x[0])
    else:
        context_column = proposals.index.to_series().str.split('/', expand=False).apply(lambda x: x[0] + "/" + x[1])

    for i, p in proposals.iterrows():
        key = (p.task_pred, p.dataset_pred, p.metric_pred)
        proposals_context = proposals[context_column == context_column[p.name]]
        proposals_context = proposals_context[~proposals_context.parsed_pred.isna()]
        proposals_context = proposals_context[
            (proposals_context.task_pred == p.task_pred) &
            (proposals_context.dataset_pred == p.dataset_pred) &
            (proposals_context.metric_pred == p.metric_pred)
            ]
        d = 0
        if key in metrics_info:
            d = metrics_info[key]
        elif p.metric_pred in metrics_info:
            d = metrics_info[p.metric_pred]
        elif 'error' in p.metric_pred.lower():
            d = -1
        elif 'accuracy' in p.metric_pred.lower():
            d = 1

        if d >= 0:
            d = 1
        else:
            d = -1

        # the minimal threshold above which all superior results are ignored
        which = d * proposals_context.parsed_pred > d * p.parsed_pred
        if np.any(which.values):
            all_proposals.at[i, "min_threshold"] = proposals_context[which].confidence.values.max()
        else:
            which = proposals_context[proposals_context.parsed_pred == p.parsed_pred].iloc[0]
            if which.name != p.name:
                all_proposals.at[i, "min_threshold"] = which.confidence

    return all_proposals


def update_cm(proposal, cm, is_activated):
    d = 1 if is_activated else -1
    if proposal.equal and proposal.pred_positive and proposal.gold_positive:
        cm = replace(cm, tp=cm.tp + d, fn=cm.fn - d)
    if proposal.equal and not proposal.pred_positive and not proposal.gold_positive:
        cm = replace(cm, tn=cm.tn + d)
    if proposal.pred_positive and (not proposal.equal or not proposal.gold_positive):
        cm = replace(cm, fp=cm.fp + d)
    #     if proposal.gold_positive and (not proposal.equal or not proposal.pred_positive):
    #         cm = replace(cm, fn = cm.fn+d)
    return cm


def sweep_thresholds(df):
    cm = CM(fn=sum(df.gold_positive))
    df = df[df.min_threshold < df.max_threshold]

    sweeps = df.reset_index().melt(id_vars="cell_ext_id", value_vars=["min_threshold", "max_threshold"],
                                   var_name="threshold_type", value_name="threshold")

    sweeps = sweeps.sort_values(by=["threshold", "threshold_type"]).reset_index(drop=True)

    steps = sweeps.threshold.drop_duplicates().index

    results = []
    for i, idx1 in enumerate(steps[:-1]):
        th1 = sweeps.threshold[idx1]

        to_restore = cm
        for j, idx2 in enumerate(steps[i + 1:], i + 1):
            th2 = sweeps.threshold[idx2]
            precision = cm.tp / (cm.tp + cm.fp + 1e-8)
            recall = cm.tp / (cm.tp + cm.fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            result = dict(threshold1=th1, threshold2=sweeps.threshold[idx2 - 1], tp=cm.tp, tn=cm.tn, fp=cm.fp, fn=cm.fn,
                          precision=precision, recall=recall, f1=f1)
            results.append(result)
            for _, row in sweeps[sweeps.threshold == sweeps.threshold[idx2 - 1]].iterrows():
                proposal = df.loc[row.cell_ext_id]
                is_activated = row.threshold_type == 'min_threshold'
                if not is_activated and proposal.min_threshold < th1:
                    cm = update_cm(proposal, cm, is_activated)

        precision = cm.tp / (cm.tp + cm.fp + 1e-8)
        recall = cm.tp / (cm.tp + cm.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        result = dict(threshold1=th1, threshold2=th2, tp=cm.tp, tn=cm.tn, fp=cm.fp, fn=cm.fn,
                      precision=precision, recall=recall, f1=f1)
        results.append(result)

        cm = to_restore

        for _, row in sweeps[sweeps.threshold == th1].iterrows():
            proposal = df.loc[row.cell_ext_id]

            is_activated = row.threshold_type == 'min_threshold'
            cm = update_cm(proposal, cm, is_activated)

    return df, sweeps, steps, pd.DataFrame(results)


class PRResults:
    def __init__(self, results):
        self.results = results

    def plot(self):
        plt.figure(figsize=(6, 6))
        plt.plot(self.results["precision"], self.results["recall"], '.')
        plt.xlabel("precision")
        plt.ylabel("recall")

    def _best(self, results, metric):
        b = results.loc[results[metric].idxmax()]
        x = ["precision", "recall", "f1"]
        x.remove(metric)
        y = [b[m] for m in x]
        print(f"Best {metric}={b[metric]:0.2f} (with {x[0]}={y[0]:.2f} and {x[1]}={y[1]:.2f})"
              f" is achieved with threshold1={b.threshold1} and threshold2={b.threshold2}")

    def best(self, min_precision=0, min_recall=0, min_f1=0):
        results = self.results
        results = results[
            (results.precision >= min_precision) &
            (results.recall >= min_recall) &
            (results.f1 >= min_f1)
            ]
        if not len(results):
            print("No results with this criteria")
        else:
            self._best(results, "precision")
            self._best(results, "recall")
            self._best(results, "f1")

    def threshold_map(self, metric):
        lin = np.linspace(0, 1, 64)

        triang = tri.Triangulation(self.results.threshold1.values, self.results.threshold2.values)
        interpolator = tri.LinearTriInterpolator(triang, self.results[metric])
        Xi, Yi = np.meshgrid(lin, lin)
        zi = interpolator(Xi, Yi)
        plt.figure(figsize=(6, 6))
        img = plt.imshow(zi[::-1], extent=[0, 1, 0, 1])
        plt.colorbar(img)
        plt.xlabel("threshold1")
        plt.ylabel("threshold2")


def optimize_filters(explainer, metrics_info):
    df = merge_gold_records(explainer)
    df = find_threshold_intervals(df, metrics_info, context="paper")
    df, sweeps, steps, results = sweep_thresholds(df)
    return PRResults(results)
