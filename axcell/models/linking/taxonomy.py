#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pathlib import Path
import json
from collections import OrderedDict
from axcell.models.linking.manual_dicts import complementary_metrics


class Taxonomy:
    def __init__(self, taxonomy, metrics_info):
        self.taxonomy = self._get_taxonomy(taxonomy)
        self.metrics_info, self.metrics_range = self._read_metrics_info(metrics_info)
        self.tasks = self._get_axis('task')
        self.datasets = self._get_axis('dataset')
        self.metrics = self._get_axis('metric')

    def normalize_metric(self, task, dataset, metric):
        if (task, dataset, metric) in self._complementary:
            return self._complementary[(task, dataset, metric)][2]
        return metric

    def _read_json(self, path):
        with open(path, "rt") as f:
            return json.load(f)

    def _get_complementary_metrics(self):
        complementary = []
        self._complementary = {}
        for record in self.canonical_records:
            metric = record["metric"]
            if metric.lower() in complementary_metrics:
                task = record["task"]
                dataset = record["dataset"]
                comp_metric = complementary_metrics[metric.lower()]
                complementary.append(
                    dict(
                        task=task,
                        dataset=dataset,
                        metric=comp_metric
                    )
                )

                self._complementary[(task, dataset, comp_metric)] = (task, dataset, metric)
        return complementary

    def _get_taxonomy(self, path):
        self.canonical_records = self._read_json(path)
        self.records = self.canonical_records + self._get_complementary_metrics()
        return [(r["task"], r["dataset"], r["metric"]) for r in self.records]

    def _get_axis(self, axis):
        return set(x[axis] for x in self.records)

    def _read_metrics_info(self, path):
        records = self._read_json(path)
        metrics_info = {}
        metrics_range = {}
        mr = {}
        for r in records:
            task, dataset, metric = r['task'], r['dataset'], r['metric']
            key = (task, dataset, metric)
            d = 1 if r['higher_is_better'] else -1
            rng = r['range']
            metrics_info[key] = d
            metrics_info[metric] = metrics_info.get(metric, 0) + d
            metrics_range[key] = rng
            s = mr.get(metric, {})
            s[rng] = s.get(rng, 0) + 1
            mr[metric] = s
        for metric in mr:
            metrics_range[metric] = sorted(mr[metric].items(), key=lambda x: x[1])[-1][0]
        return metrics_info, metrics_range
