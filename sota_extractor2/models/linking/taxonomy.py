from pathlib import Path
import json
from collections import OrderedDict
from sota_extractor2.models.linking.manual_dicts import complementary_metrics


class Taxonomy:
    def __init__(self, taxonomy, metrics_info):
        self.taxonomy = self._get_taxonomy(taxonomy)
        self.metrics_info = self._read_metrics_info(metrics_info)
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

    def _get_complementary_metrics(self, records):
        complementary = []
        self._complementary = {}
        for record in records:
            metric = record["metric"]
            if metric in complementary_metrics:
                task = record["task"]
                dataset = record["dataset"]
                comp_metric = complementary_metrics[record["metric"]]
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
        records = self._read_json(path)
        self._records = records + self._get_complementary_metrics(records)
        return [(r["task"], r["dataset"], r["metric"]) for r in self._records]

    def _get_axis(self, axis):
        return set(x[axis] for x in self._records)

    def _read_metrics_info(self, path):
        records = self._read_json(path)
        metrics_info = {}
        for r in records:
            task, dataset, metric = r['task'], r['dataset'], r['metric']
            d = 1 if r['higher_is_better'] else -1
            metrics_info[(task, dataset, metric)] = d
            metrics_info[metric] = metrics_info.get(metric, 0) + d
        return metrics_info
