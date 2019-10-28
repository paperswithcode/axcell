from pathlib import Path
import json
from collections import OrderedDict



class Taxonomy:
    def __init__(self, taxonomy, metrics_info):
        self.taxonomy = self._read_taxonomy(taxonomy)
        self.metrics_info = self._read_metrics_info(metrics_info)

    def _read_json(self, path):
        with open(path, "rt") as f:
            return json.load(f)

    def _read_taxonomy(self, path):
        records = self._read_json(path)
        return [(r["task"], r["dataset"], r["metric"]) for r in records]

    def _read_metrics_info(self, path):
        records = self._read_json(path)
        metrics_info = {}
        for r in records:
            task, dataset, metric = r['task'], r['dataset'], r['metric']
            d = 1 if r['higher_is_better'] else -1
            metrics_info[(task, dataset, metric)] = d
            metrics_info[metric] = metrics_info.get(metric, 0) + d
        return metrics_info
