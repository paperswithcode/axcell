import pandas as pd
import numpy as np
import json


class Tabular:
    def __init__(self, data, caption):
        self.data = data
        self.cell_tags = pd.DataFrame().reindex_like(data).fillna('')
        self.datasets = set()
        self.metrics = set()
        self.caption = caption

    def mark_with_metric(self, metric_name):
        self.metrics.add(metric_name)

    def mark_with_dataset(self, dataset_name):
        self.datasets.add(dataset_name)


