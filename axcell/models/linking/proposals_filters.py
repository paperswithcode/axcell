#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from ...pipeline_logger import pipeline_logger
import pandas as pd
from enum import Enum


class FilterOutReason(Enum):
    TrainDataset = "train-dataset"
    DevDataset = "dev-dataset"
    EmptyModelName = "empty-model-name"
    ModelCompeting = "model-competing"


class ProposalsFilter:
    step = "proposals_filtering"

    def _filter(self, proposals):
        raise NotImplementedError

    def filter(self, proposals):
        which, reason = self._filter(proposals)
        self.log(proposals=proposals, which=which, reason=reason)
        return which, reason

    def __rshift__(self, other):
        return CompoundFilter([self, other])

    def __call__(self, proposals):
        which, reason = self.filter(proposals)
        return proposals[which]

    def log(self, **kwargs):
        pipeline_logger(f"filtering::{self.step}::filtered", **kwargs)


class CompoundFilter(ProposalsFilter):
    step = "compound_filtering"

    def __init__(self, filters):
        self.filters = filters

    def _filter(self, proposals):
        agg_which = pd.Series(data=True, index=proposals.index)
        agg_reason = pd.Series(data="", index=proposals.index)

        for f in self.filters:
            which, reason = f.filter(proposals)
            agg_reason[agg_which & ~which] = reason
            agg_which &= which
            proposals = proposals[which]
        return agg_which, agg_reason[~agg_which]


class NopFilter(ProposalsFilter):
    step = "nop_filtering"

    def _filter(self, proposals):
        which = pd.Series(data=True, index=proposals.index)
        reason = pd.Series()
        return which, reason


# filter proposals for which structure prediction
# * was unable to find model type or
# * found dataset cell containing "dev" or "train"
# this filter could be applied before taxonomy linking,
# but to make error analysis easier it's applied after
class StructurePredictionFilter(ProposalsFilter):
    step = "structure_filtering"

    def _filter(self, proposals):
        which = (proposals.struct_model_type != '') \
                         & ~proposals.struct_dataset.str.contains('dev') \
                         & ~proposals.struct_dataset.str.contains('train')
        reason = pd.Series(data="", index=proposals.index)
        reason[proposals.struct_dataset.str.contains('train')] = "train-dataset"
        reason[proposals.struct_dataset.str.contains('dev')] = "dev-dataset"
        reason[proposals.struct_model_type == ''] = "empty-model-type"

        return which, reason[~which]


class ConfidenceFilter(ProposalsFilter):
    step = "confidence_filtering"

    def __init__(self, confidence=-1):
        self.confidence = confidence

    def _filter(self, proposals):
        which = proposals.confidence >= self.confidence
        reason = "confidence " + proposals[~which].confidence.round(2).astype(str) + f" < {self.confidence}"
        return which, reason[~which]

    def log(self, **kwargs):
        super().log(**kwargs, confidence=self.confidence)


class BestResultFilter(ProposalsFilter):
    step = "best_result_filtering"

    def __init__(self, taxonomy, context="paper"):
        assert context in ["paper", "table"]
        self.metrics_info = taxonomy.metrics_info
        self.context = context

    def _filter(self, proposals):
        reason = pd.Series(data="", index=proposals.index)
        indices = []

        if self.context == "paper":
            context_column = proposals.index.to_series().str.split('/', expand=False).apply(lambda x: x[0])
        else:
            context_column = proposals.index.to_series().str.split('/', expand=False).apply(lambda x: x[0] + "/" + x[1])

        for key_all, group in proposals[(proposals.model_type == 'model-best') & ~proposals.parsed.isna()].groupby(
                by=["dataset", "metric", "task", context_column]):
            dataset, metric, task, paper = key_all
            key = (task, dataset, metric)
            d = 0
            if key in self.metrics_info:
                d = self.metrics_info[key]
            elif metric in self.metrics_info:
                d = self.metrics_info[metric]
            elif 'error' in metric.lower():
                d = -1
            elif 'accuracy' in metric.lower():
                d = 1

            if d >= 0:
                index = group.parsed.idxmax()
            else:
                index = group.parsed.idxmin()
            indices.append(index)
            reason[group.index[group.index != index]] = "replaced by " + str(index)

        reason[proposals.struct_model_type == 'model-competing'] = "model-competing"
        which = proposals.index.to_series().isin(indices)
        return which, reason[~which]

    def log(self, **kwargs):
        super().log(**kwargs, context=self.context)
