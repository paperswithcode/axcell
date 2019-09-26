class ProposalsFilter:
    def __rshift__(self, other):
        return CompoundFilter([self, other])


class CompoundFilter(ProposalsFilter):
    def __init__(self, filters):
        self.filters = filters

    def __call__(self, df, all_proposals=None):
        for f in self.filters:
            df = f(df, all_proposals)
        return df


class NopFilter(ProposalsFilter):
    def __call__(self, df, all_proposals=None):
        return df


# filter proposals for which structure prediction
# * was unable to find model type or
# * found dataset cell containing "dev" or "train"
# this filter could be applied before taxonomy linking,
# but to make error analysis easier it's applied after
class StructurePredictionFilter(ProposalsFilter):
    def __call__(self, proposals, all_proposals=None):
        if all_proposals is not None:
            indices = proposals.index
            all_proposals.loc[indices[proposals.struct_dataset.str.contains('train')], "reason"] = "train-dataset"
            all_proposals.loc[indices[proposals.struct_dataset.str.contains('dev')], "reason"] = "dev-dataset"
            all_proposals.loc[indices[proposals.struct_model_type == ''], "reason"] = "empty-model-type"

        return proposals[(proposals.struct_model_type != '') \
                         & ~proposals.struct_dataset.str.contains('dev') \
                         & ~proposals.struct_dataset.str.contains('train')]

class ConfidenceFilter(ProposalsFilter):
    def __init__(self, confidence=-1):
        self.confidence = confidence

    def __call__(self, proposals, all_proposals=None):
        #         proposals.loc[proposals.debug_reason.isna() & (proposals.confidence <= self.confidence), "debug_reason"] = \
        #             f"confidence<{self.confidence:.02f}"
        #         return proposals
        which = proposals.confidence > self.confidence
        if all_proposals is not None:
            all_proposals.loc[proposals.index[~which], "reason"] = \
                "confidence " + proposals[~which].confidence.round(2).astype(str) + f" <= {self.confidence}"
        return proposals[which]


# does not filter per se, but changes model-best -> model-paper
# for inferior models
class BestResultFilter(ProposalsFilter):
    def __init__(self, taxonomy, context="paper", log=False):
        assert context in ["paper", "table"]
        self.metrics_info = taxonomy.metrics_info
        self.context = context
        self.log = log

    def __call__(self, proposals, all_proposals=None):
        proposals = proposals.copy(deep=True)
        indices = []
        if self.log:
            print("filtering")
            print(proposals)

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
            if all_proposals is not None:
                all_proposals.loc[group.index[group.index != index], "reason"] = "replaced by " + str(index)

        if self.log:
            print("after filtering")
            print(proposals.loc[indices])

        which = (proposals.model_type == 'model-best') & ~(proposals.index.isin(indices))
        proposals.loc[which, "model_type"] = 'model-paper'

        # proposals.loc[(proposals.model_type == 'model-best') & ~(proposals.index.isin(indices)), "model_type"] = 'not-present'
        # proposals.model_type = 'model-paper'
        # proposals.loc[indices, "model_type"] = 'model-best'

        return proposals  # .loc[indices]