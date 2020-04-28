#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from axcell.models.linking.metrics import Metrics
from ..models.structure import TableType
from ..loggers import StructurePredictionEvaluator, LinkerEvaluator, FilteringEvaluator
import pandas as pd
import numpy as np
from ..helpers.jupyter import table_to_html
from axcell.models.linking.format import extract_value
from axcell.helpers.optimize import optimize_filters


class Reason:
    pass


class IrrelevantTable(Reason):
    def __init__(self, paper, table, table_type, probs):
        self.paper = paper
        self.table = table
        self.table_type = table_type
        self.probs = pd.DataFrame(probs, columns=["type", "probability"])

    def __str__(self):
        return f"Table {self.table.name} was labelled as {self.table_type.name}."

    def _repr_html_(self):
        prediction = f'<div>{self}</div>'
        caption = f'<div>Caption: {self.table.caption}</div>'
        probs = self.probs.style.format({"probability": "{:.2f}"})._repr_html_()
        return prediction + caption + probs


class MislabeledCell(Reason):
    def __init__(self, paper, table, row, col, probs):
        self.paper = paper
        self.table = table


class TableExplanation:
    def __init__(self, paper, table, table_type, proposals, reasons, topk):
        self.paper = paper
        self.table = table
        self.table_type = table_type
        self.proposals = proposals
        self.reasons = reasons
        self.topk = topk

    def _format_tooltip(self, proposal):
        return f"dataset: {proposal.dataset}\n" \
            f"metric: {proposal.metric}\n" \
            f"task: {proposal.task}\n" \
            f"score: {proposal.parsed}\n" \
            f"confidence: {proposal.confidence:0.2f}"

    def _format_topk(self, topk):
        return ""

    def _repr_html_(self):
        matrix = self.table.matrix_html.values
        predictions = np.zeros_like(matrix, dtype=object)
        tooltips = np.zeros_like(matrix, dtype=object)
        for cell_ext_id, proposal in self.proposals.iterrows():
            paper_id, table_name, rc = cell_ext_id.split("/")
            row, col = [int(x) for x in rc.split('.')]
            if cell_ext_id in self.reasons:
                reason = self.reasons[cell_ext_id]
                tooltips[row, col] = reason
                if reason.startswith("replaced by "):
                    tooltips[row, col] += "\n\n" + self._format_tooltip(proposal)
                elif reason.startswith("confidence "):
                    tooltips[row, col] += "\n\n" + self._format_topk(self.topk[row, col])
            else:
                predictions[row, col] = 'final-proposal'
                tooltips[row, col] = self._format_tooltip(proposal)

        table_type_html = f'<div>Table {self.table.name} was labelled as {self.table_type.name}.</div>'
        caption_html = f'<div>Caption: {self.table.caption}</div>'
        table_html = table_to_html(matrix,
                                   self.table.matrix_tags.values,
                                   self.table.matrix_layout.values,
                                   predictions,
                                   tooltips)
        html = table_type_html + caption_html + table_html
        proposals = self.proposals[~self.proposals.index.isin(self.reasons.index)]
        if len(proposals):
            proposals = proposals[["dataset", "metric", "task", "model", "parsed"]]\
                .reset_index(drop=True).rename(columns={"parsed": "score"})
            html2 = proposals._repr_html_()
            return f"<div><div>{html}</div><div>Proposals</div><div>{html2}</div></div>"
        return html


class Explainer:
    _sota_record_columns = ['task', 'dataset', 'metric', 'format', 'model', 'model_type', 'raw_value', 'parsed']

    def __init__(self, pipeline_logger, paper_collection, gold_sota_records=None):
        self.paper_collection = paper_collection
        self.gold_sota_records = gold_sota_records
        self.spe = StructurePredictionEvaluator(pipeline_logger, paper_collection)
        self.le = LinkerEvaluator(pipeline_logger)
        self.fe = FilteringEvaluator(pipeline_logger)

    def explain(self, paper, cell_ext_id):
        paper_id, table_name, rc = cell_ext_id.split('/')
        if paper.paper_id != paper_id:
            return "No such cell"

        table_type, probs = self.spe.get_table_type_predictions(paper_id, table_name)

        if table_type == TableType.IRRELEVANT:
            return IrrelevantTable(paper, paper.table_by_name(table_name), table_type, probs)

        all_proposals = self.le.proposals[paper_id]
        reasons = self.fe.reason
        table_ext_id = f"{paper_id}/{table_name}"
        table_proposals = all_proposals[all_proposals.index.str.startswith(table_ext_id+"/")]
        topk = {(row, col): topk for (pid, tn, row, col), topk in self.le.topk.items()
                if (pid, tn) == (paper_id, table_name)}

        return TableExplanation(paper, paper.table_by_name(table_name), table_type, table_proposals, reasons, topk)

        row, col = [int(x) for x in rc.split('.')]

        reason = self.fe.reason.get(cell_ext_id)
        if reason is None:
            pass
        else:
            return reason

    def _get_table_sota_records(self, table):

        first_model = lambda x: ([a for a in x if a.startswith('model')] + [''])[0]
        if len(table.sota_records):
            matrix = table.matrix.values
            tags = table.matrix_tags
            model_type_col = tags.apply(first_model)
            model_type_row = tags.T.apply(first_model)
            sota_records = table.sota_records.copy()
            sota_records['model_type'] = ''
            sota_records['raw_value'] = ''
            for cell_ext_id, record in sota_records.iterrows():
                name, rc = cell_ext_id.split('/')
                row, col = [int(x) for x in rc.split('.')]
                record.model_type = model_type_col[col] or model_type_row[row]
                record.raw_value = matrix[row, col]

            sota_records["parsed"] = sota_records[["raw_value", "format"]].apply(
                lambda row: float(extract_value(row.raw_value, row.format)), axis=1)

            sota_records = sota_records[sota_records["parsed"] == sota_records["parsed"]]

            strip_cols = ["task", "dataset", "format", "metric", "raw_value", "model", "model_type"]
            sota_records = sota_records.transform(
                lambda x: x.str.strip() if x.name in strip_cols else x)
            return sota_records[self._sota_record_columns]
        else:
            empty = pd.DataFrame(columns=self._sota_record_columns)
            empty.index.rename("cell_ext_id", inplace=True)
            return empty

    def _get_sota_records(self, paper):
        if not len(paper.tables):
            empty = pd.DataFrame(columns=self._sota_record_columns)
            empty.index.rename("cell_ext_id", inplace=True)
            return empty
        records = [self._get_table_sota_records(table) for table in paper.tables]
        records = pd.concat(records)
        records.index = paper.paper_id + "/" + records.index
        records.index.rename("cell_ext_id", inplace=True)
        return records

    def linking_metrics(self, experiment_name="unk", topk_metrics=False, filtered=True, confidence=0.0):
        paper_ids = list(self.le.proposals.keys())

        proposals = pd.concat(self.le.proposals.values())

        # if not topk_metrics:
        if filtered:
            proposals = proposals[~proposals.index.isin(self.fe.reason.index)]
        if confidence:
            proposals = proposals[proposals.confidence > confidence]

        papers = {paper_id: self.paper_collection.get_by_id(paper_id) for paper_id in paper_ids}
        missing = [paper_id for paper_id, paper in papers.items() if paper is None]
        if missing:
            print("Missing papers in paper collection:")
            print(", ".join(missing))
        papers = [paper for paper in papers.values() if paper is not None]

        # if not len(papers):
        #     gold_sota_records = pd.DataFrame(columns=self._sota_record_columns)
        #     gold_sota_records.index.rename("cell_ext_id", inplace=True)
        # else:
        #     gold_sota_records = pd.concat([self._get_sota_records(paper) for paper in papers])
        if self.gold_sota_records is None:
            gold_sota_records = pd.DataFrame(columns=self._sota_record_columns)
            gold_sota_records.index.rename("cell_ext_id", inplace=True)
        else:

            gold_sota_records = self.gold_sota_records
            which = gold_sota_records.index.to_series().str.split("/", expand=True)[0]\
                .isin([paper.paper_id for paper in papers])
            gold_sota_records = gold_sota_records[which]

        df = gold_sota_records.merge(proposals, 'outer', left_index=True, right_index=True, suffixes=['_gold', '_pred'])
        df = df.reindex(sorted(df.columns), axis=1)
        df = df.fillna('not-present')
        if "experiment_name" in df.columns:
            del df["experiment_name"]

        metrics = Metrics(df, experiment_name=experiment_name, topk_metrics=topk_metrics)
        return metrics


    def optimize_filters(self, metrics_info):
        results = optimize_filters(self, metrics_info)
        return results
