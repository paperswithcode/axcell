#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pandas as pd
from django.db import connection
from IPython.core.display import display

from axcell.models.linking.metrics import Metrics
from axcell.models.linking.format import extract_value


def q(query, limit=10, index_col=None):
    if limit is not None:
        query = query.rstrip(" ;") + f" LIMIT {limit}"
    return pd.read_sql(query, connection, index_col=index_col)

def execute_model_on_papers(model, papers):
    proposals = []
    for paper in papers:
        print("Parsing ", paper.paper_id)
        paper_proposals = model(paper.paper_id, paper, paper.tables)
        proposals.append(paper_proposals)
    proposals = pd.concat(proposals)
    proposals["experiment_name"] = model.__name__
    return proposals.set_index('cell_ext_id')


def fetch_gold_sota_records():
    gold_sota_records = q("""
    SELECT sc.id as cell_id,
        st.paper_id, 
        CONCAT(st.paper_id, '/', st.name, '/', sr.row,'.', sr.col) as cell_ext_id, 
        (SELECT gold_tags FROM sota_cell WHERE (row=sc.row or col=sc.col) and table_id=sc.table_id and gold_tags LIKE 'model%' LIMIT 1) as model_type,
        task, dataset, metric, model, format, sc.value as raw_value
    FROM 
        sota_record sr 
    JOIN sota_cell sc USING (table_id, row, col)
    JOIN sota_table st ON (sc.table_id=st.id)
    WHERE parser = 'latexml' and dataset != '' and task != '' and metric != '' and model != '';""", limit=None)
    gold_sota_records["parsed"] = gold_sota_records[["raw_value", "format"]].apply(
        lambda row: float(extract_value(row.raw_value, row.format)), axis=1)

    unparsed = gold_sota_records[gold_sota_records["parsed"] != gold_sota_records["parsed"]]
    if len(unparsed):
        print("Found unparsed values")
        display(unparsed.style.format({'cell_ext_id':
            lambda x: f'<a target="labeler" href="http://10.0.1.145:8001/paper/{x}">{x}</a>'})
        )

    gold_sota_records = gold_sota_records[gold_sota_records["parsed"] == gold_sota_records["parsed"]]

    strip_cols=["task", "dataset", "format", "metric",  "raw_value", "model", "model_type"]
    gold_sota_records = gold_sota_records.transform(
        lambda x: x.str.strip() if x.name in strip_cols else x)
    gold_sota_records = gold_sota_records.set_index('cell_ext_id')
    return gold_sota_records

def fetch_gold_sota_papers():
    return q("""
    SELECT st.paper_id
    FROM 
        sota_record sr 
    JOIN sota_cell sc USING (table_id, row, col)
    JOIN sota_table st ON (sc.table_id=st.id)
    WHERE parser = 'latexml' and dataset != '' and task != '' and metric != '' and model != ''
    GROUP BY st.paper_id;""", limit=None)["paper_id"].tolist()

class Evaluator():
    def __init__(self, model, paper_collection):
        self.model = model
        self.pc = paper_collection
        self.annotated_papers = fetch_gold_sota_papers()
        self.raw_proposals = None

    def run_model(self):
        papers = [paper for paper in self.pc if paper.paper_id in self.annotated_papers]
        self.raw_proposals = execute_model_on_papers(model=self.model, papers=papers)

    def evaluate(self, proposals_filter, track_proposals=False):
        if self.raw_proposals is None:
            self.run_model()
        if track_proposals:
            all_proposals = self.raw_proposals.copy(deep=True)
        else:
            all_proposals = None
        proposals = proposals_filter(self.raw_proposals, all_proposals)
        gold_sota_records = fetch_gold_sota_records()
        df = gold_sota_records.merge(proposals, 'outer', left_index=True, right_index=True, suffixes=['_gold', '_pred'])
        df = df.reindex(sorted(df.columns), axis=1)
        df = df.fillna('not-present')
        if "experiment_name" in df.columns:
            del df["experiment_name"]

        metrics = Metrics(df, experiment_name=self.model.__name__)
        if track_proposals:
            return metrics, all_proposals
        else:
            return metrics
