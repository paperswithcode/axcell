import pandas as pd
from django.db import connection

from sota import models
from sota.pipeline.format import extract_value
from sota.pipeline.metrics import Metrics


def q(query, limit=10, index_col=None):
    if limit is not None:
        query = query.rstrip(" ;") + f" LIMIT {limit}"
    return pd.read_sql(query, connection, index_col=index_col)

def execute_model_on_papers(model, papers):
    papers = models.Paper.objects.filter(pk__in=papers)
    proposals = []
    for paper in papers:
        print("Parsing ", paper.id)
        paper_proposals = model(paper.id, paper.table_set.all())
        proposals.append(paper_proposals)
    proposals = pd.concat(proposals)
    proposals["parsed"]=proposals[["raw_value", "format"]].apply(
        lambda row: float(extract_value(row.raw_value, row.format)), axis=1)
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
    WHERE dataset != '' and task != '' and metric != '' and model != '';""", limit=None)
    gold_sota_records["parsed"] = gold_sota_records[["raw_value", "format"]].apply(
        lambda row: float(extract_value(row.raw_value, row.format)), axis=1)

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
    WHERE dataset != '' and task != '' and metric != '' and model != ''
    GROUP BY st.paper_id;""", limit=None)["paper_id"].tolist()

class Evaluator():
    def __init__(self, model):
        self.model = model
        self.annotated_papers = fetch_gold_sota_papers()
        self.raw_proposals = None

    def run_model(self):
        self.raw_proposals = execute_model_on_papers(model=self.model, papers=self.annotated_papers)

    def evaluate(self, confidence=-1):
        if self.raw_proposals is None:
            self.run_model()
        proposals = self.raw_proposals[self.raw_proposals['confidence'] > confidence]
        gold_sota_records = fetch_gold_sota_records()
        df = gold_sota_records.merge(proposals, 'outer', left_index=True, right_index=True, suffixes=['_gold', '_pred'])
        df = df.reindex(sorted(df.columns), axis=1)
        df = df.fillna('not-present')
        if "experiment_name" in df.columns:
            del df["experiment_name"]

        return Metrics(df, experiment_name=self.model.__name__)