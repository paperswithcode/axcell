#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re
from decimal import Decimal, localcontext, InvalidOperation
from dataclasses import dataclass
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, client
import logging
#from .extractors import DatasetExtractor
import spacy
from scispacy.abbreviation import AbbreviationDetector
from axcell.models.linking.format import extract_value


@dataclass()
class Value:
    type: str
    value: str
    def __str__(self):
        return self.value


@dataclass()
class Cell:
    cell_ext_id: str
    table_ext_id: str
    row: int
    col: int


@dataclass()
class Proposal:
    cell: Cell
    dataset_values: list
    table_description: str
    model_values: list  # best paper competing
    model_params: dict = None
    raw_value: str = ""

    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {}

    @property
    def dataset(self):
        return ' '.join(map(str, self.dataset_values)).strip()

    @property
    def model_name(self):
        return ' '.join(map(str, self.model_values)).strip()

    @property
    def model_type(self):
        types = [v.type for v in self.model_values] + ['']
        if 'model-competing' in types:
            return 'model-competing' # competing model is different from model-paper and model-best so we return it first
        return types[0]

    def __str__(self):
        return f"{self.model_name}: {self.raw_value} on {self.dataset}"


class MetricValue:
    value: Decimal
    unit: str = None

    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

    def to_unitless(self):
        return self.value

    def to_absolute(self):
        return self.value / Decimal(100) if self.unit is '%' else self.value

    # unit = None means that no unit was specified, so we have to guess the unit.
    # if there's a value "21" in a table's cell, then we guess if it's 21 or 0.21 (i.e., 21%)
    # based on the target metric properties.
    def to_percentage(self):
        if self.unit is None and 0 < self.value < 1:
            return self.value * 100
        return self.value

    def complement(self):
        if self.unit is None:
            if 1 < self.value < 100:
                value = 100 - self.value
            else:
                value = 1 - self.value
        else:
            value = 100 - self.value
        return MetricValue(value, self.unit)

    def __repr__(self):
        return f"MetricValue({self.value}, {repr(self.unit)})"

    def __str__(self):
        return str(self.value)


def mkquery_ngrams(query):
    return {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["dataset^3", "dataset.ngrams^1", "metric^1", "metric.ngrams^1", "task^1",
                               "task.ngrams^1"]
         }
      }
    }


def mkquery_fullmatch(query):
    return {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["dataset^3", "metric^1", "task^1"]
            }
        }
    }

class MatchSearch:
    def __init__(self, mkquery=mkquery_ngrams, es=None):
        self.case = True
        self.all_fields = True
        self.es = es or Elasticsearch()
        self.log = logging.getLogger(__name__)
        self.mkquery = mkquery

        self.nlp = spacy.load("en_core_web_sm")
        abbreviation_pipe = AbbreviationDetector(self.nlp)
        self.nlp.add_pipe(abbreviation_pipe)
        self.nlp.disable_pipes("tagger", "ner", "parser")

    def match_abrv(self, dataset, datasets):
        abrvs = []
        for ds in datasets:
            # "!" is a workaround to scispacy error
            doc = self.nlp(f"! {ds} ({dataset})")
            for abrv in doc._.abbreviations:
                if str(abrv) == dataset and str(abrv._.long_form) == ds:
                    abrvs.append(str(abrv._.long_form))
        abrvs = list(set(abrvs))
        if len(abrvs) == 1:
            print(f"abrv. for {dataset}: {abrvs[0]}")
            return abrvs[0]
        elif len(abrvs) == 0:
            return None
        else:
            print(f"Multiple abrvs. for {dataset}: {abrvs}")
            return None

    def preproc(self, val, datasets=None):
        val = val.strip(',- ')
        val = re.sub("dataset", '', val, flags=re.I)
        if datasets:
            abrv = self.match_abrv(val, datasets)
            if abrv:
                val += " " + abrv
        #         if self.case:
        #             val += (" " +re.sub("([a-z])([A-Z])", r'\1 \2', val)
        #                     +" " +re.sub("([a-zA-Z])([0-9])", r'\1 \2', val)
        #                    )
        return val

    def search(self, query, explain_doc_id=None):
        body = self.mkquery(query)
        if explain_doc_id is not None:
            return self.es.explain('et_taxonomy', doc_type='doc', id=explain_doc_id, body=body)
        return self.es.search('et_taxonomy', doc_type='doc', body=body)["hits"]

    def __call__(self, query, datasets, caption):
        split_re = re.compile('([^a-zA-Z0-9])')
        query = self.preproc(query, datasets).strip()
        if caption:
            query += " " + self.preproc(caption).strip()[:400]
        results = self.search(query)
        hits = results["hits"][:3]
        df = pd.DataFrame.from_records([
            dict(**hit["_source"],
                 confidence=hit["_score"] / len(split_re.split(query)),
                 # Roughly normalize the score not to ignore query length
                 evidence=query) for hit in hits
        ], columns=["dataset", "metric", "task", "confidence", "evidence"])
        if not len(df):
            self.log.debug("Elastic query didn't produce any output", query, hits)
        else:
            scores = []
            for dataset in df["dataset"]:
                r = self.search(dataset)
                scores.append(
                    dict(ok_score=r['hits'][0]['_score'] / len(split_re.split(dataset)),
                         bad_score=r['hits'][1]['_score'] / len(split_re.split(dataset))))

            scores = pd.DataFrame.from_records(scores)
            df['confidence'] = ((scores['ok_score'] - scores['bad_score']) / scores['bad_score']) * df['confidence'] / scores['ok_score']
        return df[["dataset", "metric", "task", "confidence", "evidence"]]

float_pm_re = re.compile(r"(±?)([+-]?\s*(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s*(%?)")
whitespace_re = re.compile(r"\s+")
def handle_pm(value):
    "handle precentage metric"
    for match in float_pm_re.findall(value):
        if not match[0]:
            try:
                percent = bool(match[-1])
                value = Decimal(whitespace_re.sub("", match[1])) / (100 if percent else 1)
                yield MetricValue(value, "%" if percent else None)
            except:
                pass
            # %%


def convert_metric(raw_value, rng, complementary):
    format = "{x}"

    percentage = '%' in raw_value
    if percentage:
        format += '%'

    with localcontext() as ctx:
        ctx.traps[InvalidOperation] = 0
        parsed = extract_value(raw_value, format)
        parsed = MetricValue(parsed, '%' if percentage else None)

        if complementary:
            parsed = parsed.complement()
        if rng == '0-1':
            parsed = parsed.to_percentage() / 100
        elif rng == '1-100':
            parsed = parsed.to_percentage()
        elif rng == 'abs':
            parsed = parsed.to_absolute()
        else:
            parsed = parsed.to_unitless()
    return parsed

proposal_columns = ['dataset', 'metric', 'task', 'format', 'raw_value', 'model', 'model_type', 'cell_ext_id',
                    'confidence', 'parsed', 'struct_model_type', 'struct_dataset']


# generator of all result-like cells
def generate_cells_proposals(table_ext_id,  matrix, structure, desc):
    # %%
    # Proposal generation
    def consume_cells(matrix):
        for row_id, row in enumerate(matrix):
            for col_id, cell in enumerate(row):
                yield (row_id, col_id, cell)


    def annotations(r, c, type='model'):
        for nc in range(0, c):
            if type in structure[r, nc]:
                yield Value(structure[r, nc], matrix[r, nc])
        for nr in range(0, r):
            if type in structure[nr, c]:
                yield Value(structure[nr, c], matrix[nr, c])

    number_re = re.compile(r'(^[± Ee/()^0-9.%,_+-]{2,}$)|(^\s*[0-9]\s*$)')

    proposals = [Proposal(
        cell=Cell(cell_ext_id=f"{table_ext_id}/{r}.{c}",
                  table_ext_id=table_ext_id,
                  row=r,
                  col=c
                  ),
        # TODO Add table type: sota / error ablation
        table_description=desc,
        model_values=list(annotations(r, c, 'model')),
        dataset_values=list(annotations(r, c, 'dataset')),
        raw_value=val)
        for r, c, val in consume_cells(matrix)
        if structure[r, c] == '' and number_re.match(matrix[r, c].strip())]
    return proposals


def link_cells_proposals(proposals, desc, taxonomy_linking,
                         paper_context, abstract_context, table_context, topk=1):
    for prop in proposals:
        # heuristyic to handle accuracy vs error
        format = "{x}"

        percentage = '%' in prop.raw_value
        if percentage:
            format += '%'

        df = taxonomy_linking(prop.dataset, paper_context, abstract_context, table_context,
                              desc, topk=topk, debug_info=prop)
        for _, row in df.iterrows():
            raw_value = prop.raw_value
            task = row['task']
            dataset = row['dataset']
            metric = row['metric']

            complementary = False
            if metric != row['true_metric']:
                metric = row['true_metric']
                complementary = True

            # todo: pass taxonomy directly to proposals generation
            ranges = taxonomy_linking.taxonomy.metrics_range
            key = (task, dataset, metric)
            rng = ranges.get(key, '')
            if not rng: rng = ranges.get(metric, '')

            parsed = float(convert_metric(raw_value, rng, complementary))

            linked = {
                'dataset': dataset,
                'metric': metric,
                'task': task,
                'format': format,
                'raw_value': raw_value,
                'model': prop.model_name,
                'model_type': prop.model_type,
                'cell_ext_id': prop.cell.cell_ext_id,
                'confidence': row['confidence'],
                'struct_model_type': prop.model_type,
                'struct_dataset': prop.dataset,
                'parsed': parsed
            }
            yield linked



def generate_proposals_for_table(table_ext_id,  matrix, structure, desc, taxonomy_linking,
                                 paper_context, abstract_context, table_context, topk=1):

    # def empty_proposal(cell_ext_id, reason):
    #     np = "not-present"
    #     return dict(
    #         dataset=np, metric=np, task=np, format=np, raw_value=np, model=np,
    #         model_type=np, cell_ext_id=cell_ext_id, confidence=-1, debug_reason=reason
    #     )



    proposals = generate_cells_proposals(table_ext_id, matrix, structure, desc)
    proposals = link_cells_proposals(proposals, desc, taxonomy_linking, paper_context, abstract_context,
                                     table_context, topk=topk)

    # specify columns in case there's no proposal
    proposals = pd.DataFrame.from_records(list(proposals), columns=proposal_columns)
    return proposals


def linked_proposals(paper_ext_id, paper, annotated_tables, taxonomy_linking=None,
                     dataset_extractor=None, topk=1):
    #                     dataset_extractor=DatasetExtractor()):
    proposals = []
    paper_context, abstract_context = dataset_extractor.from_paper(paper)
    table_contexts = dataset_extractor.get_table_contexts(paper, annotated_tables)
    #print(f"Extracted datasets: {datasets}")
    for idx, (table, table_context) in enumerate(zip(annotated_tables, table_contexts)):
        matrix = np.array(table.matrix)
        structure = np.array(table.matrix_tags)
        tags = 'sota'
        desc = table.caption
        table_ext_id = f"{paper_ext_id}/{table.name}"

        if 'sota' in tags and 'no_sota_records' not in tags: # only parse tables that are marked as sota
            proposals.append(
                generate_proposals_for_table(
                    table_ext_id, matrix, structure, desc, taxonomy_linking,
                    paper_context, abstract_context, table_context,
                    topk=topk
                )
            )
    if len(proposals):
        return pd.concat(proposals)
    return pd.DataFrame(columns=proposal_columns)


def test_link_taxonomy():
    link_taxonomy_raw = MatchSearch()
    results = link_taxonomy_raw.search(link_taxonomy_raw.preproc("miniImageNet 5-way 1-shot"))
    # assert "Mini-ImageNet - 1-Shot Learning" == results["hits"][0]["_source"]["dataset"], results
    results = link_taxonomy_raw.search(link_taxonomy_raw.preproc("CoNLL2003"))
    assert "CoNLL 2003 (English)" == results["hits"][0]["_source"]["dataset"], results
    results = link_taxonomy_raw.search(link_taxonomy_raw.preproc("AGNews"))
    assert "AG News" == results["hits"][0]["_source"]["dataset"], results
    link_taxonomy_raw("miniImageNet 5-way 1-shot")
    # %%
    split_re = re.compile('([^a-zA-Z0-9])')

    # %%
    q = "miniImageNet 5-way 1-shot Mini ImageNet 1-Shot Learning" * 1
    r = link_taxonomy_raw.search(q)
    f = len(split_re.split(q))
    r['hits'][0]['_score'] / f, r['hits'][1]['_score'] / f, r['hits'][0]['_source']
    # %%
    q = "Mini ImageNet 1-Shot Learning" * 1
    r = link_taxonomy_raw.search(q)
    f = len(split_re.split(q))
    r['hits'][0]['_score'] / f, r['hits'][1]['_score'] / f, r['hits'][0]['_source']
    # %%
    q = "Mini ImageNet 1-Shot" * 1
    r = link_taxonomy_raw.search(q)
    f = len(split_re.split(q))
    r['hits'][0]['_score'] / f, r['hits'][1]['_score'] / f, r['hits'][0]['_source']
    #
    # # %%
    # prop = proposals[1]
    # print(prop)
    # # todo issue with STS-B matching IJB-B
    # link_taxonomy_raw(prop.dataset)


