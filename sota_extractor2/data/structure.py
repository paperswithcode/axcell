import re
import pandas as pd
from collections import namedtuple
import hashlib
from fastai.text import progress_bar
from .elastic import Fragment
from .json import *

def get_all_tables(papers):
    for paper in papers:
        for table in paper.table_set.all():
            if 'trash' not in table.gold_tags and table.gold_tags != '':
                table.paper_id = paper.arxiv_id
                yield table

def consume_cells(*matrix):
    Cell = namedtuple('AnnCell', 'row col vals')
    for row_id, row in enumerate(zip(*matrix)):
        for col_id, cell_val in enumerate(zip(*row)):
            yield Cell(row=row_id, col=col_id, vals=cell_val)


def fetch_evidence(cell_content, paper_id, paper_limit=10, corpus_limit=10):
    evidence_query = Fragment.search().highlight(
        'text', pre_tags="<b>", post_tags="</b>", fragment_size=400)
    cell_content = cell_content.replace("\xa0", " ")
    query = {
        "query": cell_content,
        "slop": 2
    }
    paper_fragments = list(evidence_query
                           .filter('term', paper_id=paper_id)
                           .query('match_phrase', text=query)[:paper_limit])
    other_fagements = list(evidence_query
                           .exclude('term', paper_id=paper_id)
                           .query('match_phrase', text=query)[:corpus_limit])
    return paper_fragments + other_fagements

fix_refs_re = re.compile('\(\?\)|\s[?]+(\s|$)')


def fix_refs(text):
    return fix_refs_re.sub(' xref-unkown ', fix_refs_re.sub(' xref-unkown ', text))


highlight_re = re.compile("</?b>")


def create_evidence_records(textfrag, cell, table):
    for text_highlited in textfrag.meta['highlight']['text']:
        text_highlited = fix_refs(text_highlited)
        text = highlight_re.sub("", text_highlited)
        text_sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()

        cell_ext_id = f"{table.ext_id}/{cell.row}/{cell.col}"

        if len(text.split()) > 50:
            yield {"text_sha1": text_sha1,
                   "text_highlited": text_highlited,
                   "text": text,
                   "header": textfrag.header,
                   "cell_type": cell.vals[1],
                   "cell_content": fix_refs(cell.vals[0]),
                   "this_paper": textfrag.paper_id == table.paper_id,
                   "row": cell.row,
                   "col": cell.col,
                   "ext_id": cell_ext_id
                   #"table_id":table_id
                   }


def filter_cells(cell):
    return re.search("[a-zA-Z]{2,}", cell.vals[1]) is not None


def evidence_for_table(table, paper_limit=10, corpus_limit=1):
    records = [
        record
            for cell in consume_cells(table.matrix, table.matrix_gold_tags) if filter_cells(cell)
            for evidence in fetch_evidence(cell.vals[0], paper_id=table.paper_id, paper_limit=paper_limit, corpus_limit=corpus_limit)
            for record in create_evidence_records(evidence, cell, table=table)
    ]
    df = pd.DataFrame.from_records(records)
    return df


def evidence_for_tables(tables, paper_limit=100, corpus_limit=20):
    return pd.concat([evidence_for_table(table,  paper_limit=paper_limit, corpus_limit=corpus_limit) for table in progress_bar(tables)])

def prepare_data(tables, csv_path):
    df = evidence_for_tables(tables)
    df = df.drop_duplicates(
        ["cell_content", "text_highlited", "cell_type", "this_paper"])
    print("Number of text fragments ", len(df))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=None)
