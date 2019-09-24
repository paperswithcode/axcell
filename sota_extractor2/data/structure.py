import re
import pandas as pd
from collections import namedtuple
import hashlib
from fastai.text import progress_bar
from .elastic import Fragment, setup_default_connection
from .json import *
from .table import reference_re, remove_text_styles, remove_references, style_tags_re

def get_all_tables(papers):
    for paper in papers:
        for table in paper.table_set.filter(parser="latexml"):
            if 'trash' not in table.gold_tags and table.gold_tags != '':
                table.paper_id = paper.arxiv_id
                yield table

def consume_cells(table):
    Cell = namedtuple('AnnCell', 'row col vals')
    for row_id, row in enumerate(table.df.values):
        for col_id, cell in enumerate(row):
            vals = [
                remove_text_styles(remove_references(cell.raw_value)),
                "",
                cell.refs[0] if cell.refs else "",
                cell.layout,
                bool(style_tags_re.search(cell.raw_value))
            ]
            yield Cell(row=row_id, col=col_id, vals=vals)


reference_re = re.compile(r"\[[^]]*\]")
ours_re = re.compile(r"\(ours?\)")
all_parens_re = re.compile(r"\([^)]*\)")


def clear_cell(s):
    for pat in [reference_re, all_parens_re]:
        s = pat.sub("", s)
    s = s.strip()
    return s


def empty_fragment(paper_id):
    fragment = Fragment(paper_id=paper_id)
    fragment.meta['highlight'] = {'text': ['']}
    return fragment


def fetch_evidence(cell_content, cell_reference, paper_id, table_name, row, col, paper_limit=10, corpus_limit=10):
    if not filter_cells(cell_content):
        return [empty_fragment(paper_id)]
    cell_content = clear_cell(cell_content)
    if cell_content == "" and cell_reference == "":
        return [empty_fragment(paper_id)]

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
    if cell_reference != "":
        reference_fragments = list(evidence_query
                                   .filter('term', paper_id=paper_id)
                                   .query('match_phrase', text={
                                        "query": cell_reference,
                                        "slop": 1
                                    })[:paper_limit])
    else:
        reference_fragments = []
    other_fagements = list(evidence_query
                           .exclude('term', paper_id=paper_id)
                           .query('match_phrase', text=query)[:corpus_limit])

    ext_id = f"{paper_id}/{table_name}/{row}.{col}"
    ####print(f"{ext_id} |{cell_content}|: {len(paper_fragments)} paper fragments, {len(reference_fragments)} reference fragments, {len(other_fagements)} other fragments")
    # if not len(paper_fragments) and not len(reference_fragments) and not len(other_fagements):
    #     print(f"No evidences for '{cell_content}' of {paper_id}")
    if not len(paper_fragments) and not len(reference_fragments):
        paper_fragments = [empty_fragment(paper_id)]
    return paper_fragments + reference_fragments + other_fagements

fix_refs_re = re.compile('\(\?\)|\s[?]+(\s|$)')


def fix_refs(text):
    return fix_refs_re.sub(' xref-unkown ', fix_refs_re.sub(' xref-unkown ', text))


highlight_re = re.compile("</?b>")
partial_highlight_re = re.compile(r"\<b\>xxref\</b\>-(?!\<b\>)")


def fix_reference_hightlight(s):
    return partial_highlight_re.sub("xxref-", s)


def create_evidence_records(textfrag, cell, paper, table):
    for text_highlited in textfrag.meta['highlight']['text']:
        text_highlited = fix_reference_hightlight(fix_refs(text_highlited))
        text = highlight_re.sub("", text_highlited)
        text_sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()

        cell_ext_id = f"{paper.paper_id}/{table.name}/{cell.row}/{cell.col}"

        yield {"text_sha1": text_sha1,
               "text_highlited": text_highlited,
               "text": text,
               "header": textfrag.header,
               "cell_type": cell.vals[1],
               "cell_content": fix_refs(cell.vals[0]),
               "cell_reference": cell.vals[2],
               "cell_layout": cell.vals[3],
               "cell_styles": cell.vals[4],
               "this_paper": textfrag.paper_id == paper.paper_id,
               "row": cell.row,
               "col": cell.col,
               "row_context": " border ".join([str(s) for s in table.matrix.values[cell.row]]),
               "col_context": " border ".join([str(s) for s in table.matrix.values[:, cell.col]]),
               "ext_id": cell_ext_id
               #"table_id":table_id
               }


def filter_cells(cell_content):
    return re.search("[a-zA-Z]{2,}", cell_content) is not None


interesting_types = ["model-paper", "model-best", "model-competing", "dataset", "dataset-sub",  "dataset-task"]


def evidence_for_table(paper, table, paper_limit, corpus_limit):
    records = [
        record
            for cell in consume_cells(table)
            for evidence in fetch_evidence(cell.vals[0], cell.vals[2], paper_id=paper.paper_id, table_name=table.name,
                                           row=cell.row, col=cell.col, paper_limit=paper_limit, corpus_limit=corpus_limit)
            for record in create_evidence_records(evidence, cell, paper=paper, table=table)
    ]
    df = pd.DataFrame.from_records(records)
    return df


def prepare_data(paper, tables, csv_path, limit_type='interesting'):
    df = pd.concat([evidence_for_table(paper, table,
                                       paper_limit=100,
                                       corpus_limit=20,
                                       limit_type=limit_type) for table in progress_bar(tables)])
    #moved to experiment preprocessing
    #df = df.drop_duplicates(
    #    ["cell_content", "text_highlited", "cell_type", "this_paper"])
    print("Number of text fragments ", len(df))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=None)


class CellEvidenceExtractor:
    def __init__(self):
        # todo: make sure can be called more than once or refactor to singleton
        setup_default_connection()

    def __call__(self, paper, tables, paper_limit=30, corpus_limit=10):
        dfs = [evidence_for_table(paper, table, paper_limit, corpus_limit) for table in tables]
        if len(dfs):
            return pd.concat(dfs)
        return pd.DataFrame()