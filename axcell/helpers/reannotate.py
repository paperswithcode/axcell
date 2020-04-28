#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import requests
from axcell import config
from axcell.data.paper_collection import _load_annotated_papers


def run_graphql_query(query):
    request = requests.post(config.graphql_url, json={'query': query})
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception(f"Query error: status code {request.status_code}")


def reannotate_paper(paper, annotations):
    paper._annotations = annotations
    paper.gold_tags = annotations.gold_tags.strip()
    for table in paper.tables:
        table._set_annotations(annotations.table_set.filter(name=table.name, parser="latexml")[0])


def reannotate_papers(papers, annotations):
    for paper in papers:
        ann = annotations.get(paper.arxiv_no_version)
        if ann is not None:
            reannotate_paper(paper, ann)


def query_annotations():
    raw = run_graphql_query("""
    query {
      allPapers {
        edges {
          node {
            arxivId
            goldTags
            tableSet {
              edges {
                node {
                  name
                  datasetText
                  notes
                  goldTags
                  matrixGoldTags
                  cellsSotaRecords
                  parser
                }
              }
            }
          }
        }
      }
    }
    """)
    return _load_annotated_papers(raw)


def reannotate_papers_with_db(papers):
    annotations = query_annotations()
    reannotate_papers(papers, annotations)
