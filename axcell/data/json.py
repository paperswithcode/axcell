#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#%%
import json
import re
import gzip
import pprint
import requests
from axcell import config
#%%
def to_snake_case(name):
    #https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def to_camel_case(name):
    components = name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def wrap_dict(d):
    if "edges" in d:
        return EdgeWrap(d['edges'])
    elif "node" in d:
        return NodeWrap(d["node"])
    return NodeWrap(d)

class EdgeWrap(list):

    def all(self):
        return self
    
    def filter(self, **kwargs):
        return [n for n in self if n.matches(**kwargs)]
    
    def __add__(self, rhs):
        return EdgeWrap(list.__add__(self, rhs))
    
    def __iter__(self):
        return (wrap_dict(d) for d in super().__iter__())

    def __getitem__(self, key):
        vals = list.__getitem__(self, key)
        if isinstance(vals, dict):
            return wrap_dict(vals)
        if isinstance(vals, list):
            return EdgeWrap(vals)
        return vals

    def __repr__(self):
        val = "\n".join(repr(k) for k in self)
        return f"EdgeWrap([{val}])"


class NodeWrap(dict):

    def matches(self, **kwargs):
        return all(getattr(self, k) == v for k,v in kwargs.items())

    def __getattr__(self, name):
        camel_name = to_camel_case(name)
        if camel_name in self:
            val = self[camel_name]
            if isinstance(val, (dict)):
                return wrap_dict(val)
            return val
        return super().__getattribute__(name)

    def __repr__(self):
        def cut(s, length=20):
            return s[:length] if len(s) <= length else  s[:length] + '...'
        vals = pprint.pformat({to_snake_case(k): cut(str(self[k]))  for k in self.keys()})
        return f"NodeWrap({vals})"


def _annotations_to_gql(annotations):
    nodes = []
    for a in annotations:
        tables = []
        for t in a['tables']:
            tags = []
            if t['leaderboard']:
                tags.append('leaderboard')
            if t['ablation']:
                tags.append('ablation')
            if not tags:
                tags = ['irrelevant']

            records = {}
            for r in t['records']:
                d = dict(r)
                del d['row']
                del d['column']
                records[f'{r["row"]}.{r["column"]}'] = d
            table = {
                'node': {
                    'name': f'table_{t["index"] + 1:02}.csv',
                    'datasetText': t['dataset_text'],
                    'notes': '',
                    'goldTags': ' '.join(tags),
                    'matrixGoldTags': t['segmentation'],
                    'cellsSotaRecords': json.dumps(records),
                    'parser': 'latexml'
                }
            }
            tables.append(table)
        node = {
            'arxivId': a['arxiv_id'],
            'goldTags': a['fold'],
            'tableSet': {'edges': tables}
        }
        nodes.append({'node': node})
    return {
        'data': {
            'allPapers': {
                'edges': nodes
            }
        }
    }


def load_gql_dump(data_or_file, compressed=True):
    if isinstance(data_or_file, dict) or isinstance(data_or_file, list):
        papers_data = data_or_file
    else:
        open_fn = gzip.open if compressed else open
        with open_fn(data_or_file, "rt") as f:
            papers_data = json.load(f)
    if "data" not in papers_data:
        papers_data = _annotations_to_gql(papers_data)
    data = papers_data["data"]
    return {k:wrap_dict(v) for k,v in data.items()}

#%%
def gql(query, **variables):
    query = { 'query' : query}
    r = requests.post(url=config.graphql_url, json=query)
    return json.loads(r.text)

def gql_papers(goldtags_regex="sst2"):
    return  gql("""
query{
    papers: allPapers(goldTags_Regex:"sst2", first:10) {
        edges{
            node{
                arxivId
                title
                abstract
                tableSet {
                    edges {
                        node {
                            id
                            matrix
                            desc
                            matrixGoldTags
                            goldTags
                            cellsGoldTags
                        }
                    }
                }
            }
        }
    }
}
""", goldTags_Regex=goldtags_regex)

def load_annotated_papers():
    return load_gql_dump(config.goldtags_dump)["allPapers"]

def test__wrapping():
    papers = load_gql_dump(papers_data)
    assert papers[0].arxiv_id == '1511.08630v2'
    papers[0].table_set[0].matrix 
    papers[0].table_set[0].matrix_gold_tags
 
    a = load_gql_dump(d)['allPapers']
    a[0].arxiv_id
    next(iter(a)).arxiv_id
    a

#%%
