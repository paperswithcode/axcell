import pandas as pd

def read_arxiv_papers(path):
    return pd.read_csv(path)

def read_tables(path):
    return pd.read_json(path)

