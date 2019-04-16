#!/usr/bin/env python
import sys

from bs4 import BeautifulSoup
import fire
from pathlib import Path
import pandas as pd

def flatten_tables(soup):
    inners = soup.select("div.tabular table table")
    for inner in inners:
        inner.name = 'div'
        for elem in inner.select("tr, td, colgroup, tbody, col"):
            elem.name = 'div'

def html2data(filename, table):
    data = pd.read_html(str(table), match='')
    if len(data) > 1:
        print(f"{filename}: {len(data)}")
    return data

def save_table(data, filename):
    data.dropna(how='all').dropna(axis='columns', how='all').to_csv(filename, header=None, index=None)

def extract_tables(filename, outdir):
    with open(filename, "rb") as f:
        html = f.read()
    outdir = Path(outdir)
    soup = BeautifulSoup(html, "html.parser")
    flatten_tables(soup)
    tables = soup.select("div.tabular")

    data = []
    for table in tables:
        if table.find("table") is not None:
            data.extend(html2data(filename, table))
    for num, table in enumerate(data, 1):
        save_table(table, outdir / f"table_{num:02}.csv")

if __name__ == "__main__": fire.Fire(extract_tables)
