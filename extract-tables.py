#!/usr/bin/env python

import sys
from bs4 import BeautifulSoup
import fire
from pathlib import Path
import pandas as pd
import json


class Tabular:
    def __init__(self, data, caption):
        self.data = data
        self.caption = caption


def flatten_tables(soup):
    inners = soup.select("div.tabular table table")
    for inner in inners:
        inner.name = 'div'
        for elem in inner.select("tr, td, colgroup, tbody, col"):
            elem.name = 'div'

def html2data(filename, table):
    data = pd.read_html(str(table), match='')
    if len(data) > 1:
        raise ValueError(f"<table> element in '{filename}' contains wrong number of tables: {len(data)}")
    return data[0] if len(data) == 1 else None


def save_table(data, filename):
    data.dropna(how='all').dropna(axis='columns', how='all').to_csv(filename, header=None, index=None)


def save_tables(data, outdir):
    metadata = []

    for num, table in enumerate(data, 1):
        filename = f"table_{num:02}.csv"
        save_table(table.data, outdir / filename)
        metadata.append(dict(filename=filename, caption=table.caption))
    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f)


def deepclone(elem):
    return BeautifulSoup(str(elem), "html.parser")


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
            float_div = table.find_parent("div", class_="float")
            tab = html2data(filename, table)
            if tab is None:
                continue

            caption = None
            if float_div is not None:
                float_div = deepclone(float_div)
                for t in float_div.find_all("table"):
                    t.extract()
                caption = float_div.get_text()

            data.append(Tabular(tab, caption))

    save_tables(data, outdir)

if __name__ == "__main__": fire.Fire(extract_tables)
