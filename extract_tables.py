#!/usr/bin/env python

import sys
from bs4 import BeautifulSoup, Comment, NavigableString
import fire
from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
from ast import literal_eval

from tabular import Tabular


def flatten_tables(soup):
    inners = soup.select("div.tabular table table")
    for inner in inners:
        inner.name = 'div'
        for elem in inner.select("tr, td, colgroup, tbody, col"):
            elem.name = 'div'


def escape(s):
    return repr(s)


def unescape(r):
    return literal_eval(r)


multirow_re = re.compile(r"^\s*rows=(P<rows>\d+)\s*$")
whitespace_re = re.compile(r'[\r\n]+|\s{2,}')

def escape_table_content(soup):
    for item in soup.find_all(["td", "th"]):
        escaped = escape(whitespace_re.sub(" ", item.get_text().strip()))

        multirow = item.find("div", class_="multirow", recursive=False)
        if multirow and multirow.contents and isinstance(multirow.contents[0], Comment):
            match = multirow_re.match(str(multirow.contents[0]))
            if match:
                escaped = f"multirow={match.group('rows')};{escaped}"

        item.string = escaped


def fix_htlatex_multirow(df):
    rows, cols = df.shape

    for col in range(cols):
        for row in range(rows):
            cell = df.iloc[row, col]
            if cell.startswith("multirow="):
                pos = cell.find(';')
                multirows = int(cell[9:pos])
                assert df.iloc[row+1: row+multirows, col].isna().all()
                df.iloc[row: row+multirows, col] = cell[pos+1:]


def unescape_table_content(df):
    return df.applymap(unescape)


def fix_table(df):
    df = df.fillna(repr(''))
    fix_htlatex_multirow(df)
    df = df.replace("''", np.NaN).dropna(how='all').dropna(axis='columns', how='all').fillna("''")
    return unescape_table_content(df)


def move_out_references(table):
    for anchor in table.select('a[href^="#"]'):
        anchor.append(NavigableString("[xxref-"+anchor["href"][1:]+"]"))


def html2data(table):
    data = pd.read_html(str(table), match='')
    if len(data) > 1:
        raise ValueError(f"<table> element contains wrong number of tables: {len(data)}")
    return data[0] if len(data) == 1 else None


def save_table(data, filename):
    data.to_csv(filename, header=None, index=None)


def save_tables(data, outdir):
    metadata = []

    for num, table in enumerate(data, 1):
        filename = f"table_{num:02}.csv"
        save_table(table.data, outdir / filename)
        metadata.append(dict(filename=filename, caption=table.caption, figure_id=table.figure_id))
    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f)


def deepclone(elem):
    return BeautifulSoup(str(elem), "lxml")


def set_ids_by_labels(soup):
    captions = soup.select(".caption")
    prefix = "tex4ht:label?:"
    for caption in captions:
        el = caption.next_sibling
        if isinstance(el, Comment) and el.string.startswith(prefix):
            label = el.string[len(prefix):].strip()
            for table in caption.parent.select("table"):
                table["data-figure-id"] = label


def extract_tables(filename, outdir):
    with open(filename, "rb") as f:
        html = f.read()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    soup = BeautifulSoup(html, "lxml")
    flatten_tables(soup)
    set_ids_by_labels(soup)
    tables = soup.select("div.tabular")

    data = []
    for table in tables:
        table_el = table.find("table")
        if table_el is not None:
            float_div = table.find_parent("div", class_="float")
            #print(table)
            move_out_references(table)
            escape_table_content(table)
            #print(table)
            tab = html2data(table)
            if tab is None:
                continue

            tab = fix_table(tab)

            caption = None
            if float_div is not None:
                float_div = deepclone(float_div)
                for t in float_div.find_all("table"):
                    t.extract()
                caption = float_div.get_text()
            figure_id = table_el.get("data-figure-id")
            data.append(Tabular(tab, caption, figure_id))

    save_tables(data, outdir)

if __name__ == "__main__": fire.Fire(extract_tables)
