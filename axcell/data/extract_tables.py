#!/usr/bin/env python

#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
from bs4 import BeautifulSoup, Comment, NavigableString
import fire
from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
from ast import literal_eval
from collections import OrderedDict
from dataclasses import dataclass
from typing import Set

from axcell.data.table import Table

# begin of dirty hack
# pandas parsing of html tables is really nice
# but it has a lot of defaults that can't be
# modified

# one of the defaults is forcing thead rows
# into column names, ignoring value of `header`
# param

# the second issue is parsing numerical-looking
# values into floats

_old_data_to_frame = pd.io.html._data_to_frame
def _new_data_to_frame(**kwargs):
  head, body, foot = kwargs.pop("data")
  if head:
    body = head + body
  if foot:
    body += foot
  return _old_data_to_frame(data=(None, body, None), **kwargs)
pd.io.html._data_to_frame = _new_data_to_frame
# end of dirty hack



def flatten_tables(soup):
    inners = soup.select(".ltx_tabular .ltx_tabular")
    for inner in inners:
        inner.name = 'div'
        for elem in inner.select("tr, td, th, colgroup, tbody, thead, tfoot, col"):
            elem.name = 'div'


def escape(s):
    return repr(s)


def unescape(r):
    return literal_eval(r)

whitespace_re = re.compile(r'[\r\n]+|\s{2,}')

def clear_ws(s):
    return whitespace_re.sub(" ", s.strip())

def escape_table_content(soup):
    for item in soup.find_all(["td", "th"]):
        escaped = escape(clear_ws(item.get_text()))
        item.string = escaped

def unescape_table_content(df):
    return df.applymap(unescape)


@dataclass
class LayoutCell:
    borders: Set[str]
    align: Set[str]
    header: bool
    colspan: int
    rowspan: int
    span: Set[str]

    def __str__(self):
        borders = ['border-'+x for x in sorted(list(self.borders))]
        align = ['align-'+x for x in sorted(list(self.align))]
        span = ['span-'+x for x in sorted(list(self.span))]
        header = ["header"] if self.header else []
        return ' '.join(borders + align + span + header)

def to_layout(s):
    if s == "":
        return LayoutCell(set(), set(), False, 1, 1, set())
    borders, align, header, colspan, rowspan = s.split(",")
    borders = set(borders.split())
    align = set(align.split())
    header = (header == "True")
    colspan = int(colspan)
    rowspan = int(rowspan)
    return LayoutCell(borders, align, header, colspan, rowspan, set())


def fix_layout(layout):
    rowspan = 1
    for index, row in layout.iterrows():
        colspan = 1
        for cell in row:
            colspan -= 1
            if colspan == 0:
                colspan = cell.colspan
            if cell.colspan > 1:
                if colspan == 1:
                    cell.span.add("ce")
                    cell.borders -= {"l", "ll"}
                elif colspan == cell.colspan:
                    cell.span.add("cb")
                    cell.borders -= {"r", "rr"}
                else:
                    cell.span.add("ci")
                    cell.borders -= {"l", "ll", "r", "rr"}
    for col in layout:
        rowspan = 1
        for cell in layout[col]:
            rowspan -= 1
            if rowspan == 0:
                rowspan = cell.rowspan
            if cell.rowspan > 1:
                if rowspan == 1:
                    cell.span.add("re")
                    cell.borders -= {"t", "tt"}
                elif rowspan == cell.rowspan:
                    cell.span.add("rb")
                    cell.borders -= {"b", "bb"}
                else:
                    cell.span.add("ri")
                    cell.borders -= {"b", "bb", "t", "tt"}


# does not deal with nested tags
# f.e., </bold></red><red><bold>
# or <bold><bold>
whitespace_tag_re = re.compile(r"<(bold|italic|red|green|blue)>(\s*)</\1>")
dummy_close_tag_re = re.compile(r"</(bold|italic|red|green|blue)>(\s*)<\1>")
def clear_cell(s):
    s = whitespace_tag_re.sub(r"\2", s)
    s = dummy_close_tag_re.sub(r"\2", s)
    return s.strip()


def decouple_layout(df):
    split = df.applymap(lambda x: ("", "") if x == "" else x.split(";", 1))
    tab = split.applymap(lambda x: clear_cell(x[1]))
    layout = split.applymap(lambda x: to_layout(x[0]))
    fix_layout(layout)
    return tab, layout


def fix_table(df):
    df = df.fillna(repr(''))
    df = df.replace("''", np.NaN).dropna(how='all').dropna(axis='columns', how='all').fillna("''")
    df = unescape_table_content(df)
    return decouple_layout(df)


def is_table_empty(df):
    return (df.applymap(lambda x: x.strip()).values == "").all()

def fix_id(s):
    return s.replace(".", "-")


def wrap_elem_content(elem, begin, end):
    elem.insert(0, NavigableString(begin))
    elem.append(NavigableString(end))


def move_out_references(table):
    for anchor in table.select('a[href^="#"]'):
        wrap_elem_content(anchor, f"<ref id='{fix_id(anchor['href'][1:])}'>", "</ref>")


bold_font_weight_re = re.compile(r"(^|;)\s*font-weight:\s*(bold|700|800|900)\s*(;|$)")
bold_mathjax_font_re = re.compile(r"^MJXc-TeX-\w*-BI?$")
italic_font_style_re = re.compile(r"(^|;)\s*font-style:\s*italic\s*(;|$)")
italic_mathjax_font_re = re.compile(r"^MJXc-TeX-\w*-B?I$")

def _has_font_class(classes, font_re):
    return any(font_re.match(cls) for cls in classes)


font_color_re = re.compile(r"(^|;)\s*color:\s*(?P<color>#[0-9A-Fa-f]{3,6}|red|green|blue)\s*(;|$)")
def _extract_color_from_style(style):
    m = font_color_re.search(style)
    if m:
        color = m["color"]
        if color[0] == "#":
            color = color[1:]
            if len(color) != 6:
                color = (color + color)[:6]
            r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
            if r > 2 * g and r > 2 * b:
                color = "red"
            elif g > 2 * r and g > 2 * b:
                color = "green"
            elif b > 2 * r and b > 2 * g:
                color = "blue"
            else:
                return
        return color
    return


def move_out_text_styles(table):
    for elem in table.select('.ltx_font_bold, [style*="font-weight"], [class*="MJXc-TeX-"]'):
        classes = elem.get("class", [])
        style   = elem.get("style", "")
        if "ltx_font_bold" in classes or bold_font_weight_re.search(style) \
                or _has_font_class(classes, bold_mathjax_font_re):
                    wrap_elem_content(elem, "<bold>", "</bold>")

    for elem in table.select('.ltx_font_italic, [style*="font-style"], [class*="MJXc-TeX-"]'):
        classes = elem.get("class", [])
        style   = elem.get("style", "")
        if "ltx_font_italic" in classes or italic_font_style_re.search(style) \
                or _has_font_class(classes, italic_mathjax_font_re):
                    wrap_elem_content(elem, "<italic>", "</italic>")

    for elem in table.select('[style*="color"]'):
        style = elem.get("style")
        color = _extract_color_from_style(style)
        if color:
            wrap_elem_content(elem, f"<{color}>", f"</{color}>")


def move_out_cell_styles(table):
    ltx_border = 'ltx_border_'
    ltx_align = 'ltx_align_'
    ltx_th = 'ltx_th'

    for elem in table.select('td, th'):
        borders = []
        align = []
        header = False
        for c in elem.attrs["class"]:
            if c.startswith(ltx_border):
                borders.append(c[len(ltx_border):])
            elif c.startswith(ltx_align):
                align.append(c[len(ltx_align):])
            elif c == ltx_th:
                header = True
        b = ' '.join(borders)
        a = ' '.join(align)
        colspan = elem.attrs.get("colspan", "1")
        rowspan = elem.attrs.get("rowspan", "1")
        wrap_elem_content(elem, f"{b},{a},{header},{colspan},{rowspan};", "")


def remove_ltx_errors(soup):
    for span in soup.select('span.ltx_ERROR'):
        span.extract()


def html2data(table):
    data = pd.read_html(str(table), match='')
    if len(data) > 1:
        raise ValueError(f"<table> element contains wrong number of tables: {len(data)}")
    return data[0] if len(data) == 1 else None


def save_tables(data, outdir):
    metadata = []

    for num, table in enumerate(data, 1):
        filename = f"table_{num:02}.csv"
        layout = f"layout_{num:02}.csv"
        table.save(outdir, filename, layout)
        metadata.append(dict(filename=filename, layout=layout, caption=table.caption, figure_id=table.figure_id))
    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f)


def load_tables(path):
    path = Path(path)
    with open(path / "metadata.json", "r") as f:
        metadata = json.load(f)

    return [Table.from_file(
        path,
        table_metadata) for table_metadata in metadata]


def set_ids_by_labels(soup):
    captions = soup.select(".ltx_caption")
    for caption in captions:
        fig = caption.parent
        if fig.name == "figure" and fig.has_attr("id"):
            label = fig.attrs["id"]
            for table in fig.select(".ltx_tabular"):
                table["data-figure-id"] = label


alg_id_re = re.compile(r"^alg(orithm)?[0-9]+")
def perhaps_not_tabular(table, float_div):
    classes = float_div.attrs.get("class", [])
    if 'ltx_table' in classes:
        return False
    if 'ltx_figure' in classes:
        if table.find("img", class_="ltx_graphics"):
            return True
    if 'ltx_float' in classes:
        if 'biography' in classes:
            return True
        if 'ltx_float_algorithm':
            return True
        if 'ltx_lstlisting':
            return True
        if float_div.id and alg_id_re.match(float_div.id):
            return True
    return False

def is_figure(tag):
    return tag.name == "figure"
#    classes = tag.attrs.get("class", [])
#    return "ltx_figure" in classes or "ltx_float" in classes

def fix_span_tables(soup):
    classes = OrderedDict([("ltx_tabular", "table"), ("ltx_tr", "tr"), ("ltx_th", "th"),
               ("ltx_tbody", "tbody"), ("ltx_thead", "thead"), ("ltx_td", "td"),
               ("ltx_tfoot", "tfoot")])

    query = ','.join(["span." + c for c in classes.keys()])
    for elem in soup.select(query):
        for k, v in classes.items():
            if k in elem.attrs["class"]:
                elem.name = v
                break

# pandas.read_html treats th differently
# by trying in a few places to get column names
# for now <th>s are changed to <td>s, but we still
# have classes (ltx_th) to distinguish them
def fix_th(soup):
    for elem in soup.find_all("th"):
        elem.name = "td"

def remove_footnotes(soup):
    for elem in soup.select(".ltx_role_footnote"):
        elem.extract()


def extract_tables(html):
    soup = BeautifulSoup(html, "lxml")
    set_ids_by_labels(soup)
    fix_span_tables(soup)
    fix_th(soup)
    remove_ltx_errors(soup)
    flatten_tables(soup)
    tables = soup.find_all("table", class_="ltx_tabular")

    data = []
    for table in tables:
        if table.find_parent(class_="ltx_authors") is not None:
            continue

        float_div = table.find_parent(is_figure)
        if float_div and perhaps_not_tabular(table, float_div):
            continue
        remove_footnotes(table)
        move_out_references(table)
        move_out_text_styles(table)
        move_out_cell_styles(table)
        escape_table_content(table)
        tab = html2data(table)
        if tab is None:
            continue

        tab, layout = fix_table(tab)
        if is_table_empty(tab):
            continue

        caption = None
        if float_div is not None:
            cap_el = float_div.find("figcaption")
            if cap_el is not None:
                caption = clear_ws(cap_el.get_text())
        figure_id = table.get("data-figure-id")
        data.append(Table(f"table_{len(data)+1:02}", tab, layout.applymap(str), caption, figure_id))
    return data

def extract_tables_cmd(filename, outdir):
    with open(filename, "rb") as f:
        html = f.read()
    tables = extract_tables(html)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_tables(tables, outdir)

if __name__ == "__main__": fire.Fire(extract_tables_cmd)
