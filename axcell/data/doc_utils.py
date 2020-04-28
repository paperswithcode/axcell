#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re
from bs4 import BeautifulSoup, Comment, Tag, NavigableString
import codecs

def _handle_reference(el):
    if el.get('href', "").startswith("#"):
        r = str(el.get('href'))
        el.clear()  # to remove it's content from the descendants iterator
        return "xxref-" + _simplify_anchor(r[1:])


_anchor_like_classes = {
        'ltx_appendix', 'ltx_bibliography', 'ltx_figure', 'ltx_float', 'ltx_graphics', 'ltx_note',
        'ltx_paragraph', 'ltx_picture', 'ltx_section', 'ltx_subsection', 'ltx_subsubsection', 'ltx_theorem',
        'ltx_title_section', 'ltx_title_subsection'
}

def _insert_anchor(el, anchor_id, prefix="xxanchor"):
    el.insert(0, NavigableString(f' {prefix}-{anchor_id} '))

def put_dummy_anchors(soup):
    for elem in soup.select(
            '.ltx_bibitem, ' + \
            '.ltx_figure, .ltx_float, ' + \
            '.ltx_picture, .ltx_theorem'):
        id_str = elem.get('id', '')
        if id_str:
            _insert_anchor(elem, _simplify_anchor(id_str))
    for elem in soup.select('h2, h3, h4, h5, h6'):
        sec = elem.find_parent("section")
        if sec:
            id_str = sec.get('id')
            if id_str:
                _insert_anchor(elem, _simplify_anchor(id_str))
    for elem in soup.select(".ltx_table"):
        id_str = elem.get('id', "xxunk")
        _insert_anchor(elem, _simplify_anchor(id_str), "xxtable-xxanchor")
    for elem in soup.select(".ltx_tabular"):
        elem.extract()

    for elem in soup.select('a[href^="#"]'):
        r = str(elem.get('href'))
        elem.string = "xxref-" + _simplify_anchor(r[1:])

    put_footnote_anchors(soup)

def put_footnote_anchors(soup):
    for elem in soup.select('.ltx_note_content > .ltx_note_mark'):
        elem.extract()

    for elem in soup.select('.ltx_role_footnote > .ltx_note_mark'):
        ft = elem.parent
        id_str = ft.get('id')
        if id_str:
            elem.string = f" xxref-{_simplify_anchor(id_str)} "

    for elem in soup.select('.ltx_note_content > .ltx_tag_note'):
        ft = elem.find_parent(class_="ltx_role_footnote")
        if ft:
            id_str = ft.get('id')
            elem.string = f" xxanchor-{_simplify_anchor(id_str)} "

# remove . from latexml ids (f.e., S2.SS5) so they can be searched for in elastic
# without disambiguations
def _simplify_anchor(s):
    return s.replace('.', '')


def _handle_anchor(el):
    if el.name.lower() == 'a' and el.get('id', ""):
        id_str = el.get('id', "")
        el.clear()  # to remove it's content from the descendants iterator
        return "xxanchor-" + id_str
#    classes = get_classes(el)
#    id_str = el.get('id')
#    if 'ltx_title_section' in classes  or 'ltx_title_subsection' in classes:
#        print(el.get_text())
#    print(el.name)
#    if 'ltx_title_section' in classes or 'ltx_title_subsection' in classes:
#        print(el.get_text())
#        # this is workaround to deal with differences between
#        # htlatex and latexml html structure
#        # it would be better to make use of latexml structure
#        sec = el.find_parent("section")
#        if sec:
#            id_str = sec.get('id')
#            print(id_str, el.get_text())
#
#    if id_str and classes:
#        classes = set(classes)
#        if classes.intersection(_anchor_like_classes):
#            print('xxanchor-'+id_str)
#            el.clear()  # to remove it's content from the descendants iterator
#            return "xxanchor-" + id_str


def _handle_table(el):
    if 'ltx_table' in get_classes(el):
        id_str = el.get('id', "xxunk")
        el.clear()  # to remove it's content from the descendants iterator
        return f"xxtable-xxanchor-" + id_str


_transforms_el = [
    _handle_reference,
    _handle_table,
    _handle_anchor,
]


def transform(el):
    if isinstance(el, Tag):
#        for f in _transforms_el:
#            r = f(el)
#            if r is not None:
#                return transform(r)
        return el.get_text()
    elif not isinstance(el, Comment):
        return str(el)
    return ''


def clean_abstract(t):
    return re.sub("^\s*[aA]bstract ?", "", t)


def get_text(*els):
#    t = " ".join([transform(t)
#                  for el in els for t in getattr(el, 'descendants', [el])])
    t = " ".join([transform(e) for e in els])
    t = re.sub("[ \n\xa0]+", " ", t)
    t = re.sub("[;,()]* (#[A-Za-z0-9]+) [;,()]*", r" \1 ", t)
    t = re.sub(r" (#[A-Za-z0-9]+) *\1 ", r" \1 ", t)
    return t.strip()


def content_in_section(header, names=['h2', 'h3'], skip_comments=True):
    for el in header.next_siblings:
        if getattr(el, 'name', '') in names:
            break
        if skip_comments and isinstance(el, Comment):
            continue
        yield el


def get_classes(el):
    if hasattr(el, 'get'):
        return el.get('class', [])
    else:
        return []


def get_name(el):
    return hasattr(el, 'name') and el.name or ''


def _group_bibliography(el):
    if 'ltx_bibliography' in get_classes(el):
        return [get_text(i) for i in el.select('li.ltx_bibitem')]
    return []


def _group_table(el):
    if 'ltx_table' in get_classes(el):
        return [get_text(el)]
    return []


class ParagraphGrouper:
    def __init__(self):
        self.els = []
        self.join_next_p = False

    def collect(self, el):
        if get_name(el) == 'table':
            self.join_next_p = True
        elif 'ltx_para' in get_classes(el):
            if self.join_next_p:
                self.join_next_p = False
                self.els.append(el)
            else:
                return self.flush(new_els=[el])
        else:
            self.els.append(el)
        return []

    def flush(self, new_els=None):
        text = get_text(*self.els)
        if new_els is None:
            new_els = []
        if isinstance(new_els, Tag):  # allow for one tag to be passed
            new_els = [new_els]
        self.els = new_els
        if text:
            return [text]
        return []

    def reset(self):
        self.els = []


_group_el = [
    _group_bibliography,
    _group_table,
]


def group_content2(elements):
    par_gruop = ParagraphGrouper()
    for el in elements:
        fragments = [frag for grouper in _group_el for frag in grouper(el)]
        if fragments:
            fragments = par_gruop.flush() + fragments
        else:
            fragments = par_gruop.collect(el)
        for frag in fragments:
            yield frag

    for frag in par_gruop.flush():
        yield frag


def walk(elem):
    for el in elem.children:
        classes = get_classes(el)
        if el.name == 'section' or 'ltx_biblist' in classes:
            yield from walk(el)
        else:
            yield el

class Grouper:
    def __init__(self):
        self.out = []
        self.section_idx = -1
        self.subsection_idx = 0
        self.header = ""
        self.in_section = False # move elements before first section into that section
        self.section_output = False # if a section is empty and new section begins, output it for keep header

    def get_output_text(self):
        return " ".join(self.out)

    def flush(self):
        if self.in_section:
            r = max(self.section_idx, 0), self.subsection_idx, self.header, self.get_output_text()
            self.out = []
            self.section_output = True
            self.subsection_idx += 1
            yield r

    def new_section(self, header_el):
        if not self.section_output or self.out: # output (possibly) empty section so header won't be lost
            yield from self.flush()
        self.section_output = False
        self.in_section = True
        self.section_idx += 1
        self.subsection_idx = 0
        self.header = get_text(header_el)

    def append(self, el):
        t = get_text(el).strip()
        if t != "":
            self.out.append(t)
            return True
        return False

    def group_content(self, doc):
        for el in walk(doc):
            classes = get_classes(el)
            if el.name in ["h2", "h3"]:
                yield from self.new_section(el)
            elif el.name == "h1":
                continue
            elif 'ltx_para' in classes or el.name == "figure" or 'ltx_bibitem' in classes:
                has_content = self.append(el)
                if has_content:
                    yield from self.flush()
            else:
                self.append(el)
        self.in_section = True
        if not self.section_output or self.out:
            yield from self.flush()


def group_content(doc):
    yield from Grouper().group_content(doc)

def group_content3(doc):
    out = []
    section_idx = -1
    subsection_idx = 0
    header = ""
    has_paragraph = False
    for el in walk(doc):
        classes = get_classes(el)
        if el.name in ["h2", "h3"]:
            if len(out) and has_paragraph:
                yield (max(section_idx, 0), subsection_idx, header, " ".join([get_text(o) for o in out]))
                out = []
            section_idx += 1
            subsection_idx = 0
            header = get_text(el)
            continue
        elif 'ltx_title' in classes and el.name != "h1":
            if len(out) and has_paragraph:
                yield (max(section_idx, 0), subsection_idx, header, " ".join([get_text(o) for o in out]))
                out = []
            out += [el]

        elif 'ltx_title_document' in classes:
            continue
        elif 'ltx_para' in classes or el.name == "figure" or 'ltx_bibitem' in classes:
            if len(out) and has_paragraph:
                yield (max(section_idx, 0), subsection_idx, header, " ".join([get_text(o) for o in out]))
                subsection_idx += 1
                out = []
            has_paragraph = True
            out += [el]
        else:
            out.append(el)
    if len(out):
        yield (max(section_idx, 0), subsection_idx, header, " ".join([get_text(o) for o in out]))

def read_html(file):
    with codecs.open(file, 'r', encoding='UTF-8') as f:
        text = f.read()
    return BeautifulSoup(text, "html.parser")
