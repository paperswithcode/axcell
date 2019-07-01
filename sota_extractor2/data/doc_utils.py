import re
from bs4 import BeautifulSoup, Comment, Tag
import codecs

def _handle_reference(el):
    if el.get('href', "").startswith("#"):
        r = str(el.get('href'))
        el.clear()  # to remove it's content from the descendants iterator
        return "xxref-" + r[1:]


def _handle_anchor(el):
    if el.get('id', ""):
        id_str = el.get('id', "")
        el.clear()  # to remove it's content from the descendants iterator
        return "xxanchor-" + id_str


def _handle_table(el):
    if el.name.lower() == 'table':
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
        for f in _transforms_el:
            r = f(el)
            if r is not None:
                return transform(r)
    elif not isinstance(el, Comment):
        return str(el)
    return ''


def get_text(*els):
    t = " ".join([transform(t)
                  for el in els for t in getattr(el, 'descendants', [el])])
    t = re.sub("^[aA]bstract ?", "", t)
    t = re.sub("[ \n\xa0]+", " ", t)
    t = re.sub("[;,()]* (#[A-Za-z0-9]+) [;,()]*", r" \1 ", t)
    t = re.sub(r" (#[A-Za-z0-9]+) *\1 ", r" \1 ", t)
    return t.strip()


def content_in_section(header, names=['h3', 'h4'], skip_comments=True):
    for el in header.next_siblings:
        if getattr(el, 'name', '') in names:
            break
        if skip_comments and isinstance(el, Comment):
            continue
        yield el


def get_class(el):
    if hasattr(el, 'get'):
        # fixme: less convoluted way to return '' if calss is not found
        return (el.get('class', [''])+[''])[0]
    else:
        return ''


def get_name(el):
    return hasattr(el, 'name') and el.name or ''


def _group_bibliography(el):
    if get_class(el) == 'thebibliography':
        return [get_text(i) for i in el.select('p.bibitem')]
    return []


def _group_table(el):
    if get_class(el) == 'table':
        return [get_text(el)]
    return []


class ParagraphGrouper:
    def __init__(self):
        self.els = []
        self.join_next_p = False

    def collect(self, el):
        if get_name(el) == 'table':
            self.join_next_p = True
        elif get_name(el) == "p":
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


def group_content(elements):
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


def set_ids_by_labels(soup):
    captions = soup.select(".caption")
    prefix = "tex4ht:label?:"
    for caption in captions:
        el = caption.next_sibling
        if isinstance(el, Comment) and el.string.startswith(prefix):
            label = el.string[len(prefix):].strip()
            for table in caption.parent.select("table"):
                table["id"] = label

def read_html(file):
    with codecs.open(file, 'r', encoding='UTF-8') as f:
        text = f.read()
    return BeautifulSoup(text, "html.parser")
