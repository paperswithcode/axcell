#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from bs4 import BeautifulSoup
import pandas as pd
import re
from dataclasses import asdict

from elasticsearch_dsl import Document, Boolean, Object, \
    analyzer, InnerDoc, Keyword, Text, Integer, tokenizer, token_filter, Date
from elasticsearch_dsl.serializer import serializer

from IPython.display import display, Markdown

from elasticsearch_dsl import connections

from axcell.data.doc_utils import get_text, content_in_section, group_content, read_html, put_dummy_anchors, clean_abstract
from .. import config
from pathlib import Path
import sys


def setup_default_connection():
    # TODO: extract that to settings / configuraiton
    connections.create_connection(**config.elastic)


def printmd(*args):  # fixme: make it work without jupyter notebook
    display(Markdown(" ".join(map(str, args))))


class Fragments(list):

    def get_toc(self):
        header = None
        for f in self:
            if header != f.header:
                header = f.header
                yield header

    def print(fs, clean_up=lambda x: x):
        header = None
        for f in fs:
            if header != f.header:
                header = f.header
                printmd(f"# {header}")
            printmd(clean_up(f.text))
            printmd(f.order)
            printmd()


html_strip = analyzer('html_strip',
                      tokenizer="standard",
                      filter=["standard", "lowercase", "stop", "snowball"],
                      char_filter=["html_strip"])

filter_shingle = token_filter('filter_shingle',
                              type="shingle",
                              min_shingle_size=2,
                              max_shingle_size=5,
                              output_unigrams=True
                              )

filter_word_delimiter2 = token_filter('word_delimiter2',
                                      type='word_delimiter',
                                      generate_word_parts=True,    # "PowerShot" ⇒ "Power" "Shot"
                                      generate_number_parts=True,  # "500-42" ⇒ "500" "42"
                                      catenate_words=True,         # "wi-fi" ⇒ "wifi"
                                      split_on_case_change=True,
                                      stem_english_possessive=True,
                                      preserve_original=True
                                      )


taxonomy_standard = analyzer('taxonomy_standard',
                             tokenizer="standard",
                             filter=[filter_word_delimiter2,
                                     "lowercase", filter_shingle],
                             char_filter=[])

taxonomy_ngrams = analyzer('taxonomy_ngrams',
                           tokenizer=tokenizer('trigram', 'nGram', min_gram=2, max_gram=4, token_chars=[
                                               "letter", "digit"]),
                           filter=["standard", "lowercase", ],
                           char_filter=[])


class ETTaxonomy(Document):
    dataset = Text(analyzer=taxonomy_standard, fields={
                   'ngrams': Text(analyzer=taxonomy_ngrams)})  # 'GigaWord'
    task = Text(analyzer=taxonomy_standard, fields={'ngrams': Text(
        analyzer=taxonomy_ngrams)})  # 'Text Summarization',
    metric = Text(analyzer=taxonomy_standard, fields={
                  'ngrams': Text(analyzer=taxonomy_ngrams)})   # 'ROUGE-2',
    custom = Text(analyzer=taxonomy_standard, fields={
                  'ngrams': Text(analyzer=taxonomy_ngrams)})

    class Index:
        name = 'et_taxonomy'

    def __repr__(self):
        return f"{self.dataset} / {self.task} / {self.metric}"


class Fragment(Document):
    paper_id = Keyword()
    order = Integer()
    header = Text(fields={'raw': Keyword()})
    text = Text(
        analyzer=html_strip
    )
    outer_headers = Text(analyzer=html_strip, )

    class Meta:
        doc_type = '_doc'

    class Index:
        doc_type = '_doc'
        name = 'paper-fragments'

    @classmethod
    def from_json(cls, json):
        if isinstance(json, str):
            source = serializer.loads(json)
        else:
            source = json
        data = dict(
            _source = source,
            _id = f"{source['paper_id']}_{source['order']}",
            _index = 'paper-fragments',
            _type = 'doc')
        return cls.from_es(data)


    def __repr__(self):
        return f"# {self.header},\n" \
            f"{self.text}" \
            f"<Fragment(meta.id={self.meta.id}, order={self.order})>\n"


class Paper(Document):
    title = Text()
    authors = Keyword() #TODO: change this to Text() otherwise we can't search using this field.
    abstract = Text(
        analyzer=html_strip
    )

    class Meta:
        doc_type = '_doc'

    class Index:
        doc_type = '_doc'
        name = 'papers'

    def to_json(self):
        data = self.to_dict()
        return serializer.dumps(data)

    @classmethod
    def from_json(cls, json, paper_id=None):
        if isinstance(json, str):
            source = serializer.loads(json)
        else:
            source = json
        fragments = source.pop('fragments', [])
        data = dict(
            _source = source,
            _index = 'papers',
            _type = 'doc')
        if paper_id is not None:
            data['_id'] = paper_id

        paper = cls.from_es(data)
        paper.fragments = Fragments([Fragment.from_json(f) for f in fragments])
        return paper

    @classmethod
    def from_file(cls, path, paper_id=None):
        path = Path(path)
        if paper_id is None:
            paper_id = path.parent.name
        with open(path, "rt") as f:
            json = f.read()
        return cls.from_json(json, paper_id)

    def to_df(self):
        return pd.DataFrame({'header': [f.header for f in self.fragments],
                             'text': [f.text for f in self.fragments],
                             'len(text)': [len(f.text) for f in self.fragments]})
    # TODO fix this as right now fragments are being saved to elastic as part of paprer

    def save(self, **kwargs):
        if hasattr(self, 'fragments'):
            fragments = self.fragments
            for f in fragments:
                f.save()
            del self.fragments
            r = super().save(**kwargs)  # so it isn't saved
            self.fragments = fragments
            return r
        else:
            return super().save(**kwargs)

    def delete(self, **kwargs):
        if hasattr(self, 'fragments'):
            for f in self.fragments:
                f.delete()
        return super().delete(**kwargs)

    @classmethod
    def parse_html(cls, soup, paper_id):
        put_dummy_anchors(soup)
        abstract = soup.select("div.ltx_abstract")
        author = soup.select("div.ltx_authors")
        p = cls(title=get_text(soup.title),
                authors=get_text(*author),
                abstract=clean_abstract(get_text(*abstract)),
                meta={'id': paper_id})
        for el in abstract + author:
            el.extract()

        fragments = Fragments()
        doc = soup.find("article")
        if doc:
            footnotes = doc.select(".ltx_role_footnote > .ltx_note_outer")
            for ft in footnotes:
                ft.extract()

            idx = 0
            for idx, idx2, section_header, content in group_content(doc):
                content = content.strip()
                if p.abstract == "" and "abstract" in section_header.lower():
                    p.abstract = clean_abstract(content)
                else:
                    order = (idx + 1) * 1000 + idx2
                    f = Fragment(
                        paper_id=paper_id,
                        order=order,
                        header=section_header,
                        text=content,
                        meta={'id': f"{paper_id}-{order}"}
                    )
                    fragments.append(f)
            idx += 1
            idx2 = 0
            for ft in footnotes:
                order = (idx + 1) * 1000 + idx2
                f = Fragment(
                        paper_id=paper_id,
                        order=order,
                        header="xxanchor-footnotes Footnotes",
                        text=get_text(ft),
                        meta={'id': f"{paper_id}-{order}"}
                )
                fragments.append(f)
                idx2 += 1
        else:
            print(f"No article found for {paper_id}", file=sys.stderr)
        p.fragments = fragments
        return p

    def get_toc(self):
        return self.fragments.get_toc()

    def print_toc(self):
        for header in self.get_toc():
            printmd(f"{header}")

    def print_section(self, name, clean_up=lambda x: x):
        Fragments(f for f in self.fragments if name in f.header).print(clean_up)

    @classmethod
    def read_html(cls, file):
        return read_html(file)

    @classmethod
    def from_html(cls, html, paper_id):
        soup = BeautifulSoup(html, "html.parser")
        return cls.parse_html(soup, paper_id)

    @classmethod
    def parse_paper(cls, file, paper_id=None):
        file = Path(file)
        soup = cls.read_html(file)
        if paper_id is None:
            paper_id = file.stem
        return cls.parse_html(soup, paper_id)


class Author(InnerDoc):
    name = Text()
    ids = Integer()


class Reference(Document):
    title = Text()
    authors = Object(Author)
    abstract = Text()
    in_citations = Keyword()
    out_citations = Keyword()
    urls = Keyword()
    is_ml = Boolean()

    class Meta:
        doc_type = '_doc'

    class Index:
        doc_type = '_doc'
        name = 'references'

    def __repr__(self):
        return f"{self.title} / {self.authors}"


ID_LIMIT=480


class Author2(InnerDoc):
    forenames = Text(fields={'keyword': Keyword()})
    surname = Text(fields={'keyword': Keyword()})


class Reference2(Document):
    title = Text()
    authors = Object(Author2)

    idno = Keyword()
    date = Date()
    ptr = Keyword()

    arxiv_id = Keyword()
    pwc_slug = Keyword()
    orig_refs = Text()

    class Meta:
        doc_type = '_doc'

    class Index:
        doc_type = '_doc'
        name = 'references2'

    def add_ref(self, ref):
        # if not hasattr(self, 'refs'):
        #     self.refs = []
        # self.refs.append(asdict(ref))
        if ref.arxiv_id:
            self.arxiv_id = ref.arxiv_id
        if ref.pwc_slug:
            self.pwc_slug = ref.pwc_slug
        if ref.idno:
            if hasattr(ref.idno, 'values'):
                self.idno = ([None]+[v for v in ref.idno.values() if v.startswith("http")]).pop()
            elif isinstance(ref.idno, str):
                self.idno = ref.idno
        # if ref.date:
        #     self.date = ref.date
        if ref.ptr:
            self.ptr = ref.ptr
        self.orig_refs = self.orig_refs if self.orig_refs else []
        self.orig_refs.append(ref.orig_ref)
        self.orig_refs = list(set(self.orig_refs))

        # TODO Update authors
        # titles = Counter([norm_title] + [normalize_title(ref.title) for ref in merged])
        # norm_title = titles.most_common(1)[0][0]

    @property
    def stable_id(self):
        return self.meta.id

    def unique_id(self):
        return self.meta.id

    @classmethod
    def from_ref(cls, ref):
        #title = ref.title
        #first_author = ref.authors[0].short() if len(ref.authors) > 0 else "unknown"
        # Todo figure out what to do here so stable_id is recoverable, and it has no collisions
        #  stable_id = first_author + "-" + normalize_title(until_first_nonalphanumeric(title))[:50]
        stable_id = ref.unique_id()[:ID_LIMIT]

        self = cls(meta={"id":stable_id},
                   title=ref.title,
                   authors=[asdict(a) for a in ref.authors if a is not None])

        return self

#
# arxiv = Path('data/arxiv')
# html = arxiv/'html'
# htmls=list(html.glob('[0-9]*'))
#
# broken=[]
# for h in (arxiv/"html-f").glob("[0-9]*"):
#     try:
#         p, fs = parse_paper(h)
#         p.save()
#         save_paper(p, fs)
#         print(h, "saved")
#     except elasticsearch.exceptions.RequestError as e:
#         print (h, "error", str(e))
#         broken.append((h,e))
# - find all references of a table in text
# - find the contributions section
# - find ulmfit, and what it means
# - upload all to elastic
# - find all papers that are talking about F1
# - find all papers that are using squad
# `


_reference_re = re.compile(r'xxref-\w{40}', re.UNICODE)

def cell_type_heuristic(orig_text, text, query):
    flags = set()
    if query in orig_text:
        flags.add("exact")
    if "model" in text:
        flags.add(
            "<span style='background-color:orange'>model</<span style='background-color:yellow'>")
    if "our" in text:
        flags.add("<span style='background-color:orange'>model-best</span>")
    if "dataset" in text:
        flags.add("<span style='background-color:lightblue'>dataset</span>")
    if "data set" in text:
        flags.add("<span style='background-color:lightblue'>dataset</span>")
    if _reference_re.search(text):
        flags.add("<span style='background-color:yellow'>external</span>")
    if _reference_re.search(orig_text):
        flags.add("<span style='background-color:yellow'>external?</span>")

    if flags:
        return " ".join(flags)


def display_fragment(f, cell_type="", display=True):
    from axcell.helpers.jupyter import display_html
    cell_type = f" - <b> {cell_type} </b>" if cell_type else ""
    pre = f"<br/>{f.header}{cell_type}<br/>"
    body = " ... ".join(f.meta.highlight.text)
    html = pre + body + f"<br/>paper_id:{f.paper_id}"
    if display:
        display_html(html)
    return html


def query_for_evidences(paper_id, values, topk=5, fragment_size=50):
    evidence_query = Fragment.search().highlight(
        'text', pre_tags="<b>", post_tags="</b>", fragment_size=fragment_size)

    query = {
        "query": ' '.join(values)
    }

    fragments = list(evidence_query
                     .filter('term', paper_id=paper_id)
                     .query('match', text=query)[:topk]
                     )

    return '\n'.join([' '.join(f.meta['highlight']['text']) for f in fragments])
