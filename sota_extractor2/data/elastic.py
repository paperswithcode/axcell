import pandas as pd
import re

from elasticsearch_dsl import Document, Boolean, Object, \
    analyzer, InnerDoc, Keyword, Text, Integer, tokenizer, token_filter
from elasticsearch_dsl.serializer import serializer

from IPython.display import display, Markdown

from elasticsearch_dsl import connections

from sota_extractor2.data.doc_utils import get_text, content_in_section, group_content, set_ids_by_labels, read_html
from .. import config
from pathlib import Path


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

    class Index:
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
    authors = Keyword()
    abstract = Text(
        analyzer=html_strip
    )

    class Index:
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
    def from_file(cls, path):
        path = Path(path)
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

    @classmethod
    def parse_html(cls, soup, paper_id):
        set_ids_by_labels(soup)
        abstract = soup.select("div.abstract")
        author = soup.select("div.author")
        p = cls(title=get_text(soup.title),
                authors=get_text(*author),
                abstract=get_text(*abstract),
                meta={'id': paper_id})
        for el in abstract + author:
            el.extract()

        fragments = Fragments()
        for idx, h in enumerate(soup.find_all(['h3', 'h4'])):
            section_header = get_text(h)
            if p.abstract == "" and section_header.lower() == "abstract":
                p.abstract = get_text(*list(content_in_section(h)))
            else:
                for idx2, content in enumerate(group_content(content_in_section(h))):
                    order = (idx + 1) * 1000 + idx2
                    f = Fragment(
                        paper_id=paper_id,
                        order=order,
                        header=section_header,
                        text=content,
                        meta={'id': f"{paper_id}-{order}"}
                    )
                    fragments.append(f)
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
    def parse_paper(cls, file):
        soup = cls.read_html(file)
        return cls.parse_html(soup, file.stem)


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

    class Index:
        name = 'references'

    def __repr__(self):
        return f"{self.title} / {self.authors}"

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
    from sota_extractor2.helpers.jupyter import display_html
    cell_type = f" - <b> {cell_type} </b>" if cell_type else ""
    pre = f"<br/>{f.header}{cell_type}<br/>"
    body = " ... ".join(f.meta.highlight.text)
    html = pre + body + f"<br/>paper_id:{f.paper_id}"
    if display:
        display_html(html)
    return html
