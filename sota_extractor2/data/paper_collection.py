from .elastic import Paper as PaperText
from .table import Table, read_tables
from .json import load_gql_dump
from pathlib import Path
import re

class Paper:
    def __init__(self, text, tables, annotations):
        self.text = text
        self.tables = tables
        if annotations is not None:
            self.gold_tags = annotations.gold_tags.strip()
        else:
            self.gold_tags = ''


arxiv_version_re = re.compile(r"v\d+$")
def clean_arxiv_version(arxiv_id):
    return arxiv_version_re.sub("", arxiv_id)


class PaperCollection:
    def __init__(self, path, load_texts=True, load_tables=True):
        self.path = path
        self.load_texts = load_texts
        self.load_tables = load_tables

        if self.load_texts:
            texts = self._load_texts()
        else:
            texts = {}

        annotations = self._load_annotated_papers()
        if self.load_tables:
            tables = self._load_tables(annotations)
        else:
            tables = {}
            annotations = {}
        outer_join = set(texts).union(set(tables))

        self._papers = {k: Paper(texts.get(k), tables.get(k), annotations.get(k)) for k in outer_join}
        self.annotations = annotations

    def __len__(self):
        return len(self._papers)

    def __getitem__(self, idx):
        return self._papers[idx]

    def __iter__(self):
        return iter(self._papers)

    def _load_texts(self):
        texts = {}

        for f in (self.path / "texts").glob("**/*.json"):
            text = PaperText.from_file(f)
            texts[clean_arxiv_version(text.meta.id)] = text
        return texts


    def _load_tables(self, annotations):
        tables = {}

        for f in (self.path / "tables").glob("**/metadata.json"):
            paper_dir = f.parent
            tbls = read_tables(paper_dir, annotations)
            tables[clean_arxiv_version(paper_dir.name)] = tbls
        return tables

    def _load_annotated_papers(self):
        dump = load_gql_dump(self.path / "structure-annotations.json.gz", compressed=True)["allPapers"]
        annotations = {}
        for a in dump:
            arxiv_id = clean_arxiv_version(a.arxiv_id)
            annotations[arxiv_id] = a
        return annotations
