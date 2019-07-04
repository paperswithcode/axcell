from .elastic import Paper as PaperText
from .table import Table, read_tables
from .json import load_gql_dump
from pathlib import Path
import re
from fastprogress import progress_bar
import pickle

class Paper:
    def __init__(self, text, tables, annotations):
        self.text = text
        self.tables = tables
        if annotations is not None:
            self.gold_tags = annotations.gold_tags.strip()
        else:
            self.gold_tags = ''


arxiv_version_re = re.compile(r"v\d+$")
def clear_arxiv_version(arxiv_id):
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

        self.papers = {k: Paper(texts.get(k), tables.get(k, []), annotations.get(k)) for k in outer_join}
        self._annotations = annotations

    def _load_texts(self):
        files = list((self.path / "texts").glob("**/*.json"))
        texts = [PaperText.from_file(f) for f in progress_bar(files)]
        return {clear_arxiv_version(text.meta.id): text for text in texts}


    def _load_tables(self, annotations):
        files = list((self.path / "tables").glob("**/metadata.json"))
        return {clear_arxiv_version(f.parent.name): read_tables(f.parent, annotations) for f in progress_bar(files)}

    def _load_annotated_papers(self):
        dump = load_gql_dump(self.path / "structure-annotations.json", compressed=False)["allPapers"]
        annotations = {}
        for a in dump:
            arxiv_id = clear_arxiv_version(a.arxiv_id)
            annotations[arxiv_id] = a
        return annotations

    def to_pickle(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
