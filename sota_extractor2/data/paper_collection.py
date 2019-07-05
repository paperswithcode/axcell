from .elastic import Paper as PaperText, Fragments
from .table import Table, read_tables
from .json import load_gql_dump
from pathlib import Path
import re
import pickle
from joblib import Parallel, delayed
from collections import UserList

class Paper:
    def __init__(self, paper_id, text, tables, annotations):
        self.paper_id = paper_id
        if text is not None:
            self.text = text
        else:
            self.text = PaperText()
            self.text.fragments = Fragments()
        self.tables = tables
        self._annotations = annotations
        if annotations is not None:
            self.gold_tags = annotations.gold_tags.strip()
        else:
            self.gold_tags = ''


arxiv_version_re = re.compile(r"v\d+$")
def remove_arxiv_version(arxiv_id):
    return arxiv_version_re.sub("", arxiv_id)


def _load_texts(path, jobs):
    files = list((path / "texts").glob("**/*.json"))
    texts = Parallel(n_jobs=jobs, prefer="processes")(delayed(PaperText.from_file)(f) for f in files)
    return {remove_arxiv_version(text.meta.id): text for text in texts}


def _load_tables(path, annotations, jobs):
    files = list((path / "tables").glob("**/metadata.json"))
    tables = Parallel(n_jobs=jobs, prefer="processes")(delayed(read_tables)(f.parent, annotations.get(f.parent.name)) for f in files)
    return {remove_arxiv_version(f.parent.name): tbls for f, tbls in zip(files, tables)}

def _load_annotated_papers(path):
    dump = load_gql_dump(path / "structure-annotations.json", compressed=False)["allPapers"]
    annotations = {}
    for a in dump:
        arxiv_id = remove_arxiv_version(a.arxiv_id)
        annotations[arxiv_id] = a
    return annotations


class PaperCollection(UserList):
    def __init__(self, data=None):
        super().__init__(data)

    @classmethod
    def from_files(cls, path, load_texts=True, load_tables=True, jobs=-1):
        if load_texts:
            texts = _load_texts(path, jobs)
        else:
            texts = {}

        annotations = _load_annotated_papers(path)
        if load_tables:
            tables = _load_tables(path, annotations, jobs)
        else:
            tables = {}
            annotations = {}
        outer_join = set(texts).union(set(tables))

        papers = [Paper(k, texts.get(k), tables.get(k, []), annotations.get(k)) for k in outer_join]
        return cls(papers)

    def get_by_id(self, paper_id):
        paper_id = remove_arxiv_version(paper_id)
        for p in self.data:
            if p.paper_id == paper_id:
                return p
        return None


    def to_pickle(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
