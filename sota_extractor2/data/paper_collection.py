from .elastic import Paper as PaperText, Fragments
from .table import Table, read_tables
from .json import load_gql_dump
from pathlib import Path
import re
import pickle
from joblib import Parallel, delayed
from collections import UserList
from ..helpers.jupyter import display_table

class Paper:
    def __init__(self, paper_id, text, tables, annotations):
        self.paper_id = paper_id
        self.arxiv_no_version = remove_arxiv_version(paper_id)
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
    files = list(path.glob("**/text.json"))
    texts = Parallel(n_jobs=jobs, prefer="processes")(delayed(PaperText.from_file)(f) for f in files)
    return {text.meta.id: text for text in texts}


def _load_tables(path, annotations, jobs, migrate):
    files = list(path.glob("**/metadata.json"))
    tables = Parallel(n_jobs=jobs, prefer="processes")(delayed(read_tables)(f.parent, annotations.get(f.parent.name), migrate) for f in files)
    return {f.parent.name: tbls for f, tbls in zip(files, tables)}


def _load_annotated_papers(path):
    dump = load_gql_dump(path, compressed=path.suffix == ".gz")["allPapers"]
    annotations = {remove_arxiv_version(a.arxiv_id): a for a in dump}
    annotations.update({a.arxiv_id: a for a in dump})
    return annotations


class PaperCollection(UserList):
    def __init__(self, data=None):
        super().__init__(data)

    @classmethod
    def from_files(cls, path, annotations_path=None, load_texts=True, load_tables=True, jobs=-1, migrate=False):
        path = Path(path)
        if annotations_path is None:
            annotations_path = path / "structure-annotations.json"
        if load_texts:
            texts = _load_texts(path, jobs)
        else:
            texts = {}

        annotations = _load_annotated_papers(annotations_path)
        if load_tables:
            tables = _load_tables(path, annotations, jobs, migrate)
        else:
            tables = {}
            annotations = {}
        outer_join = set(texts).union(set(tables))

        papers = [Paper(k, texts.get(k), tables.get(k, []), annotations.get(k)) for k in outer_join]
        return cls(papers)

    def get_by_id(self, paper_id, ignore_version=True):
        if ignore_version:
            paper_id = remove_arxiv_version(paper_id)
            for p in self.data:
                if p.arxiv_no_version == paper_id:
                    return p
            return None
        else:
            for p in self.data:
                if p.paper_id == paper_id:
                    return p
            return None

    @classmethod
    def cells_gold_tags_legend(cls):
        tags = [
            ("Tag", "description"),
            ("model-best", "model that has results that author most likely would like to have exposed"),
            ("model-paper", "an example of a generic model, (like LSTM)"),
            ("model-competing", "model from another paper used for comparison"),
            ("dataset-task", "Task"),
            ("dataset", "Dataset"),
            ("dataset-sub", "Subdataset"),
            ("dataset-metric", "Metric"),
            ("model-params", "Params, f.e., number of layers or inference time"),
            ("table-meta", "Cell describing other header cells"),
            ("trash", "Parsing erros")
        ]
        anns = [(t[0], "") for t in tags]
        anns[0] = ("", "")
        display_table(tags, anns)


    def to_pickle(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
