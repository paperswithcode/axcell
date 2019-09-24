from ..data.elastic import Paper as PaperText
from ..data.paper_collection import Paper
from extract_tables import extract_tables
import string
import random


# todo: make sure multithreading/processing won't cause collisions
def random_id():
    return "temp_" + ''.join(random.choice(string.ascii_lowercase) for i in range(10))


class TempPaper(Paper):
    """Similar to Paper, but can be used as context manager, temporarily saving the paper to elastic"""
    def __init__(self, html):
        paper_id = random_id()
        text = PaperText.from_html(html, paper_id)
        tables = extract_tables(html)
        super().__init__(paper_id=paper_id, text=text, tables=tables, annotations=None)

    def __enter__(self):
        self.text.save()
        return self

    def __exit__(self, exc, value, tb):
        self.text.delete()
