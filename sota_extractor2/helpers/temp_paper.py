from ..data.elastic import Paper as PaperText
from ..data.paper_collection import Paper
from extract_tables import extract_tables
import string
import random

# todo: make sure multithreading/processing won't cause collisions
def random_id():
    return "temp_" + ''.join(random.choice(string.ascii_lowercase) for i in range(10))


def temp_paper(path):
    text = PaperText.parse_paper(path, random_id())
    tables = extract_tables(path)
    return Paper(paper_id=text.meta['id'], text=text, tables=tables, annotations=None)
