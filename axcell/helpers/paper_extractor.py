#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pathlib import Path
from axcell.helpers import LatexConverter, Unpack
from axcell.errors import UnpackError, LatexConversionError
from axcell.data.elastic import Paper as PaperText
import axcell.data.extract_tables as table_extraction

import re
import warnings

arxiv_re = re.compile(r"^(?P<arxiv_id>\d{4}\.\d+(v\d+)?)(\..*)?$")


class PaperExtractor:
    def __init__(self, root):
        self.root = Path(root)
        self.unpack = Unpack()
        self.latex = LatexConverter()

    def __call__(self, source):
        source = Path(source)

        m = arxiv_re.match(source.name)
        if not m:
            warnings.warn(f'Unable to infer arxiv_id from "{source.name}" filename')
            arxiv_id = source.name
        else:
            arxiv_id = m.group('arxiv_id')

        subpath = source.relative_to(self.root / 'sources').parent / arxiv_id
        unpack_path = self.root / 'unpacked_sources' / subpath
        try:
            self.unpack(source, unpack_path)
        except UnpackError as e:
            if e.args[0].startswith('The paper has been withdrawn'):
                return 'withdrawn'
            return 'no-tex'
        html_path = self.root / 'htmls' / subpath / 'index.html'
        try:
            html = self.latex.to_html(unpack_path)
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(html, 'utf-8')
        except LatexConversionError:
            return 'processing-error'

        text_path = self.root / 'papers' / subpath / 'text.json'
        doc = PaperText.from_html(html, arxiv_id)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(doc.to_json(), 'utf-8')

        tables_path = self.root / 'papers' / subpath
        tables = table_extraction.extract_tables(html)
        table_extraction.save_tables(tables, tables_path)

        return 'success'
