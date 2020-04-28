#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import spacy
from scispacy.abbreviation import AbbreviationDetector
from .utils import normalize_cell, normalize_dataset

class AcronymExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_sci_sm")
        abbreviation_pipe = AbbreviationDetector(self.nlp)
        self.nlp.add_pipe(abbreviation_pipe)
        self.nlp.disable_pipes("tagger", "ner", "parser")

    def __call__(self, text):
        doc = self.nlp(text)
        abbrvs = {}
        for abrv in doc._.abbreviations:
            # abbrvs.setdefault(normalize_cell(str(abrv)), Counter())[str(abrv._.long_form)] += 1
            norm = normalize_cell(normalize_dataset(str(abrv)))
            if norm != '':
                abbrvs[norm] = normalize_cell(normalize_dataset(str(abrv._.long_form)))
        return abbrvs
