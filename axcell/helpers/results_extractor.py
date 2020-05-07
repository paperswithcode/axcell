#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from axcell.data.structure import CellEvidenceExtractor
from axcell.models.structure import TableType, TableStructurePredictor, TableTypePredictor
from axcell.models.linking import *
from pathlib import Path


class ResultsExtractor:
    def __init__(self, models_path):
        models_path = Path(models_path)
        self.cell_evidences = CellEvidenceExtractor()
        self.ttp = TableTypePredictor(models_path, "table-type-classifier.pth")
        self.tsp = TableStructurePredictor(models_path, "table-structure-classifier.pth")
        self.taxonomy = Taxonomy(taxonomy=models_path / "taxonomy.json", metrics_info=models_path / "metrics.json")

        self.evidence_finder = EvidenceFinder(self.taxonomy, abbreviations_path=models_path / "abbreviations.json")
        self.context_search = ContextSearch(self.taxonomy, self.evidence_finder)
        self.dataset_extractor = DatasetExtractor(self.evidence_finder)

        self.linker = Linker("linking", self.context_search, self.dataset_extractor)
        self.filters = StructurePredictionFilter() >> ConfidenceFilter(0.8) >> \
            BestResultFilter(self.taxonomy, context="paper") >> ConfidenceFilter(0.85)

    def __call__(self, paper, tables=None, in_place=False):
        if tables is None:
            tables = paper.tables
        tables_types = self.ttp.predict(paper, tables)
        if in_place:
            types = {
                TableType.SOTA: 'leaderboard',
                TableType.ABLATION: 'ablation',
                TableType.IRRELEVANT: 'irrelevant'
            }
            for table, table_type in zip(paper.tables, tables_types):
                table.gold_tags = types[table_type]
        sota_tables = [
            table for table, table_type in zip(paper.tables, tables_types)
            if table_type != TableType.IRRELEVANT
        ]
        paper.text.save()
        evidences = self.cell_evidences(paper, sota_tables)
        labeled_tables = self.tsp.label_tables(paper, sota_tables, evidences, in_place=in_place, use_crf=False)

        proposals = self.linker(paper, labeled_tables)
        proposals = self.filters(proposals)
        proposals = proposals[["dataset", "metric", "task", "model", "parsed"]] \
            .reset_index(drop=True).rename(columns={"parsed": "score"})
        return proposals
