from ..models.structure import TableType
from ..loggers import StructurePredictionEvaluator, LinkerEvaluator, FilteringEvaluator
import pandas as pd


class TableTypeExplainer:
    def __init__(self, paper, table, table_type, probs):
        self.paper = paper
        self.table = table
        self.table_type = table_type
        self.probs = pd.DataFrame(probs, columns=["type", "probability"])

    def __str__(self):
        return f"Table {self.table.name} was labelled as {self.table_type}."

    def display(self):
        print(self)
        self.probs.display()


class Explainer:
    def __init__(self, pipeline_logger, paper_collection):
        self.spe = StructurePredictionEvaluator(pipeline_logger, paper_collection)
        self.le = LinkerEvaluator(pipeline_logger, paper_collection)
        self.fe = FilteringEvaluator(pipeline_logger)

    def explain(self, paper, cell_ext_id):
        paper_id, table_name, rc = cell_ext_id.split('/')
        if paper.paper_id != paper_id:
            return "No such cell"

        row, col = [int(x) for x in rc.split('.')]

        table_type, probs = self.spe.get_table_type_predictions(paper_id, table_name)

        if table_type == TableType.IRRELEVANT:
            return TableTypeExplainer(paper, paper.table_by_name(table_name), table_type, probs)

        reason = self.fe.reason.get(cell_ext_id)
        if reason is None:
            pass
        else:
            return reason
