import sys
import pandas as pd
from .models.structure.experiment import Experiment, label_map, Labels
from .models.structure.type_predictor import TableType


class BaseLogger:
    def __init__(self, pipeline_logger, pattern=".*"):
        pipeline_logger.register(pattern, self)

    def __call__(self, step, **kwargs):
        raise NotImplementedError()


class StdoutLogger:
    def __init__(self, pipeline_logger, file=sys.stdout):
        self.file = file
        pipeline_logger.register(".*", self)

    def __call__(self, step, **kwargs):
        print(f"[STEP] {step}: {kwargs}", file=self.file)


class StructurePredictionEvaluator:
    def __init__(self, pipeline_logger, pc):
        pipeline_logger.register("structure_prediction::tables_labelled", self.on_tables_labelled)
        pipeline_logger.register("type_prediction::predicted", self.on_type_predicted)
        self.pc = pc
        self.results = {}
        self.type_predictions = {}

    def on_type_predicted(self, step, paper, tables, predictions):
        self.type_predictions[paper.paper_id] = predictions

    def on_tables_labelled(self, step, paper, tables):
        golds = [p for p in self.pc if p.text.title == paper.text.title]
        paper_id = paper.paper_id
        type_results = []
        cells_results = []
        if len(golds) == 1:
            gold = golds[0]
            for gold_table, table, table_type in zip(gold.tables, paper.tables, self.type_predictions.get(paper.paper_id, [])):
                is_important = table_type == TableType.SOTA or table_type == TableType.ABLATION
                gold_is_important = "sota" in gold_table.gold_tags or "ablation" in gold_table.gold_tags
                type_results.append({"predicted": is_important, "gold": gold_is_important, "name": table.name})
                if not is_important:
                    continue
                rows, cols = table.df.shape
                for r in range(rows):
                    for c in range(cols):
                        cells_results.append({
                            "predicted": table.df.iloc[r, c].gold_tags,
                            "gold": gold_table.df.iloc[r, c].gold_tags,
                            "ext_id": f"{table.name}/{r}.{c}",
                            "content": table.df.iloc[r, c].value
                        })

        self.results[paper_id] = {
            'type': pd.DataFrame.from_records(type_results),
            'cells': pd.DataFrame.from_records(cells_results)
        }

    def map_tags(self, tags):
        mapping = dict(label_map)
        mapping[""] = Labels.EMPTY.value
        return tags.str.strip().apply(lambda x: mapping.get(x, 0))

    def metrics(self, paper_id):
        if paper_id not in self.results:
            print(f"No annotations for {paper_id}")
            return
        print("Structure prediction:")
        results = self.results[paper_id]
        cells_df = results['cells']
        e = Experiment()
        e._set_results(paper_id, self.map_tags(results['cells'].predicted), self.map_tags(results['cells'].gold))
        e.show_results(paper_id, normalize=True)


class LinkerEvaluator:
    def __init__(self, pipeline_logger, pc):
        pipeline_logger.register("linking::call", self.on_before_linking)
        pipeline_logger.register("linking::taxonomy_linking::call", self.on_before_taxonomy)
        pipeline_logger.register("linking::taxonomy_linking::topk", self.on_taxonomy_topk)
        pipeline_logger.register("linking::linked", self.on_after_linking)
        self.proposals = {}
        self.topk = {}

    def on_before_linking(self, step, paper, tables):
        pass

    def on_after_linking(self, step, paper, tables, proposals):
        self.proposals[paper.paper_id] = proposals.copy(deep=True)

    def on_before_taxonomy(self, step, ext_id, query, datasets, caption):
        pass

    def on_taxonomy_topk(self, step, ext_id, topk):
        paper_id, table_name, rc = ext_id.split('/')
        row, col = [int(x) for x in rc.split('.')]
        self.topk[paper_id, table_name, row, col] = topk.copy(deep=True)

    def top_matches(self, paper_id, table_name, row, col):
        return self.topk[(paper_id, table_name, row, col)]
