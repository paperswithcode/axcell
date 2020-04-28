#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
import pandas as pd
from .models.structure.experiment import Experiment, label_map, Labels
from .models.structure.type_predictor import TableType
from copy import deepcopy
import pickle



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


class SessionRecorder:
    def __init__(self, pipeline_logger):
        self.pipeline_logger = pipeline_logger
        self.session = []
        self._recording = False

    def __call__(self, step, **kwargs):
        self.session.append((step, deepcopy(kwargs)))

    def reset(self):
        self.session = []

    def record(self):
        if not self._recording:
            self.pipeline_logger.register(".*", self)
            self._recording = True

    def stop(self):
        if self._recording:
            self.pipeline_logger.unregister(".*", self)
            self._recording = False

    def replay(self):
        self.stop()
        for step, kwargs in self.session:
            self.pipeline_logger(step, **kwargs)

    def save_session(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.session, f)

    def load_session(self, path):
        with open(path, "rb") as f:
            self.session = pickle.load(f)


class StructurePredictionEvaluator:
    def __init__(self, pipeline_logger, pc):
        pipeline_logger.register("structure_prediction::evidences_split", self.on_evidences_split)
        pipeline_logger.register("structure_prediction::tables_labeled", self.on_tables_labeled)
        pipeline_logger.register("type_prediction::predicted", self.on_type_predicted)
        pipeline_logger.register("type_prediction::multiclass_predicted", self.on_type_multiclass_predicted)
        self.pc = pc
        self.results = {}
        self.type_predictions = {}
        self.type_multiclass_predictions = {}
        self.evidences = pd.DataFrame()

    def on_type_multiclass_predicted(self, step, paper, tables, threshold, predictions):
        for table, prediction in zip(tables, predictions):
            self.type_multiclass_predictions[paper.paper_id, table.name] = {
                TableType.SOTA: prediction[0],
                TableType.ABLATION: prediction[1],
                TableType.IRRELEVANT: threshold
            }

    def on_type_predicted(self, step, paper, tables, predictions):
        for table, prediction in zip(tables, predictions):
            self.type_predictions[paper.paper_id, table.name] = prediction

    def on_evidences_split(self, step, evidences, evidences_num):
        self.evidences = pd.concat([self.evidences, evidences])

    def on_tables_labeled(self, step, paper, labeled_tables):
        golds = [p for p in self.pc if p.text.title == paper.text.title]
        paper_id = paper.paper_id
        type_results = []
        cells_results = []
        labeled_tables = {table.name: table for table in labeled_tables}
        if len(golds) == 1:
            gold = golds[0]
            for gold_table, table, in zip(gold.tables, paper.tables):
                table_type = self.type_predictions[paper.paper_id, table.name]
                is_important = table_type == TableType.SOTA or table_type == TableType.ABLATION
                gold_is_important = "sota" in gold_table.gold_tags or "ablation" in gold_table.gold_tags
                type_results.append({"predicted": is_important, "gold": gold_is_important, "name": table.name})
                if not is_important:
                    continue
                table = labeled_tables[table.name]
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

    def get_table_type_predictions(self, paper_id, table_name):
        prediction = self.type_predictions.get((paper_id, table_name))
        multi_predictions = self.type_multiclass_predictions.get((paper_id, table_name))
        if prediction is not None:
            multi_predictions = sorted(multi_predictions.items(), key=lambda x: x[1], reverse=True)
            return prediction, [(k.name, v) for k, v in multi_predictions
                                ]


class LinkerEvaluator:
    def __init__(self, pipeline_logger):
        pipeline_logger.register("linking::call", self.on_before_linking)
        pipeline_logger.register("linking::taxonomy_linking::call", self.on_before_taxonomy)
        pipeline_logger.register("linking::taxonomy_linking::topk", self.on_taxonomy_topk)
        pipeline_logger.register("linking::linked", self.on_after_linking)
        self.proposals = {}
        self.topk = {}
        self.queries = {}

    def on_before_linking(self, step, paper, tables):
        pass

    def on_after_linking(self, step, paper, tables, proposals):
        self.proposals[paper.paper_id] = proposals.copy(deep=True)

    def on_before_taxonomy(self, step, ext_id, query, paper_context, abstract_context, table_context, caption):
        self.queries[ext_id] = (query, paper_context, abstract_context, table_context, caption)

    def on_taxonomy_topk(self, step, ext_id, topk):
        paper_id, table_name, rc = ext_id.split('/')
        row, col = [int(x) for x in rc.split('.')]
        self.topk[paper_id, table_name, row, col] = topk.copy(deep=True)

    def top_matches(self, paper_id, table_name, row, col):
        return self.topk[(paper_id, table_name, row, col)]


class FilteringEvaluator:
    def __init__(self, pipeline_logger):
        pipeline_logger.register("filtering::.*::filtered", self.on_filtered)
        self.proposals = {}
        self.which = {}
        self.reason = pd.Series(dtype=str)

    def on_filtered(self, step, proposals, which, reason, **kwargs):
        _, filter_step, _ = step.split('::')
        if filter_step != "compound_filtering":
            if filter_step in self.proposals:
                self.proposals[filter_step] = pd.concat([self.proposals[filter_step], proposals])
                self.which[filter_step] = pd.concat([self.which[filter_step], which])
            else:
                self.proposals[filter_step] = proposals
                self.which[filter_step] = which
            self.reason = self.reason.append(reason)


