#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fastai.text import *
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from .experiment import Labels, label_map
from .ulmfit_experiment import ULMFiTExperiment
import re
from .ulmfit import ULMFiT_SP
from ...pipeline_logger import pipeline_logger
from copy import deepcopy


def load_crf(path):
    with open(path, "rb") as f:
        return pickle.load(f)


with_letters_re = re.compile(r"(?:^\s*[a-zA-Z])|(?:[a-zA-Z]{2,})")

def cut_ulmfit_head(model):
    pooling = PoolingLinearClassifier([1], [])
    pooling.layers = model[1].layers[:-2]
    return SequentialRNN(model[0], pooling)


# todo: move to TSP
n_ulmfit_features = 50
n_fasttext_features = 0
n_layout_features = 16
n_features = n_ulmfit_features + n_fasttext_features + n_layout_features
n_classes = 5


class TableStructurePredictor(ULMFiT_SP):
    step = "structure_prediction"

    def __init__(self, path, file, crf_path=None, crf_model=None,
                 sp_path=None, sp_model="spm.model", sp_vocab="spm.vocab"):
        super().__init__(path, file, sp_path, sp_model, sp_vocab)

        self._full_learner = deepcopy(self.learner)
        self.learner.model = cut_ulmfit_head(self.learner.model)
        self.learner.loss_func = None

        if crf_model is not None:
            crf_path = Path(path) if crf_path is None else Path(crf_path)
            self.crf = load_crf(crf_path / crf_model)
        else:
            self.crf = None

        # todo: clean Experiment from older approaches
        self._e = ULMFiTExperiment(remove_num=False, drop_duplicates=False,
               this_paper=True, merge_fragments=True, merge_type='concat',
               evidence_source='text_highlited', split_btags=True, fixed_tokenizer=True,
               fixed_this_paper=True, mask=True, evidence_limit=None, context_tokens=None,
               lowercase=True, drop_mult=0.15, fp16=True, train_on_easy=False)

    def preprocess_df(self, raw_df):
        return self._e.transform_df(raw_df)

    @staticmethod
    def keep_alphacells(df):
        # which = df.cell_content.str.contains(with_letters_re)
        which = df.cell_content.str.contains(with_letters_re)
        return df[which], df[~which]

    def df2tl(self, df):
        text_cols = ["cell_styles", "cell_layout", "text", "cell_content", "row_context", "col_context",
                     "cell_reference"]
        df = df[text_cols]
        return TextList.from_df(df, cols=text_cols)

    def get_features(self, evidences, use_crf=True):
        if use_crf:
            learner = self.learner
        else:
            learner = self._full_learner
        if len(evidences):
            tl = self.df2tl(evidences)
            learner.data.add_test(tl)

            preds, _ = learner.get_preds(DatasetType.Test, ordered=True)
            return preds.cpu().numpy()
        return np.zeros((0, n_ulmfit_features if use_crf else n_classes))

    @staticmethod
    def to_tables(df, transpose=False, n_ulmfit_features=n_ulmfit_features):
        X_tables = []
        Y_tables = []
        ids = []
        C_tables = []
        for table_id, frame in df.groupby("table_id"):
            rows, cols = frame.row.max()+1, frame.col.max()+1
            x_table = np.zeros((rows, cols, n_features))
            ###y_table = np.ones((rows, cols), dtype=np.int) * n_classes
            c_table = np.full((rows, cols), "", dtype=np.object)
            for i, r in frame.iterrows():
                x_table[r.row, r.col, :n_ulmfit_features] = r.features
                c_table[r.row, r.col] = r.cell_content
                #x_table[r.row, r.col, n_ulmfit_features:n_ulmfit_features+n_fasttext_features] = ft_model[r.text]
                # if n_fasttext_features > 0:
                #     x_table[r.row, r.col, n_ulmfit_features:n_ulmfit_features+n_fasttext_features] = ft_model[r.cell_content]
                ###y_table[r.row, r.col] = r.label
                if n_layout_features > 0:
                    offset = n_ulmfit_features+n_fasttext_features
                    layout = r.cell_layout
                    x_table[r.row, r.col, offset] = 1 if 'border-t' in layout or 'border-tt' in layout else -1
                    x_table[r.row, r.col, offset+1] = 1 if 'border-b' in layout or 'border-bb' in layout else -1
                    x_table[r.row, r.col, offset+2] = 1 if 'border-l' in layout or 'border-ll' in layout else -1
                    x_table[r.row, r.col, offset+3] = 1 if 'border-r' in layout or 'border-rr' in layout else -1
                    x_table[r.row, r.col, offset+4] = 1 if r.cell_reference == "True" else -1
                    x_table[r.row, r.col, offset+5] = 1 if r.cell_styles == "True" else -1
                    for span_idx, span in enumerate(["cb", "ci", "ce", "rb", "ri", "re"]):
                        x_table[r.row, r.col, offset+6+span_idx] = 1 if f'span-{span}' in r.cell_layout else -1
                    x_table[r.row, r.col, offset+12] = 1 if r.row == 0 else -1
                    x_table[r.row, r.col, offset+13] = 1 if r.row == rows-1 else -1
                    x_table[r.row, r.col, offset+14] = 1 if r.col == 0 else -1
                    x_table[r.row, r.col, offset+15] = 1 if r.col == cols-1 else -1
                #x_table[r.row, r.col, -n_fasttext_features:] = ft_model[r.cell_content]
            X_tables.append(x_table)
            ###Y_tables.append(y_table)
            C_tables.append(c_table)
            ids.append(table_id)
            if transpose:
                X_tables.append(x_table.transpose((1, 0, 2)))
                ###Y_tables.append(y_table.transpose())
                C_tables.append(c_table.transpose())
                ids.append(table_id)
        ###return (X_tables, Y_tables), C_tables, ids
        return X_tables, C_tables, ids

    @staticmethod
    def merge_with_preds(df, preds):
        if not len(df):
            return []
        ext_id = df.ext_id.str.split("/", expand=True)
        return list(zip(ext_id[0] + "/" + ext_id[1], ext_id[2].astype(int), ext_id[3].astype(int),
                        preds, df.text, df.cell_content, df.cell_layout, df.cell_styles, df.cell_reference, df.label))

    @staticmethod
    def merge_all_with_preds(df, df_num, preds, use_crf=True):
        columns = ["table_id", "row", "col", "features", "text", "cell_content", "cell_layout",
                   "cell_styles", "cell_reference", "label"]

        alpha = TableStructurePredictor.merge_with_preds(df, preds)
        nums = TableStructurePredictor.merge_with_preds(df_num, np.zeros((len(df_num), n_ulmfit_features if use_crf else n_classes)))

        df1 = pd.DataFrame(alpha, columns=columns)
        df2 = pd.DataFrame(nums, columns=columns)
        df2.label = n_classes
        return df1.append(df2, ignore_index=True)

    # todo: fix numeric cells being labelled as meta / other
    @staticmethod
    def format_predictions(tables_preds, test_ids):
        num2label = {v: k for k, v in label_map.items()}
        num2label[0] = "table-meta"
        num2label[Labels.PAPER_MODEL.value] = 'model-paper'
        num2label[Labels.DATASET.value] = 'dataset'
        num2label[max(label_map.values()) + 1] = ''

        flat = []
        for preds, ext_id in zip(tables_preds, test_ids):
            paper_id, table_id = ext_id.split("/")
            labels = pd.DataFrame(preds).applymap(num2label.get).values
            flat.extend(
                [(paper_id, table_id, r, c, labels[r, c]) for r in range(len(labels)) for c in range(len(labels[r])) if
                 labels[r, c]])
        return pd.DataFrame(flat, columns=["paper", "table", "row", "col", "predicted_tags"])

    def predict_tags(self, raw_evidences, use_crf=True):
        evidences, evidences_num = self.keep_alphacells(self.preprocess_df(raw_evidences))
        pipeline_logger(f"{TableStructurePredictor.step}::evidences_split", evidences=evidences, evidences_num=evidences_num)
        features = self.get_features(evidences, use_crf)
        df = self.merge_all_with_preds(evidences, evidences_num, features, use_crf)
        tables, contents, ids = self.to_tables(df, n_ulmfit_features=n_ulmfit_features if use_crf else n_classes)
        if use_crf:
            preds = self.crf.predict(tables)
        else:
            preds = []
            for table in tables:
                p = table[..., :n_classes].argmax(axis=-1)
                p[table[..., :n_classes].max(axis=-1) == 0.0] = n_classes
                preds.append(p)
        return self.format_predictions(preds, ids)

    # todo: consider adding sota/ablation information
    @staticmethod
    def label_table(paper, table, annotations, in_place):
        structure = pd.DataFrame().reindex_like(table.matrix).fillna("")
        ext_id = (paper.paper_id, table.name)
        if ext_id in annotations:
            for _, entry in annotations[ext_id].iterrows():
                # todo: add model-ensemble support
                structure.iloc[entry.row, entry.col] = entry.predicted_tags if entry.predicted_tags != "model-paper" else "model-best"
        if not in_place:
            table = deepcopy(table)
        table.set_tags(structure.values)
        return table

    # todo: take EvidenceExtractor in constructor
    def label_tables(self, paper, tables, raw_evidences, in_place=False, use_crf=True):
        pipeline_logger(f"{TableStructurePredictor.step}::label_tables", paper=paper, tables=tables, raw_evidences=raw_evidences)
        if len(raw_evidences):
            tags = self.predict_tags(raw_evidences, use_crf)
            annotations = dict(list(tags.groupby(by=["paper", "table"])))
        else:
            annotations = {}  # just deep-copy all tables
        pipeline_logger(f"{TableStructurePredictor.step}::annotations", paper=paper, tables=tables, annotations=annotations)
        labeled = [self.label_table(paper, table, annotations, in_place) for table in tables]
        pipeline_logger(f"{TableStructurePredictor.step}::tables_labeled", paper=paper, labeled_tables=labeled)
        return labeled
