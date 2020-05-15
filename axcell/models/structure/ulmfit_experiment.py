#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from functools import partial

from .experiment import Experiment, label_map_ext
from axcell.models.structure.nbsvm import *
from sklearn.metrics import confusion_matrix
from .nbsvm import preds_for_cell_content, preds_for_cell_content_max, preds_for_cell_content_multi
import dataclasses
from dataclasses import dataclass
from typing import Tuple
from axcell.helpers.training import set_seed
from fastai.text import *
from fastai.text.learner import _model_meta
import torch
import numpy as np
from pathlib import Path
import json


@dataclass
class ULMFiTExperiment(Experiment):
    seed: int = 42
    schedule: Tuple = (
        (1, 1e-2),   # (a,b) -> fit_one_cyclce(a, b)
        (1, 5e-3/2., 5e-3),  # (a, b) -> freeze_to(-2); fit_one_cycle(a, b)
        (8, 2e-3/100, 2e-3)  # (a, b) -> unfreeze(); fit_one_cyccle(a, b)
    )
    moms: Tuple = None
    drop_mult: float = 0.75
    fp16: bool = False
    pretrained_lm: str = "pretrained-on-papers_enc.pkl"
    dataset: str = None
    train_on_easy: bool = True
    BS: int = 64
    valid_split: str = 'speech_rec'
    test_split: str = 'img_class'
    n_layers: int = 3

    has_predictions: bool = False   # similar to has_model, but to avoid storing pretrained models we only keep predictions
                                    # that can be later used by CRF

    def _save_predictions(self, path):
        self._dump_pickle([self._preds, self._phases], path)

    def _load_predictions(self, path):
        self._preds, self._phases = self._load_pickle(path)
        return self._preds

    def load_predictions(self):
        path = self._path.parent / f"{self._path.stem}.preds"
        return self._load_predictions(path)

    # todo: make it compatible with Experiment
    def get_trained_model(self, data_clas):
        self._model = self.train_model(data_clas)
        self.has_model = True
        return self._model

    def new_experiment(self, **kwargs):
        kwargs.setdefault("has_predictions", False)
        return super().new_experiment(**kwargs)

    def _schedule(self, clas, i):
        s = self.schedule[i]
        cyc_len = s[0]
        if len(s) == 2:
            max_lr = s[1]
        else:
            max_lr = slice(s[1], s[2])

        if self.moms is None:
            clas.fit_one_cycle(cyc_len, max_lr)
        else:
            clas.fit_one_cycle(cyc_len, max_lr, moms=self.moms)

    def _add_phase(self, state):
        del state['opt']
        del state['train_dl']
        self._phases.append(state)

    def _get_train_metrics(self):
        return None

    # todo: make it compatible with Experiment
    def train_model(self, data_clas):
        set_seed(self.seed, "clas")
        cfg = _model_meta[AWD_LSTM]['config_clas'].copy()
        cfg['n_layers'] = self.n_layers

        metrics = self._get_train_metrics()
        clas = text_classifier_learner(data_clas, AWD_LSTM, config=cfg, drop_mult=self.drop_mult, metrics=metrics)
        clas.load_encoder(self.pretrained_lm)
        if self.fp16:
            clas = clas.to_fp16()

        self._phases = []

        if self.schedule[0][0]:
            self._schedule(clas, 0)
            self._add_phase(clas.recorder.get_state())

        if self.schedule[1][0]:
            clas.freeze_to(-2)
            self._schedule(clas, 1)
            self._add_phase(clas.recorder.get_state())

        if self.schedule[2][0]:
            clas.unfreeze()
            self._schedule(clas, 2)
            self._add_phase(clas.recorder.get_state())

        return clas

    def _save_model(self, path):
        self._model.save(path)


    # todo: move to Experiment
    def save(self, dir_path):
        dir_path = Path(dir_path)
        dir_path.mkdir(exist_ok=True, parents=True)
        filename = self._get_next_exp_name(dir_path)
        j = dataclasses.asdict(self)
        with open(filename, "wt") as f:
            json.dump(j, f)
        self.save_model(dir_path / f"{filename.stem}.model")
        if hasattr(self, "_preds"):
            self._save_predictions(dir_path / f"{filename.stem}.preds")

        return filename.name


    def evaluate(self, model, train_df, valid_df, test_df):
        valid_probs = model.get_preds(ds_type=DatasetType.Valid, ordered=True)[0].cpu().numpy()
        test_probs = model.get_preds(ds_type=DatasetType.Test, ordered=True)[0].cpu().numpy()
        train_probs = model.get_preds(ds_type=DatasetType.Train, ordered=True)[0].cpu().numpy()
        self._preds = []

        for prefix, tdf, probs in zip(["train", "valid", "test"],
                                      [train_df, valid_df, test_df],
                                      [train_probs, valid_probs, test_probs]):
            preds = np.argmax(probs, axis=1)

            if self.merge_fragments and self.merge_type != "concat":
                if self.merge_type == "vote_maj":
                    vote_results = preds_for_cell_content(tdf, probs)
                elif self.merge_type == "vote_avg":
                    vote_results = preds_for_cell_content_multi(tdf, probs)
                elif self.merge_type == "vote_max":
                    vote_results = preds_for_cell_content_max(tdf, probs)
                preds = vote_results["pred"]
                true_y = vote_results["true"]
            else:
                true_y = tdf["label"]
                true_y_ext = tdf["cell_type"].apply(lambda x: label_map_ext.get(x, 0))
            self._set_results(prefix, preds, true_y, true_y_ext)
            self._preds.append(probs)


def multipreds2preds(preds, threshold=0.5):
    bs = preds.shape[0]
    return torch.cat([preds, preds.new_full((bs,1), threshold)], dim=-1).argmax(dim=-1)


def accuracy_multilabel(input, target, sigmoid=True, irrelevant_as_class=False, threshold=0.5):
    if sigmoid:
        if irrelevant_as_class:
            input = torch.sigmoid(input).argmax(dim=-1)
            target = target.argmax(dim=-1)
            return (input == target).float().mean()
        else:
            input = torch.sigmoid(input)
            input = multipreds2preds(input, threshold)
            targs = multipreds2preds(target, threshold)
            return (input == targs).float().mean()
    else:
        return accuracy(input, target)


def accuracy_binary(input, target, sigmoid=True, irrelevant_as_class=False, threshold=0.5):
    if sigmoid:
        if irrelevant_as_class:
            input = torch.sigmoid(input).argmax(dim=-1)
            target = target.argmax(dim=-1)
            input[input == 1] = 0
            target[target == 1] = 0
            return (input == target).float().mean()
        else:
            input = torch.sigmoid(input)
            input = multipreds2preds(input, threshold)
            target = multipreds2preds(target, threshold)
            input[input == 1] = 0
            target[target == 1] = 0
            return (input == target).float().mean()
    else:
        input = input.argmax(dim=-1)
        input[input == 1] = 0
        target[target == 1] = 0
        return (input == target).float().mean()


@dataclass
class ULMFiTTableTypeExperiment(ULMFiTExperiment):
    sigmoid: bool = True
    distinguish_ablation: bool = True
    irrelevant_as_class: bool = False
    caption: bool = True
    first_row: bool = False
    first_column: bool = False
    referencing_sections: bool = False
    dedup_seqs: bool = False

    def _save_model(self, path):
        pass

    def _get_train_metrics(self):
        if self.distinguish_ablation:
            return [
                partial(accuracy_multilabel, sigmoid=self.sigmoid, irrelevant_as_class=self.irrelevant_as_class),
                partial(accuracy_binary, sigmoid=self.sigmoid, irrelevant_as_class=self.irrelevant_as_class)
            ]
        else:
            return [accuracy]

    def _transform_df(self, df):
        df = df.copy(True)
        if self.distinguish_ablation:
            df["label"] = 2
            df.loc[df.ablation, "label"] = 1
            df.loc[df.sota, "label"] = 0
        else:
            df["label"] = 1
            df.loc[df.sota, "label"] = 0
            df.loc[df.ablation, "label"] = 0

        if self.sigmoid:
            if self.irrelevant_as_class:
                df["irrelevant"] = ~(df["sota"] | df["ablation"])
            if not self.distinguish_ablation:
                df["sota"] = df["sota"] | df["ablation"]
                df = df.drop(columns=["ablation"])
        else:
            df["class"] = df["label"]

        drop_columns = []
        if not self.caption:
            drop_columns.append("caption")
        if not self.first_column:
            drop_columns.append("col0")
        if not self.first_row:
            drop_columns.append("row0")
        if not self.referencing_sections:
            drop_columns.append("sections")
        df = df.drop(columns=drop_columns)
        return df

    def evaluate(self, model, train_df, valid_df, test_df):
        valid_probs = model.get_preds(ds_type=DatasetType.Valid, ordered=True)[0].cpu().numpy()
        test_probs = model.get_preds(ds_type=DatasetType.Test, ordered=True)[0].cpu().numpy()
        train_probs = model.get_preds(ds_type=DatasetType.Train, ordered=True)[0].cpu().numpy()
        self._preds = []

        def multipreds2preds(preds, threshold=0.5):
            bs = preds.shape[0]
            return np.concatenate([probs, np.ones((bs, 1)) * threshold], axis=-1).argmax(-1)

        for prefix, tdf, probs in zip(["train", "valid", "test"],
                                      [train_df, valid_df, test_df],
                                      [train_probs, valid_probs, test_probs]):

            if self.sigmoid and not self.irrelevant_as_class:
                preds = multipreds2preds(probs)
            else:
                preds = np.argmax(probs, axis=1)

            true_y = tdf["label"]
            self._set_results(prefix, preds, true_y)
            self._preds.append(probs)

    def _set_results(self, prefix, preds, true_y, true_y_ext=None):
        def metrics(preds, true_y):
            y = true_y
            p = preds

            if self.distinguish_ablation:
                g = {0: 0, 1: 0, 2: 1}.get
                bin_y = np.array([g(x) for x in y])
                bin_p = np.array([g(x) for x in p])
                irr = 2
            else:
                bin_y = y
                bin_p = p
                irr = 1

            acc = (p == y).mean()
            tp = ((y != irr) & (p == y)).sum()
            fp = ((p != irr) & (p != y)).sum()
            fn = ((y != irr) & (p == irr)).sum()

            bin_acc = (bin_p == bin_y).mean()
            bin_tp = ((bin_y != 1) & (bin_p == bin_y)).sum()
            bin_fp = ((bin_p != 1) & (bin_p != bin_y)).sum()
            bin_fn = ((bin_y != 1) & (bin_p == 1)).sum()

            prec = tp / (fp + tp)
            reca = tp / (fn + tp)
            bin_prec = bin_tp / (bin_fp + bin_tp)
            bin_reca = bin_tp / (bin_fn + bin_tp)
            return {
                "precision": prec,
                "accuracy": acc,
                "recall": reca,
                "TP": tp,
                "FP": fp,
                "bin_precision": bin_prec,
                "bin_accuracy": bin_acc,
                "bin_recall": bin_reca,
                "bin_TP": bin_tp,
                "bin_FP": bin_fp,
            }

        m = metrics(preds, true_y)
        r = {}
        r[f"{prefix}_accuracy"] = m["accuracy"]
        r[f"{prefix}_precision"] = m["precision"]
        r[f"{prefix}_recall"] = m["recall"]
        r[f"{prefix}_bin_accuracy"] = m["bin_accuracy"]
        r[f"{prefix}_bin_precision"] = m["bin_precision"]
        r[f"{prefix}_bin_recall"] = m["bin_recall"]
        r[f"{prefix}_cm"] = confusion_matrix(true_y, preds).tolist()
        self.update_results(**r)

    def get_cm_labels(self, cm):
        if len(cm) == 3:
            return ["SOTA", "ABLATION", "IRRELEVANT"]
        else:
            return ["SOTA", "IRRELEVANT"]
