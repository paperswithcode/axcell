from .experiment import Experiment
from .nbsvm import preds_for_cell_content, preds_for_cell_content_max, preds_for_cell_content_multi
import dataclasses
from dataclasses import dataclass
from typing import Tuple
from sota_extractor2.helpers.training import set_seed
from fastai.text import *
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
    drop_mult: float = 0.75
    fp16: bool = False
    pretrained_lm: str = "pretrained-on-papers_enc.pkl"
    dataset: str = None
    train_on_easy: bool = True
    BS: int = 64

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
        if len(s) == 2:
            clas.fit_one_cycle(s[0], s[1])
        else:
            clas.fit_one_cycle(s[0], slice(s[1], s[2]))

    def _add_phase(self, state):
        del state['opt']
        del state['train_dl']
        self._phases.append(state)

    # todo: make it compatible with Experiment
    def train_model(self, data_clas):
        set_seed(self.seed, "clas")
        clas = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=self.drop_mult)
        clas.load_encoder(self.pretrained_lm)
        if self.fp16:
            clas = clas.to_fp16()

        self._schedule(clas, 0)
        self._phases = []
        self._add_phase(clas.recorder.get_state())

        clas.freeze_to(-2)
        self._schedule(clas, 1)
        self._add_phase(clas.recorder.get_state())

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
            self._set_results(prefix, preds, true_y)
            self._preds.append(probs)
