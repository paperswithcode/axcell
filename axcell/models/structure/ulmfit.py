#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fastai.text import *
from pathlib import Path

class ULMFiT_SP:
    def __init__(self, path, file, sp_path=None, sp_model="spm.model", sp_vocab="spm.vocab"):
        path = Path(path)
        sp_path = path if sp_path is None else Path(sp_path)
        self.learner = load_learner(path=path, file=file)
        import sys, os
        print(f"[PID {os.getpid()}] Load model {file}", file=sys.stderr)
        sys.stderr.flush()
        self._fix_sp_processor(sp_path, sp_model, sp_vocab)

        # disable multiprocessing to avoid celery deamon issues
        for dl in self.learner.data.dls:
            dl.num_workers = 0

    def _fix_sp_processor(self, sp_path, sp_model, sp_vocab):
        for processor in self.learner.data.label_list.valid.x.processor:
            if isinstance(processor, SPProcessor):
                processor.sp_model = sp_path / sp_model
                processor.sp_vocab = sp_path / sp_vocab
                processor.n_cpus = 1

                #todo: see why it wasn't set on save
                processor.mark_fields = True
