from fastai.text import *
from pathlib import Path

class ULMFiT_SP:
    def __init__(self, path, file, sp_path=None, sp_model="spm.model", sp_vocab="spm.vocab"):
        path = Path(path)
        sp_path = path if sp_path is None else Path(sp_path)
        self.learner = load_learner(path=path, file=file)
        self._fix_sp_processor(sp_path, sp_model, sp_vocab)

    def _fix_sp_processor(self, sp_path, sp_model, sp_vocab):
        for processor in self.learner.data.label_list.valid.x.processor:
            if isinstance(processor, SPProcessor):
                processor.sp_model = sp_path / sp_model
                processor.sp_vocab = sp_path / sp_vocab
