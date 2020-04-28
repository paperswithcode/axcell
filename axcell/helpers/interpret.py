#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fastai.text.interpret import TextClassificationInterpretation as AbsTextClassificationInterpretation, _eval_dropouts
from fastai.basic_data import DatasetType
import torch


__all__ = ["TextClassificationInterpretation", "TextMultiClassificationInterpretation"]


class TextClassificationInterpretation(AbsTextClassificationInterpretation):
    @classmethod
    def from_learner(cls, learner):
        empty_preds = torch.Tensor([[1]])
        return cls(learner, empty_preds, None, None)

    def intrinsic_attention(self, text:str, class_id:int=None):
        """Calculate the intrinsic attention of the input w.r.t to an output `class_id`, or the classification given by the model if `None`.
        Similar as in base class, but does not apply abs() before summing gradients.
        """
        self.model.train()
        _eval_dropouts(self.model)
        self.model.zero_grad()
        self.model.reset()
        ids = self.data.one_item(text)[0]
        emb = self.model[0].module.encoder(ids).detach().requires_grad_(True)
        lstm_output = self.model[0].module(emb, from_embeddings=True)
        self.model.eval()
        cl = self.model[1](lstm_output + (torch.zeros_like(ids).byte(),))[0].softmax(dim=-1)
        if class_id is None: class_id = cl.argmax()
        cl[0][class_id].backward()
        # attn = emb.grad.squeeze().abs().sum(dim=-1)
        # attn /= attn.max()
        attn = emb.grad.squeeze().sum(dim=-1)
        attn = attn / attn.abs().max() * 0.5 + 0.5
        tokens = self.data.single_ds.reconstruct(ids[0])
        return tokens, attn


class TextMultiClassificationInterpretation(TextClassificationInterpretation):
    def intrinsic_attention(self, text:str, class_id:int=None):
        """Calculate the intrinsic attention of the input w.r.t to an output `class_id`, or the classification given by the model if `None`.
        Similar as in base class, but uses sigmoid instead of softmax and does not apply abs() before summing gradients.
        """
        self.model.train()
        _eval_dropouts(self.model)
        self.model.zero_grad()
        self.model.reset()
        ids = self.data.one_item(text)[0]
        emb = self.model[0].module.encoder(ids).detach().requires_grad_(True)
        lstm_output = self.model[0].module(emb, from_embeddings=True)
        self.model.eval()
        cl = self.model[1](lstm_output + (torch.zeros_like(ids).byte(),))[0].sigmoid()
        if class_id is None: class_id = cl.argmax()
        cl[0][class_id].backward()
        # attn = emb.grad.squeeze().abs().sum(dim=-1)
        # attn /= attn.max()
        attn = emb.grad.squeeze().sum(dim=-1)
        attn = attn / attn.abs().max() * 0.5 + 0.5
        tokens = self.data.single_ds.reconstruct(ids[0])
        return tokens, attn
