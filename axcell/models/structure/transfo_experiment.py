#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import time

from .experiment import Experiment
from .nbsvm import preds_for_cell_content, preds_for_cell_content_max, preds_for_cell_content_multi
import dataclasses
from dataclasses import dataclass
from typing import Tuple
from axcell.helpers.training import set_seed
from fastai.text import *
import numpy as np
from pathlib import Path
import json

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from fastai.text import * # for utilty functions

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import tensorflow_datasets

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          DistilBertConfig,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizer, DataProcessor, InputExample, AutoConfig)

from transformers import AdamW, WarmupLinearSchedule

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import AutoTokenizer, AutoModelForSequenceClassification, glue_convert_examples_to_features
from transformers.data.processors.glue import glue_processors


logger = logging.getLogger(__name__)


def train(args, train_dataset, valid_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = args.get_summary_writer()

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = range(int(args.num_train_epochs))
    set_seed(args.seed, "Training", all_gpus=(args.n_gpu > 1))  # Added here for reproductibility (even between python 2 and 3)
    mb = master_bar(train_iterator)
    mb.first_bar.comment = f'Epochs'
    results={}
    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, display=args.local_rank not in [-1, 0], parent=mb)

        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 and not args.tpu:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    mb.child.comment = f"loss: {loss}"
                    tb_writer.add_scalar('train/lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('train/loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
                #mb.first_bar.comment = f'first bar stat'
                #mb.write(f'Finished loop {i}.')
            if args.tpu:
                args.xla_model.optimizer_step(optimizer, barrier=True)
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
            results = evaluate(args, model, valid_dataset)
            for key, value in results.items():
                tb_writer.add_scalar('eval/{}'.format(key), value, global_step)
            mb.first_bar.comment = str(results['acc'])
        mb.write(f"Epoch: {epoch} {loss} Accuracy: {results.get('acc', 0)}")

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    hparams_dict = {k: v for k, v in dataclasses.asdict(args).items() if isinstance(v, (int, float, str, bool,))}
    tb_writer.add_hparams(hparam_dict=hparams_dict, metric_dict=results)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset, prefix="", eval_output_dir="/tmp/out"):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    results = {}
    eval_task = args.task_name
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    mb = progress_bar(eval_dataloader)
    for batch in mb:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(eval_task, preds, out_label_ids)
    results.update(result)
    results['loss'] = eval_loss
    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return results

def prepare_glue_examples(tokenizer, task_name='mrpc', split_name='train'):
    processor = glue_processors[task_name]()

    def tf_mrpc_to_pytorch(d):
        for ex in d:
            ex = processor.get_example_from_tensor_dict(ex)
            #        ex = processor.tfds_map(ex)
            yield ex

    tf_data = tensorflow_datasets.load(f"glue/{task_name}")[split_name]
    examples = tf_mrpc_to_pytorch(tf_data)
    features = glue_convert_examples_to_features(examples,
                                                 tokenizer,
                                                 max_length=128,
                                                 task='mrpc')

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def strip_tensors(r):
    nr = {}
    for k,v in r.items():
        v = v.numpy()
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        else:
            v = v.item()
        nr[k] = v
    return nr

def glue_dataset_to_df(task_name):
    data = tensorflow_datasets.load(f"glue/{task_name}")
    new_dict = {}
    for name, dataset in data.items():
        new_dict[name] = pd.DataFrame.from_records([strip_tensors(r) for r in dataset],
                                                   columns=dataset.output_shapes.keys(),
                                                   index='idx')
    return new_dict.get('train', None), new_dict.get('validation', None), new_dict.get('test', None)

def convert_df_to_examples(df, text_a='sentence1', text_b='sentence2', label='label'):
    return [InputExample(
                idx,
                row[text_a],
                row[text_b],
                str(row[label]))
            for idx, row in df.iterrows()]

def convert_df_to_dataset(tokenizer, df, max_length=128, task='mrpc', text_a='sentence1', text_b='sentence2', label='label', return_labels=False):
    label_list = list(sorted(map(str, df[label].unique())))
    examples = convert_df_to_examples(df, text_a, text_b, label)
    features = glue_convert_examples_to_features(examples,
                                                 tokenizer,
                                                 max_length=max_length,
                                                 label_list=label_list,
                                                 output_mode='classification',
                                                 task=None)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    if return_labels:
        return dataset, label_list
    return dataset

@dataclass
class TransfoLearner():
    model: nn.Module
    tokenizer: Any
    data: Any

def get_preds(args, model, dataset, ordered=True):
    eval_dataset = dataset
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if isinstance(eval_sampler, DistributedSampler) and ordered:
        # Note that DistributedSampler samples randomly
        raise ValueError("Unable to run distributed get_preds with ordered == True")
    logger.info("Num examples = %d", len(eval_dataset))
    logger.info("Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    mb = progress_bar(eval_dataloader)
    preds = []
    labels = []
    try:
        with torch.no_grad():
            model.to(args.device)
            model.eval()
            for batch in mb:
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                preds.append(logits.detach().cpu())
                labels.append(inputs['labels'].detach().cpu()) # add non_blocking=True but somehow it isn't avaliabe in our torch
            return torch.cat(preds, dim=0), torch.cat(labels, dim=0)
    finally:
        model.to("cpu")

@dataclass
class TransfoDatabunch():
    num_labels: int
    train_ds: Any
    valid_ds: Any
    test_ds: Any

@dataclass
class TransfoExperiment(Experiment):
    test_split: str = None
    valid_split: str = None
    text_a: str = 'text'
    text_b: str = 'cell_content'
    label: str = 'label'
    #@help("Model type selected in the list: ...")
    model_type: str = None
    #@help("Path to pre-trained model or shortcut name selected in the list: ...")
    pretrained_name: str = None
    #@help("The name of the task to train selected in the list: " + "".join(processors.keys()))
    task_name: str = None
    #@help("Pretrained config name or path if not the same as model_name")
    config_name: str = ""
    #@help("Pretrained tokenizer name or path if not the same as model_name")
    tokenizer_name: str = ""
    #@help("Where do you want to store the pre-trained models downloaded from s3")
    cache_dir: str = ""
    #@help("The maximum total input sequence length after tokenization. Sequences longer  than this will be truncated sequences shorter will be padded.")
    max_seq_length: int = 128
    #@help("Whether to run training.")
    do_train: bool = False
    #@help("Whether to run eval on the dev set.")
    do_eval: bool = False
    #@help("Rul evaluation during training at each logging step.")
    evaluate_during_training: bool = False
    #@help("Batch size per GPU/CPU for training.")
    per_gpu_train_batch_size: int = 8
    #@help("Batch size per GPU/CPU for evaluation.")
    per_gpu_eval_batch_size: int = 8
    #@help("Number of updates steps to accumulate before performing a backward/update pass.")
    gradient_accumulation_steps: int = 1
    #@help("The initial learning rate for Adam.")
    learning_rate: float = 5e-5
    #@help("Weight deay if we apply some.")
    weight_decay: float = 0.0
    #@help("Epsilon for Adam optimizer.")
    adam_epsilon: float = 1e-8
    #@help("Max gradient norm.")
    max_grad_norm: float = 1.0
    #@help("Total number of training epochs to perform.")
    num_train_epochs: float = 3.0
    #@help("If > 0: set total number of training steps to perform. Override num_train_epochs.")
    max_steps: int = -1
    #@help("Linear warmup over warmup_steps.")
    warmup_steps: int = 0
    #@help("Log every X updates steps.")
    logging_steps: int = 10
    #@help("Save checkpoint every X updates steps.")
    save_steps: int = 50
    #@help("Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    eval_all_checkpoints: bool = False
    #@help("Avoid using CUDA when available")
    no_cuda: bool = False
    #@help("Overwrite the cached training and evaluation sets")
    overwrite_cache: bool = False
    #@help("random seed for initialization")
    seed: int = 42
    #@help("Whether to run on the TPU defined in the environment variables")
    tpu: bool = False
    #@help("TPU IP address if none are set in the environment variables")
    tpu_ip_address: str = ''
    #@help("TPU name if none are set in the environment variables")
    tpu_name: str = ''
    #@help("XRT TPU config if none are set in the environment variables")
    xrt_tpu_config: str = ''

    #@help("Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    fp16: bool = False
    #@help("For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2' and 'O3']. See details at https://nvidia.github.io/apex/amp.html")
    fp16_opt_level: str = 'O1'
    #@help("For distributed training: local_rank")
    local_rank: int = -1
    #@help("For distant debugging.")
    server_ip: str = ''
    #@help("For distant debugging.")
    server_port: str = ''

    seed: int = 42
    # Unused

    #@help("The input data dir. Should contain the .tsv files (or other data files) for the task.")
    data_dir: str = "/tmp/data"

    #@help("The output directory where the model predictions and checkpoints will be written.")
    output_dir: str = "/tmp/tmp_output_dir"

    #@help("Overwrite the content of the output directory")
    overwrite_output_dir: bool = True

    def __post_init__(self):
        if os.path.exists(self.output_dir) and os.listdir(
                self.output_dir) and self.do_train and not self.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    self.output_dir))

        # Setup distant debugging if needed
        if self.server_ip and self.server_port:
            # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
            import ptvsd
            print("Waiting for debugger attach")
            ptvsd.enable_attach(address=(self.server_ip, self.server_port), redirect_output=True)
            ptvsd.wait_for_attach()

        # Setup CUDA, GPU & distributed training
        if self.local_rank == -1 or self.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            self.n_gpu = 1
        self.device = device
        self.output_mode = "classification"

        self.train_batch_size = self.per_gpu_train_batch_size * max(1, self.n_gpu)
        self.eval_batch_size = self.per_gpu_eval_batch_size * max(1, self.n_gpu)
        self._tokenizer = None
        self._model = None
        self._data_cache = None
        self.train_started = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained_name)
        return self._tokenizer

    @property
    def experiment_name(self):
        from datetime import datetime
        import socket
        if not self.name:
            now = datetime.now()
            d = now.strftime("%y%m%d_%H%M%S")
            h = "_".join(socket.gethostname().split('-'))

            def short_name(name):
                return "".join([p[0] for p in name.split('_')])

            def short_val(val):
                if isinstance(val, bool):
                    return int(val)
                return val

            relevant_params = {k: v for k, v in dataclasses.asdict(self).items()
                               if not k.startswith('_') and hasattr(TransfoExperiment, k) and getattr(TransfoExperiment,
                                                                                                      k) != v}
            params = [f"{short_name(k)}_{v}" for k, v in relevant_params.items() if not isinstance(v, bool)]
            bool_flags = [f"{short_name(k)}" for k, v in relevant_params.items() if isinstance(v, bool) and v]
            params_str =  ".".join(params + bool_flags)

            self.name = f"{d}.{h}.{params_str}"
        return self.name

    def get_summary_writer(self):
        return SummaryWriter("runs/"+self.experiment_name)

    def _save_predictions(self, path):
        self._dump_pickle([self._preds, self._phases], path)

    def _load_predictions(self, path):
        self._preds, self._phases = self._load_pickle(path)
        return self._preds

    def load_predictions(self):
        path = self._path.parent / f"{self._path.stem}.preds"
        return self._load_predictions(path)

    # todo: make it compatible with Experiment
    def get_trained_model(self, data: TransfoDatabunch):
        self._model = self.train_model(data)
        self.has_model = True
        return self._model

    def get_glue_databunch(self):
        return TransfoDatabunch(
            train_ds = prepare_glue_examples(self.tokenizer, self.task_name, 'train'),
            valid_ds = prepare_glue_examples(self.tokenizer, self.task_name, 'validation'),
            test_ds = None
        )

    def get_databunch(self, train_df, valid_df, test_df):
        data_key = (id(train_df), id(valid_df), id(test_df))

        if self._data_cache is not None and self._data_cache.key != data_key:
            self._data_cache = None

        self.tokenizer.max_len = 999999
        if self._data_cache is None:
            common_args = dict(text_a=self.text_a, text_b=self.text_b, label=self.label)
            train_ds, label_list = convert_df_to_dataset(self.tokenizer, train_df, return_labels=True, **common_args)
            data =  TransfoDatabunch(
                num_labels=len(label_list),
                train_ds=train_ds,
                valid_ds=convert_df_to_dataset(self.tokenizer, valid_df, **common_args),
                test_ds=convert_df_to_dataset(self.tokenizer, test_df, **common_args)
            )
            data.key = data_key
            self._data_cache = data
        return self._data_cache

    def new_experiment(self, **kwargs):
        #kwargs.setdefault("has_predictions", False)
        return super().new_experiment(**kwargs)

    def _add_phase(self, state):
        del state['opt']
        del state['train_dl']
        self._phases.append(state)

    def set_seed(self, name):
        return set_seed(self.seed, name, all_gpus=(self.n_gpu > 1))

    # todo: make it compatible with Experiment
    def train_model(self, data: TransfoDatabunch):
        self.set_seed("class")
        self.train_started = time.time()
        num_labels = data.num_labels
        config = AutoConfig.from_pretrained(self.pretrained_name, num_labels=num_labels) #, finetuning_task=args.task_name
        model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_name, config=config)
        train(self, data.train_ds, data.valid_ds, model.to(self.device), self._tokenizer)
        model.to("cpu")
        return model

    def _save_model(self, path):
        model_to_save = self._model.module if hasattr(self._model,
                                                'module') else self._model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(path)
        logger.info("Saving model checkpoint to %s", path)

    # todo: move to Experiment
    def save(self, dir_path):
        dir_path = Path(dir_path)
        dir_path.mkdir(exist_ok=True, parents=True)
        filename = self._get_next_exp_name(dir_path)
        j = dataclasses.asdict(self)
        with open(filename, "wt") as f:
            json.dump(j, f)
        self._save_model(dir_path / f"{filename.stem}.model")
        if hasattr(self, "_preds"):
            self._save_predictions(dir_path / f"{filename.stem}.preds")

        return filename.name

    def evaluate_transformers(self, data):
        return evaluate(self, self._model.to(self.device), data.valid_ds, prefix="")

    def evaluate(self, model, train_df, valid_df, test_df):
        data = self.get_databunch(train_df, valid_df, test_df)
        valid_probs = get_preds(self, model, data.valid_ds, ordered=True)[0].cpu().numpy()
        test_probs = get_preds(self, model, data.test_ds, ordered=True)[0].cpu().numpy()
        train_probs = get_preds(self, model, data.train_ds, ordered=True)[0].cpu().numpy()
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
                print(true_y.shape)
            self._set_results(prefix, preds, true_y)
            self._preds.append(probs)

# # schedule: Tuple = (
# #     (1, 1e-2),   # (a,b) -> fit_one_cyclce(a, b)
# #     (1, 5e-3/2., 5e-3),  # (a, b) -> freeze_to(-2); fit_one_cycle(a, b)
# #     (8, 2e-3/100, 2e-3)  # (a, b) -> unfreeze(); fit_one_cyccle(a, b)
# # )
# # # drop_mult: float = 0.75
# # fp16: bool = False
# pretrained_lm: str = "bert_base_cased"
# # dataset: str = None
# # train_on_easy: bool = True
# # BS: int = 64
# #
# # has_predictions: bool = False   # similar to has_model, but to avoid storing pretrained models we only keep predictions
# #                                 # that can be later used by CRF


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['premise'].numpy().decode('utf-8'),
                            tensor_dict['hypothesis'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

