#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# metrics[taxonomy name] is a list of normalized evidences for taxonomy name
from collections import Counter, OrderedDict

from axcell.models.linking.acronym_extractor import AcronymExtractor
from axcell.models.linking.probs import get_probs, reverse_probs
from axcell.models.linking.utils import normalize_dataset, normalize_dataset_ws, normalize_cell, normalize_cell_ws
from scipy.special import softmax
import re
import pandas as pd
import numpy as np
import json
import ahocorasick
from numba import njit, typed, types
from pathlib import Path

from axcell.pipeline_logger import pipeline_logger

from axcell.models.linking import manual_dicts


def dummy_item(reason):
    return pd.DataFrame(dict(dataset=[reason], task=[reason], metric=[reason], evidence=[""], confidence=[0.0]))


class EvidenceFinder:
    single_letter_re = re.compile(r"\b\w\b")
    init_letter_re = re.compile(r"\b\w")
    end_letter_re = re.compile(r"\w\b")
    letter_re = re.compile(r"\w")

    def __init__(self, taxonomy, abbreviations_path=None, use_manual_dicts=False):
        self.abbreviations_path = abbreviations_path
        self.use_manual_dicts = use_manual_dicts
        self._init_structs(taxonomy)

    @staticmethod
    def evidences_from_name(key):
        x = normalize_dataset_ws(key)
        y = [w for w in x.split() if w not in manual_dicts.stop_words]
        return [x] + y if len(y) > 1 else [x]

    @staticmethod
    def get_basic_dicts(taxonomy):
        tasks = {ts: [normalize_dataset_ws(ts)] for ts in taxonomy.tasks}
        datasets = {ds: EvidenceFinder.evidences_from_name(ds) for ds in taxonomy.datasets}
        metrics = {ms: EvidenceFinder.evidences_from_name(ms) for ms in taxonomy.metrics}
        return tasks, datasets, metrics

    @staticmethod
    def merge_evidences(target, source):
        for name, evs in source.items():
            target.setdefault(name, []).extend(evs)

    @staticmethod
    def make_trie(names):
        trie = ahocorasick.Automaton()
        for name in names:
            norm = name.replace(" ", "")
            trie.add_word(norm, (len(norm), name))
        trie.make_automaton()
        return trie

    @staticmethod
    def get_auto_evidences(name, abbreviations, abbrvs_trie):
        frags = EvidenceFinder.find_names(normalize_dataset_ws(name), abbrvs_trie)
        evidences = []
        for f in frags:
            evidences.extend(abbreviations[f])
        return list(set(evidences))

    @staticmethod
    def find_names(text, names_trie):
        text = text.lower()
        profile = EvidenceFinder.letter_re.sub("i", text)
        profile = EvidenceFinder.init_letter_re.sub("b", profile)
        profile = EvidenceFinder.end_letter_re.sub("e", profile)
        profile = EvidenceFinder.single_letter_re.sub("x", profile)
        text = text.replace(" ", "")
        profile = profile.replace(" ", "")
        s = Counter()
        for (end, (l, word)) in names_trie.iter(text):
            if profile[end] in ['e', 'x'] and profile[end - l + 1] in ['b', 'x']:
                s[word] += 1
        return s

    def find_datasets(self, text):
        return EvidenceFinder.find_names(text, self.all_datasets_trie)

    def find_metrics(self, text):
        return EvidenceFinder.find_names(text, self.all_metrics_trie)

    def find_tasks(self, text):
        return EvidenceFinder.find_names(text, self.all_tasks_trie)

    def init_evidence_dicts(self, taxonomy):
        self.tasks, self.datasets, self.metrics = EvidenceFinder.get_basic_dicts(taxonomy)

        if self.use_manual_dicts:
            EvidenceFinder.merge_evidences(self.tasks, manual_dicts.tasks)
            EvidenceFinder.merge_evidences(self.datasets, manual_dicts.datasets)
            EvidenceFinder.merge_evidences(self.metrics, manual_dicts.metrics)

        if self.abbreviations_path is not None:
            with Path(self.abbreviations_path).open('rt') as f:
                abbreviations = json.load(f)
            abbrvs_trie = EvidenceFinder.make_trie(list(abbreviations.keys()))

            ds_auto = {x: EvidenceFinder.get_auto_evidences(x, abbreviations, abbrvs_trie) for x in taxonomy.datasets}
            ms_auto = {x: EvidenceFinder.get_auto_evidences(x, abbreviations, abbrvs_trie) for x in taxonomy.metrics}

            EvidenceFinder.merge_evidences(self.datasets, ds_auto)
            EvidenceFinder.merge_evidences(self.metrics, ms_auto)

        self.datasets = {k: (v + ['test'] if 'val' not in k else v + ['validation', 'dev', 'development']) for k, v in
                    self.datasets.items()}
        if self.use_manual_dicts:
            self.datasets.update({
                'LibriSpeech dev-clean': ['libri speech dev clean', 'libri speech', 'dev', 'clean', 'dev clean', 'development'],
                'LibriSpeech dev-other': ['libri speech dev other', 'libri speech', 'dev', 'other', 'dev other', 'development', 'noisy'],
            })

    def _init_structs(self, taxonomy):
        self.init_evidence_dicts(taxonomy)

        self.datasets = {k: set(v) for k, v in self.datasets.items()}
        self.metrics = {k: set(v) for k, v in self.metrics.items()}
        self.tasks = {k: set(v) for k, v in self.tasks.items()}

        self.all_datasets = set(normalize_cell_ws(normalize_dataset(y)) for x in self.datasets.values() for y in x)
        self.all_metrics = set(normalize_cell_ws(y) for x in self.metrics.values() for y in x)
        self.all_tasks = set(normalize_cell_ws(normalize_dataset(y)) for x in self.tasks.values() for y in x)

        self.all_datasets_trie = EvidenceFinder.make_trie(self.all_datasets)
        self.all_metrics_trie = EvidenceFinder.make_trie(self.all_metrics)
        self.all_tasks_trie = EvidenceFinder.make_trie(self.all_tasks)


@njit
def axis_logprobs(evidences_for, reverse_probs, found_evidences, noise, pb, max_repetitions):
    logprob = 0.0
    empty = typed.Dict.empty(types.unicode_type, types.float64)
    short_probs = reverse_probs.get(evidences_for, empty)
    for evidence, count in found_evidences.items():
        logprob += min(count, max_repetitions) * np.log(noise * pb + (1 - noise) * short_probs.get(evidence, 0.0))
    return logprob


# compute log-probabilities in a given context and add them to logprobs
@njit
def compute_logprobs(taxonomy, tasks, datasets, metrics,
                     reverse_merged_p, reverse_metrics_p, reverse_task_p,
                     dss, mss, tss, noise, ms_noise, ts_noise, ds_pb, ms_pb, ts_pb,
                     max_repetitions):
    task_cache = typed.Dict.empty(types.unicode_type, types.float64)
    dataset_cache = typed.Dict.empty(types.unicode_type, types.float64)
    metric_cache = typed.Dict.empty(types.unicode_type, types.float64)
    logprobs = np.zeros(len(taxonomy))
    axes_logprobs = (
        np.zeros(len(tasks)),
        np.zeros(len(datasets)),
        np.zeros(len(metrics))
    )
    for i, (task, dataset, metric) in enumerate(taxonomy):
        if dataset not in dataset_cache:
            dataset_cache[dataset] = axis_logprobs(dataset, reverse_merged_p, dss, noise, ds_pb, 1)
        if metric not in metric_cache:
            metric_cache[metric] = axis_logprobs(metric, reverse_metrics_p, mss, ms_noise, ms_pb, 1)
        if task not in task_cache:
            task_cache[task] = axis_logprobs(task, reverse_task_p, tss, ts_noise, ts_pb, max_repetitions)

        logprobs[i] += dataset_cache[dataset] + metric_cache[metric] + task_cache[task]
    for i, task in enumerate(tasks):
        axes_logprobs[0][i] += task_cache[task]

    for i, dataset in enumerate(datasets):
        axes_logprobs[1][i] += dataset_cache[dataset]

    for i, metric in enumerate(metrics):
        axes_logprobs[2][i] += metric_cache[metric]
    return logprobs, axes_logprobs


def _to_typed_list(iterable):
    l = typed.List()
    for i in iterable:
        l.append(i)
    return l


class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def __getitem__(self, key):
        self.cache.move_to_end(key)
        return self.cache[key]

    def __setitem__(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def __contains__(self, item):
        return item in self.cache

    def __repr__(self):
        return f"LRUCache(capacity={self.capacity}, {repr(dict(self.cache))})"


class ContextSearch:
    def __init__(self, taxonomy, evidence_finder,
                 context_noise=(0.99, 1.0, 1.0, 0.25, 0.01),
                 metric_noise=(0.99, 1.0, 1.0, 0.25, 0.01),
                 task_noise=(0.1, 1.0, 1.0, 0.1, 0.1),
                 ds_pb=0.001, ms_pb=0.01, ts_pb=0.01,
                 include_independent=True, debug_gold_df=None):
        merged_p = \
        get_probs({k: Counter([normalize_cell(normalize_dataset(x)) for x in v]) for k, v in evidence_finder.datasets.items()})[1]
        metrics_p = \
        get_probs({k: Counter([normalize_cell(normalize_dataset(x)) for x in v]) for k, v in evidence_finder.metrics.items()})[1]
        tasks_p = \
        get_probs({k: Counter([normalize_cell(normalize_dataset(x)) for x in v]) for k, v in evidence_finder.tasks.items()})[1]

        self.queries = LRUCache(10_000)
        self.logprobs_cache = LRUCache(10_000)
        self.taxonomy = taxonomy
        self.evidence_finder = evidence_finder

        self._taxonomy = _to_typed_list(self.taxonomy.taxonomy)
        self._taxonomy_tasks = _to_typed_list(self.taxonomy.tasks)
        self._taxonomy_datasets = _to_typed_list(self.taxonomy.datasets)
        self._taxonomy_metrics = _to_typed_list(self.taxonomy.metrics)

        self.extract_acronyms = AcronymExtractor()
        self.context_noise = context_noise
        self.metrics_noise = metric_noise if metric_noise else context_noise
        self.task_noise = task_noise if task_noise else context_noise
        self.ds_pb = ds_pb
        self.ms_pb = ms_pb
        self.ts_pb = ts_pb
        self.reverse_merged_p = self._numba_update_nested_dict(reverse_probs(merged_p))
        self.reverse_metrics_p = self._numba_update_nested_dict(reverse_probs(metrics_p))
        self.reverse_tasks_p = self._numba_update_nested_dict(reverse_probs(tasks_p))
        self.debug_gold_df = debug_gold_df
        self.max_repetitions = 3
        self.include_independent = include_independent

    def _numba_update_nested_dict(self, nested):
        d = typed.Dict()
        for key, dct in nested.items():
            d2 = typed.Dict()
            d2.update(dct)
            d[key] = d2
        return d

    def _numba_extend_list(self, lst):
        l = typed.List.empty_list((types.unicode_type, types.int32))
        for x in lst:
            l.append(x)
        return l

    def _numba_extend_dict(self, dct):
        d = typed.Dict.empty(types.unicode_type, types.int64)
        d.update(dct)
        return d

    def _hash_counter(self, d):
        items = list(d.items())
        items = sorted(items)
        return ";".join([x[0]+":"+str(x[1]) for x in items])

    def compute_context_logprobs(self, context, noise, ms_noise, ts_noise, logprobs, axes_logprobs):
        if isinstance(context, str) or context is None:
            context = context or ""
            #abbrvs = self.extract_acronyms(context)
            context = normalize_cell_ws(normalize_dataset_ws(context))
            #dss = set(self.evidence_finder.find_datasets(context)) | set(abbrvs.keys())
            dss = self.evidence_finder.find_datasets(context)
            mss = self.evidence_finder.find_metrics(context)
            tss = self.evidence_finder.find_tasks(context)

            dss -= mss
            dss -= tss
        else:
            tss, dss, mss = context

        dss = {normalize_cell(ds): count for ds, count in dss.items()}
        mss = {normalize_cell(ms): count for ms, count in mss.items()}
        tss = {normalize_cell(ts): count for ts, count in tss.items()}
        ###print("dss", dss)
        ###print("mss", mss)
        dss = self._numba_extend_dict(dss)
        mss = self._numba_extend_dict(mss)
        tss = self._numba_extend_dict(tss)

        key = (self._hash_counter(tss), self._hash_counter(dss), self._hash_counter(mss), noise, ms_noise, ts_noise)
        if key not in self.logprobs_cache:
            lp, alp = compute_logprobs(self._taxonomy, self._taxonomy_tasks, self._taxonomy_datasets, self._taxonomy_metrics,
                             self.reverse_merged_p, self.reverse_metrics_p, self.reverse_tasks_p,
                             dss, mss, tss, noise, ms_noise, ts_noise, self.ds_pb, self.ms_pb, self.ts_pb,
                             self.max_repetitions)
            self.logprobs_cache[key] = (lp, alp)
        else:
            lp, alp = self.logprobs_cache[key]
        logprobs += lp
        axes_logprobs[0] += alp[0]
        axes_logprobs[1] += alp[1]
        axes_logprobs[2] += alp[2]

    def match(self, contexts):
        assert len(contexts) == len(self.context_noise)
        n = len(self._taxonomy)
        context_logprobs = np.zeros(n)
        axes_context_logprobs = _to_typed_list([
            np.zeros(len(self._taxonomy_tasks)),
            np.zeros(len(self._taxonomy_datasets)),
            np.zeros(len(self._taxonomy_metrics)),
        ])

        for context, noise, ms_noise, ts_noise in zip(contexts, self.context_noise, self.metrics_noise, self.task_noise):
            self.compute_context_logprobs(context, noise, ms_noise, ts_noise, context_logprobs, axes_context_logprobs)
        keys = self.taxonomy.taxonomy
        logprobs = context_logprobs
        #keys, logprobs = zip(*context_logprobs.items())
        probs = softmax(np.array(logprobs))
        axes_probs = [softmax(np.array(a)) for a in axes_context_logprobs]
        return (
            zip(keys, probs),
            zip(self._taxonomy_tasks, axes_probs[0]),
            zip(self._taxonomy_datasets, axes_probs[1]),
            zip(self._taxonomy_metrics, axes_probs[2])
        )

    def __call__(self, query, paper_context, abstract_context, table_context, caption, topk=1, debug_info=None):
        cellstr = debug_info.cell.cell_ext_id
        pipeline_logger("linking::taxonomy_linking::call", ext_id=cellstr, query=query,
                        paper_context=paper_context, abstract_context=abstract_context, table_context=table_context,
                        caption=caption)

        paper_hash = ";".join(",".join(sorted(s.elements())) for s in paper_context)
        abstract_hash = ";".join(",".join(sorted(s.elements())) for s in abstract_context)
        mentions_hash = ";".join(",".join(sorted(s.elements())) for s in table_context)
        key = (paper_hash, abstract_hash, mentions_hash, caption, query, topk)
        ###print(f"[DEBUG] {cellstr}")
        ###print("[DEBUG]", debug_info)
        ###print("query:", query, caption)
        if key in self.queries:
            # print(self.queries[key])
            # for context in key:
            #     abbrvs = self.extract_acronyms(context)
            #     context = normalize_cell_ws(normalize_dataset(context))
            #     dss = set(find_datasets(context)) | set(abbrvs.keys())
            #     mss = set(find_metrics(context))
            #     dss -= mss
                ###print("dss", dss)
                ###print("mss", mss)

            ###print("Taking result from cache")
            p = self.queries[key]
        else:
            dists = self.match((paper_context, abstract_context, table_context, caption, query))

            all_top_results = [sorted(list(dist), key=lambda x: x[1], reverse=True)[:max(topk, 5)] for dist in dists]
            top_results, top_results_t, top_results_d, top_results_m = all_top_results

            entries = []
            for it, prob in top_results:
                task, dataset, metric = it
                entry = dict(task=task, dataset=dataset, metric=metric)
                entry.update({"evidence": "", "confidence": prob})
                entries.append(entry)

            if self.include_independent:
                best_independent = dict(
                    task=top_results_t[0][0],
                    dataset=top_results_d[0][0],
                    metric=top_results_m[0][0])
                best_independent.update({
                    "evidence": "",
                    "confidence": 0.79
                })
                entries.append(best_independent)

            # entries = []
            # for i in range(5):
            #     best_independent = dict(
            #         task=top_results_t[i][0],
            #         dataset=top_results_d[i][0],
            #         metric=top_results_m[i][0])
            #     best_independent.update({
            #         "evidence": "",
            #         "confidence": np.power(top_results_t[i][1] * top_results_d[i][1] * top_results_m[i][1], 1.0/3.0)
            #     })
            #     entries.append(best_independent)
                #entries = [best_independent] + entries

            # best, best_p = sorted(dist, key=lambda x: x[1], reverse=True)[0]
            # entry = et[best]
            # p = pd.DataFrame({k:[v] for k, v in entry.items()})
            # p["evidence"] = ""
            # p["confidence"] = best_p
            p = pd.DataFrame(entries).sort_values("confidence", ascending=False)

            self.queries[key] = p

        ###print(p)

        # error analysis only
        if self.debug_gold_df is not None:
            if cellstr in self.debug_gold_df.index:
                gold_record = self.debug_gold_df.loc[cellstr]
                if p.iloc[0].dataset == gold_record.dataset:
                    print("[EA] Matching gold sota record (dataset)")
                else:
                    print(
                        f"[EA] Proposal dataset ({p.iloc[0].dataset}) and gold dataset ({gold_record.dataset}) mismatch")
            else:
                print("[EA] No gold sota record found for the cell")
        # end of error analysis only
        pipeline_logger("linking::taxonomy_linking::topk", ext_id=cellstr, topk=p.head(5))

        q = p.head(topk).copy()
        q["true_metric"] = q.apply(lambda row: self.taxonomy.normalize_metric(row.task, row.dataset, row.metric), axis=1)
        return q


# todo: compare regex approach (old) with find_datasets(.) (current)
# todo: rename it
class DatasetExtractor:
    def __init__(self, evidence_finder):
        self.evidence_finder = evidence_finder
        self.dataset_prefix_re = re.compile(r"[A-Z]|[a-z]+[A-Z]+|[0-9]")
        self.dataset_name_re = re.compile(r"\b(the)\b\s*(?P<name>((?!(the)\b)\w+\W+){1,10}?)(test|val(\.|idation)?|dev(\.|elopment)?|train(\.|ing)?\s+)?\bdata\s*set\b", re.IGNORECASE)

    def find_references(self, text, references):
        refs = r"\bxxref-(" + "|".join([re.escape(ref) for ref in references]) + r")\b"
        return set(re.findall(refs, text))

    def get_table_contexts(self, paper, tables):
        ref_tables = [table for table in tables if table.figure_id and table.figure_id.replace(".", "")]
        refs = [table.figure_id.replace(".", "") for table in ref_tables]
        if not refs:
            return [[Counter(), Counter(), Counter()] for table in tables]
        ref_contexts = {ref: [Counter(), Counter(), Counter()] for ref in refs}
        if hasattr(paper.text, "fragments"):
            for fragment in paper.text.fragments:
                found_refs = self.find_references(fragment.text, refs)
                if found_refs:
                    ts, ds, ms = self(fragment.header + "\n" + fragment.text)
                    for ref in found_refs:
                        ref_contexts[ref][0] += ts
                        ref_contexts[ref][1] += ds
                        ref_contexts[ref][2] += ms
        table_contexts = [
            ref_contexts.get(
                table.figure_id.replace(".", ""),
                [Counter(), Counter(), Counter()]
            ) if table.figure_id else [Counter(), Counter(), Counter()]
            for table in tables
        ]
        return table_contexts

    def from_paper(self, paper):
        abstract = paper.text.abstract
        text = ""
        if hasattr(paper.text, "fragments"):
            text += " ".join(f.text for f in paper.text.fragments)
        return self(text), self(abstract)

    def __call__(self, text):
        text = normalize_cell_ws(normalize_dataset_ws(text))
        ds = self.evidence_finder.find_datasets(text)
        ts = self.evidence_finder.find_tasks(text)
        ms = self.evidence_finder.find_metrics(text)
        ds -= ts
        ds -= ms
        return ts, ds, ms
