# metrics[taxonomy name] is a list of normalized evidences for taxonomy name
from collections import Counter

from sota_extractor2.models.linking.acronym_extractor import AcronymExtractor
from sota_extractor2.models.linking.probs import get_probs, reverse_probs
from sota_extractor2.models.linking.utils import normalize_dataset, normalize_dataset_ws, normalize_cell, normalize_cell_ws
from scipy.special import softmax
import re
import pandas as pd
import numpy as np
import ahocorasick
from numba import njit, typed, types

from sota_extractor2.pipeline_logger import pipeline_logger

from sota_extractor2.models.linking import manual_dicts

def dummy_item(reason):
    return pd.DataFrame(dict(dataset=[reason], task=[reason], metric=[reason], evidence=[""], confidence=[0.0]))


class EvidenceFinder:
    single_letter_re = re.compile(r"\b\w\b")
    init_letter_re = re.compile(r"\b\w")
    end_letter_re = re.compile(r"\w\b")
    letter_re = re.compile(r"\w")

    def __init__(self, taxonomy):
        self._init_structs(taxonomy)

    @staticmethod
    def evidences_from_name(key):
        x = normalize_dataset_ws(key)
        y = x.split()
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
    def find_names(text, names_trie):
        text = text.lower()
        profile = EvidenceFinder.letter_re.sub("i", text)
        profile = EvidenceFinder.init_letter_re.sub("b", profile)
        profile = EvidenceFinder.end_letter_re.sub("e", profile)
        profile = EvidenceFinder.single_letter_re.sub("x", profile)
        text = text.replace(" ", "")
        profile = profile.replace(" ", "")
        s = set()
        for (end, (l, word)) in names_trie.iter(text):
            if profile[end] in ['e', 'x'] and profile[end - l + 1] in ['b', 'x']:
                s.add(word)
        return s

    def find_datasets(self, text):
        return EvidenceFinder.find_names(text, self.all_datasets_trie)

    def find_metrics(self, text):
        return EvidenceFinder.find_names(text, self.all_metrics_trie)

    def find_tasks(self, text):
        return EvidenceFinder.find_names(text, self.all_tasks_trie)

    def _init_structs(self, taxonomy):
        self.tasks, self.datasets, self.metrics = EvidenceFinder.get_basic_dicts(taxonomy)
        EvidenceFinder.merge_evidences(self.tasks, manual_dicts.tasks)
        EvidenceFinder.merge_evidences(self.datasets, manual_dicts.datasets)
        EvidenceFinder.merge_evidences(self.metrics, manual_dicts.metrics)
        self.datasets = {k: (v + ['test'] if 'val' not in k else v + ['validation', 'dev', 'development']) for k, v in
                    self.datasets.items()}
        self.datasets.update({
            'LibriSpeech dev-clean': ['libri speech dev clean', 'libri speech', 'dev', 'clean', 'dev clean', 'development'],
            'LibriSpeech dev-other': ['libri speech dev other', 'libri speech', 'dev', 'other', 'dev other', 'development', 'noisy'],
        })

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
def axis_logprobs(evidences_for, reverse_probs, found_evidences, noise, pb):
    logprob = 0.0
    empty = typed.Dict.empty(types.unicode_type, types.float64)
    short_probs = reverse_probs.get(evidences_for, empty)
    for evidence in found_evidences:
        logprob += np.log(noise * pb + (1 - noise) * short_probs.get(evidence, 0.0))
    return logprob


# compute log-probabilities in a given context and add them to logprobs
@njit
def compute_logprobs(taxonomy, reverse_merged_p, reverse_metrics_p, reverse_task_p,
                     dss, mss, tss, noise, ms_noise, ts_noise, ds_pb, ms_pb, ts_pb, logprobs):
    task_cache = typed.Dict.empty(types.unicode_type, types.float64)
    dataset_cache = typed.Dict.empty(types.unicode_type, types.float64)
    metric_cache = typed.Dict.empty(types.unicode_type, types.float64)
    for i, (task, dataset, metric) in enumerate(taxonomy):
        if dataset not in dataset_cache:
            dataset_cache[dataset] = axis_logprobs(dataset, reverse_merged_p, dss, noise, ds_pb)
        if metric not in metric_cache:
            metric_cache[metric] = axis_logprobs(metric, reverse_metrics_p, mss, ms_noise, ms_pb)
        if task not in task_cache:
            task_cache[task] = axis_logprobs(task, reverse_task_p, tss, ts_noise, ts_pb)

        logprobs[i] += dataset_cache[dataset] + metric_cache[metric] + task_cache[task]


class ContextSearch:
    def __init__(self, taxonomy, evidence_finder, context_noise=(0.5, 0.2, 0.1), metrics_noise=None, task_noise=None,
                 ds_pb=0.001, ms_pb=0.01, ts_pb=0.01, debug_gold_df=None):
        merged_p = \
        get_probs({k: Counter([normalize_cell(normalize_dataset(x)) for x in v]) for k, v in evidence_finder.datasets.items()})[1]
        metrics_p = \
        get_probs({k: Counter([normalize_cell(normalize_dataset(x)) for x in v]) for k, v in evidence_finder.metrics.items()})[1]
        tasks_p = \
        get_probs({k: Counter([normalize_cell(normalize_dataset(x)) for x in v]) for k, v in evidence_finder.tasks.items()})[1]

        self.queries = {}
        self.taxonomy = taxonomy
        self.evidence_finder = evidence_finder
        self._taxonomy = typed.List()
        for t in self.taxonomy.taxonomy:
            self._taxonomy.append(t)
        self.extract_acronyms = AcronymExtractor()
        self.context_noise = context_noise
        self.metrics_noise = metrics_noise if metrics_noise else context_noise
        self.task_noise = task_noise if task_noise else context_noise
        self.ds_pb = ds_pb
        self.ms_pb = ms_pb
        self.ts_pb = ts_pb
        self.reverse_merged_p = self._numba_update_nested_dict(reverse_probs(merged_p))
        self.reverse_metrics_p = self._numba_update_nested_dict(reverse_probs(metrics_p))
        self.reverse_tasks_p = self._numba_update_nested_dict(reverse_probs(tasks_p))
        self.debug_gold_df = debug_gold_df

    def _numba_update_nested_dict(self, nested):
        d = typed.Dict()
        for key, dct in nested.items():
            d2 = typed.Dict()
            d2.update(dct)
            d[key] = d2
        return d

    def _numba_extend_list(self, lst):
        l = typed.List.empty_list(types.unicode_type)
        for x in lst:
            l.append(x)
        return l

    def compute_context_logprobs(self, context, noise, ms_noise, ts_noise, logprobs):
        context = context or ""
        abbrvs = self.extract_acronyms(context)
        context = normalize_cell_ws(normalize_dataset(context))
        dss = set(self.evidence_finder.find_datasets(context)) | set(abbrvs.keys())
        mss = set(self.evidence_finder.find_metrics(context))
        tss = set(self.evidence_finder.find_tasks(context))
        dss -= mss
        dss -= tss
        dss = [normalize_cell(ds) for ds in dss]
        mss = [normalize_cell(ms) for ms in mss]
        tss = [normalize_cell(ts) for ts in tss]
        ###print("dss", dss)
        ###print("mss", mss)
        dss = self._numba_extend_list(dss)
        mss = self._numba_extend_list(mss)
        tss = self._numba_extend_list(tss)
        compute_logprobs(self._taxonomy, self.reverse_merged_p, self.reverse_metrics_p, self.reverse_tasks_p,
                         dss, mss, tss, noise, ms_noise, ts_noise, self.ds_pb, self.ms_pb, self.ts_pb, logprobs)

    def match(self, contexts):
        assert len(contexts) == len(self.context_noise)
        n = len(self._taxonomy)
        context_logprobs = np.zeros(n)

        for context, noise, ms_noise, ts_noise in zip(contexts, self.context_noise, self.metrics_noise, self.task_noise):
            self.compute_context_logprobs(context, noise, ms_noise, ts_noise, context_logprobs)
        keys = self.taxonomy.taxonomy
        logprobs = context_logprobs
        #keys, logprobs = zip(*context_logprobs.items())
        probs = softmax(np.array(logprobs))
        return zip(keys, probs)

    def __call__(self, query, datasets, caption, topk=1, debug_info=None):
        cellstr = debug_info.cell.cell_ext_id
        pipeline_logger("linking::taxonomy_linking::call", ext_id=cellstr, query=query, datasets=datasets, caption=caption)
        datasets = " ".join(datasets)
        key = (datasets, caption, query, topk)
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
            dist = self.match((datasets, caption, query))
            top_results = sorted(dist, key=lambda x: x[1], reverse=True)[:max(topk, 5)]

            entries = []
            for it, prob in top_results:
                task, dataset, metric = it
                entry = dict(task=task, dataset=dataset, metric=metric)
                entry.update({"evidence": "", "confidence": prob})
                entries.append(entry)

            # best, best_p = sorted(dist, key=lambda x: x[1], reverse=True)[0]
            # entry = et[best]
            # p = pd.DataFrame({k:[v] for k, v in entry.items()})
            # p["evidence"] = ""
            # p["confidence"] = best_p
            p = pd.DataFrame(entries)

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
class DatasetExtractor:
    def __init__(self, evidence_finder):
        self.evidence_finder = evidence_finder
        self.dataset_prefix_re = re.compile(r"[A-Z]|[a-z]+[A-Z]+|[0-9]")
        self.dataset_name_re = re.compile(r"\b(the)\b\s*(?P<name>((?!(the)\b)\w+\W+){1,10}?)(test|val(\.|idation)?|dev(\.|elopment)?|train(\.|ing)?\s+)?\bdata\s*set\b", re.IGNORECASE)

    def from_paper(self, paper):
        text = paper.text.abstract
        if hasattr(paper.text, "fragments"):
            text += " ".join(f.text for f in paper.text.fragments)
        return self(text)

    def __call__(self, text):
        text = normalize_cell_ws(normalize_dataset(text))
        return self.evidence_finder.find_datasets(text) | self.evidence_finder.find_tasks(text)
