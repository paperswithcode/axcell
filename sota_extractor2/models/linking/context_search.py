# metrics[taxonomy name] is a list of normalized evidences for taxonomy name
from collections import Counter

from sota_extractor2.models.linking.acronym_extractor import AcronymExtractor
from sota_extractor2.models.linking.probs import get_probs, reverse_probs
from sota_extractor2.models.linking.utils import normalize_dataset, normalize_cell, normalize_cell_ws
from scipy.special import softmax
import re
import pandas as pd
import numpy as np
import ahocorasick
from numba import njit, typed, types

from sota_extractor2.pipeline_logger import pipeline_logger

metrics = {
    'BLEU': ['bleu'],
    'BLEU score': ['bleu'],
    'Character Error Rate': ['cer', 'cers'],
    'Error': ['error'],
    'Exact Match Ratio': ['exact match'],
    'F1': ['f1', 'f1 score'],
    'F1 score': ['f1', 'f1 score'],
    'MAP': ['map'],
    'Percentage error': ['wer', 'per', 'wers', 'pers', 'word error rate', 'word error rates', 'phoneme error rates',
                         'phoneme error rate', 'error', 'error rate', 'error rates'],
    'Word Error Rate': ['wer', 'wers', 'word error rate', 'word error rates', 'error', 'error rate', 'error rates'],
    'Word Error Rate (WER)': ['wer', 'wers', 'word error rate', 'word error rates', 'error', 'error rate', 'error rates'],
    'ROUGE-1': ['r1'],
    'ROUGE-2': ['r2'],
    'ROUGE-F': ['rf'],
    'Precision': ['precision'],
    'Recall': ['recall'],
    # RAIN REMOVAL
    'PSNR': ['psnr', 'psnr (db)', 'mean psnr'],
    'SSIM': ['ssim'],
    'UQI': ['uqi'],
    'VIF': ['vif'],
    'SSEQ': ['sseq'],
    'NIQE': ['niqe'],
    'BLINDS-II': ['blinds-ii'],
    'FSIM': ['fsim'],
    # SEMANTIC SEGMENTATION
    'Mean iOU': ['miou', 'mean iou', 'mean iu'],
    'Pixel Accuracy': ['pixel accuracy', 'pixel acc', 'pixel acc.'],
    'Class iOU': ['class iou', 'iou cla.'],
    'Category iOU': ['cat iou', 'iou cat.'],
    'Class iiOU': ['class iiou', 'iiou cla.'],
    'Category iiOU': ['cat iiou', 'iiou cat.'],
}

# datasets[taxonomy name] is a list of normalized evidences for taxonomy name
datasets = {
    'Hub5\'00 Average': ['avg', 'full', 'hub5', 'sum', 'evaluation'],
    'Hub5\'00 Switchboard': ['swbd', 'swb', 'hub5 swb', 'hub5 swbd', 'switchboard'],
    'Hub5\'00 CallHome': ['ch', 'hub5 ch', 'call home', 'chm'],
    'TIMIT': ['timit'],
    'WSJ eval92': ['wsj eval 92', 'eval 92', 'wsj'],
    'WSJ eval93': ['wsj eval 93', 'eval 93', 'wsj'],
    'LibriSpeech test-clean': ['libri speech test clean', 'libri speech', 'test', 'tst', 'clean', 'test clean'],
    'LibriSpeech test-other': ['libri speech test other', 'libri speech', 'test', 'tst', 'other', 'test other',
                               'noisy'],
    'Babel Cebuano': ['babel cebuano', 'babel', 'cebuano', 'ceb'],
    'Babel Kazakh': ['babel kazakh', 'babel', 'kazakh', 'kaz'],
    'Babel Kurmanji': ['babel kurmanji', 'babel', 'kurmanji', 'kur'],
    'Babel Lithuanian': ['babel lithuanian', 'babel', 'lithuanian', 'lit'],
    'Babel Telugu': ['babel telugu', 'babel', 'telugu', 'tel'],
    'Babel Tok Pisin': ['babel tok pisin', 'babel', 'tok pisin', 'tok'],

    'Ask Ubuntu': ['ask ubuntu', 'ask u', 'ubuntu'],
    'Chatbot': ['chatbot'],
    'Web Apps': ['web apps'],
    'CHiME clean': ['chime clean', 'chime', 'clean'],
    'CHiME real': ['chime real', 'chime', 'real'],
    'CHiME simu': ['chime simu', 'chime', 'simu', 'sim', 'simulated'],
    'CHiME-4 real 6ch': ['chime 4 real 6 ch', 'chime 4', 'real', '6 channel'],
    'AG News': ['ag news', 'ag'],
    'GigaWord': ['gigaword', 'giga'],
    'GEOTEXT': ['geotext', 'geo'],
    'IWSLT 2015 English-Vietnamese': ["iwslt 2015 english vietnamese", "iwslt", "2015", "english vietnamese", "en vi",
                                      "iwslt 15 english vietnamese", "iwslt 15 en vi", "english", "en", "vietnamese",
                                      "vi"],
    'IWSLT2011 English TED Talks': ["iwslt 2011 english ted talks", "iwslt", "2011", "english", "en", "eng", "ted",
                                    "ted talks", "english ted talks"],
    'IWSLT2012 English TED Talks': ["iwslt 2012 english ted talks", "iwslt", "2012", "english", "en", "eng", "ted",
                                    "ted talks", "english ted talks"],
    'IWSLT2014 English-German': ["iwslt 2014 english german", "iwslt", "2014", "english german", "en de", "en", "de",
                                 "english", "german"],
    'Rich Transcription 2002': ["rich transcription 2002", "rich transcription 02", "rt 2002", "2002", "rt 02", "rich",
                                "transcription"],
    'Rich Transcription 2003': ["richt ranscription 2003", "rich transcription 03", "rt 2003", "2003", "rt 03", "rich",
                                "transcription"],
    'Rich Transcription 2004': ["rich transcription 2004", "rich transcription 04", "rt 2004", "2004", "rt 04", "rich",
                                "transcription"],
    'DIRHA English WSJ real': ['dirha english wsj real', 'dirha', 'english', 'en', 'eng', 'real', 'wsj'],
    'DIRHA English WSJ simu': ['dirha english wsj simu', 'dirha', 'english', 'en', 'eng', 'simu', 'wsj', 'simulated'],
    'VCTK clean': ["vctk clean", "vctk", "clean"],
    'VCTK noisy': ["vctk noisy", "vctk", "noisy"],
    'VoxForge American-Canadian': ["vox forge american canadian", "vox forge", "vox", "forge", "american canadian",
                                   "american", "canadian", "us ca"],
    'VoxForge Commonwealth': ["vox forge common wealth", "vox forge", "common wealth", "vox", "forge", "common",
                              "wealth"],
    'VoxForge European': ["vox forge european", "vox forge", "european", "vox", "forge", "eu"],
    'VoxForge Indian': ["vox forge indian", "vox forge", "indian", "vox", "forge"],
    # RAIN REMOVAL
    'Raindrop': ['raindrop'],
    'Rain100H': ['rain100h'],
    'Rain100L': ['rain100l'],
    'Rain12': ['rain12'],
    'Rain800': ['rain800'],
    'Rain1400': ['rain1400'],
    'Real Rain': ['real rain'],
    'Rain in Surveillance': ['ris'],
    'Rain in Driving': ['rid'],
    'DID-MDN': ['did-mdn'],
    'SOTS': ['sots'],
    'Test 1': ['test 1'],
    'RainSynLight25': ['rainsynlight25'],
    'RainSynComplex25': ['rainsyncomplex25'],
    'NTURain': ['nturain'],
    'RainSynAll100': ['rainsynall100'],
    'SPA-DATA': ['spa-data'],
    'LasVR': ['lasvar'],
    # SEMANTIC SEGMENTATION
    'PASCAL VOC 2012': ['voc 2012', 'pascal voc 2012'],
    'ADE20K': ['ade20k'],
    'ImageNet': ['imagenet'],
    'Cityscapes': ['cityscapes'],
    'PASCAL-Context': ['pascal-context'],
    'PASCAL-Person-Part': ['pascal-person-part'],
    'ParseNet': ['parsenet'],
    'LIP': ['lip'],
}

datasets = {k:(v+['test']) for k,v in datasets.items()}
datasets.update({
    'LibriSpeech dev-clean': ['libri speech dev clean', 'libri speech', 'dev', 'clean', 'dev clean', 'development'],
    'LibriSpeech dev-other': ['libri speech dev other', 'libri speech', 'dev', 'other', 'dev other', 'development', 'noisy'],
})

tasks = {}

# escaped_ws_re = re.compile(r'\\\s+')
# def name_to_re(name):
#     return re.compile(r'(?:^|\s+)' + escaped_ws_re.sub(r'\\s*', re.escape(name.strip())) + r'(?:$|\s+)', re.I)

#all_datasets = set(k for k,v in merged_p.items() if k != '' and not re.match("^\d+$", k) and v.get('NOMATCH', 0.0) < 0.9)
all_datasets = set(normalize_cell_ws(normalize_dataset(y)) for x in datasets.values() for y in x)
all_metrics = set(normalize_cell_ws(y) for x in metrics.values() for y in x)
all_tasks = set(normalize_cell_ws(normalize_dataset(y)) for x in tasks.values() for y in x)

#all_metrics = set(metrics_p.keys())

# all_datasets_re = {x:name_to_re(x) for x in all_datasets}
# all_metrics_re = {x:name_to_re(x) for x in all_metrics}
#all_datasets = set(x for v in merged_p.values() for x in v)

# def find_names(text, names_re):
#     return set(name for name, name_re in names_re.items() if name_re.search(text))


def make_trie(names):
    trie = ahocorasick.Automaton()
    for name in names:
        norm = name.replace(" ", "")
        trie.add_word(norm, (len(norm), name))
    trie.make_automaton()
    return trie


single_letter_re = re.compile(r"\b\w\b")
init_letter_re = re.compile(r"\b\w")
end_letter_re = re.compile(r"\w\b")
letter_re = re.compile(r"\w")


def find_names(text, names_trie):
    text = text.lower()
    profile = letter_re.sub("i", text)
    profile = init_letter_re.sub("b", profile)
    profile = end_letter_re.sub("e", profile)
    profile = single_letter_re.sub("x", profile)
    text = text.replace(" ", "")
    profile = profile.replace(" ", "")
    s = set()
    for (end, (l, word)) in names_trie.iter(text):
        if profile[end] in ['e', 'x'] and profile[end - l + 1] in ['b', 'x']:
            s.add(word)
    return s


all_datasets_trie = make_trie(all_datasets)
all_metrics_trie = make_trie(all_metrics)
all_tasks_trie = make_trie(all_tasks)


def find_datasets(text):
    return find_names(text, all_datasets_trie)

def find_metrics(text):
    return find_names(text, all_metrics_trie)

def find_tasks(text):
    return find_names(text, all_tasks_trie)

def dummy_item(reason):
    return pd.DataFrame(dict(dataset=[reason], task=[reason], metric=[reason], evidence=[""], confidence=[0.0]))



@njit
def compute_logprobs(taxonomy, reverse_merged_p, reverse_metrics_p, reverse_task_p,
                     dss, mss, tss, noise, ms_noise, ts_noise, ds_pb, ms_pb, ts_pb, logprobs):
    empty = typed.Dict.empty(types.unicode_type, types.float64)
    for i, (task, dataset, metric) in enumerate(taxonomy):
        logprob = 0.0
        short_probs = reverse_merged_p.get(dataset, empty)
        met_probs = reverse_metrics_p.get(metric, empty)
        task_probs = reverse_task_p.get(task, empty)
        for ds in dss:
            #                 for abbrv, long_form in abbrvs.items():
            #                     if ds == abbrv:
            #                         ds = long_form
            #                         break
            # if merged_p[ds].get('NOMATCH', 0.0) < 0.5:
            logprob += np.log(noise * ds_pb + (1 - noise) * short_probs.get(ds, 0.0))
        for ms in mss:
            logprob += np.log(ms_noise * ms_pb + (1 - ms_noise) * met_probs.get(ms, 0.0))
        for ts in tss:
            logprob += np.log(ts_noise * ts_pb + (1 - ts_noise) * task_probs.get(ts, 0.0))
        logprobs[i] += logprob
        #logprobs[(dataset, metric)] = logprob


class ContextSearch:
    def __init__(self, taxonomy, context_noise=(0.5, 0.2, 0.1), metrics_noise=None, task_noise=None,
                 ds_pb=0.001, ms_pb=0.01, ts_pb=0.01, debug_gold_df=None):
        merged_p = \
        get_probs({k: Counter([normalize_cell(normalize_dataset(x)) for x in v]) for k, v in datasets.items()})[1]
        metrics_p = \
        get_probs({k: Counter([normalize_cell(normalize_dataset(x)) for x in v]) for k, v in metrics.items()})[1]
        tasks_p = \
        get_probs({k: Counter([normalize_cell(normalize_dataset(x)) for x in v]) for k, v in tasks.items()})[1]

        self.queries = {}
        self.taxonomy = taxonomy
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
        dss = set(find_datasets(context)) | set(abbrvs.keys())
        mss = set(find_metrics(context))
        tss = set(find_tasks(context))
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

    def __call__(self, query, datasets, caption, debug_info=None):
        cellstr = debug_info.cell.cell_ext_id
        pipeline_logger("linking::taxonomy_linking::call", ext_id=cellstr, query=query, datasets=datasets, caption=caption)
        datasets = " ".join(datasets)
        key = (datasets, caption, query)
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
            dist = self.match(key)
            topk = sorted(dist, key=lambda x: x[1], reverse=True)[0:5]

            entries = []
            for it, prob in topk:
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
        pipeline_logger("linking::taxonomy_linking::topk", ext_id=cellstr, topk=p)
        return p.head(1)


# todo: compare regex approach (old) with find_datasets(.) (current)
class DatasetExtractor:
    def __init__(self):
        self.dataset_prefix_re = re.compile(r"[A-Z]|[a-z]+[A-Z]+|[0-9]")
        self.dataset_name_re = re.compile(r"\b(the)\b\s*(?P<name>((?!(the)\b)\w+\W+){1,10}?)(test|val(\.|idation)?|dev(\.|elopment)?|train(\.|ing)?\s+)?\bdata\s*set\b", re.IGNORECASE)

    def from_paper(self, paper):
        text = paper.text.abstract
        if hasattr(paper.text, "fragments"):
            text += " ".join(f.text for f in paper.text.fragments)
        return self(text)

    def __call__(self, text):
        return find_datasets(normalize_cell_ws(normalize_dataset(text)))
