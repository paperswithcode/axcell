# metrics[taxonomy name] is a list of normalized evidences for taxonomy name
from collections import Counter

from sota_extractor2.models.linking.acronym_extractor import AcronymExtractor
from sota_extractor2.models.linking.probs import get_probs, reverse_probs
from sota_extractor2.models.linking.utils import normalize_dataset, normalize_cell, normalize_cell_ws
from scipy.special import softmax
import re
import pandas as pd
import numpy as np

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

escaped_ws_re = re.compile(r'\\\s+')
def name_to_re(name):
    return re.compile(r'(?:^|\s+)' + escaped_ws_re.sub(r'\\s*', re.escape(name.strip())) + r'(?:$|\s+)', re.I)

#all_datasets = set(k for k,v in merged_p.items() if k != '' and not re.match("^\d+$", k) and v.get('NOMATCH', 0.0) < 0.9)
all_datasets = set(y for x in datasets.values() for y in x)
all_metrics = set(y for x in metrics.values() for y in x)
#all_metrics = set(metrics_p.keys())

all_datasets_re = {x:name_to_re(x) for x in all_datasets}
all_metrics_re = {x:name_to_re(x) for x in all_metrics}
#all_datasets = set(x for v in merged_p.values() for x in v)

def find_names(text, names_re):
    return set(name for name, name_re in names_re.items() if name_re.search(text))

def find_datasets(text):
    return find_names(text, all_datasets_re)

def find_metrics(text):
    return find_names(text, all_metrics_re)

def dummy_item(reason):
    return pd.DataFrame(dict(dataset=[reason], task=[reason], metric=[reason], evidence=[""], confidence=[0.0]))




class ContextSearch:
    def __init__(self, taxonomy, context_noise=(0.5, 0.2, 0.1), debug_gold_df=None):
        merged_p = \
        get_probs({k: Counter([normalize_cell(normalize_dataset(x)) for x in v]) for k, v in datasets.items()})[1]
        metrics_p = \
        get_probs({k: Counter([normalize_cell(normalize_dataset(x)) for x in v]) for k, v in metrics.items()})[1]


        self.queries = {}
        self.taxonomy = taxonomy
        self.extract_acronyms = AcronymExtractor()
        self.context_noise = context_noise
        self.reverse_merged_p = reverse_probs(merged_p)
        self.reverse_metrics_p = reverse_probs(metrics_p)
        self.debug_gold_df = debug_gold_df

    def compute_logprobs(self, dss, mss, abbrvs, noise, logprobs):
        for dataset, metric in self.taxonomy.taxonomy:
            logprob = logprobs.get((dataset, metric), 1.0)
            short_probs = self.reverse_merged_p.get(dataset, {})
            met_probs = self.reverse_metrics_p.get(metric, {})
            for ds in dss:
                ds = normalize_cell(ds)
                #                 for abbrv, long_form in abbrvs.items():
                #                     if ds == abbrv:
                #                         ds = long_form
                #                         break
                # if merged_p[ds].get('NOMATCH', 0.0) < 0.5:
                logprob += np.log(noise * 0.001 + (1 - noise) * short_probs.get(ds, 0.0))
            for ms in mss:
                ms = normalize_cell(ms)
                logprob += np.log(noise * 0.01 + (1 - noise) * met_probs.get(ms, 0.0))
            logprobs[(dataset, metric)] = logprob

    def compute_context_logprobs(self, context, noise, logprobs):
        abbrvs = self.extract_acronyms(context)
        context = normalize_cell_ws(normalize_dataset(context))
        dss = set(find_datasets(context)) | set(abbrvs.keys())
        mss = set(find_metrics(context))
        dss -= mss
        ###print("dss", dss)
        ###print("mss", mss)
        self.compute_logprobs(dss, mss, abbrvs, noise, logprobs)

    def match(self, contexts):
        assert len(contexts) == len(self.context_noise)
        context_logprobs = {}

        for context, noise in zip(contexts, self.context_noise):
            self.compute_context_logprobs(context, noise, context_logprobs)
        keys, logprobs = zip(*context_logprobs.items())
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
            for context in key:
                abbrvs = self.extract_acronyms(context)
                context = normalize_cell_ws(normalize_dataset(context))
                dss = set(find_datasets(context)) | set(abbrvs.keys())
                mss = set(find_metrics(context))
                dss -= mss
                ###print("dss", dss)
                ###print("mss", mss)

            ###print("Taking result from cache")
            p = self.queries[key]
        else:
            dist = self.match(key)
            topk = sorted(dist, key=lambda x: x[1], reverse=True)[0:5]

            entries = []
            for it, prob in topk:
                entry = dict(self.taxonomy.taxonomy[it])
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
