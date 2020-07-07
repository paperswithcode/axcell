#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Open source tools lfoppiano/grobid
from collections import OrderedDict, Counter, Iterable, defaultdict
from dataclasses import dataclass, field
from warnings import warn

import json
import regex as re
from unidecode import unidecode
import time
import requests
import shelve
import xmltodict
from elasticsearch import ConflictError
from joblib import Parallel, delayed
from pathlib import Path

import diskcache as dc

from axcell import config
from axcell.data.elastic import Reference2, ID_LIMIT


def just_letters(s, _tok_re = re.compile(r'[^\p{L}]+')):
    return _tok_re.sub(' ', s).strip()

def ensure_list(a):
    if isinstance(a, (list, tuple)):
        return a
    else:
        return [a]

_anchor_re=re.compile(r'^((?:\s+[^\s]+){1,5}\s*[\(\[][12]\d{3}[a-zA-Z]{0,3}[\]\)] |\s*\[[a-zA-Z0-9_ .]{1,30}\]\s*)')
def strip_anchor(ref_str):
    return _anchor_re.sub('[1] ', ' '+ref_str)

_tokenizer_re = re.compile(r'[^/a-z0-9\\:?#\[\]\(\).-–]+')
def normalize_title(s, join=True):
    toks = _tokenizer_re.split(unidecode(s).lower())
    return "-".join(toks).strip() if join else toks

def to_normal_dict(d):
    if isinstance(d, list):
        return [to_normal_dict(x) for x in d]
    if isinstance(d, OrderedDict):
        return {k:to_normal_dict(v) for k,v in d.items()}
    return d

class GrobidClient:
    def __init__(self, cache_path=None, host='127.0.0.1', port=8070, max_tries=4, retry_wait=2):
        self.host = host
        self.port = port
        self.max_tries = max(max_tries, 1)
        self.retry_wait = retry_wait
        self.cache_path_shelve = Path.home()/'.cache'/'refs' /'gobrid'/'gobrid.pkl' if cache_path is None else Path(cache_path)
        self.cache_path = Path.home() / '.cache' / 'refs' /'gobrid' / 'gobrid.db' if cache_path is None else Path(cache_path)
        self.cache = None

    def get_cache(self):
        if self.cache is None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache = dc.Cache(self.cache_path)
        return self.cache

    def migrate(self):
        """Migrate from shelve to diskcache for thread safty."""
        cache = self.get_cache()
        old_cache = shelve.open(str(self.cache_path_shelve))
        count = 0
        for k in old_cache.keys():
            cache[k] = old_cache[k]
            count += 1
        old_cache.close()
        return count

    def _post(self, data):
        tries = 0
        while tries < self.max_tries:
            with requests.Session() as s:
                r = s.post(
                    f'http://{self.host}:{self.port}/api/processCitation',
                    data=data,
                    headers={'Connection': 'close'}
                )
            if r.status_code in [200, 204]:
                return r.content.decode("utf-8")
            if r.status_code != 503:
                raise RuntimeError(f"{r.status_code} {r.reason}\n{r.content}")
            tries += 1
            if tries < self.max_tries:
                time.sleep(self.retry_wait)
        raise ConnectionRefusedError(r.reason)

    def parse_ref_str_to_tei_dict(self, ref_str):
        cache = self.get_cache()
        d = cache.get(ref_str)
        if d is None:  # potential multiple recomputation in multithreading case
            content = self._post(data={'citations': ref_str})
            d = xmltodict.parse(content)
            d = to_normal_dict(d)
            cache[ref_str] = d
        return d

    def close(self):
        self.cache.close()
        self.cache = None


def pop_first(dictionary, *path):
    if dictionary is None:
        return None
    try:
        d = dictionary
        for p in path:
            d = d.pop(p, None)
            if d is None:
                return None
            if isinstance(d, (list, tuple)):
                d = d[0]
            if isinstance(d, str):
                return d
        return d
    except Exception as err:
        warn(f"{err} - Unable to pop path {path}, from {dictionary}")

@dataclass(eq=True, frozen=True)
class PAuthor:
    forenames: tuple
    surname: str

    @classmethod
    def from_tei_dict(cls, d):
        try:
            p = d['persName']
            forename = p.get('forename', [])
            if isinstance(forename, dict):
                forename = [forename]
            return cls(
                forenames=tuple(fname['#text'] for fname in forename),
                surname=p['surname']
            )
        except KeyError as err:
            warn(f"{err} - Unable to parse {d} as Author")
            print(d)
        except TypeError as err:
            warn(f"{err} - Unable to parse {d} as Author")
            print(d)

    @classmethod
    def from_fullname(cls, fullname):
        names = fullname.split()
        return cls(
            forenames=tuple(names[:-1]),
            surname=names[-1]
        )

    def __repr__(self):
        fnames = ', '.join(self.forenames)
        return f'"{self.surname}; {fnames}"'

    def short_names(self):
        fnames = [just_letters(f)[:1].capitalize() for f in self.forenames]
        surname = just_letters(self.surname).capitalize()
        return [surname] + fnames

    def short(self):
        return " ".join(self.short_names())
        return f'{surname} {fnames}'


conferences = [
    'nips', 'neurips', 'emnlp', 'acl', 'corr', 'ai magazine', 'machine learning', 'arxiv.org'
]
conferences_re = re.compile(f'\\b({"|".join(re.escape(c) for c in conferences)})[\\s0-9.]*', re.IGNORECASE)
def strip_conferences(title):
    return conferences_re.sub('', title)

_arxiv_id_re = re.compile(r'((?:arxiv|preprint|corr abs?)\s*)*[:/]([12]\d{3}\.\d{4,5})(v\d+)?', re.IGNORECASE)
def extract_arxivid(ref_str):
    arxiv_id = None
    m = _arxiv_id_re.search(ref_str)
    if m:
        arxiv_id = m.group(2)
        b,e = m.span()
        if m.group(1) and m.group(1).strip() != "": # we only remove arxivid it was prefixed with arxiv / pre print etc to keep urls intact.
            ref_str = ref_str[:b] + " " +ref_str[e:]
    return ref_str, arxiv_id


def is_publication_venue(word):
    return word.lower() in conferences

latex_artefacts_re=re.compile("|".join(re.escape(a) for a in {
    '\\BBA',
     '\\BBCP',
     '\\BBCQ',
     '\\BBOP',
     '\\BBOQ',
     '\\BCAY',
     '\\Bem',
     '\\citename'
}))

def strip_latex_artefacts(text):
    #text=text.replace('\\Bem', '.')
    return latex_artefacts_re.sub('',text)

def post_process_title(title, is_surname, is_publication_venue):
    if title is None:
        return title
    parts = title.split('. ')
    if len(parts) == 1:
        title = parts[0]
    else:
        def score_sent(part, idx):
            words = part.split(' ')

            return ((10 - idx) +
                    len(words)*2 +
                    sum(2 for w in words if w.isupper() and len(w) > 2) +
                    sum(-1 for w in words if w.istitle()) +
                    sum(-4 for w in words if is_surname(w)) +
                    sum(-10 for w in words if is_publication_venue(w)) +
                    (-100 if "in proceedings" in part.lower() else 0)
                    )

        scores = [(score_sent(part, idx=idx), part) for idx, part in enumerate(parts)]

        title = max(scores)[1]

    # title = strip_conferences(title)
    title = title.rstrip(' .')
    return title

@dataclass
class PReference:
    orig_key: tuple = None
    title: str = None
    authors: list = None
    idno: str = None
    date: str = None
    ptr: str = None

    extra: dict = None
    alt: dict = None
    orig_ref: str = field(repr=False, default_factory=lambda:None)

    arxiv_id: str = None
    pwc_slug: str = None

    def unique_id(self):
        if not self.title:
            return None
        norm_title = normalize_title(self.title)[:ID_LIMIT]  # elastic limit
        return norm_title

    @classmethod
    def from_tei_dict(cls, citation, **kwargs):
        bibstruct = dict(citation['biblStruct'])
        monogr = bibstruct.pop('monogr', {})
        paper = bibstruct.pop('analytic', monogr)
        if paper is None:
            raise ValueError("Analytic and mongr are both None")

        title = (pop_first(paper, 'title', '#text')
                or pop_first(monogr, 'title', '#text')
                or pop_first(bibstruct, 'note')
                )

        if not isinstance(title, str):
            bibstruct['note'] = title  # note was not string so let's revers pop
            title = None

        return cls(
            title=title,
            authors=[PAuthor.from_tei_dict(a) for a in ensure_list(paper.pop('author', []))],
            idno=pop_first(paper, 'idno'),
            date=pop_first(paper, 'imprint', 'date', '@when'),
            ptr=pop_first(paper, 'ptr', '@target'),
            extra={k: paper[k] for k in paper if k not in ['title', 'author']},
            alt=bibstruct,
            **kwargs
        )

    @classmethod
    def parse_ref_str(cls, ref_str, grobid_client, orig_key=None, is_surname=None, is_publication_venue=is_publication_venue):
        try:
            clean_ref_str = strip_latex_artefacts(ref_str)
            clean_ref_str = strip_anchor(clean_ref_str)
            clean_ref_str, arxiv_id = extract_arxivid(clean_ref_str)
            d = grobid_client.parse_ref_str_to_tei_dict(clean_ref_str)
            ref = cls.from_tei_dict(d, orig_ref=ref_str, arxiv_id=arxiv_id)
            ref.orig_key = orig_key

            ref.title = post_process_title(ref.title, is_surname, is_publication_venue)

            return ref
        except (KeyError, TypeError,ValueError) as err:
            warn(f"{err} - Unable to parse {d} as Ref")


nonalphanumeric_re = re.compile(r'[^a-z0-9 ]', re.IGNORECASE)
def until_first_nonalphanumeric(string):
    return nonalphanumeric_re.split(string)[0]

class ReferenceStore:
    def __init__(self, grobid_client,
                 surnames_path='/mnt/efs/pwc/data/ref-names.json',
                 use_cache=True):
        self.grobid_client = grobid_client
        self.refdb = {}
        self.tosync = []
        self.surnames_db = defaultdict(lambda: 0)
        self._load_surnames(surnames_path)
        self.use_cache = use_cache

    def _load_surnames(self, path):
        with Path(path).open() as f:
            self.preloaded_surnames_db = json.load(f)

    def is_surname(self, word):
        return word in self.preloaded_surnames_db  #or self.surnames_db[word] > 5

    def get_reference(self, key):
        if self.use_cache:
            if key not in self.refdb:
                self.refdb[key] = Reference2.mget([key])[0]
            return self.refdb[key]
        return Reference2.mget([key])[0]

    def add_or_merge(self, ref):
        curr_uid = ref.unique_id()
        if not self.use_cache or curr_uid not in self.refdb:
            curr_ref = Reference2.mget([curr_uid])[0] or Reference2.from_ref(ref)
        else:
            curr_ref = self.refdb[curr_uid]
        curr_ref.add_ref(ref)  # to fill all other fields but title
        if self.use_cache:
            self.refdb[curr_uid] = curr_ref
            self.refdb[ref.unique_id()] = curr_ref
            self.tosync.append(curr_ref)
        else:
            try:
                curr_ref.save()
            except ConflictError:
                pass
        for author in ref.authors:
            if author is not None:
                self.surnames_db[author.surname] += 1

        return curr_ref.stable_id

    def add_reference_string(self, ref_str):
        ref = PReference.parse_ref_str(ref_str, self.grobid_client, is_surname=self.is_surname)
        if ref is None or ref.unique_id() is None:
            for r in Reference2.search().query('match', orig_refs=ref_str)[:10]:
                if r.stable_id in normalize_title(ref_str):
                    return r.stable_id
            return None

        stable_id = self.add_or_merge(ref)
        return stable_id

    def add_batch(self, ref_strs):
        if isinstance(ref_strs, str):
            ref_strs = [ref_strs]
        def add_ref(ref_str):
            return self.add_reference_string(ref_str)
        if len(ref_strs) < 1000:
            yield from (add_ref(ref_str) for ref_str in ref_strs)
        else:
            yield from Parallel(n_jobs=10, require='sharedmem')(
                delayed(add_ref)(ref_str) for ref_str in ref_strs)

    def sync(self):
        tosync = self.tosync
        self.tosync = []
        for p in tosync:
            try:
                p.save()
            except ConflictError:
                pass


def get_refstrings(p):
    paper = p.text if hasattr(p, 'text') else p
    if not hasattr(paper, 'fragments'):
        return
    fragments = paper.fragments
    ref_sec_started = False
    for f in reversed(fragments):
        if f.header.startswith('xxanchor-bib'):
            ref_sec_started = True
            yield f.text
        elif ref_sec_started:
            # todo: check if a paper can have multiple bibliography sections
            # (f.e., one in the main paper and one in the appendix)
            break  # the refsection is only at the end of paper



_ref_re = re.compile(r'^\s*(?:xxanchor-bib\s)?xxanchor-([a-zA-Z0-9-]+)\s(.+)$')
def extract_refs(p):
    for ref in get_refstrings(p):
        m = _ref_re.match(ref)
        if m:
            ref_id, ref_str = m.groups()
            yield {
                "ref_id": ref_id,
                "ref_str": ref_str.strip(r'\s')
            }
