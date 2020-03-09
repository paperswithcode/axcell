import re
import json
from pathlib import Path
from collections import Counter
from sota_extractor2.data.elastic import Reference2, setup_default_connection
from sota_extractor2.data.references import PReference, PAuthor, ReferenceStore
from tqdm import tqdm
from elasticsearch.helpers import bulk
from elasticsearch_dsl.connections import connections
import http.client
import xml.etree.ElementTree as ET

# required for bulk saving
http.client._MAXHEADERS = 1000

setup_default_connection()

papers_path = Path("/data/dblp/papers/papers-with-abstracts.json")


def read_pwc_papers(path):
    with open(path, "rt") as f:
        return json.load(f)


arxiv_url_re = re.compile(r"^(?:https?://(?:www.)?arxiv.org/(?:abs|pdf|e-print)/)?(?P<arxiv_id>\d{4}\.\d+)(?:v\d+)?(?:\.pdf)?$")
arxiv_url_only_re = re.compile(r"^(?:https?://(?:www.)?arxiv.org/(?:abs|pdf|e-print)/)(?P<arxiv_id>\d{4}\.\d+)(?:v\d+)?(?:\.pdf)?$")
pwc_url_re = re.compile(r"^(?:https?://(?:www.)?)paperswithcode.com/paper/(?P<slug>[^/]*)/?$")


def from_paper_dict(paper):
    authors = [PAuthor.from_fullname(a) for a in paper["authors"] if a.strip()]
    arxiv_id = None
    if paper["arxiv_id"]:
        arxiv_id = paper["arxiv_id"]
    elif paper["url_abs"]:
        m = arxiv_url_re.match(paper["url_abs"])
        if m:
            arxiv_id = m.group("arxiv_id")
    title = None
    if paper["title"]:
        title = paper["title"].rstrip(" .")
    slug = None
    if paper["paper_url"]:
        m = pwc_url_re.match(paper["paper_url"])
        if m:
            slug = m.group("slug")
    return PReference(
        title=title,
        authors=authors,
        ptr=paper["url_pdf"] or paper["url_abs"],
        arxiv_id=arxiv_id,
        pwc_slug=slug,
        date=paper["date"],
        orig_ref=f"{', '.join(paper['authors'])}. {paper['title']}.",
    )


def _text(elem): return "".join(elem.itertext())


def from_paper_elem(elem):
    authors_str = [_text(a).strip() for a in elem.findall("author")]
    authors_str = [s for s in authors_str if s]
    authors = [PAuthor.from_fullname(a) for a in authors_str]
    arxiv_id = None
    url = None
    for ee in elem.findall("ee"):
        if url is None or "oa" in ee.attrib: # prefere open access urls
            url = _text(ee)
        m = arxiv_url_only_re.match(_text(ee))
        if m:
            url = _text(ee) # prefere arxiv urls
            arxiv_id = m.group("arxiv_id")
            break
    title = None
    title_elem = elem.find("title")
    if title_elem is not None:
        title = _text(title_elem).rstrip(" .")
    return PReference(
        title=title,
        authors=authors,
        ptr=url,
        arxiv_id=arxiv_id,
        orig_ref=f"{', '.join(authors_str)}. {title}.",
    )


def merge_references(p_references, elastic_references):
    uids = Counter([p_ref.unique_id() for p_ref in p_references])
    for p_ref in tqdm(p_references):
        uid = p_ref.unique_id()
        # ignore papers with too common title
        # (often these are "Editorial", "Preface", "Letter")
        if uids[uid] > 5:
            continue
        e_ref = elastic_references.get(uid)
        if not e_ref:
            e_ref = Reference2.from_ref(p_ref)
            elastic_references[uid] = e_ref
        e_ref.add_ref(p_ref)


def save_all(docs):
    bulk(connections.get_connection(), (d.to_dict(True) for d in docs), chunk_size=500)


def get_elastic_references(unique_ids, chunk_size=1000):
    elastic_references = {}
    i = 0
    while i < len(unique_ids):
        ids = unique_ids[i:i+chunk_size]
        i += chunk_size
        elastic_references.update({
            uid: ref for uid, ref in zip(ids, Reference2.mget(ids))
            if ref
        })
    return elastic_references


def init_pwc():
    # read list of ML papers (titles, abstracts, arxiv ids, etc.)
    all_papers = read_pwc_papers(papers_path)

    # change dicts into PReferences
    p_references = [from_paper_dict(paper) for paper in all_papers]

    # keep references with valid ids
    p_references = [ref for ref in p_references if ref.unique_id()]

    all_ids = list(set(ref.unique_id() for ref in p_references))
    elastic_references = get_elastic_references(all_ids)
    merge_references(p_references, elastic_references)
    save_all(elastic_references.values())


def init_dblp():
    dblp_xml = ET.parse(str(Path("/data") / "dblp" / "dblp-noent.xml"))
    #dblp_xml = ET.parse(str(Path("/data") / "dblp" / "dblp-small-noent.xml"))
    root = dblp_xml.getroot()
    p_references = [from_paper_elem(elem) for elem in root]
    p_references = [ref for ref in p_references if ref.unique_id()]

    all_ids = list(set(ref.unique_id() for ref in p_references))
    # todo: add references2 index initialization
    elastic_references = {} #get_elastic_references(all_ids)

    merge_references(p_references, elastic_references)
    save_all(elastic_references.values())

# Reference2._index.delete()
Reference2.init()
init_dblp()
init_pwc()
