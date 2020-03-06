import re
import json
from pathlib import Path
from sota_extractor2.data.elastic import Reference2
from sota_extractor2.data.references import PReference, PAuthor, ReferenceStore
from tqdm import tqdm
from elasticsearch.helpers import bulk
from elasticsearch_dsl.connections import connections
import http.client
import xml.etree.ElementTree as ET

# required for bulk saving
http.client._MAXHEADERS = 1000

# papers_path = Path("/tmp/papers/papers-with-abstracts.json")
papers_path = Path("/tmp/papers/papers-with-abstracts-duplicates.json")


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


def from_paper_elem(elem):
    authors_str = [a.text.strip() for a in elem.findall("author") if a.text.strip()]
    authors = [PAuthor.from_fullname(a) for a in authors_str]
    arxiv_id = None
    url = None
    for ee in elem.findall("ee"):
        if url is None or "oa" in ee.attrib: # prefere open access urls
            url = ee.text
        m = arxiv_url_only_re.match(ee.text)
        if m:
            url = ee.text
            arxiv_id = m.group("arxiv_id")
            break
    title = None
    title_elem = elem.find("title")
    if title_elem is not None:
        title = title_elem.text.rstrip(" .")
    return PReference(
        title=title,
        authors=authors,
        ptr=url,
        arxiv_id=arxiv_id,
        orig_ref=f"{', '.join(authors_str)}. {title}.",
    )


def merge_references(p_references):
    for p_ref in tqdm(p_references):
        uid = p_ref.unique_id()
        e_ref = elastic_references.get(uid)
        if not e_ref:
            e_ref = Reference2.from_ref(p_ref)
            elastic_references[uid] = e_ref
        e_ref.add_ref(p_ref)


def save_all(docs):
    bulk(connections.get_connection(), (d.to_dict(True) for d in docs), chunk_size=500)


def init_pwc():
    # read list of ML papers (titles, abstracts, arxiv ids, etc.)
    all_papers = read_pwc_papers(papers_path)

    # change dicts into PReferences
    p_references = [from_paper_dict(paper) for paper in all_papers]

    # keep references with valid ids
    p_references = [ref for ref in p_references if ref.unique_id()]

    all_ids = list(set(ref.unique_id() for ref in p_references))
    elastic_references = {
        uid: ref for uid, ref in zip(all_ids, Reference2.mget(all_ids))
        if ref
    }

    merge_references(p_references)
    save_all(elastic_references.values())


def init_dblp():
    dblp_xml = ET.parse(str(Path.home() / "data" / "dblp" / "dblp-10k-noent.xml"))
    root = dblp_xml.getroot()
    p_references = [from_paper_elem(elem) for elem in root.getchildren()]
    p_references = [ref for ref in p_references if ref.unique_id()]

    all_ids = list(set(ref.unique_id() for ref in p_references))
    elastic_references = {
        uid: ref for uid, ref in zip(all_ids, Reference2.mget(all_ids))
        if ref
    }

    merge_references(p_references)
    save_all(elastic_references.values())

init_dblp()
#init_pwc()
