import fire
from unidecode import unidecode
from pathlib import Path
import string
import ahocorasick
import pickle
from multiprocessing import Pool
from sota_extractor2.data.doc_utils import get_text, read_html

punctuation_table = str.maketrans('', '', string.punctuation)

def normalize_title(title):
    return unidecode(title.strip().lower().replace(' ', '')).translate(punctuation_table)

def resolve_references(reference_trie, bibitems):
    if len(bibitems) == 0:
        return {}
    bib_ids = list(bibitems.keys())
    texts = list(bibitems.values())
    found = 0
    resolved = {}
    for bib_id, text in zip(bib_ids, texts):
        references =  [ref for _, ref in reference_trie.iter(normalize_title(text)) if len(normalize_title(ref['title'])) >= 6]
        references = sorted(references, key=lambda ref: len(normalize_title(ref['title'])), reverse=True)
        for ref in references:
            for author in ref['authors']:
                if normalize_title(author['name'].split(' ')[-1]) not in normalize_title(text):
                    break
            else:
                found += 1
                resolved[bib_id] = ref['id']
                break
    print(f"Found {found} ({found / len(bibitems)})")
    return resolved

def update_references(html, mapping):
    anchors = html.select('[href^="#"]')
    for anchor in anchors:
        target = anchor['href'][1:]
        anchor['href'] = '#' + mapping.get(target, target)
    anchors = html.select('a[id]:not([id=""])')
    for anchor in anchors:
        bib_id = anchor['id']
        anchor['id'] = mapping.get(bib_id, bib_id)

def get_bibitems(html):
    elems = html.select(".thebibliography p.bibitem")
    bibitems = {}
    for elem in elems:
        anchors = elem.select('a[id]:not([id=""])')
        if anchors:
            bib_id = anchors[0]['id']
            bibitems[bib_id] = get_text(elem)
    return bibitems

def save_html(path, html):
    with open(path, 'w') as f:
        f.write(str(html))

def resolve_references_in_html(args):
    file, output = args
    output.parent.mkdir(exist_ok=True, parents=True)
    html = read_html(file)
    bibitems = get_bibitems(html)
    mapping = resolve_references(reference_trie, bibitems)
    update_references(html, mapping)
    save_html(output, html)

#DUMP_REFERENCES_PATH = Path("/home/ubuntu/pwc/mycache/references-short.json")

#TRIE_PATH = Path("/home/ubuntu/pwc/mycache/automaton.pkl")

def normalize_references(source_path, target_path, automaton, jobs=1):
    global reference_trie
    source_path = Path(source_path)
    target_path = Path(target_path)
    with open(automaton, 'rb') as f:
        reference_trie = pickle.load(f)
    with Pool(jobs) as p:
        params = [(file, target_path / file.relative_to(source_path)) for file in source_path.glob("**/*.html")]
        p.map(resolve_references_in_html, params)

if __name__ == "__main__": fire.Fire(normalize_references)
