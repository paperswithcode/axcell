#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import regex
import diskcache

from fastai.text import progress_bar

from sota_extractor2.data.references import *
from functools import lru_cache
from sota_extractor2.data.elastic import *

from sota_extractor2.data.paper_collection import PaperCollection

connections.create_connection(hosts=['10.0.1.145'], timeout=20)

pc = PaperCollection.from_pickle("/mnt/efs/pwc/data/pc-small-noann.pkl")




class PaperCollectionReferenceParser:
    def __init__(self):
        self.refsdb = ReferenceStore()
        self.cache = diskcache.Cache(Path.home() / '.cache' / 'refs' / 'refs_ids.db')


    def parse_refs(self, p):
        for d in extract_refs(p):
            if not d["ref_id"].startswith("pwc-"):
                key = d["paper_arxiv_id"] + d["ref_id"]
                if key not in self.cache:
                    new_id = self.refsdb.add_reference_string(d['ref_str'])
                    if new_id is not None:
                        new_id = "pwc-" + new_id
                    self.cache[key] = new_id
                if self.cache[key] and len(self.cache[key]) > ID_LIMIT:  # fix to self.cache to make the id compatible with elastic
                    self.cache[key] = self.cache[key][:ID_LIMIT]
                yield d["ref_id"], self.cache[key]
        self.refsdb.sync()


    def update_references(self, pc):
        def update_paper(p_idx):
            p = pc[p_idx]
            for old_ref_id, new_ref_id in self.parse_refs(p):
                if new_ref_id is not None:
                    for f in p.text.fragments:
                        f.text = f.text.replace(old_ref_id, new_ref_id)

        Parallel(n_jobs=8, require='sharedmem')(
            delayed(update_paper)(p_idx) for p_idx in progress_bar(range(len(pc))))


    def update_references_pickle(self, data_pkl_path="/mnt/efs/pwc/data/pc-small-noann.pkl"):
        print("Loading pickle", data_pkl_path)
        pc = PaperCollection.from_pickle(data_pkl_path)
        self.update_references(pc)
        print()
        print("Saving pickle", data_pkl_path)
        pc.to_pickle(data_pkl_path)
        return pc

def main(data_pkl_path="/home/ubuntu/data/pc2.pkl"):
    with PaperCollectionReferenceParser() as worker:
        worker.update_references_pickle(data_pkl_path)
