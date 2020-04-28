#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fire import Fire
from pathlib import Path
from axcell.data.paper_collection import PaperCollection
from axcell.data.structure import CellEvidenceExtractor
from elasticsearch_dsl import connections
from tqdm import tqdm
import pandas as pd
from joblib import delayed, Parallel

class Helper:
    def split_pc_pickle(self, path, outdir="pc-parts", parts=8):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        pc = PaperCollection.from_pickle(path)
        step = (len(pc) + parts - 1) // parts
        for idx, i in enumerate(range(0, len(pc), step)):
            part = PaperCollection(pc[i:i + step])
            part.to_pickle(outdir / f"pc-part-{idx:02}.pkl")

    def _evidences_for_pc(self, path):
        path = Path(path)
        pc = PaperCollection.from_pickle(path)
        cell_evidences = CellEvidenceExtractor()
        connections.create_connection(hosts=['10.0.1.145'], timeout=20)
        raw_evidences = []
        for paper in tqdm(pc):
            raw_evidences.append(cell_evidences(paper, paper.tables, paper_limit=100, corpus_limit=20))
        raw_evidences = pd.concat(raw_evidences)
        path = path.with_suffix(".evidences.pkl")
        raw_evidences.to_pickle(path)

    def evidences_for_pc(self, pattern="pc-parts/pc-part-??.pkl", jobs=-1):
        pickles = sorted(Path(".").glob(pattern))
        Parallel(backend="multiprocessing", n_jobs=jobs)(delayed(self._evidences_for_pc)(path) for path in pickles)

    def merge_evidences(self, output="evidences.pkl", pattern="pc-parts/pc-part-*.evidences.pkl"):
        pickles = sorted(Path(".").glob(pattern))
        evidences = [pd.read_pickle(pickle) for pickle in pickles]
        evidences = pd.concat(evidences)
        evidences.to_pickle(output)


if __name__ == "__main__": Fire(Helper())
