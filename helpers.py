from fire import Fire
from pathlib import Path
from sota_extractor2.data.paper_collection import PaperCollection
from sota_extractor2.data.structure import CellEvidenceExtractor
from elasticsearch_dsl import connections
from tqdm import tqdm
import pandas as pd

class Helper:
    def split_pc_pickle(self, path, outdir="pc-parts", parts=8):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        pc = PaperCollection.from_pickle(path)
        step = (len(pc) + parts - 1) // parts
        for idx, i in enumerate(range(0, len(pc), step)):
            part = PaperCollection(pc[i:i + step])
            part.to_pickle(outdir / f"pc-part-{idx:02}.pkl")

    def evidences_for_pc(self, path):
        path = Path(path)
        pc = PaperCollection.from_pickle(path)
        cell_evidences = CellEvidenceExtractor()
        connections.create_connection(hosts=['10.0.1.145'], timeout=20)
        raw_evidences = []
        for paper in tqdm(pc):
            raw_evidences.append(cell_evidences(paper, paper.tables))
        raw_evidences = pd.concat(raw_evidences)
        path = path.with_suffix(".evidences.pkl")
        raw_evidences.to_pickle(path)

    def merge_evidences(self, output="evidences.pkl", pattern="pc-parts/pc-part-*.evidences.pkl"):
        pickles = sorted(Path(".").glob(pattern))
        evidences = [pd.read_pickle(pickle) for pickle in pickles]
        evidences = pd.concat(evidences)
        evidences.to_pickle(output)


if __name__ == "__main__": Fire(Helper())
