from .bm25_naive import linked_proposals
from ...pipeline_logger import pipeline_logger


class Linker:
    step = "linking"

    def __init__(self, name, taxonomy_linking, dataset_extractor):
        self.taxonomy_linking = taxonomy_linking
        self.dataset_extractor = dataset_extractor
        self.__name__ = name

    def __call__(self, paper, tables, topk=1):
        pipeline_logger(f"{Linker.step}::call", paper=paper, tables=tables)
        proposals = linked_proposals(paper.paper_id, paper, tables,
                                     taxonomy_linking=self.taxonomy_linking,
                                     dataset_extractor=self.dataset_extractor,
                                     topk=topk)

        if topk == 1:
            proposals = proposals.set_index('cell_ext_id')
            best = proposals
        else:
            best = proposals.groupby('cell_ext_id').head(1).set_index('cell_ext_id')

        pipeline_logger(f"{Linker.step}::linked", paper=paper, tables=tables, proposals=best)
        return proposals
