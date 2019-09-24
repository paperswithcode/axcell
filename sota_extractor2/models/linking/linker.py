from .bm25_naive import linked_proposals


class Linker:
    def __init__(self, name, taxonomy_linking, dataset_extractor):
        self.taxonomy_linking = taxonomy_linking
        self.dataset_extractor = dataset_extractor
        self.__name__ = name

    def __call__(self, paper, tables):
        proposals = linked_proposals(paper.paper_id, paper, tables,
                                     taxonomy_linking=self.taxonomy_linking,
                                     dataset_extractor=self.dataset_extractor)
        return proposals.set_index('cell_ext_id')
