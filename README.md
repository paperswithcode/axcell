# AxCell: Automatic Extraction of Results from Machine Learning Papers
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/axcell-automatic-extraction-of-results-from/scientific-results-extraction-on-pwc)](https://paperswithcode.com/sota/scientific-results-extraction-on-pwc?p=axcell-automatic-extraction-of-results-from)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/axcell-automatic-extraction-of-results-from/scientific-results-extraction-on-nlp-tdms-exp)](https://paperswithcode.com/sota/scientific-results-extraction-on-nlp-tdms-exp?p=axcell-automatic-extraction-of-results-from)

This repository is the official implementation of [AxCell: Automatic Extraction of Results from Machine Learning Papers](https://arxiv.org/abs/2004.14356).

![pipeline](https://user-images.githubusercontent.com/13535078/81287158-33e01000-905a-11ea-8573-d716373efbdd.png)

## Requirements

To create a [conda](https://www.anaconda.com/distribution/) environment named `axcell` and install requirements run:

```setup
conda env create -f environment.yml
```

Additionally, `axcell` requires `docker` (that can be run without `sudo`). Run `scripts/pull_docker_images.sh` to download necessary images.

## Datasets
We publish the following datasets:
* [ArxivPapers](https://github.com/paperswithcode/axcell/releases/download/v1.0/arxiv-papers.csv.xz)
* [SegmentedTables & LinkedResults](https://github.com/paperswithcode/axcell/releases/download/v1.0/segmented-tables.json.xz)
* [PWCLeaderboards](https://github.com/paperswithcode/axcell/releases/download/v1.0/pwc-leaderboards.json.xz)

See [datasets](notebooks/datasets.ipynb) notebook for an example of how to load the datasets provided below. The [extraction](notebooks/extraction.ipynb) notebook shows how to use `axcell` to extract text and tables from papers.

## Evaluation

See the [evaluation](notebooks/evaluation.ipynb) notebook for the full example on how to evaluate AxCell on the PWCLeaderboards dataset. 

## Training

* [pre-training language model](notebooks/training/lm.ipynb) on the ArxivPapers dataset 
* [table type classifier](notebooks/training/table-type-classifier.ipynb) and [table segmentation](notebooks/training/table-segmentation.ipynb) on the SegmentedResults dataset 

## Pre-trained Models

You can download pretrained models here:

- [axcell](https://github.com/paperswithcode/axcell/releases/download/v1.0/models.tar.xz) &mdash; an archive containing the taxonomy, abbreviations, table type classifier and table segmentation model. See the [results-extraction](notebooks/results-extraction.ipynb) notebook for an example of how to load and run the models 
- [language model](https://github.com/paperswithcode/axcell/releases/download/v1.0/lm.pth.xz) &mdash; [ULMFiT](https://arxiv.org/abs/1801.06146) language model pretrained on the ArxivPapers dataset

## Results

AxCell achieves the following performance:

### 


| Dataset | Macro F1 | Micro F1 |
| ---------- |---------------- | -------------- |
| [PWC Leaderboards](https://paperswithcode.com/sota/scientific-results-extraction-on-pwc)     |     21.1         |      28.7       |
| [NLP-TDMS](https://paperswithcode.com/sota/scientific-results-extraction-on-nlp-tdms-exp)    |     19.7         |      25.8       |



## License

AxCell is released under the [Apache 2.0 license](LICENSE).

## Citation
The pipeline is described in the following paper:
```bibtex
@inproceedings{axcell,
    title={AxCell: Automatic Extraction of Results from Machine Learning Papers},
    author={Marcin Kardas and Piotr Czapla and Pontus Stenetorp and Sebastian Ruder and Sebastian Riedel and Ross Taylor and Robert Stojnic},
    year={2020},
    booktitle={2004.14356}
}
```
