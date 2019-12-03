# Scripts for extracting tables

Dependencies:
 * [jq](https://stedolan.github.io/jq/) (`sudo apt install jq`)
 * docker (run without `sudo`)
 * [conda](https://www.anaconda.com/distribution/)

Directory structure:
```
.
└── data
    ├── annotations
    │   └── evaluation-tables.json.gz     # current annotations
    └── arxiv
        ├── sources                       # gzip archives with e-prints
        ├── unpacked\_sources             # automatically extracted latex sources
        ├── htmls                         # automatically generated htmls
        ├── htmls-clean                   # htmls fixed by chromium
        └── tables                        # extracted tables
```


To preprocess data and extract tables and texts, run:
```
make pull_images
conda env create -f environment.yml
source activate xtables
make -j 8 -i extract_all > stdout.log 2> stderr.log
```
where `8` is number of jobs to run simultaneously. Optionally one can specify path to data directory, f.e., `make DATA_DIR=mydata ...`.

## Test
To test the whole extraction on a single file run
```
make test
```
