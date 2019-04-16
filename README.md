Scripts for data preparation.

Dependencies:
 * [jq](https://stedolan.github.io/jq/) (`sudo apt install jq`)
 * docker

Directory structure:
```
.
├── data
│   ├── annotations
│   │   └── evaluation-tables.json.gz     # current annotations
│   └── arxiv
│       ├── sources                       # gzip archives with e-prints
│       ├── unpacked\_sources             # automatically extracted latex sources
│       └── htmls                         # automatically generated htmls
└── prepare-data
```

To preprocess data, run:
```cd prepare-data
make -j 8 -i```
where `8` is number of jobs to run simultaneously.
