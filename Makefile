DATA_DIR = data
ARXIV_DIR = $(DATA_DIR)/arxiv
ARCHIVES_DIR = $(ARXIV_DIR)/sources
UNPACKED_DIR = $(ARXIV_DIR)/unpacked_sources
HTMLS_DIR = $(ARXIV_DIR)/htmls
FIXED_HTMLS_DIR = $(ARXIV_DIR)/htmls-clean
TABLES_DIR = $(ARXIV_DIR)/papers
TEXTS_DIR = $(ARXIV_DIR)/papers

ARCHIVES    := $(shell find $(ARCHIVES_DIR) -name '*.gz' -type f 2>/dev/null)
UNPACKS     := $(patsubst $(ARCHIVES_DIR)/%.gz,$(UNPACKED_DIR)/%,$(ARCHIVES))
HTMLS       := $(patsubst $(ARCHIVES_DIR)/%.gz,$(HTMLS_DIR)/%.html,$(ARCHIVES))
FIXED_HTMLS := $(patsubst $(ARCHIVES_DIR)/%.gz,$(FIXED_HTMLS_DIR)/%.html,$(ARCHIVES))
TABLES      := $(patsubst $(ARCHIVES_DIR)/%.gz,$(TABLES_DIR)/%,$(ARCHIVES))
TEXTS       := $(patsubst $(ARCHIVES_DIR)/%.gz,$(TEXTS_DIR)/%/text.json,$(ARCHIVES))

.PHONY: all
all:	extract_all

.PHONY: test
test: DATA_DIR = test/data
test:
	mkdir -p $(ARCHIVES_DIR)
	tar czf $(ARCHIVES_DIR)/paper.gz -C test/src .
	$(MAKE) DATA_DIR=$(DATA_DIR) --always-make extract_all
	diff -r $(TABLES_DIR) test/expected

.PHONY: extract_all extract_texts extract_tables fix_htmls_all convert_all unpack_all

extract_all: extract_tables extract_texts

extract_texts: $(TEXTS)

$(TEXTS): $(TEXTS_DIR)/%/text.json: $(FIXED_HTMLS_DIR)/%.html
	python ./extract_texts.py $^ $@


extract_tables: $(TABLES)

fix_htmls_all: $(FIXED_HTMLS)

convert_all: $(HTMLS)

$(TABLES): $(TABLES_DIR)/%: $(FIXED_HTMLS_DIR)/%.html
	python ./extract_tables.py $^ --outdir $@

$(FIXED_HTMLS): $(FIXED_HTMLS_DIR)/%: $(HTMLS_DIR)/%
	./clean_html.sh $^ $@

$(HTMLS): $(HTMLS_DIR)/%.html: $(UNPACKED_DIR)/%
	./docker-latex2html.sh $^ $@

unpack_all: $(UNPACKS)

$(UNPACKS): $(UNPACKED_DIR)/%: $(ARCHIVES_DIR)/%.gz
	./unpack-sources.sh $^ $@

.PHONY: pull_images
pull_images:
	docker pull arxivvanity/engrafo:b3db888fefa118eacf4f13566204b68ce100b3a6
	docker pull zenika/alpine-chrome:73

.PHONY: clean
clean :
	#rm -f *.gz
