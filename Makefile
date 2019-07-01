DATA_DIR = ../data
ANNOTATIONS_DIR = $(DATA_DIR)/annotations
ARXIV_DIR = $(DATA_DIR)/arxiv
ARCHIVES_DIR = $(ARXIV_DIR)/sources
UNPACKED_DIR = $(ARXIV_DIR)/unpacked_sources
HTMLS_DIR = $(ARXIV_DIR)/htmls
FIXED_HTMLS_DIR = $(ARXIV_DIR)/htmls-clean
TABLES_DIR = $(ARXIV_DIR)/tables

ARCHIVES    = $(wildcard $(ARCHIVES_DIR)/**.gz)
UNPACKS     = $(patsubst $(ARCHIVES_DIR)/%.gz,$(UNPACKED_DIR)/%,$(ARCHIVES))
HTMLS       = $(patsubst $(ARCHIVES_DIR)/%.gz,$(HTMLS_DIR)/%.html,$(ARCHIVES))
FIXED_HTMLS = $(patsubst $(ARCHIVES_DIR)/%.gz,$(FIXED_HTMLS_DIR)/%.html,$(ARCHIVES))
TABLES      = $(patsubst $(ARCHIVES_DIR)/%.gz,$(TABLES_DIR)/%,$(ARCHIVES))

$(shell mkdir -p "$(DATA_DIR)" "$(ANNOTATIONS_DIR)" "$(UNPACKED_DIR)" "$(HTMLS_DIR)" "$(FIXED_HTMLS_DIR)" "$(TABLES_DIR)")

.PHONY: all
all:	$(ANNOTATIONS_DIR)/pdfs-urls.csv $(ANNOTATIONS_DIR)/sources-urls.csv extract_all

.PHONY: test
test: DATA_DIR = test/data
test:
	mkdir -p $(ARCHIVES_DIR)
	tar czf $(ARCHIVES_DIR)/paper.gz -C test/src .
	$(MAKE) DATA_DIR=$(DATA_DIR) extract_all
	cat $(TABLES_DIR)/paper/table_01.csv
	diff $(TABLES_DIR)/paper/table_01.csv test/src/table_01.csv

extract_all: $(TABLES)

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

$(ANNOTATIONS_DIR)/pdfs-urls.csv:	$(ANNOTATIONS_DIR)/papers-urls.csv
	sed -e 's#/abs/#/pdf/#' -e 's#$$#.pdf#' $^ > $@

$(ANNOTATIONS_DIR)/sources-urls.csv:	$(ANNOTATIONS_DIR)/papers-urls.csv
	sed -e 's#/abs/#/e-print/#' $^ > $@

$(ANNOTATIONS_DIR)/papers-urls.csv:	$(ANNOTATIONS_DIR)/evaluation-tables.json get_papers_links.sh
	./get_papers_links.sh $< > $@

$(ANNOTATIONS_DIR)/%:	$(ANNOTATIONS_DIR)/%.gz
	gunzip -kf $^

$(ANNOTATIONS_DIR)/evaluation-tables.json.gz:
	wget https://paperswithcode.com/media/about/evaluation-tables.json.gz -O $@


.PHONY : clean
clean :
	cd "$(ANNOTATIONS_DIR)" && rm -f *.json *.csv
	#rm -f *.gz
