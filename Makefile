DATA_DIR = ../data
ANNOTATIONS_DIR = $(DATA_DIR)/annotations
ARCHIVES_DIR = $(DATA_DIR)/arxiv/sources
UNPACKED_DIR = $(DATA_DIR)/arxiv/unpacked_sources
HTMLS_DIR = $(DATA_DIR)/arxiv/htmls

ARCHIVES = $(wildcard $(ARCHIVES_DIR)/*)
UNPACKS  = $(patsubst $(ARCHIVES_DIR)/%,$(UNPACKED_DIR)/%,$(ARCHIVES))
HTMLS    = $(patsubst $(ARCHIVES_DIR)/%,$(HTMLS_DIR)/%.html,$(ARCHIVES))

$(shell mkdir -p "$(DATA_DIR)")
$(shell mkdir -p "$(HTMLS_DIR)")

.PHONY: all
all:	$(ANNOTATIONS_DIR)/pdfs-urls.csv $(ANNOTATIONS_DIR)/sources-urls.csv convert_all

convert_all: $(HTMLS)

$(HTMLS): $(HTMLS_DIR)/%.html: $(UNPACKED_DIR)/%
	./docker-latex2html.sh $(HTMLS_DIR) $^

unpack_all: $(UNPACKS)

$(UNPACKS): $(UNPACKED_DIR)/%: $(ARCHIVES_DIR)/%
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
