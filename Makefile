DATA_DIR = ../data
ANNOTATIONS_DIR = $(DATA_DIR)/annotations
ARCHIVES_DIR = $(DATA_DIR)/sources
$(shell mkdir -p "$(DATA_DIR)")

.PHONY: all
all:	$(ANNOTATIONS_DIR)/pdfs-urls.csv $(ANNOTATIONS_DIR)/sources-urls.csv

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
