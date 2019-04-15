all:	pdfs-urls.csv sources-urls.csv

pdfs-urls.csv:	papers-urls.csv
	sed -e 's#/abs/#/pdf/#' -e 's#$$#.pdf#' $^ > $@

sources-urls.csv:	papers-urls.csv
	sed -e 's#/abs/#/e-print/#' $^ > $@

papers-urls.csv:	evaluation-tables.json get_papers_links.sh
	./get_papers_links.sh evaluation-tables.json > $@

%:	%.gz
	gunzip -k $^

evaluation-tables.json.gz:
	wget https://paperswithcode.com/media/about/evaluation-tables.json.gz

.PHONY : clean
clean :
	rm -f *.json *.gz *.csv
