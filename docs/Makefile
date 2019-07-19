MARKDOWN = $(wildcard *.md)
PDF = $(patsubst %.md,%.pdf,$(MARKDOWN))

all: $(PDF)

%.pdf: %.md
	pandoc --filter pandoc-citeproc --bibliography=sdot.bib --csl=ieee.csl \
		-M date="`LC_ALL=C date "+%B %e, %Y"`" -s $< -o $@

clean:
	@rm -fv $(PDF)
