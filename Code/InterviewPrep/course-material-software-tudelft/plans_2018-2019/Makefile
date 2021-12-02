FILENAME = article
DEPENDANCIES = $(notdir $(wildcard *.tex)) 
FIGDEPENDANCIES = $(wildcard figures/*)
BIBDEPENDANCIES = $(wildcard references/*.bib)

all: $(FILENAME).pdf 


%.pdf: %.tex Makefile $(DEPENDANCIES) $(FIGDEPENDANCIES) $(BIBDEPENDANCIES)
	latexmk -pdf $<

clean: 
	latexmk -C $(FILENAME).tex

spell: $(DEPANDANCIES)
	for file in $(DEPENDANCIES); do \
		aspell check -t --dont-backup --lang=en_GB $$file; \
	done

.PHONY: all clean spell
