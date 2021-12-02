FILENAME = lecture
DEPENDANCIES = $(notdir $(wildcard *.tex)) $(wildcard **/*.tex)
FIGDEPENDANCIES = $(wildcard figures/*)
BIBDEPENDANCIES = $(wildcard references/*.bib)

.PHONY: all clean spell

all: $(FILENAME).pdf 

%.pdf: %.tex Makefile $(DEPENDANCIES) $(FIGDEPENDANCIES) $(BIBDEPENDANCIES)
	latexmk -xelatex -pdf $<
	latexmk -c $<

clean: 
	-@rm *.log *.aux *.out *.toc *.bbl *.blg $(FILENAME).pdf
spell: $(DEPANDANCIES)
	for file in $(DEPENDANCIES); do \
		aspell check -t --dont-backup --lang=en_GB $$file; \
	done

