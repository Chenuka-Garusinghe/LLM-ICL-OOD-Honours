PROJ = thesis

.PHONY: all bib figures pdf wc clean

all: pdf

# references.bib is generated from ../docs/references.bib (APA-ready: protected titles,
# subtitles, arXiv DOIs); edit the source there, then run `make bib`.
bib:
	python3 tools/apa_bib.py ../docs/references.bib references.bib

# figures/*.pdf are drawn from the result files (run from the project root)
figures:
	cd .. && .venv/bin/python thesis/tools/figures.py

pdf:
	latexmk -pdf $(PROJ)

wc:
	wc `find . -name "*.tex"`

clean:
	rm -f `find . -name '*.log'`
	rm -f `find . -name '*.dvi'`
	rm -f `find . -name '*.ps'`
	rm -f `find . -name '*.aux'`
	rm -f `find . -name '*.blg'`
	rm -f `find . -name '*.bbl'`
	rm -f `find . -name '*.toc'`
	rm -f `find . -name '*.lof'`
	rm -f `find . -name '*.lot'`
	rm -f `find . -name '*.out'`
	rm -f `find . -name '*~'`
