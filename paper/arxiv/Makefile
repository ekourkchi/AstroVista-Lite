# Build the arXiv manuscript, and the bundle to upload to arXiv.
#   make            main.pdf
#   make arxiv      astrovista-lite-arxiv.tar.gz (with main.bbl: arXiv does not run BibTeX)
#   make figures    regenerate the figures and numbers (../make_figures.py)
# AASTeX 7 (aastex701.cls, aasjournalv7.bst) is fetched from CTAN when it is missing.

AASTEX = aastex701.cls aasjournalv7.bst

main.pdf: main.tex paper.bib $(AASTEX) figures/profiles.pdf
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	bibtex main >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	@grep -E "Warning: (Citation|Reference)" main.log || echo "no undefined citations or references"

paper.bib: ../paper.bib
	cp ../paper.bib paper.bib

$(AASTEX):
	curl -sL -o aastex.zip https://mirrors.ctan.org/macros/latex/contrib/aastex.zip
	unzip -oqj aastex.zip 'aastex/aastex701.cls' 'aastex/aasjournalv7.bst'
	rm -f aastex.zip

figures/profiles.pdf:
	cd .. && python make_figures.py

figures:
	cd .. && python make_figures.py

arxiv: main.pdf
	tar czf astrovista-lite-arxiv.tar.gz main.tex main.bbl $(AASTEX) figures/*.pdf figures/*.png
	@echo "upload astrovista-lite-arxiv.tar.gz to arXiv"

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf paper.bib $(AASTEX) astrovista-lite-arxiv.tar.gz

.PHONY: arxiv clean figures
