OPTIONS = -pdf --shell-escape -use-make #-e '$$max_repeat=7'
MAIN = main
#MAIN = iclr


.PHONY :  all clean show publish $(MAIN).*

all: $(MAIN).pdf

open: $(MAIN).pdf
	open $(MAIN).pdf

edit:
	vim $(MAIN).tex

# no, %.pdf doesn't get phony'd so this is the only way..
#
$(MAIN).pdf: $(MAIN).tex
	latexmk $(OPTIONS) $<

# Sometimes, a good cleanup is required to get rid of some errors
clean:
	#latexmk -C # -c is simple clean
	latexmk -CA # -CA is better clean
	rm -rf svg-inkscape
	rm -rf $(MAIN).bbl
	#rm -rf *.bbl #why not
	rm -rf *.snm
	rm -rf *.nav
	rm -rf *.aux
	rm -rf *.log
	rm -rf _minted-$(MAIN)
	rm -rf comment.cut
	rm -rf *.run.xml
	# it just never ends does it? No, it's not covered by latexmk.
	rm -rf *.gls *.glg *.ist *.glo pdflatex*.fls

