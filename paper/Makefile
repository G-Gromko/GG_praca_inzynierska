SRC = inzynierska.tex
AUX = inzynierska.aux

inzynierska.pdf: $(SRC)
	pdflatex $<
	pdflatex $<
	bibtex $(AUX)
	pdflatex $<
	pdflatex $<

clean:
	rm -f *~ *.{pdf,log,lof,lol,lot,aux,blg,bbl,toc,nav,out,snm}

