
all: prepare

prepare:
	pdflatex paper.tex

view: prepare
	open -a Skim paper.pdf

develop: prepare
	fswatch -i 'paper.tex' -e '.*' . | xargs -t -n1 -I % bash -c "pdflatex % || osascript -e 'display notification \"Latex compilation failed\" with title \"ERROR\"'"

clean:
	rm -rf *.aux *.listing *.pdf *.out *.log
