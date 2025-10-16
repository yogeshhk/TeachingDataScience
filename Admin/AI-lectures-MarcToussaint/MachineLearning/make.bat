@echo off
for /r %%i in (*.tex) do texify -cp --tex-option="--interaction=nonstopmode" %%i
makeindex script.idx
pdflatex -interaction=nonstopmode script.tex > /dev/null