for input in ./*.fig
do
    echo '-----------------------------------'
    echo 'compiling' ${input}
    rm -f z.tex z.pdf
    fig2dev -Lpdftex  -m.5 ${input} z.pdf
    fig2dev -Lpdftex_t -m.5 -p z.pdf ${input} z.tex
    pdflatex fig2pdf.tex
    pdfcrop fig2pdf.pdf ${input%.*}.pdf
done
