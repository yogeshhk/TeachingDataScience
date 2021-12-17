for input in ./*.tex
do
    echo '-----------------------------------'
    echo 'compiling' ${input}
    pdflatex -interaction=nonstopmode ${input} > /dev/null
    grep "Warning" ${input%.*}.log
    grep "Missing" ${input%.*}.log
    grep -A2 "Undefined control sequence" ${input%.*}.log
    grep -A2 "Error" ${input%.*}.log
done

echo '-----------------------------------'
echo 'compiling script'
makeindex script.idx
pdflatex -interaction=nonstopmode script.tex > /dev/null
