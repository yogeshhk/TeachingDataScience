for input in ./*.fig
do
    echo '-----------------------------------'
    rm ${input%.*}.pdf
done
