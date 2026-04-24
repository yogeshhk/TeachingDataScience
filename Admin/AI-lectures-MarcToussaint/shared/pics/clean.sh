for input in ./*.fig
do
    echo '-----------------------------------' $input
    rm ${input%.*}.pdf
done
