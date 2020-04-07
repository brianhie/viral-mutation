#for file in target/clusters/*.fa
#do
#    /usr/local/bin/mafft --thread 40 --auto --inputorder $file > $file.clustal
#done
#
#CONS_DIR=/scratch2/brianhie/conservation_code
#for file in target/clusters/*.clustal
#do
#    sed -i 's/B/N/g' $file
#    sed -i 's/J/I/g' $file
#    python2 $CONS_DIR/score_conservation.py -m $CONS_DIR/matrix/blosum62.bla $file > $file.cons
#done

for file in target/flu/clusters/*.cons
do
    echo $file
    cut -f2 $file | tail -n+3 | awk '$1 >= 0' | awk '{ total += $1 } END { print total/NR }'
    cut -f2 $file | tail -n+3 | awk '$1 >= 0' | sort -n | head -n1
done
