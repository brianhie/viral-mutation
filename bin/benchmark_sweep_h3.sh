cutoffs=(0. 0.25 0.5 0.75 1. 1.25 1.5 1.75 2. 2.25 2.5 2.75 3. 3.25 3.5 4. 4.5 5. 5.5 6. 6.5 7)
virus="h3"
cache="results/flu/semantics/analyze_semantics_flu_h3_bilstm_512.txt"

declare -a methods=("bepler" "evcouplings" "freq" "tape" "unirep")

for cutoff in ${cutoffs[@]}
do
    echo "cutoff = "$cutoff

    for method in ${methods[@]}
    do
        python bin/escape_energy.py $method $virus --cutoff $cutoff
    done

    python bin/cached_semantics.py $cache $cutoff
done
