cutoffs=(0 0.02 0.04 0.06 0.08 0.10 0.11 0.12 0.14 0.16 0.18 0.2 0.22 0.24)
virus="bg505"
cache="results/hiv/semantics/analyze_semantics_hiv_bilstm_512.txt"

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
