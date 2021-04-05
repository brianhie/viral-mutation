cutoffs=(0 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3)
virus="h1"
cache="results/flu/semantics/analyze_semantics_flu_h1_bilstm_512.txt"

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
