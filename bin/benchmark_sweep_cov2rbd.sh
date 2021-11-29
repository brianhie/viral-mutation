cutoffs=(0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)
virus="cov2rbd"
cache="results/cov/semantics/analyze_semantics_cov2rbd_bilstm_512.txt"

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
