cutoffs=(-2.4 -2.3 -2.2 -2.1 -2 -1.9 -1.8 -1.7 -1.6 -1.5 -1.4 -1.3 -1.2 -1.1 -1 -.9 -.8 -.7 -.6 -.5 -.4 -.3 -.2 -.1 )
virus="cov2rbd"
cache="results/cov/semantics/analyze_semantics_cov2rbd_bilstm_512.txt"

for cutoff in ${cutoffs[@]}
do
    echo "cutoff = "$cutoff
    python bin/cached_semantics.py $cache 0.3 $cutoff
done
