#declare -a thresholds=(0.15 0.14 0.13 0.12 0.11 0.10)
#
#for threshold in ${thresholds[@]}
#do
#    python bin/TreeCluster.py \
#           -t $threshold \
#           -i target/flu/clusters/all_oneline.tree \
#           -o target/flu/clusters/all.clusters_$threshold.txt
#done

declare -a thresholds=(0.30 0.26 0.23 0.20 0.18)

for threshold in ${thresholds[@]}
do
    python bin/TreeCluster.py \
           -t $threshold \
           -i target/hiv/clusters/all_oneline.tree \
           -o target/hiv/clusters/all.clusters_$threshold.txt
done
