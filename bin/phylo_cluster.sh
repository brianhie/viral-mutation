declare -a thresholds=(0.119 0.117 0.115 0.113 0.111)

for threshold in ${thresholds[@]}
do
    python bin/TreeCluster.py \
           -t $threshold \
           -i target/flu/clusters/all_oneline.tree \
           -o target/flu/clusters/all.clusters_$threshold.txt
done

declare -a thresholds=(0.449 0.445 0.443 0.44)

for threshold in ${thresholds[@]}
do
    python bin/TreeCluster.py \
           -t $threshold \
           -i target/hiv/clusters/all_oneline.tree \
           -o target/hiv/clusters/all.clusters_$threshold.txt
done
