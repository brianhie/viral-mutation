# MAFFT tree

/usr/local/bin/mafft \
    --thread 40 --auto --inputorder \
    data/influenza/ird_influenzaA_HA_allspecies.fa \
    > target/flu/clusters/all.fasta

/usr/local/bin/mafft \
    --thread 40 --auto --inputorder \
    data/hiv/HIV-1_env_samelen.fa \
    > target/hiv/clusters/all.fasta

# Clustal Omega tree

clustal \
    -i data/influenza/ird_influenzaA_HA_allspecies.fa \
    --threads 40 \
    --guidetree-out target/flu/clusters/clustal_omega.guidetree.txt \
    --clustering-out target/flu/clusters/clustal_omega.cluster.txt \
    -o target/flu/clusters/clustal_omega.align.txt --outfmt clu \
    > clustalomega_flu.log 2>&1

clustal \
    -i data/hiv/HIV-1_env_samelen.fa \
    --threads 40 \
    --guidetree-out target/hiv/clusters/clustal_omega.guidetree.txt \
    --clustering-out target/hiv/clusters/clustal_omega.cluster.txt \
    -o target/hiv/clusters/clustal_omega.align.txt --outfmt clu \
    > clustalomega_hiv.log 2>&1

# PhyML tree

python bin/fasta2phylip.py target/flu/clusters/all.fasta target/flu/all.phylip
python bin/fasta2phylip.py target/hiv/clusters/all.fasta target/hiv/all.phylip
phyml -i target/flu/all.phylip -d aa > phyml_flu.log 2>&1
phyml -i target/hiv/all.phylip -d aa > phyml_hiv.log 2>&1
