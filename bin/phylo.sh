# MAFFT tree

/usr/local/bin/mafft \
    --thread 40 --auto --treeout --inputorder \
    data/influenza/all.fa \
    > target/flu/clusters/all_nice.fasta

/usr/local/bin/mafft \
    --thread 40 --auto --treeout --inputorder \
    data/hiv/all.fa \
    > target/hiv/clusters/all_nice.fasta

# Clustal Omega tree

clustal \
    -i data/influenza/all.fa \
    --threads 40 \
    --guidetree-out target/flu/clusters/clustal_omega.guidetree.txt \
    --clustering-out target/flu/clusters/clustal_omega.cluster.txt \
    -o target/flu/clusters/clustal_omega.align.txt --outfmt clu \
    > clustalomega_flu.log 2>&1
clustal \
    -i data/hiv/all.fa \
    --threads 40 \
    --guidetree-out target/hiv/clusters/clustal_omega.guidetree.txt \
    --clustering-out target/hiv/clusters/clustal_omega.cluster.txt \
    -o target/hiv/clusters/clustal_omega.align.txt --outfmt clu \
    > clustalomega_hiv.log 2>&1

python bin/clustal2newick.py \
       target/flu/clusters/clustal_omega.guidetree.txt \
       target/flu/clusters/clustal_omega.newick
python bin/clustal2newick.py \
       target/hiv/clusters/clustal_omega.guidetree.txt \
       target/hiv/clusters/clustal_omega.newick

# RAxML tree

python bin/fasta2phylip.py target/flu/clusters/all_nice.fasta target/flu/clusters/all.phylip
python bin/fasta2phylip.py target/hiv/clusters/all_nice.fasta target/hiv/clusters/all.phylip
raxml -T 40 -D -F -m PROTCATBLOSUM62 \
      -s target/flu/clusters/all.phylip -f E \
      -n raxml_flu.tree -p 1 \
      > raxml_flu.log 2>&1
raxml -T 40 -D -F -m PROTCATBLOSUM62 \
      -s target/hiv/clusters/all.phylip -f E \
      -n raxml_hiv.tree -p 1 \
      > raxml_hiv.log 2>&1

# MrBayes

python bin/fasta2nexus.py target/flu/clusters/all_nice.fasta target/flu/clusters/all.nex
python bin/fasta2nexus.py target/hiv/clusters/all_nice.fasta target/hiv/clusters/all.nex

# FastTree

FastTree -fastest target/flu/clusters/all_nice.fasta > target/flu/clusters/fasttree.tree
FastTree -fastest target/hiv/clusters/all_nice.fasta > target/hiv/clusters/fasttree.tree
