
/usr/local/bin/mafft \
    --thread 40 --auto --inputorder \
    data/influenza/ird_influenzaA_HA_allspecies.fa \
    > target/flu/clusters/all.fasta

/usr/local/bin/mafft \
    --thread 40 --auto --inputorder \
    data/hiv/HIV-1_env_samelen.fa \
    > target/hiv/clusters/all.fasta

python bin/benchmark_subset.py

python bin/escape_energy.py

cd MPF-BML/
matlab -r "fasta_name = '../target/flu/clusters/all.fasta'; mut_name = '../target/flu/mutation/mutations_h1.fa'; main_MPF_BML(fasta_name, mut_name)"
matlab -r "fasta_name = '../target/flu/clusters/all.fasta'; mut_name = '../target/flu/mutation/mutations_h3.fa'; main_MPF_BML(fasta_name, mut_name)"
matlab -r "fasta_name = '../target/hiv/clusters/all_BG505.fasta'; mut_name = '../target/hiv/mutation/mutations_hiv.fa'; main_MPF_BML(fasta_name, mut_name)"
