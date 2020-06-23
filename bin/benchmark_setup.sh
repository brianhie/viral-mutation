
/usr/local/bin/mafft \
    --thread 40 --auto --inputorder \
    data/influenza/ird_influenzaA_HA_allspecies.fa \
    > target/flu/clusters/all.fasta

/usr/local/bin/mafft \
    --thread 40 --auto --inputorder \
    data/hiv/HIV-1_env_samelen.fa \
    > target/hiv/clusters/all.fasta

python bin/benchmark_subset.py

##########################
## Potts (Louie et al.) ##
##########################

cd MPF-BML/
matlab -r "fasta_name = '../target/flu/clusters/all_h1.fasta'; mut_name = '../target/flu/mutation/mutations_h1.fa'; main_MPF_BML(fasta_name, mut_name)"
matlab -r "fasta_name = '../target/flu/clusters/all_h3.fasta'; mut_name = '../target/flu/mutation/mutations_h3.fa'; main_MPF_BML(fasta_name, mut_name)"
matlab -r "fasta_name = '../target/hiv/clusters/all_BG505.fasta'; mut_name = '../target/hiv/mutation/mutations_hiv.fa'; main_MPF_BML(fasta_name, mut_name)"
matlab -r "fasta_name = '../target/hiv/clusters/all_BF520.fasta'; mut_name = '../target/hiv/mutation/mutations_bf520.fa'; main_MPF_BML(fasta_name, mut_name)"
cd ..

#########################
## Potts (EVcouplings) ##
#########################

evcouplings_runcfg data/evcouplings/flu_h1_config.yaml > \
                   evcouplings_flu_h1.log 2>&1
evcouplings_runcfg data/evcouplings/flu_h3_config.yaml > \
                   evcouplings_flu_h3.log 2>&1
evcouplings_runcfg data/evcouplings/hiv_env_config.yaml > \
                   evcouplings_hiv_env.log 2>&1
evcouplings_runcfg data/evcouplings/hiv_bf520_config.yaml > \
                   evcouplings_hiv_bf520.log 2>&1

mkdir -p target/flu/evcouplings
mkdir -p target/hiv/evcouplings
mv flu_h1 target/flu/evcouplings/
mv flu_h3 target/flu/evcouplings/
mv hiv_env target/hiv/evcouplings/
mv hiv_bf520 target/hiv/evcouplings/

######################
## TAPE Transformer ##
######################

sed 's/-//g' target/flu/mutation/mutations_h1.fa > \
    target/flu/mutation/mutations_clean_h1.fasta
sed 's/-//g' target/flu/mutation/mutations_h3.fa > \
    target/flu/mutation/mutations_clean_h3.fasta
sed 's/-//g' target/hiv/mutation/mutations_hiv.fa > \
    target/hiv/mutation/mutations_clean_hiv.fasta

tape-embed transformer \
           target/flu/mutation/mutations_clean_h1.fasta \
           target/flu/embedding/tape_transformer_h1.npz \
           bert-base \
           --tokenizer iupac \
           --batch_size 256
tape-embed transformer \
           target/flu/mutation/mutations_clean_h3.fasta \
           target/flu/embedding/tape_transformer_h3.npz \
           bert-base \
           --tokenizer iupac \
           --batch_size 256
tape-embed transformer \
           target/hiv/mutation/mutations_clean_hiv.fasta \
           target/hiv/embedding/tape_transformer_hiv.npz \
           bert-base \
           --tokenizer iupac \
           --batch_size 128

tape-embed unirep \
           target/flu/mutation/mutations_clean_h1.fasta \
           target/flu/embedding/unirep_h1.npz \
           babbler-1900 \
           --tokenizer unirep \
           --batch_size 256
tape-embed unirep \
           target/flu/mutation/mutations_clean_h3.fasta \
           target/flu/embedding/unirep_h3.npz \
           babbler-1900 \
           --tokenizer unirep \
           --batch_size 256
tape-embed unirep \
           target/hiv/mutation/mutations_clean_hiv.fasta \
           target/hiv/embedding/unirep_hiv.npz \
           babbler-1900 \
           --tokenizer unirep \
           --batch_size 128

##########################
## Fitness calculations ##
##########################

declare -a methods=("energy" "evcouplings" "freq")
declare -a viruses=("h1" "bf520" "bf520")

for method in ${methods[@]}
do
    for virus in ${viruses[@]}
    do
        python bin/fitness_energy.py $method $virus
    done
done

#########################
## Escape calculations ##
#########################

declare -a methods=("bepler" "energy" "evcouplings" "freq" "tape" "unirep")
declare -a viruses=("h1" "h3" "hiv")

for method in ${methods[@]}
do
    for virus in ${viruses[@]}
    do
        python bin/escape_energy.py $method $virus
    done
done
