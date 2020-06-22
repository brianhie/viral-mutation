from utils import *

from Bio import SeqIO
from sklearn.metrics import auc

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark methods')
    parser.add_argument('method', type=str,
                        help='Benchmarking method to use')
    parser.add_argument('virus', type=str,
                        help='Viral type to test')
    args = parser.parse_args()
    return args

def print_result(fitnesses, predictions, virus):
    tprint('Results for {}:'.format(virus))
    tprint('Spearman r = {}, P = {}'
           .format(*ss.spearmanr(fitnesses, predictions)))
    tprint('Pearson r = {}, P = {}'
           .format(*ss.pearsonr(fitnesses, predictions)))

def fitness_energy(virus):
    if virus == 'h1':
        from combinatorial_fitness import load_doud2016
        strains, seqs_fitness = load_doud2016()
        strain = 'h1'
        train_fname = 'target/flu/clusters/all_h1.fasta'
        mut_fname = 'target/flu/mutation/mutations_h1.fa'
        energy_fname = 'target/flu/clusters/all_h1.fasta.E.txt'
    elif virus == 'bf520':
        from combinatorial_fitness import load_haddox2018
        strains, seqs_fitness = load_haddox2018()
        strain = 'BF520'
        train_fname = ''
        mut_fname = ''
        energy_fname = ''
    elif virus == 'bg505':
        from combinatorial_fitness import load_haddox2018
        strains, seqs_fitness = load_haddox2018()
        strain = 'BG505'
        train_fname = 'target/hiv/clusters/all_BG505.fasta'
        mut_fname = 'target/hiv/mutation/mutations_hiv.fa'
        energy_fname = 'target/hiv/clusters/all_BG505.fasta.E.txt'
    elif virus == 'cov2':
        from combinatorial_fitness import load_starr2020
        strains, seqs_fitness = load_starr2020()
        strain = 'sars_cov_2'
        train_fname = ''
        mut_fname = ''
        energy_fname = ''
    else:
        raise ValueError('invalid option {}'.format(virus))

    train_seqs = list(SeqIO.parse(train_fname, 'fasta'))
    mut_seqs = list(SeqIO.parse(mut_fname, 'fasta'))
    energies = np.loadtxt(energy_fname)
    assert(len(energies) == len(train_seqs) + len(mut_seqs))

    fitnesses = [
        seqs_fitness[(mut_seq, strain)][0]['preference']
        for mut_seq in mut_seqs
    ]

    predictions = energies[len(train_seqs):]

    print_result(fitnesses, predictions, virus + ' (Potts)')

def fitness_evcouplings(virus):
    if virus == 'h1':
        from combinatorial_fitness import load_doud2016
        strains, seqs_fitness = load_doud2016()
        strain = 'h1'
        train_fname = 'target/flu/clusters/all_h1.fasta'
        mut_fname = 'target/flu/mutation/mutations_h1.fa'
        energy_fname = ('target/flu/evcouplings/flu_h1/mutate/'
                        'flu_h1_single_mutant_matrix.csv')
        anchor_id = ('gb:LC333185|ncbiId:BBB04702.1|UniProtKB:-N/A-|'
                     'Organism:Influenza')
    elif virus == 'bf520':
        from combinatorial_fitness import load_haddox2018
        strains, seqs_fitness = load_haddox2018()
        strain = 'BF520'
        train_fname = ''
        mut_fname = ''
        energy_fname = ''
        anchor_id = ''
    elif virus == 'bg505':
        from combinatorial_fitness import load_haddox2018
        strains, seqs_fitness = load_haddox2018()
        strain = 'BG505'
        train_fname = 'target/hiv/clusters/all_BG505.fasta'
        mut_fname = 'target/hiv/mutation/mutations_hiv.fa'
        energy_fname = ('target/hiv/evcouplings/hiv_env/mutate/'
                        'hiv_env_single_mutant_matrix.csv')
        anchor_id = 'A1.KE.-.BG505_W6M_ENV_C2.DQ208458'
    elif virus == 'cov2':
        from combinatorial_fitness import load_starr2020
        strains, seqs_fitness = load_starr2020()
        strain = 'sars_cov_2'
        train_fname = ''
        mut_fname = ''
        energy_fname = ''
        anchor_id = ''
    else:
        raise ValueError('invalid option {}'.format(virus))

    anchor = None
    for idx, record in enumerate(SeqIO.parse(train_fname, 'fasta')):
        if record.id == anchor_id:
            anchor = str(record.seq).replace('-', '')
    assert(anchor is not None)

    pos_aa_score_epi = {}
    pos_aa_score_ind = {}
    with open(energy_fname) as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(',')
            pos = int(fields[2]) - 1
            orig, mut = fields[3], fields[4]
            assert(anchor[pos] == orig)
            pos_aa_score_epi[(pos, mut)] = float(fields[7])
            pos_aa_score_ind[(pos, mut)] = float(fields[8])

    mutations = [
        str(record.seq)
        for record in SeqIO.parse(mut_fname, 'fasta')
    ]

    mut_scores_epi, mut_scores_ind = [], []
    fitnesses = []
    for mut_idx, mutation in enumerate(mutations):
        mutation = mutation.replace('-', '')
        didx = [ c1 != c2
                 for c1, c2 in zip(anchor, mutation) ].index(True)
        if (didx, mutation[didx]) in pos_aa_score_epi:
            mut_scores_epi.append(pos_aa_score_epi[(didx, mutation[didx])])
            mut_scores_ind.append(pos_aa_score_ind[(didx, mutation[didx])])
        else:
            mut_scores_epi.append(0)
            mut_scores_ind.append(0)
        fitnesses.append(seqs_fitness[(mutation, strain)][0]['preference'])

    print_result(mut_scores_epi, fitnesses,
                 virus + ' (EVcouplings epistatic)')
    print_result(mut_scores_ind, fitnesses,
                 virus + ' (EVcouplings independent)')

def fitness_freq(virus):
    if virus == 'h1':
        from combinatorial_fitness import load_doud2016
        strains, seqs_fitness = load_doud2016()
        strain = 'h1'
        train_fname = 'target/flu/clusters/all_h1.fasta'
        mut_fname = 'target/flu/mutation/mutations_h1.fa'
    elif virus == 'bf520':
        from combinatorial_fitness import load_haddox2018
        strains, seqs_fitness = load_haddox2018()
        strain = 'BF520'
        train_fname = ''
        mut_fname = ''
    elif virus == 'bg505':
        from combinatorial_fitness import load_haddox2018
        strains, seqs_fitness = load_haddox2018()
        strain = 'BG505'
        train_fname = 'target/hiv/clusters/all_BG505.fasta'
        mut_fname = 'target/hiv/mutation/mutations_hiv.fa'
    elif virus == 'cov2':
        from combinatorial_fitness import load_starr2020
        strains, seqs_fitness = load_starr2020()
        strain = 'sars_cov_2'
        train_fname = ''
        mut_fname = ''
    else:
        raise ValueError('invalid option {}'.format(virus))

    anchor = None
    pos_aa_freq = {}
    for idx, record in enumerate(SeqIO.parse(train_fname, 'fasta')):
        mutation = str(record.seq)
        if record.id == anchor_id:
            anchor = mutation
        else:
            for pos, c in enumerate(mutation):
                if (pos, c) not in pos_aa_freq:
                    pos_aa_freq[(pos, c)] = 0.
                pos_aa_freq[(pos, c)] += 1.
    assert(anchor is not None)

    mutations = [
        str(record.seq)
        for record in SeqIO.parse(mut_fname, 'fasta')
    ]

    mut_freqs, fitnesses = [], []
    for mut_idx, mutation in enumerate(mutations):
        fitnesses.append()
        didx = [ c1 != c2
                 for c1, c2 in zip(anchor, mutation) ].index(True)
        if (didx, mutation[didx]) in pos_aa_freq:
            mut_freqs.append(pos_aa_freq[(didx, mutation[didx])])
        else:
            mut_freqs.append(0)
        fitnesses.append(seqs_fitness[(mutation, strain)][0]['preference'])
    mut_freqs = np.array(mut_freqs)
    assert(len(fitness_idx) == len(seqs_escape) - 1)

    print_result(mut_freqs, fitnesses, virus + ' (freq)')

if __name__ == '__main__':
    args = parse_args()

    if args.method == 'energy':
        fitness_energy(args.virus)

    elif args.method == 'evcouplings':
        fitness_evcouplings(args.virus)

    elif args.method == 'freq':
        fitness_freq(args.virus)
