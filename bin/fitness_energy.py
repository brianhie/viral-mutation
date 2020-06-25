from utils import *

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
        train_fname = 'target/hiv/clusters/all_BF520.fasta'
        mut_fname = 'target/hiv/mutation/mutations_bf520.fa'
        energy_fname = 'target/hiv/clusters/all_BF520.fasta.E.txt'
    elif virus == 'bg505':
        from combinatorial_fitness import load_haddox2018
        strains, seqs_fitness = load_haddox2018()
        strain = 'BG505'
        train_fname = 'target/hiv/clusters/all_BG505.fasta'
        mut_fname = 'target/hiv/mutation/mutations_hiv.fa'
        energy_fname = 'target/hiv/clusters/all_BG505.fasta.E.txt'
    else:
        raise ValueError('invalid option {}'.format(virus))

    train_seqs = list(SeqIO.parse(train_fname, 'fasta'))
    mut_seqs = list(SeqIO.parse(mut_fname, 'fasta'))
    energies = np.loadtxt(energy_fname)
    assert(len(energies) == len(train_seqs) + len(mut_seqs))

    mut2energy = { str(seq.seq).replace('-', ''): -energy
                   for seq, energy in
                   zip(mut_seqs, energies[len(train_seqs):]) }

    fitnesses, predictions = [], []
    for mut_seq, strain in seqs_fitness:
        if mut_seq not in mut2energy:
            continue
        fitnesses.append(seqs_fitness[(mut_seq, strain)][0]['preference'])
        predictions.append(mut2energy[mut_seq])

    print_result(fitnesses, predictions, virus + ' (Potts)')

def fitness_evcouplings(virus):
    if virus == 'h1':
        from combinatorial_fitness import load_doud2016
        strains, seqs_fitness = load_doud2016()
        strain = 'h1'
        train_fname = 'target/flu/clusters/all_h1.fasta'
        energy_fname = ('target/flu/evcouplings/flu_h1/mutate/'
                        'flu_h1_single_mutant_matrix.csv')
        anchor_id = ('gb:LC333185|ncbiId:BBB04702.1|UniProtKB:-N/A-|'
                     'Organism:Influenza')
    elif virus == 'bf520':
        from combinatorial_fitness import load_haddox2018
        strains, seqs_fitness = load_haddox2018()
        strain = 'BF520'
        train_fname = 'target/hiv/clusters/all_BF520.fasta'
        energy_fname = ('target/hiv/evcouplings/hiv_bf520/mutate/'
                        'hiv_bf520_single_mutant_matrix.csv')
        anchor_id = 'A1.KE.1994.BF520.W14M.C2.KX168094'
    elif virus == 'bg505':
        from combinatorial_fitness import load_haddox2018
        strains, seqs_fitness = load_haddox2018()
        strain = 'BG505'
        train_fname = 'target/hiv/clusters/all_BG505.fasta'
        energy_fname = ('target/hiv/evcouplings/hiv_env/mutate/'
                        'hiv_env_single_mutant_matrix.csv')
        anchor_id = 'A1.KE.-.BG505_W6M_ENV_C2.DQ208458'
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

    mutations = [ mut_seq for mut_seq, strain_i in seqs_fitness
                  if strain_i == strain ]

    mut_scores_epi, mut_scores_ind = [], []
    fitnesses = []
    for mut_idx, mutation in enumerate(mutations):
        mutation = mutation.replace('-', '')
        if mutation == anchor:
            continue
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
        anchor_id = ('gb:LC333185|ncbiId:BBB04702.1|UniProtKB:-N/A-|'
                     'Organism:Influenza')
    elif virus == 'bf520':
        from combinatorial_fitness import load_haddox2018
        strains, seqs_fitness = load_haddox2018()
        strain = 'BF520'
        train_fname = 'target/hiv/clusters/all_BF520.fasta'
        mut_fname = 'target/hiv/mutation/mutations_bf520.fa'
        anchor_id = 'A1.KE.1994.BF520.W14M.C2.KX168094'
    elif virus == 'bg505':
        from combinatorial_fitness import load_haddox2018
        strains, seqs_fitness = load_haddox2018()
        strain = 'BG505'
        train_fname = 'target/hiv/clusters/all_BG505.fasta'
        mut_fname = 'target/hiv/mutation/mutations_hiv.fa'
        anchor_id = 'A1.KE.-.BG505_W6M_ENV_C2.DQ208458'
    else:
        raise ValueError('invalid option {}'.format(virus))

    anchor = None
    pos_aa_freq = {}
    for idx, record in enumerate(SeqIO.parse(train_fname, 'fasta')):
        mutation = record.seq
        if record.id == anchor_id:
            anchor = mutation
        else:
            for pos, c in enumerate(mutation):
                if c == '-':
                    continue
                if (pos, c) not in pos_aa_freq:
                    pos_aa_freq[(pos, c)] = 0.
                pos_aa_freq[(pos, c)] += 1.
    assert(anchor is not None)

    mutations = [
        str(record.seq) for record in SeqIO.parse(mut_fname, 'fasta')
    ]

    mut_freqs, fitnesses = [], []
    for mut_idx, mutation in enumerate(mutations):
        if mutation == anchor:
            continue
        mutation_clean = str(mutation).replace('-', '')
        if (mutation_clean, strain) not in seqs_fitness:
            continue
        didx = [ c1 != c2
                 for c1, c2 in zip(anchor, mutation) ].index(True)
        if (didx, mutation[didx]) in pos_aa_freq:
            mut_freqs.append(pos_aa_freq[(didx, mutation[didx])])
        else:
            mut_freqs.append(0)
        fitnesses.append(
            seqs_fitness[(mutation_clean, strain)][0]['preference']
        )
    mut_freqs = np.array(mut_freqs)

    print_result(mut_freqs, fitnesses, virus + ' (freq)')

if __name__ == '__main__':
    args = parse_args()

    if args.method == 'energy':
        fitness_energy(args.virus)

    elif args.method == 'evcouplings':
        fitness_evcouplings(args.virus)

    elif args.method == 'freq':
        fitness_freq(args.virus)
