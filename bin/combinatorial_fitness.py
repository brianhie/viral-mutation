from Bio import SeqIO

def load_wu2020():
    mut_pos = [
        156, 158, 159, 190, 193, 196
    ]
    offset = 16 # Amino acids in prefix.
    mut_pos = [ pos - 1 + offset for pos in mut_pos ]

    names = [
        'HK68', 'Bk79', 'Bei89', 'Mos99', 'Bris07L194', 'NDako16',
    ]
    wildtypes = [
        'KGSESV', 'EESENV', 'EEYENV', 'QKYDST', 'HKFDFA', 'HNSDFA',
    ]

    # Load full wildtype sequences.

    wt_seqs = {}
    fname = 'data/influenza/fitness_wu2020/wildtypes.fa'
    for record in SeqIO.parse(fname, 'fasta'):
        strain_idx = names.index(record.description)
        wt = wildtypes[strain_idx]
        for aa, pos in zip(wt, mut_pos):
            assert(record.seq[pos] == aa)
        wt_seqs[names[strain_idx]] = record.seq

    # Load mutants.

    seqs_fitness = {}
    fname = 'data/influenza/fitness_wu2020/data_pref.tsv'
    with open(fname) as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split('\t')
            mut, strain, fitness, preference = fields
            if strain == 'Bris07P194':
                continue
            if strain == 'Bris07':
                strain = 'Bris07L194'
            fitness = float(preference)
            preference = float(preference)

            strain_idx = names.index(strain)
            wt = wildtypes[strain_idx]
            full_seq = wt_seqs[strain]

            mutable = [ aa for aa in full_seq ]
            for aa_wt, aa, pos in zip(wt, mut, mut_pos):
                assert(mutable[pos] == aa_wt)
                mutable[pos] = aa
            mut_seq = ''.join(mutable)

            if mut_seq not in seqs_fitness:
                seqs_fitness[mut_seq] = []
            seqs_fitness[mut_seq].append({
                'strain': strain,
                'fitness': fitness,
                'preference': preference,
                'wildtype': full_seq,
                'mut_pos': mut_pos,
            })

    return wt_seqs, seqs_fitness

if __name__ == '__main__':
    load_wu2020()
