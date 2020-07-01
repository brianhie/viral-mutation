from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
import sys

def msa_subset(ifname, ofname, anchor_id, cutoff=0):
    align = AlignIO.read(ifname, 'fasta')

    anchor_idx = [ align[i].id
                   for i in range(len(align)) ].index(anchor_id)
    anchor = align[anchor_idx]

    subset = []
    for idx, record in enumerate(align):
        if idx % 10000 == 9999:
            print('Record {}...'.format(idx + 1))
        n_diff = sum([
            (x1 != '-' and x2 == '-')
            for x1, x2 in zip(anchor, align[idx])
        ])
        if n_diff <= cutoff:
            subset.append(align[idx])

    print('Found {} records'.format(len(subset)))

    align_subset = MultipleSeqAlignment(subset)
    AlignIO.write(align_subset, ofname, 'fasta')

    return str(anchor.seq)

def create_mutants(aligned_str):
    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V', 'U',
    ]

    mutants, mutant_names = [], []
    for i in range(len(aligned_str)):
        if aligned_str[i] == '-':
            continue
        for aa in AAs:
            if aligned_str[i] == aa:
                continue
            name = 'mut_{}_{}'.format(i, aa)
            mutable = aligned_str[:i] + aa + aligned_str[i + 1:]
            mutant_names.append(name)
            mutants.append(mutable)
    return mutants, mutant_names

def write_mutants(mutants, mutant_names, outfile):
    with open(outfile, 'w') as of:
        for mutant, name in zip(mutants, mutant_names):
            of.write('>{}\n'.format(name))
            of.write('{}\n'.format(mutant))

if __name__ == '__main__':
    print('H1...')
    anchor = msa_subset(
        'target/flu/clusters/all.fasta',
        'target/flu/clusters/all_h1.fasta',
        'gb:LC333185|ncbiId:BBB04702.1|UniProtKB:-N/A-|'
        'Organism:Influenza', 2
    )
    mutants, mutant_names = create_mutants(anchor)
    write_mutants(mutants, mutant_names,
                  'target/flu/mutation/mutations_h1.fa')

    print('H3...')
    anchor = msa_subset(
        'target/flu/clusters/all.fasta',
        'target/flu/clusters/all_h3.fasta',
        'Reference_Perth2009_HA_coding_sequence', 0
    )
    mutants, mutant_names = create_mutants(anchor)
    write_mutants(mutants, mutant_names,
                  'target/flu/mutation/mutations_h3.fa')

    print('HIV BG505...')
    anchor = msa_subset(
        'target/hiv/clusters/all.fasta',
        'target/hiv/clusters/all_BG505.fasta',
        'A1.KE.-.BG505_W6M_ENV_C2.DQ208458', 15
    )
    mutants, mutant_names = create_mutants(anchor)
    write_mutants(mutants, mutant_names,
                  'target/hiv/mutation/mutations_hiv.fa')

    print('HIV BF520...')
    anchor = msa_subset(
        'target/hiv/clusters/all.fasta',
        'target/hiv/clusters/all_BF520.fasta',
        'A1.KE.1994.BF520.W14M.C2.KX168094', 15
    )
    mutants, mutant_names = create_mutants(anchor)
    write_mutants(mutants, mutant_names,
                  'target/hiv/mutation/mutations_bf520.fa')

    print('SARS-CoV-2...')
    anchor = msa_subset(
        'target/cov/clusters/all.fasta',
        'target/cov/clusters/all_sarscov2.fasta',
        'YP_009724390.1', 0
    )
    mutants, mutant_names = create_mutants(anchor)
    write_mutants(mutants, mutant_names,
                  'target/cov/mutation/mutations_sarscov2.fa')
