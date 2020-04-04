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


if __name__ == '__main__':
    print('H1 escape...')
    msa_subset(
        'target/flu/clusters/all.fasta',
        'target/flu/clusters/all_h1.fasta',
        'gb:LC333185|ncbiId:BBB04702.1|UniProtKB:-N/A-|'
        'Organism:Influenza', 4
    )
    print('H3 escape...')
    msa_subset(
        'target/flu/clusters/all.fasta',
        'target/flu/clusters/all_h3.fasta',
        'gb:GQ293081|ncbiId:ACS71642.1|UniProtKB:C6KNH7|'
        'Organism:Influenza', 0
    )
    print('HIV escape...')
    msa_subset(
        'target/hiv/clusters/all.fasta',
        'target/hiv/clusters/all_BG505.fasta',
        'A1.KE.-.BG505_W6M_ENV_C2.DQ208458', 25
    )
