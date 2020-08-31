from utils import *

def load_to2020():
    seq = str(SeqIO.read('data/cov/cov2_spike_wt.fasta', 'fasta').seq)
    seq = seq[:779] + 'Q' + seq[780:]

    muts = [
        [ 'L18F', 'A222V', 'D614G', 'Q780E' ],
    ]

    mutants = { 4: [] }
    for i in range(len(muts)):
        assert(len(muts[i]) == 4)
        mutable = seq
        for mut in muts[i]:
            aa_orig = mut[0]
            aa_mut = mut[-1]
            pos = int(mut[1:-1]) - 1
            assert(seq[pos] == aa_orig)
            mutable = mutable[:pos] + aa_mut + mutable[pos + 1:]
        mutants[4].append(mutable)

    return seq, mutants

if __name__ == '__main__':
    load_to2020()
