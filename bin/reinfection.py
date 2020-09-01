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

def load_ratg13():
    seq = str(SeqIO.read('data/cov/cov2_spike_wt.fasta', 'fasta').seq)

    muts = [
        [ 'N439K', 'Y449F', 'F486L', 'Q493Y',
          'S494R', 'Q498Y', 'N501D', 'Y505H' ],
    ]

    mutants = { 8: [] }
    for i in range(len(muts)):
        assert(len(muts[i]) == 8)
        mutable = seq
        for mut in muts[i]:
            aa_orig = mut[0]
            aa_mut = mut[-1]
            pos = int(mut[1:-1]) - 1
            assert(seq[pos] == aa_orig)
            mutable = mutable[:pos] + aa_mut + mutable[pos + 1:]
        mutants[8].append(mutable)

    return seq, mutants

def load_sarscov1():
    seq = str(SeqIO.read('data/cov/cov2_spike_wt.fasta', 'fasta').seq)

    muts = [
        [ 'K417V', 'N439R', 'G446T', 'L455Y',
          'F456L', 'A475P', 'F486L', 'Q493N',
          'S494D', 'Q498Y', 'N501T', 'V503I' ],
    ]

    mutants = { 12: [] }
    for i in range(len(muts)):
        assert(len(muts[i]) == 12)
        mutable = seq
        for mut in muts[i]:
            aa_orig = mut[0]
            aa_mut = mut[-1]
            pos = int(mut[1:-1]) - 1
            assert(seq[pos] == aa_orig)
            mutable = mutable[:pos] + aa_mut + mutable[pos + 1:]
        mutants[12].append(mutable)

    return seq, mutants

if __name__ == '__main__':
    load_to2020()
    load_ratg13()
    load_sarscov1()
