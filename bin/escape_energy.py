from utils import *

from Bio import SeqIO
from sklearn.metrics import auc

def escape_energy(virus):
    if virus == 'h1':
        from escape import load_lee2018
        seq, seqs_escape = load_lee2018()
        train_fname = 'target/flu/clusters/all_h1.fasta'
        mut_fname = 'target/flu/mutation/mutations_h1.fa'
        energy_fname = 'target/flu/clusters/all_h1.fasta.E.txt'
    elif virus == 'h3':
        from escape import load_lee2019
        seq, seqs_escape = load_lee2019()
        train_fname = 'target/flu/clusters/all_h3.fasta'
        mut_fname = 'target/flu/mutation/mutations_h3.fa'
        energy_fname = 'target/flu/clusters/all_h3.fasta.E.txt'
    elif virus == 'hiv':
        from escape import load_dingens2019
        seq, seqs_escape = load_dingens2019()
        train_fname = 'target/hiv/clusters/all_BG505.fasta'
        mut_fname = 'target/hiv/mutation/mutations_hiv.fa'
        energy_fname = 'target/hiv/clusters/all_BG505.fasta.E.txt'
    else:
        raise ValueError('invalid option {}'.format(virus))

    train_seqs = list(SeqIO.parse(train_fname, 'fasta'))
    mut_seqs = list(SeqIO.parse(mut_fname, 'fasta'))
    energies = np.loadtxt(energy_fname)
    assert(len(energies) == len(train_seqs) + len(mut_seqs))

    one_seq = str(mut_seqs[69].seq).replace('-', '')

    from Bio import pairwise2
    from Bio.pairwise2 import format_alignment
    alignments = pairwise2.align.globalxx(str(seq), one_seq)
    print(format_alignment(*alignments[0]))
    exit()

    escape_idx = [
        idx for idx, mut_seq in enumerate(mut_seqs)
        if str(mut_seq.seq).replace('-', '') in seqs_escape
    ]
    print((len(escape_idx), len(seqs_escape)))
    assert(len(escape_idx) == len(seqs_escape) - 1)

    mut_energies = energies[len(train_seqs):]
    acq_argsort = ss.rankdata(mut_energies)
    escape_rank_dist = acq_argsort[escape_idx]

    max_consider = len(mut_seqs)
    n_consider = np.array([ i + 1 for i in range(max_consider) ])

    n_escape = np.array([ sum(escape_rank_dist <= i + 1)
                          for i in range(max_consider) ])
    norm = max(n_consider) * max(n_escape)
    norm_auc = auc(n_consider, n_escape) / norm

    escape_frac = len(seqs_escape) / len(mut_seqs)

    plt.figure()
    plt.plot(n_consider, n_escape)
    plt.plot(n_consider, n_consider * escape_frac,
             c='gray', linestyle='--')
    plt.legend([
        r'Potts model energy, AUC = {:.3f}'.format(norm_auc),
        'Random guessing, AUC = 0.500'
    ])
    plt.xlabel('Top N')
    plt.ylabel('Number of escape mutations in top N')
    plt.savefig('figures/{}_energy_escape.png'.format(virus), dpi=300)
    plt.close()

if __name__ == '__main__':
    escape_energy(sys.argv[1])
