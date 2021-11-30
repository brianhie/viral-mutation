from Bio import SeqIO
import numpy as np
import random
import sys


AA_ALPHABET_STANDARD_ORDER = 'ARNDCQEGHILKMFPSTWYV'


def make_n_random_edits(
        seq,
        nedits,
        min_pos=None,
        max_pos=None,
):
    """
    min_pos is inclusive. max_pos is exclusive
    """
    alphabet = AA_ALPHABET_STANDARD_ORDER
    
    lseq = list(seq)
    lalphabet = list(alphabet)
    
    if min_pos is None:
        min_pos = 0
    
    if max_pos is None:
        max_pos = len(seq)
    
    # Create non-redundant list of positions to mutate.
    l = list(range(min_pos, max_pos))
    nedits = min(len(l), nedits)
    random.shuffle(l)
    pos_to_mutate = l[:nedits]    
    
    for i in range(nedits):
        pos = pos_to_mutate[i]     
        aa_to_choose_from = list(set(lalphabet) - set([seq[pos]]))
                        
        lseq[pos] = aa_to_choose_from[np.random.randint(len(aa_to_choose_from))]
        
    return ''.join(lseq)

if __name__ == '__main__':
    seq = str(SeqIO.read(sys.argv[1], 'fasta').seq)

    print('>wt_seq')
    print(seq + '\n')

    for i in range(10000):
        mut = make_n_random_edits(seq, 37)
        print(f'>mut_seq_{i}')
        print(mut + '\n')
        
