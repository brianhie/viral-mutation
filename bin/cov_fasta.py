from mutation import *

np.random.seed(1)
random.seed(1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Coronavirus sequence analysis')
    parser.add_argument('baseline_fasta', help='Baseline sequence')
    parser.add_argument('target_fasta',
                        help='Sequences on which to compute statistics')
    parser.add_argument('--checkpoint', type=str, default='models/cov.hdf5',
                        help='Model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Results output filename')
    parser.add_argument('--model_name', type=str, default='bilstm',
                        help='Type of language model (e.g., hmm, lstm)')
    parser.add_argument('--namespace', type=str, default='cov',
                        help='Model namespace')
    parser.add_argument('--dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Training minibatch size')
    parser.add_argument('--n-epochs', type=int, default=11,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    args = parser.parse_args()
    return args

def get_mutations(seq1, seq2):
    mutations = []
    from Bio import pairwise2
    alignment = pairwise2.align.globalms(
        seq1, seq2, 5, -5, -3, -.1, one_alignment_only=True,
    )[0]
    pos = 0
    for ch1, ch2 in zip(alignment[0], alignment[1]):
        if ch1 != ch2 and ch1 != '-' and ch2 != '-':
            mutations.append('{}{}{}'.format(ch1, pos + 1, ch2))
        elif ch1 == '-' and ch2 != '-':
            mutations.append('{}ins{}'.format(pos + 1, ch2))
        elif ch1 != '-' and ch2 == '-':
            mutations.append('{}{}del'.format(ch1, pos + 1))
        if ch1 != '-':
            pos += 1
    return mutations

def grammaticality_change(word_pos_prob, seq, mutations, args, vocabulary, model,
                          verbose=False,):
    if len(mutations) == 0:
        return 0

    mut_probs = []
    for mutation in mutations:
        if 'del' in mutation or 'ins' in mutation:
            continue
        aa_orig = mutation[0]
        aa_pos = int(mutation[1:-1]) - 1
        aa_mut = mutation[-1]
        if (seq[aa_pos] != aa_orig):
            print(mutation)
        assert(seq[aa_pos] == aa_orig)
        mut_probs.append(word_pos_prob[(aa_mut, aa_pos)])

    return np.mean(np.log10(mut_probs))

def compute_statistics(baseline_fname, target_fname,
                       args, model, vocabulary):

    # Load baseline sequence.
    base_seq = str(SeqIO.read(baseline_fname, 'fasta').seq)

    # Compute baseline information.
    y_pred = predict_sequence_prob(
        args, base_seq, vocabulary, model, verbose=False
    )
    word_pos_prob = {}
    for pos in range(len(base_seq)):
        for word in vocabulary:
            word_idx = vocabulary[word]
            prob = y_pred[pos + 1, word_idx]
            word_pos_prob[(word, pos)] = prob
    base_embedding = embed_seqs(
        args, model, { base_seq: [ {} ] }, vocabulary,
        use_cache=False, verbose=False
    )[base_seq][0]['embedding']

    # Compute statistics over target fasta.

    if args.output is None:
        of = sys.stdout
    else:
        of = open(args.output, 'w')

    fields = [ 'Mutations', 'Semantic change', 'Grammaticality', 'ID', 'Sequence' ]
    of.write('\t'.join([ str(field) for field in fields ]) + '\n')

    for record in SeqIO.parse(target_fname, 'fasta'):
        name, mut_seq = record.id, str(record.seq)

        mutations = get_mutations(base_seq, mut_seq)

        mut_embedding = embed_seqs(
            args, model, { mut_seq: [ {} ] }, vocabulary, verbose=False,
        )[mut_seq][0]['embedding']
        mut_change = np.sum(np.abs(mut_embedding.mean(0) - base_embedding.mean(0)))

        mut_gramm = grammaticality_change(word_pos_prob, base_seq, mutations,
                                          args, vocabulary, model)

        fields = [ ','.join(mutations), mut_change, mut_gramm, name, mut_seq ]
        of.write('\t'.join([ str(field) for field in fields ]) + '\n')

    if args.output is not None:
        of.close()

if __name__ == '__main__':
    args = parse_args()

    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V', 'X', 'Z', 'J', 'U', 'B',
    ]
    vocabulary = { aa: idx + 1 for idx, aa in enumerate(sorted(AAs)) }
    vocab_size = len(AAs) + 2

    model = get_model(args, 1584, vocab_size,
                      inference_batch_size=args.batch_size)
    model.model_.load_weights(args.checkpoint)
    tprint('Model summary:')
    tprint(model.model_.summary())

    compute_statistics(args.baseline_fasta, args.target_fasta,
                       args, model, vocabulary)
