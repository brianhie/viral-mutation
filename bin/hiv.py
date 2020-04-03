from mutation import *

from Bio import BiopythonWarning
from Bio import SeqIO

np.random.seed(1)
random.seed(1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='HIV sequence analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., hmm, lstm)')
    parser.add_argument('--namespace', type=str, default='hiv',
                        help='Model namespace')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint')
    parser.add_argument('--train', action='store_true',
                        help='Train model')
    parser.add_argument('--train-split', action='store_true',
                        help='Train model on portion of data')
    parser.add_argument('--test', action='store_true',
                        help='Test model')
    parser.add_argument('--embed', action='store_true',
                        help='Analyze embeddings')
    parser.add_argument('--dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--semantics', action='store_true',
                        help='Analyze mutational semantic change')
    args = parser.parse_args()
    return args

def load_meta(meta_fnames):
    metas = {}
    for fname in meta_fnames:
        with open(fname) as f:
            for line in f:
                if not line.startswith('>'):
                    continue
                accession = line[1:].rstrip()
                fields = line.rstrip().split('.')
                country, year, strain = fields[1], fields[2], fields[3]
                if year == '-':
                    year = None
                else:
                    year = int(year)
                metas[accession] = {
                    'country': country,
                    'year': year,
                    'strain': strain,
                }
    return metas

def process(fnames, meta_fnames):
    metas = load_meta(meta_fnames)

    seqs = {}
    for fname in fnames:
        for record in SeqIO.parse(fname, 'fasta'):
            if record.seq not in seqs:
                seqs[record.seq] = []
            accession = record.description
            meta = metas[accession]
            seqs[record.seq].append(meta)

    return seqs

def split_seqs(seqs, split_method='random'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BiopythonWarning)

        train_seqs, test_seqs = {}, {}

        old_cutoff = 1900
        new_cutoff = 2008

        tprint('Splitting seqs...')
        for seq in seqs:
            # Pick validation set based on date.
            seq_dates = [
                meta['year'] for meta in seqs[seq]
                if meta['year'] is not None
            ]
            if len(seq_dates) == 0:
                test_seqs[seq] = seqs[seq]
                continue
            if len(seq_dates) > 0:
                oldest_date = sorted(seq_dates)[0]
                if oldest_date < old_cutoff or oldest_date >= new_cutoff:
                    test_seqs[seq] = seqs[seq]
                    continue
            train_seqs[seq] = seqs[seq]
        tprint('{} train seqs, {} test seqs.'
               .format(len(train_seqs), len(test_seqs)))

    return train_seqs, test_seqs

def setup(args):
    fnames = [ 'data/hiv/HIV-1_env_samelen.fa' ]
    meta_fnames = [ 'data/hiv/HIV-1_env_samelen.fa' ]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BiopythonWarning)
        seqs = process(fnames, meta_fnames)

    seq_len = max([ len(seq) for seq in seqs ]) + 2
    vocab_size = len(AAs) + 2

    model = get_model(args, seq_len, vocab_size)

    return model, seqs

def interpret_clusters(adata):
    clusters = sorted(set(adata.obs['louvain']))
    for cluster in clusters:
        tprint('Cluster {}'.format(cluster))
        adata_cluster = adata[adata.obs['louvain'] == cluster]
        for var in [ 'year', 'country', 'strain' ]:
            tprint('\t{}:'.format(var))
            counts = Counter(adata_cluster.obs[var])
            for val, count in counts.most_common():
                tprint('\t\t{}: {}'.format(val, count))
        tprint('')

def seq_clusters(adata):
    clusters = sorted(set(adata.obs['louvain']))
    for cluster in clusters:
        adata_cluster = adata[adata.obs['louvain'] == cluster]
        counts = Counter(adata_cluster.obs['seq'])
        with open('target/clusters/cluster{}.fa'.format(cluster), 'w') as of:
            for i, (seq, count) in enumerate(counts.most_common()):
                of.write('>cluster{}_{}_{}\n'.format(cluster, i, count))
                of.write(seq + '\n\n')

def plot_umap(adata):
    sc.tl.umap(adata, min_dist=1.)
    sc.pl.umap(adata, color='year', save='_year.png')
    sc.pl.umap(adata, color='country', save='_country.png')
    sc.pl.umap(adata, color='strain', save='_strain.png')
    sc.pl.umap(adata, color='louvain', save='_louvain.png')
    sc.pl.umap(adata, color='n_seq', save='_number.png',
               s=np.log(np.array(adata.obs['n_seq']) * 100) + 1)

def analyze_embedding(args, model, seqs, vocabulary):
    seqs = embed_seqs(args, model, seqs, vocabulary)

    X, obs = [], {}
    obs['n_seq'] = []
    obs['seq'] = []
    for seq in seqs:
        meta = seqs[seq][0]
        X.append(meta['embedding'].mean(0))
        for key in meta:
            if key == 'embedding':
                continue
            if key not in obs:
                obs[key] = []
            obs[key].append(Counter([
                meta[key] for meta in seqs[seq]
            ]).most_common(1)[0][0])
        obs['n_seq'].append(len(seqs[seq]))
        obs['seq'].append(str(seq))
    X = np.array(X)

    adata = AnnData(X)
    for key in obs:
        adata.obs[key] = obs[key]

    sc.pp.neighbors(adata, n_neighbors=100, use_rep='X')
    sc.tl.louvain(adata, resolution=1.)

    sc.set_figure_params(dpi_save=500)
    plot_umap(adata)

    interpret_clusters(adata)
    #seq_clusters(adata)

def analyze_semantics(args, model, vocabulary, seq_to_mutate, escape_seqs,
                      prob_cutoff=0, beta=1., verbose=False):
    seqs = { seq_to_mutate: [ {} ] }
    X_cat, lengths = featurize_seqs(seqs, vocabulary)

    if args.model_name == 'lstm':
        from lstm import _split_and_pad
    elif args.model_name == 'bilstm':
        from bilstm import _split_and_pad
    else:
        raise ValueError('No semantics support for model {}'
                         .format(args.model_name))

    X = _split_and_pad(X_cat, lengths, model.seq_len_,
                       model.vocab_size_, verbose)[0]
    y_pred = model.model_.predict(X, batch_size=2500)
    assert(y_pred.shape[0] == len(seq_to_mutate) + 2)
    assert(y_pred.shape[1] == len(AAs) + 3)

    word_pos_prob = {}
    for i in range(len(seq_to_mutate)):
        for word in vocabulary:
            word_idx = vocabulary[word]
            prob = y_pred[i + 1, word_idx]
            if prob < prob_cutoff:
                continue
            word_pos_prob[(word, i)] = prob

    prob_sorted = sorted(word_pos_prob.items(), key=lambda x: -x[1])
    prob_seqs = { seq_to_mutate: [ {} ] }
    seq_prob = {}
    for (word, pos), prob in prob_sorted:
        mutable = seq_to_mutate[:pos] + word + seq_to_mutate[pos + 1:]
        prob_seqs[mutable] = [ {} ]
        seq_prob[mutable] = prob

    prob_seqs = embed_seqs(args, model, prob_seqs, vocabulary,
                           use_cache=False, verbose=verbose)
    base_embedding = prob_seqs[seq_to_mutate][0]['embedding']
    seq_change = {}
    for seq in prob_seqs:
        embedding = prob_seqs[seq][0]['embedding']
        # L1 distance between embedding vectors.
        seq_change[seq] = abs(base_embedding - embedding).sum()

    seqs = np.array([ str(seq) for seq in sorted(seq_prob.keys()) ])
    prob = np.array([ seq_prob[seq] for seq in seqs ])
    change = np.array([ seq_change[seq] for seq in seqs ])

    dirname = 'target/hiv/semantics/cache'
    mkdir_p(dirname)
    cache_fname = dirname + '/plot.npz'
    np.savez_compressed(cache_fname, prob, change)

if __name__ == '__main__':
    args = parse_args()

    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V', 'X', 'Z', 'J', 'U', 'B',
    ]
    vocabulary = { aa: idx + 1 for idx, aa in enumerate(sorted(AAs)) }

    model, seqs = setup(args)

    if args.checkpoint is not None:
        model.model_.load_weights(args.checkpoint)
        tprint('Model summary:')
        print(model.model_.summary())

    if args.train or args.train_split or args.test:
        train_test(args, model, seqs, vocabulary, split_seqs)

    if args.embed:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')
        no_embed = { 'hmm' }
        if args.model_name in no_embed:
            raise ValueError('Embeddings not available for models: {}'
                             .format(', '.join(no_embed)))
        analyze_embedding(args, model, seqs, vocabulary)

    if args.semantics:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')
        raise NotImplementedError()
