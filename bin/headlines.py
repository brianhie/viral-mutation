from mutation import *

np.random.seed(1)
random.seed(1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Headline analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., hmm, lstm)')
    parser.add_argument('--namespace', type=str, default='headlines',
                        help='Model namespace')
    parser.add_argument('--dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--n-epochs', type=int, default=20,
                        help='Number of training epochs')
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
    parser.add_argument('--semantics', action='store_true',
                        help='Analyze mutational semantic change')
    args = parser.parse_args()
    return args

def parse_meta(timestamp, headline):
    return {
        'timestamp': timestamp,
        'date': dparse(timestamp),
        'year': int(timestamp[:4]),
        'headline': headline,
    }

def process(fnames):
    seqs = {}
    for fname in fnames:
        with open(fname) as f:
            f.readline() # Consume header.
            for line in f:
                timestamp, headline = line.rstrip().split(',')
                seq = tuple(headline.split())
                if seq not in seqs:
                    seqs[seq] = []
                seqs[seq].append(parse_meta(timestamp, headline))
    return seqs

def split_seqs(seqs, split_method='random'):
    train_seqs, val_seqs = {}, {}

    new_cutoff = dparse('01-01-2016')

    tprint('Splitting seqs...')
    for seq in seqs:
        # Pick validation set based on date.
        seq_dates = [ meta['date'] for meta in seqs[seq] ]
        if len(seq_dates) > 0:
            oldest_date = sorted(seq_dates)[0]
            if oldest_date >= new_cutoff:
                val_seqs[seq] = seqs[seq]
                continue
        train_seqs[seq] = seqs[seq]
    tprint('{} train seqs, {} test seqs.'
           .format(len(train_seqs), len(val_seqs)))

    return train_seqs, val_seqs

def setup():
    fnames = [ 'data/headlines/abcnews-date-text.csv' ]
    seqs = process(fnames)
    vocabulary = sorted({ word for seq in seqs for word in seq })
    vocabulary = { word: idx + 1 for idx, word in enumerate(vocabulary) }
    return seqs, vocabulary

def interpret_clusters(adata):
    clusters = sorted(set(adata.obs['louvain']))
    for cluster in clusters:
        tprint('Cluster {}'.format(cluster))
        adata_cluster = adata[adata.obs['louvain'] == cluster]
        counts = Counter(adata_cluster.obs['headline'])
        for val, count in counts.most_common():
            tprint('\t\t{}: {}'.format(val, count))
        tprint('')

def plot_umap(adata):
    sc.tl.umap(adata, min_dist=1.)
    sc.pl.umap(adata, color='louvain', save='_louvain.png')
    sc.pl.umap(adata, color='year', save='_year.png')
    sc.pl.umap(adata, color='date', save='_date.png')

def analyze_embedding(args, model, seqs, vocabulary):
    seqs = embed_seqs(args, model, seqs, vocabulary,
                      use_cache=True)

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

    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
    sc.tl.louvain(adata, resolution=1.)

    sc.set_figure_params(dpi_save=500)
    plot_umap(adata)

    interpret_clusters(adata)

def analyze_headline_semantics(
        args, model, seq_to_mutate, vocabulary,
        prob_cutoff=1e-4, n_most_probable=100, beta=1.,
        plot_acquisition=False, verbose=False
):
    seqs, prob, change, _, _ = analyze_semantics(
        args, model, vocabulary, seq_to_mutate, {},
        prob_cutoff=prob_cutoff, beta=beta, plot_acquisition=False,
        cache_fname=None, verbose=verbose
    )

    headlines = np.array([ ' '.join(seq) for seq in seqs ])
    acquisition = ss.rankdata(change) + (beta * ss.rankdata(prob))

    if plot_acquisition:
        plt.figure()
        plt.scatter(np.log10(prob), change,
                    c=acquisition, cmap='viridis', alpha=0.3)
        plt.title(' '.join(seq_to_mutate))
        plt.xlabel('$ \log_{10}(p(x_i)) $')
        plt.ylabel('$ \Delta \Theta $')
        plt.savefig('figures/headline_acquisition.png', dpi=300)
        plt.close()
        exit()

    tprint('Original headline: ' + ' '.join(seq_to_mutate))
    tprint('Modifications:')
    for idx in np.argsort(-acquisition)[:n_most_probable]:
        tprint('{}: {} (change), {} (prob)'.format(
            headlines[idx], change[idx], prob[idx]
        ))
    tprint('Least change:')
    for idx in np.argsort(change)[:n_most_probable]:
        tprint('{}: {} (change), {} (prob)'.format(
            headlines[idx], change[idx], prob[idx]
        ))
    tprint('Most change:')
    for idx in np.argsort(-change)[:n_most_probable]:
        tprint('{}: {} (change), {} (prob)'.format(
            headlines[idx], change[idx], prob[idx]
        ))

if __name__ == '__main__':
    args = parse_args()

    seqs, vocabulary = setup()
    seq_len = max([ len(seq) for seq in seqs ]) + 2
    vocab_size = len(vocabulary) + 2
    model = get_model(args, seq_len, vocab_size, batch_size=5)

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
        random_sample = np.random.choice(
            [ ' '.join(seq) for seq in seqs ], 100000
        )
        for headline in random_sample[50000:]:
            tprint('')
            analyze_headline_semantics(
                args, model, headline.split(' '),
                vocabulary, n_most_probable=3,
                prob_cutoff=1e-4, beta=0.25
            )
