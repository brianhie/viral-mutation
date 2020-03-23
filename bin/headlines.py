from utils import *

from dateutil.parser import parse as dparse

# Global variables.
VOCABULARY = None
START_INT = None
END_INT = None

np.random.seed(1)
random.seed(1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Headline analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., hmm, lstm)')
    parser.add_argument('--namespace', type=str, default='headlines',
                        help='Model namespace')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint')
    parser.add_argument('--train', action='store_true',
                        help='Train model')
    parser.add_argument('--test', action='store_true',
                        help='Train model')
    parser.add_argument('--embed', action='store_true',
                        help='Analyze embeddings')
    parser.add_argument('--dim', type=int, default=256,
                        help='Embedding dimension')
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
            for line in f:
                timestamp, headline = f.split(',')
                seq = headline.split()
                if seq not in seqs:
                    seqs[seq] = []
                seqs[seq].append(parse_meta(timestamp, headline))
    return seqs

def err_model(name):
    raise ValueError('Model {} not supported'.format(name))

def get_model(args, seq_len, vocab_size,):
    if args.model_name == 'hmm':
        from hmmlearn.hmm import MultinomialHMM
        model = MultinomialHMM(
            n_components=16,
            startprob_prior=1.0,
            transmat_prior=1.0,
            algorithm='viterbi',
            random_state=1,
            n_iter=100,
            tol=0.01,
            verbose=True,
            params='ste',
            init_params='ste'
        )
    elif args.model_name == 'lstm':
        from lstm import LSTMLanguageModel
        model = LSTMLanguageModel(
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=args.dim,
            n_hidden=2,
            n_epochs=20,
            batch_size=1000,
            cache_dir='target/{}'.format(args.namespace),
            verbose=2,
        )
    elif args.model_name == 'bilstm':
        from bilstm import BiLSTMLanguageModel
        model = BiLSTMLanguageModel(
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=args.dim,
            n_hidden=2,
            n_epochs=20,
            batch_size=1000,
            cache_dir='target/{}'.format(args.namespace),
            verbose=2,
        )
    elif args.model_name == 'bilstm-a':
        from bilstm import BiLSTMLanguageModel
        model = BiLSTMLanguageModel(
            seq_len,
            vocab_size,
            attention=True,
            embedding_dim=20,
            hidden_dim=args.dim,
            n_hidden=2,
            n_epochs=20,
            batch_size=1000,
            cache_dir='target/{}'.format(args.namespace),
            verbose=2,
        )
    else:
        err_model(args.model_name)

    return model

def split_seqs(seqs, split_method='random'):
    train_seqs, val_seqs = {}, {}

    new_cutoff = dparse('01-01-2019')

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
    tprint('Done.')

    return train_seqs, val_seqs

def featurize_seqs(seqs):
    global VOCABULARY, START_INT, END_INT
    sorted_seqs = sorted(seqs.keys())
    X = np.concatenate([
        np.array([ START_INT ] + [
            VOCABULARY[word] for word in seq
        ] + [ END_INT ]) for seq in sorted_seqs
    ]).reshape(-1, 1)
    lens = np.array([ len(seq) + 2 for seq in sorted_seqs ])
    assert(sum(lens) == X.shape[0])
    return X, lens

def fit_model(name, model, seqs):
    X, lengths = featurize_seqs(seqs)

    if name == 'hmm':
        model.fit(X, lengths)
    elif name == 'lstm':
        model.fit(X, lengths)
    elif name == 'bilstm':
        model.fit(X, lengths)
    elif name == 'bilstm-a':
        model.fit(X, lengths)
    else:
        err_model(name)

    return model

def perplexity(logprob, n_samples):
    return -logprob / n_samples

def report_performance(model_name, model, train_seqs, test_seqs):
    X_train, lengths_train = featurize_seqs(train_seqs)
    logprob = model.score(X_train, lengths_train)
    tprint('Model {}, train perplexity: {}'
           .format(model_name, perplexity(logprob, len(lengths_train))))
    X_test, lengths_test = featurize_seqs(test_seqs)
    logprob = model.score(X_test, lengths_test)
    tprint('Model {}, test perplexity: {}'
           .format(model_name, perplexity(logprob, len(lengths_test))))

def setup(args):
    global VOCABULARY, START_INT, END_INT

    fnames = [ 'data/headlines/abcnews-date-text.csv' ]

    seqs = process(fnames)

    seq_len = max([ len(seq) for seq in seqs ]) + 2
    VOCABULARY = { word for word in seq for seq in seqs }
    VOCABULARY = { word: idx + 1 for idx, word in enumerate(VOCABULARY) }
    START_INT = len(VOCABULARY) + 1
    END_INT = len(VOCABULARY) + 2
    vocab_size = len(VOCABULARY) + 2

    model = get_model(args, seq_len, vocab_size)

    return model, seqs

def train_test(args, model, seqs):
    train_seqs, val_seqs = split_seqs(seqs)
    if args.train:
        model = fit_model(args.model_name, model, train_seqs)
    if args.test:
        report_performance(args.model_name, model, train_seqs, val_seqs)

def embed_seqs(args, model, seqs):
    X_cat, lengths = featurize_seqs(seqs)

    from keras.models import Model
    if args.model_name == 'lstm':
        from lstm import _iterate_lengths, _split_and_pad
        layer_name = 'lstm_{}'.format(model.n_hidden_)
    elif args.model_name == 'bilstm':
        from bilstm import _iterate_lengths, _split_and_pad
        layer_name = 'concatenate_1'
    else:
        raise ValueError('No embedding support for model {}'
                         .format(args.model_name))

    hidden = Model(
        inputs=model.model_.input,
        outputs=model.model_.get_layer(layer_name).output
    )

    mkdir_p('target/{}/embedding'.format(args.namespace))
    embed_fname = ('target/{}/embedding/{}_{}.npy'
                   .format(args.namespace, args.model_name, args.dim))
    if os.path.exists(embed_fname):
        embed_cat = np.load(embed_fname)
    else:
        X = _split_and_pad(
            X_cat, lengths,
            model.seq_len_, model.vocab_size_, model.verbose_
        )[0]
        tprint('Embedding...')
        embed_cat = hidden.predict(X, batch_size=5000,
                                   verbose=model.verbose_ > 0)
        np.save(embed_fname, embed_cat)
        tprint('Done embedding.')

    sorted_seqs = sorted(seqs)
    for seq, (start, end) in zip(
            sorted_seqs, _iterate_lengths(lengths, model.seq_len_)):
        embedding = embed_cat[start:end]
        for meta in seqs[seq]:
            meta['embedding'] = embedding

    return seqs

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

def analyze_embedding(args, model, seqs):
    seqs = embed_seqs(args, model, seqs)

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

if __name__ == '__main__':
    args = parse_args()

    model, seqs = setup(args)

    if args.checkpoint is not None:
        model.model_.load_weights(args.checkpoint)
        tprint('Model summary:')
        print(model.model_.summary())

    if args.train or args.test:
        train_test(args, model, seqs)

    if args.embed:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')
        no_embed = { 'hmm' }
        if args.model_name in no_embed:
            raise ValueError('Embeddings not available for models: {}'
                             .format(', '.join(no_embed)))
        analyze_embedding(args, model, seqs)
