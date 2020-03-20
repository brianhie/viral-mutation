from utils import *

from Bio import BiopythonWarning
from Bio import SeqIO
from dateutil.parser import parse as dparse

np.random.seed(1)
random.seed(1)

AAs = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
    'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
    'Y', 'V', 'X', 'Z', 'J', 'U', 'B', 'Z'
]
START_INT = len(AAs) + 1
END_INT = len(AAs) + 2

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Flu sequence analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., hmm, lstm)')
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

def load_meta(meta_fnames):
    metas = {}
    for fname in meta_fnames:
        with open(fname) as f:
            header = f.readline().rstrip().split('\t')
            for line in f:
                fields = line.rstrip().split('\t')
                accession = fields[1]
                meta = {}
                for key, value in zip(header, fields):
                    if key == 'Subtype':
                        meta[key] = value.strip('()').split('N')[0].split('/')[-1]
                    elif key == 'Collection Date':
                        meta[key] = int(value.split('/')[-1]) \
                                    if value != '-N/A-' else 2019
                    elif key == 'Host Species':
                        meta[key] = value.split(':')[1].split('/')[-1].lower()
                    else:
                        meta[key] = value
                metas[accession] = meta
    return metas

def process(fnames, meta_fnames):
    metas = load_meta(meta_fnames)

    seqs = {}
    for fname in fnames:
        for record in SeqIO.parse(fname, 'fasta'):
            if record.seq not in seqs:
                seqs[record.seq] = []
            accession = record.description.split('|')[0].split(':')[1]
            seqs[record.seq].append(metas[accession])
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
            verbose=2,
        )
    else:
        err_model(args.model_name)

    return model

def split_seqs(seqs, split_method='random'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BiopythonWarning)

        train_seqs, test_seqs, val_seqs = {}, {}, {}

        old_cutoff = dparse('01-01-1970')
        new_cutoff = dparse('01-01-2019')

        tprint('Splitting seqs...')
        for seq in seqs:
            # Pick validation set based on date.
            seq_dates = [ dparse(meta['Collection Date']) for meta in seqs[seq]
                          if meta['Collection Date'] != '-N/A-' ]
            if len(seq_dates) > 0:
                oldest_date = sorted(seq_dates)[0]
                if oldest_date < old_cutoff or oldest_date >= new_cutoff:
                    val_seqs[seq] = seqs[seq]
                    continue

            # Randomly separate remainder into train and test sets for tuning.
            rand = np.random.uniform()
            if rand < 0.85:
                train_seqs[seq] = seqs[seq]
            else:
                test_seqs[seq] = seqs[seq]
        tprint('Done.')

    return train_seqs, test_seqs, val_seqs

def featurize_seqs(seqs):
    aa2idx = { aa: idx + 1 for idx, aa in enumerate(sorted(AAs)) }
    sorted_seqs = sorted(seqs.keys())
    X = np.concatenate([
        np.array([ START_INT ] + [
            aa2idx[aa] for aa in seq
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
    fnames = [ 'data/influenza/ird_influenzaA_HA_allspecies.fa' ]
    meta_fnames = [ 'data/influenza/ird_influenzaA_HA_allspecies_meta.tsv' ]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BiopythonWarning)
        seqs = process(fnames, meta_fnames)

    seq_len = max([ len(seq) for seq in seqs ]) + 2
    vocab_size = len(AAs) + 2

    model = get_model(args, seq_len, vocab_size)

    return model, seqs

def train_test(args, model, seqs):
    train_seqs, test_seqs, val_seqs = split_seqs(seqs)
    if args.train:
        model = fit_model(args.model_name, model, train_seqs)
    if args.test:
        report_performance(args.model_name, model, train_seqs, test_seqs)

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

    embed_fname = ('target/embedding/{}_{}.npy'
                   .format(args.model_name, args.dim))
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
        for var in [ 'Collection Date', 'Country', 'Subtype',
                     'Flu Season', 'Host Species', 'Strain Name' ]:
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

def plot_composition(adata, var):
    years = sorted(set(adata.obs['Collection Date']))
    vals = sorted(set(adata.obs[var]))

    comps = [ [] for val in vals ]
    norms = [ [] for val in vals ]
    for year in years:
        adata_year = adata[adata.obs['Collection Date'] == year]
        val_count = { val: 0 for val in vals }
        for n_seq, val in zip(adata_year.obs['n_seq'], adata_year.obs[var]):
            val_count[val] += n_seq
        comp = np.array([ val_count[val] for val in vals ], dtype=float)
        comp_sum = float(np.sum(comp))
        for i in range(len(comp)):
            comps[i].append(comp[i])
            norms[i].append(comp[i] / comp_sum)

    x = np.array(range(len(years)))
    plt.figure(figsize=(40, 10))
    plt.stackplot(x, norms, labels=vals)
    plt.xticks(x, [ str(year) for year in years ], rotation=45)
    plt.legend()
    plt.grid(b=None)
    plt.savefig('figures/plot_composition_{}.png'.format(var), dpi=500)

    plt.figure(figsize=(10, 40))
    for i in range(len(vals)):
        plt.subplot(len(vals), 1, i + 1)
        plt.title(str(vals[i]))
        plt.fill_between(x, comps[i], 0)
        if i == len(vals) - 1:
            plt.xticks(x, [ str(year) for year in years ], rotation=45)
        plt.grid(b=None)
    plt.savefig('figures/plot_composition_separate_{}.png'
                .format(var), dpi=500)

def plot_umap(adata):
    sc.tl.umap(adata, min_dist=1.)
    sc.pl.umap(adata, color='Host Species', save='_species.png')
    sc.pl.umap(adata, color='Subtype', save='_subtype.png')
    sc.pl.umap(adata, color='Collection Date', save='_date.png')
    sc.pl.umap(adata, color='louvain', save='_louvain.png')
    sc.pl.umap(adata, color='n_seq', save='_number.png',
               s=np.log(np.array(adata.obs['n_seq']) * 100) + 1)

def analyze_embedding(args, model, seqs):
    seqs = embed_seqs(args, model, seqs)

    X, obs = [], {}
    obs['n_seq'] = []
    obs['seq'] = []
    for seq in seqs:
        for meta in seqs[seq]:
            X.append(meta['embedding'].mean(0))
            for key in meta:
                if key == 'embedding':
                    continue
                if key not in obs:
                    obs[key] = []
                obs[key].append(meta[key])
            obs['n_seq'].append(len(seqs[seq]))
            obs['seq'].append(str(seq))
            break # Pick first meta entry for sequence.
    X = np.array(X)

    adata = AnnData(X)
    for key in obs:
        adata.obs[key] = obs[key]
    adata = adata[
        np.logical_or.reduce((
            adata.obs['Host Species'] == 'human',
            adata.obs['Host Species'] == 'avian',
            adata.obs['Host Species'] == 'swine',
        ))
    ]

    sc.pp.neighbors(adata, n_neighbors=100, use_rep='X')
    sc.tl.louvain(adata, resolution=1.)

    sc.set_figure_params(dpi_save=500)
    plot_umap(adata)

    adata_human = adata[adata.obs['Host Species'] == 'human']
    plot_composition(adata_human, 'louvain')
    plot_composition(adata_human, 'Subtype')

    interpret_clusters(adata)
    seq_clusters(adata)

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
