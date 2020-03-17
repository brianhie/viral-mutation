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
                metas[accession] = { key: value for key, value in zip(header, fields) }
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

def analyze_embedding(args, model, seqs):
    from keras.models import Model
    layer_name = 'lstm_{}'.format(model.n_hidden_)
    hidden = Model(
        inputs=model.model_.input,
        outputs=model.model_.get_layer(layer_name).output
    )

    X_cat, lengths = featurize_seqs(seqs)

    if args.model_name == 'lstm':
        from lstm import _iterate_lengths, _split_and_pad
    elif args.model_name == 'bilstm':
        from bilstm import _iterate_lengths, _split_and_pad
    else:
        raise ValueError('No embedding support for model {}'
                         .format(args.model_name))

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

    sorted_seqs = sorted(seqs)
    for seq, (start, end) in zip(
            sorted_seqs, _iterate_lengths(lengths, model.seq_len_)):
        embedding = embed_cat[start:end]
        for meta in seqs[seq]:
            meta['embedding'] = embedding
    tprint('Done embedding.')

    from anndata import AnnData
    import scanpy as sc

    X, obs = [], {}
    for seq in seqs:
        for meta in seqs[seq]:
            X.append(meta['embedding'].mean(0))
            for key in meta:
                if key == 'embedding':
                    continue
                if key not in obs:
                    obs[key] = []
                obs[key].append(meta[key])
            break # DEBUG
    X = np.array(X)
    adata = AnnData(X)
    for key in obs:
        if key == 'Subtype':
            adata.obs[key] = [
                val.strip('()').split('N')[0] for val in obs[key]
            ]
        elif key == 'Collection Date':
            adata.obs[key] = [
                int(val.split('/')[-1]) if val != '-N/A-' else 2019
                for val in obs[key]
            ]
        elif key == 'Host Species':
            adata.obs[key] = [
                val.split(':')[1].lower() for val in obs[key]
            ]
        else:
            adata.obs[key] = obs[key]

    adata = adata[adata.obs['Host Species'] == 'human']

    sc.pp.neighbors(adata, n_neighbors=100, use_rep='X')
    sc.tl.umap(adata)
    sc.pl.umap(adata, color='Host Species', save='_species.png')
    sc.pl.umap(adata, color='Subtype', save='_subtype.png')
    sc.pl.umap(adata, color='Collection Date', save='_date.png')
    sc.tl.draw_graph(adata)
    sc.pl.draw_graph(adata, color='Host Species', save='_species.png')
    sc.pl.draw_graph(adata, color='Subtype', save='_subtype.png')
    sc.pl.draw_graph(adata, color='Collection Date', save='_date.png',
                     edges=True)

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
