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

def get_model(name, seq_len, vocab_size,):
    if name == 'hmm':
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
    elif name == 'lstm':
        from lstm import LSTMLanguageModel
        model = LSTMLanguageModel(
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=256,
            n_hidden=2,
            n_epochs=20,
            batch_size=1000,
            verbose=2,
        )
    elif name == 'bilstm':
        from bilstm import BiLSTMLanguageModel
        model = BiLSTMLanguageModel(
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=512,
            n_hidden=2,
            n_epochs=20,
            batch_size=1000,
            verbose=2,
        )
    else:
        err_model(name)

    return model

def split_seqs(seqs, split_method='random'):
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

if __name__ == '__main__':
    fnames = [ 'data/influenza/ird_influenzaA_HA_allspecies.fa' ]
    meta_fnames = [ 'data/influenza/ird_influenzaA_HA_allspecies_meta.tsv' ]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BiopythonWarning)
        seqs = process(fnames, meta_fnames)

    model_name = sys.argv[1]

    seq_len = max([ len(seq) for seq in seqs ]) + 2
    vocab_size = len(AAs) + 2

    model = get_model(model_name, seq_len, vocab_size)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BiopythonWarning)
        train_seqs, test_seqs, val_seqs = split_seqs(seqs)

    model = fit_model(model_name, model, train_seqs)

    report_performance(model_name, model, train_seqs, test_seqs)
