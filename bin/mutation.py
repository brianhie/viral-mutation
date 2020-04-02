from utils import *

def err_model(name):
    raise ValueError('Model {} not supported'.format(name))

def get_model(args, seq_len, vocab_size,):
    if args.model_name == 'hmm':
        from hmmlearn.hmm import MultinomialHMM
        model = MultinomialHMM(
            n_components=args.dim,
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
    elif args.model_name == 'dnn':
        from dnn import DNNLanguageModel
        model = DNNLanguageModel(
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
    else:
        err_model(args.model_name)

    return model

def featurize_seqs(seqs, vocabulary):
    start_int = len(vocabulary) + 1
    end_int = len(vocabulary) + 2
    sorted_seqs = sorted(seqs.keys())
    X = np.concatenate([
        np.array([ start_int ] + [
            vocabulary[word] for word in seq
        ] + [ end_int ]) for seq in sorted_seqs
    ]).reshape(-1, 1)
    lens = np.array([ len(seq) + 2 for seq in sorted_seqs ])
    assert(sum(lens) == X.shape[0])
    return X, lens

def fit_model(name, model, seqs, vocabulary):
    X, lengths = featurize_seqs(seqs, vocabulary)

    if name == 'hmm':
        model.fit(X, lengths)
    elif name == 'dnn':
        model.fit(X, lengths)
    elif name == 'lstm':
        model.fit(X, lengths)
    elif name == 'bilstm':
        model.fit(X, lengths)
    else:
        err_model(name)

    return model

def cross_entropy(logprob, n_samples):
    return -logprob / n_samples

def report_performance(model_name, model, vocabulary,
                       train_seqs, test_seqs):
    X_train, lengths_train = featurize_seqs(train_seqs, vocabulary)
    logprob = model.score(X_train, lengths_train)
    tprint('Model {}, train cross entropy: {}'
           .format(model_name, cross_entropy(logprob, len(lengths_train))))
    X_test, lengths_test = featurize_seqs(test_seqs, vocabulary)
    logprob = model.score(X_test, lengths_test)
    tprint('Model {}, test cross entropy: {}'
           .format(model_name, cross_entropy(logprob, len(lengths_test))))

def train_test(args, model, seqs, vocabulary, split_seqs=None):
    if args.train and args.train_split:
        raise ValueError('Training on full and split data is invalid.')

    if args.train:
        model = fit_model(args.model_name, model, seqs, vocabulary)
        return

    if split_seqs is None:
        raise ValueError('Must provide function to split train/test.')
    train_seqs, val_seqs = split_seqs(seqs)

    if args.train_split:
        model = fit_model(args.model_name, model, train_seqs, vocabulary)
    if args.test:
        report_performance(args.model_name, model, vocabulary,
                           train_seqs, val_seqs)

def load_hidden_model(args, model):
    from keras.models import Model
    if args.model_name == 'lstm':
        layer_name = 'lstm_{}'.format(model.n_hidden_)
    elif args.model_name == 'bilstm':
        layer_name = 'concatenate_1'
    elif args.model_name == 'dnn':
        layer_name = 'concatenate_1'
    else:
        raise ValueError('No embedding support for model {}'
                         .format(args.model_name))
    hidden = Model(
        inputs=model.model_.input,
        outputs=model.model_.get_layer(layer_name).output
    )
    return hidden

def embed_seqs(args, model, seqs, vocabulary,
               use_cache=True, verbose=True):
    X_cat, lengths = featurize_seqs(seqs, vocabulary)

    hidden = load_hidden_model(args, model)

    if args.model_name == 'lstm':
        from lstm import _iterate_lengths, _split_and_pad
    elif args.model_name == 'bilstm':
        from bilstm import _iterate_lengths, _split_and_pad
    elif args.model_name == 'dnn':
        from dnn import _iterate_lengths, _split_and_pad
    else:
        raise ValueError('No embedding support for model {}'
                         .format(args.model_name))

    mkdir_p('target/{}/embedding'.format(args.namespace))
    embed_fname = ('target/{}/embedding/{}_{}.npy'
                   .format(args.namespace, args.model_name, args.dim))
    if os.path.exists(embed_fname) and use_cache:
        embed_cat = np.load(embed_fname)
    else:
        X = _split_and_pad(
            X_cat, lengths,
            model.seq_len_, model.vocab_size_, verbose
        )[0]
        if verbose:
            tprint('Embedding...')
        embed_cat = hidden.predict(X, batch_size=2500, verbose=verbose)
        if use_cache:
            np.save(embed_fname, embed_cat)
        if verbose:
            tprint('Done embedding.')

    sorted_seqs = sorted(seqs)
    for seq, (start, end) in zip(
            sorted_seqs, _iterate_lengths(lengths, model.seq_len_)):
        embedding = embed_cat[start:end]
        for meta in seqs[seq]:
            meta['embedding'] = embedding

    return seqs
