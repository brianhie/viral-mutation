from utils import *

def err_model(name):
    raise ValueError('Model {} not supported'.format(name))

def get_model(args, seq_len, vocab_size,
              inference_batch_size=1500):
    if args.model_name == 'hmm':
        from hmmlearn.hmm import MultinomialHMM
        model = MultinomialHMM(
            n_components=args.dim,
            startprob_prior=1.0,
            transmat_prior=1.0,
            algorithm='viterbi',
            random_state=1,
            tol=0.01,
            verbose=True,
            params='ste',
            init_params='ste'
        )
    elif args.model_name == 'dnn':
        from language_model import DNNLanguageModel
        model = DNNLanguageModel(
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=args.dim,
            n_hidden=2,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            inference_batch_size=inference_batch_size,
            cache_dir='target/{}'.format(args.namespace),
            seed=args.seed,
            verbose=True,
        )
    elif args.model_name == 'lstm':
        from language_model import LSTMLanguageModel
        model = LSTMLanguageModel(
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=args.dim,
            n_hidden=2,
            n_epochs=args.n_epochs,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            cache_dir='target/{}'.format(args.namespace),
            seed=args.seed,
            verbose=True,
        )
    elif args.model_name == 'bilstm':
        from language_model import BiLSTMLanguageModel
        model = BiLSTMLanguageModel(
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=args.dim,
            n_hidden=2,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            inference_batch_size=inference_batch_size,
            cache_dir='target/{}'.format(args.namespace),
            seed=args.seed,
            verbose=True,
        )
    elif args.model_name == 'attention':
        from language_model import AttentionLanguageModel
        model = AttentionLanguageModel(
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=args.dim,
            n_hidden=2,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            inference_batch_size=inference_batch_size,
            cache_dir='target/{}'.format(args.namespace),
            seed=args.seed,
            verbose=True,
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
    model.fit(X, lengths)
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

def batch_train(args, model, seqs, vocabulary, batch_size=5000,
                verbose=True):
    assert(args.train)

    # Control epochs here.
    n_epochs = args.n_epochs
    args.n_epochs = 1
    model.n_epochs_ = 1

    n_batches = math.ceil(len(seqs) / float(batch_size))
    if verbose:
        tprint('Traing seq batch size: {}, N batches: {}'
               .format(batch_size, n_batches))

    for epoch in range(n_epochs):
        if verbose:
            tprint('True epoch {}/{}'.format(epoch + 1, n_epochs))
        perm_seqs = [ str(s) for s in seqs.keys() ]
        random.shuffle(perm_seqs)

        for batchi in range(n_batches):
            start = batchi * batch_size
            end = (batchi + 1) * batch_size
            seqs_batch = { seq: seqs[seq] for seq in perm_seqs[start:end] }
            train_test(args, model, seqs_batch, vocabulary)

        fname_prefix = ('target/{0}/checkpoints/{1}/{1}_{2}'
                        .format(args.namespace, args.model_name, args.dim))

        if epoch == 0:
            os.rename('{}-01.hdf5'.format(fname_prefix),
                      '{}-00.hdf5'.format(fname_prefix))
        else:
            os.rename('{}-01.hdf5'.format(fname_prefix),
                      '{}-{:02d}.hdf5'.format(fname_prefix, epoch + 1))
    os.rename('{}-00.hdf5'.format(fname_prefix),
              '{}-01.hdf5'.format(fname_prefix))

def embed_seqs(args, model, seqs, vocabulary,
               use_cache=False, verbose=True):
    X_cat, lengths = featurize_seqs(seqs, vocabulary)

    if use_cache:
        mkdir_p('target/{}/embedding'.format(args.namespace))
        embed_fname = ('target/{}/embedding/{}_{}.npy'
                       .format(args.namespace, args.model_name, args.dim))
    else:
        embed_fname = None

    if use_cache and os.path.exists(embed_fname):
        X_embed = np.load(embed_fname, allow_pickle=True)
    else:
        X_embed = model.transform(X_cat, lengths, embed_fname)
        if use_cache:
            np.save(embed_fname, X_embed)

    sorted_seqs = sorted(seqs)
    for seq_idx, seq in enumerate(sorted_seqs):
        for meta in seqs[seq]:
            meta['embedding'] = X_embed[seq_idx]

    return seqs

def predict_sequence_prob(args, seq_of_interest, vocabulary, model,
                          verbose=False):
    seqs = { seq_of_interest: [ {} ] }
    X_cat, lengths = featurize_seqs(seqs, vocabulary)

    y_pred = model.predict(X_cat, lengths)
    assert(y_pred.shape[0] == len(seq_of_interest) + 2)

    return y_pred

def analyze_comb_fitness(
        args, model, vocabulary, strain, wt_seq, seqs_fitness,
        comb_batch=None, prob_cutoff=0., beta=1., verbose=True,
):
    from copy import deepcopy

    y_pred = predict_sequence_prob(
        args, wt_seq, vocabulary, model, verbose=verbose
    )

    word_pos_prob = {}
    for pos in range(len(wt_seq)):
        for word in vocabulary:
            word_idx = vocabulary[word]
            prob = y_pred[pos + 1, word_idx]
            if prob < prob_cutoff:
                continue
            word_pos_prob[(word, pos)] = prob

    base_embedding = embed_seqs(
        args, model, { wt_seq: [ {} ] }, vocabulary,
        use_cache=False, verbose=False
    )[wt_seq][0]['embedding']

    if comb_batch is None:
        comb_batch = len(seqs_fitness)
    seqs = sorted(seqs_fitness.keys())
    n_batches = math.ceil(float(len(seqs)) / comb_batch)

    for batchi in range(n_batches):
        start = batchi * comb_batch
        end = (batchi + 1) * comb_batch
        seqs_fitness_batch = {
            seq: deepcopy(seqs_fitness[seq])
            for seq in seqs[start:end]
        }

        seqs_fitness_batch = embed_seqs(
            args, model, seqs_fitness_batch, vocabulary,
            use_cache=False, verbose=False
        )

        data = []
        for mut_seq in seqs_fitness_batch:
            assert(len(mut_seq) == len(wt_seq))
            assert(len(seqs_fitness_batch[mut_seq]) == 1)
            meta = seqs_fitness_batch[mut_seq][0]
            if meta['strain'] != strain:
                continue

            mut_pos = set(meta['mut_pos'])
            raw_probs = []
            for idx, aa in enumerate(mut_seq):
                if idx in mut_pos:
                    raw_probs.append(word_pos_prob[(aa, idx)])
                else:
                    assert(aa == wt_seq[idx])
            assert(len(raw_probs) == len(mut_pos))

            grammar = np.sum(np.log10(raw_probs))
            sem_change = abs(base_embedding - meta['embedding']).sum()

            data.append([
                meta['strain'],
                meta['fitness'],
                meta['preference'],
                grammar,
                sem_change,
                sem_change + (beta * grammar),
            ])

        del seqs_fitness_batch

    df = pd.DataFrame(data, columns=[
        'strain', 'fitness', 'preference',
        'predicted', 'sem_change', 'cscs'
    ])

    print('\nStrain: {}'.format(strain))
    print('\tGrammaticality correlation:')
    print('\t\tSpearman r = {:.4f}, P = {:.4g}'
          .format(*ss.spearmanr(df.preference, df.predicted)))
    print('\t\tPearson rho = {:.4f}, P = {:.4g}'
          .format(*ss.pearsonr(df.preference, df.predicted)))

    print('\tSemantic change correlation:')
    print('\t\tSpearman r = {:.4f}, P = {:.4g}'
          .format(*ss.spearmanr(df.preference, df.sem_change)))
    print('\t\tPearson rho = {:.4f}, P = {:.4g}'
          .format(*ss.pearsonr(df.preference, df.sem_change)))

    plt.figure()
    plt.scatter(df.preference, df.predicted, alpha=0.3)
    plt.title(strain)
    plt.xlabel('Preference')
    plt.ylabel('Grammaticality')
    plt.savefig('figures/combinatorial_fitness_grammar_{}_{}.png'
                .format(args.namespace, strain), dpi=300)
    plt.close()

    plt.figure()
    plt.scatter(df.preference, df.sem_change, alpha=0.3)
    plt.title(strain)
    plt.xlabel('Preference')
    plt.ylabel('Semantic change')
    plt.savefig('figures/combinatorial_fitness_semantics_{}_{}.png'
                .format(args.namespace, strain), dpi=300)
    plt.close()

def analyze_semantics(args, model, vocabulary, seq_to_mutate, escape_seqs,
                      prob_cutoff=0., beta=1., plot_acquisition=True,
                      plot_namespace=None, verbose=True):
    if plot_acquisition:
        dirname = ('target/{}/semantics/cache'.format(args.namespace))
        mkdir_p(dirname)
        if plot_namespace is None:
            plot_namespace = args.namespace

    y_pred = predict_sequence_prob(
        args, seq_to_mutate, vocabulary, model, verbose=verbose
    )

    word_pos_prob = {}
    for i in range(len(seq_to_mutate)):
        for word in vocabulary:
            word_idx = vocabulary[word]
            prob = y_pred[i + 1, word_idx]
            word_pos_prob[(word, i)] = prob

    prob_sorted = sorted(word_pos_prob.items(), key=lambda x: -x[1])
    prob_seqs = { seq_to_mutate: [ {} ] }
    seq_prob = {}
    for (word, pos), prob in prob_sorted:
        mutable = seq_to_mutate[:pos] + word + seq_to_mutate[pos + 1:]
        seq_prob[mutable] = prob
        if prob >= prob_cutoff:
            prob_seqs[mutable] = [ {} ]

    seqs = np.array([ str(seq) for seq in sorted(seq_prob.keys()) ])

    if plot_acquisition:
        ofname = dirname + '/{}_mutations.txt'.format(args.namespace)
        with open(ofname, 'w') as of:
            of.write('orig\tmutant\n')
            for seq in seqs:
                try:
                    didx = [
                        c1 != c2 for c1, c2 in zip(seq_to_mutate, seq)
                    ].index(True)
                    of.write('{}\t{}\t{}\n'
                             .format(didx, seq_to_mutate[didx], seq[didx]))
                except ValueError:
                    of.write('NA\n')

    prob_seqs = embed_seqs(args, model, prob_seqs, vocabulary,
                           use_cache=False, verbose=verbose)
    base_embedding = prob_seqs[seq_to_mutate][0]['embedding']
    seq_change = {}
    for seq in seqs:
        if seq in prob_seqs:
            embedding = prob_seqs[seq][0]['embedding']
            # L1 distance between embedding vectors.
            seq_change[seq] = abs(base_embedding - embedding).sum()
        else:
            seq_change[seq] = 0

    prob = np.array([ seq_prob[seq] for seq in seqs ])
    change = np.array([ seq_change[seq] for seq in seqs ])

    escape_idx = np.array([
        ((seq in escape_seqs) and
         (sum([ m['significant'] for m in escape_seqs[seq] ]) > 0))
        for seq in seqs
    ])
    viable_idx = np.array([ seq in escape_seqs for seq in seqs ])

    if plot_acquisition:
        cache_fname = dirname + ('/plot_{}_{}.npz'
                                 .format(args.model_name, args.dim))
        np.savez_compressed(
            cache_fname, prob=prob, change=change,
            escape_idx=escape_idx, viable_idx=viable_idx,
        )
        from cached_semantics import cached_escape_semantics
        cached_escape_semantics(cache_fname, beta,
                                plot=plot_acquisition,
                                namespace=plot_namespace)

    return seqs, prob, change, escape_idx, viable_idx
