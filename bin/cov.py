from mutation import *

np.random.seed(1)
random.seed(1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Coronavirus sequence analysis')
    parser.add_argument('model_name', type=str,
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
    parser.add_argument('--combfit', action='store_true',
                        help='Analyze combinatorial fitness')
    parser.add_argument('--reinfection', action='store_true',
                        help='Analyze reinfection cases')
    parser.add_argument('--ukmut', action='store_true',
                        help='Analyze reinfection cases')
    args = parser.parse_args()
    return args

def parse_viprbrc(entry):
    fields = entry.split('|')
    if fields[7] == 'NA':
        date = None
    else:
        date = fields[7].split('/')[0]
        date = dparse(date.replace('_', '-'))

    country = fields[9]
    from locations import country2continent
    if country in country2continent:
        continent = country2continent[country]
    else:
        country = 'NA'
        continent = 'NA'

    from mammals import species2group

    meta = {
        'strain': fields[5],
        'host': fields[8],
        'group': species2group[fields[8]],
        'country': country,
        'continent': continent,
        'dataset': 'viprbrc',
    }
    return meta

def parse_nih(entry):
    fields = entry.split('|')

    country = fields[3]
    from locations import country2continent
    if country in country2continent:
        continent = country2continent[country]
    else:
        country = 'NA'
        continent = 'NA'

    meta = {
        'strain': 'SARS-CoV-2',
        'host': 'human',
        'group': 'human',
        'country': country,
        'continent': continent,
        'dataset': 'nih',
    }
    return meta

def parse_gisaid(entry):
    fields = entry.split('|')

    type_id = fields[1].split('/')[1]

    if type_id in { 'bat', 'canine', 'cat', 'env', 'mink',
                    'pangolin', 'tiger' }:
        host = type_id
        country = 'NA'
        continent = 'NA'
    else:
        host = 'human'
        from locations import country2continent
        if type_id in country2continent:
            country = type_id
            continent = country2continent[country]
        else:
            country = 'NA'
            continent = 'NA'

    from mammals import species2group

    meta = {
        'strain': fields[1],
        'host': host,
        'group': species2group[host].lower(),
        'country': country,
        'continent': continent,
        'dataset': 'gisaid',
    }
    return meta

def process(fnames):
    seqs = {}
    for fname in fnames:
        for record in SeqIO.parse(fname, 'fasta'):
            if len(record.seq) < 1000:
                continue
            if str(record.seq).count('X') > 0:
                continue
            if record.seq not in seqs:
                seqs[record.seq] = []
            if fname == 'data/cov/viprbrc_db.fasta':
                meta = parse_viprbrc(record.description)
            elif fname == 'data/cov/gisaid.fasta':
                meta = parse_gisaid(record.description)
            else:
                meta = parse_nih(record.description)
            meta['accession'] = record.description
            seqs[record.seq].append(meta)

    with open('data/cov/cov_all.fa', 'w') as of:
        for seq in seqs:
            metas = seqs[seq]
            for meta in metas:
                of.write('>{}\n'.format(meta['accession']))
                of.write('{}\n'.format(str(seq)))

    return seqs

def split_seqs(seqs, split_method='random'):
    train_seqs, test_seqs = {}, {}

    tprint('Splitting seqs...')
    for idx, seq in enumerate(seqs):
        if idx % 10 < 2:
            test_seqs[seq] = seqs[seq]
        else:
            train_seqs[seq] = seqs[seq]
    tprint('{} train seqs, {} test seqs.'
           .format(len(train_seqs), len(test_seqs)))

    return train_seqs, test_seqs

def setup(args):
    fnames = [ 'data/cov/sars_cov2_seqs.fa',
               'data/cov/viprbrc_db.fasta',
               'data/cov/gisaid.fasta' ]

    seqs = process(fnames)

    seq_len = max([ len(seq) for seq in seqs ]) + 2
    vocab_size = len(AAs) + 2

    model = get_model(args, seq_len, vocab_size,
                      inference_batch_size=1200)

    return model, seqs

def interpret_clusters(adata):
    clusters = sorted(set(adata.obs['louvain']))
    for cluster in clusters:
        tprint('Cluster {}'.format(cluster))
        adata_cluster = adata[adata.obs['louvain'] == cluster]
        for var in [ 'host', 'country', 'strain' ]:
            tprint('\t{}:'.format(var))
            counts = Counter(adata_cluster.obs[var])
            for val, count in counts.most_common():
                tprint('\t\t{}: {}'.format(val, count))
        tprint('')

def plot_umap(adata, categories, namespace='cov'):
    for category in categories:
        sc.pl.umap(adata, color=category,
                   save='_{}_{}.png'.format(namespace, category))

def analyze_embedding(args, model, seqs, vocabulary):
    seqs = embed_seqs(args, model, seqs, vocabulary, use_cache=True)

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

    sc.pp.neighbors(adata, n_neighbors=20, use_rep='X')
    sc.tl.louvain(adata, resolution=1.)
    sc.tl.umap(adata, min_dist=1.)

    sc.set_figure_params(dpi_save=500)
    plot_umap(adata, [ 'host', 'group', 'continent', 'louvain' ])

    interpret_clusters(adata)

    adata_cov2 = adata[(adata.obs['louvain'] == '0') |
                       (adata.obs['louvain'] == '2')]
    plot_umap(adata_cov2, [ 'host', 'group', 'country' ],
              namespace='cov7')

def make_mutant(wt_seq, mutations):
    indiv_muts, mut_seq = [], wt_seq[:]
    for mutation in mutations:
        aa_orig = mutation[0]
        if 'del' in mutation:
            aa_pos = int(mutation[1:-3]) - 1
            aa_mut = '-'
        elif '|' in mutation and 'ins' in mutation:
            aa_pos = int(mutation.split('|')[0][1:]) - 1
            aa_mut = mutation.split('|')[1][3:]
        else:
            aa_pos = int(mutation[1:-1]) - 1
            aa_mut = mutation[-1]
        assert(mut_seq[aa_pos] == aa_orig)
        mut_seq = mut_seq[:aa_pos] + aa_mut + mut_seq[aa_pos + 1:]
    mut_seq = mut_seq.replace('-', '')
    return mut_seq

def analyze_uk_mutation(args, model, seqs, vocabulary):
    uk_mutations = [
        'H69del', 'V70del', 'Y145del', 'N501Y', 'A570D',
        'P681H', 'T716I', 'S982A',  'D1118H'
    ]
    sa_mutations = [
        'D80A', 'D215G', 'K417N', 'E484K', 'N501Y', 'A701V'
    ]
    andreano_mutations = [
        'F140del', 'E484K', 'Y248|insKTRNKSTSRRE|L249'
    ]

    names = [ 'uk', 'sa', 'andreano' ]
    mutations_list = [ uk_mutations, sa_mutations, andreano_mutations ]

    wt_seq = str(SeqIO.read('data/cov/cov2_spike_wt.fasta', 'fasta').seq)

    seqs = embed_seqs(
        args, model, seqs, vocabulary, use_cache=True, namespace='cov2_null'
    )
    wt_embedding = seqs[wt_seq][0]['embedding']

    sorted_seqs = sorted([
        seq for seq in seqs
        if ((seqs[seq][0]['strain'] == 'SARS-CoV-2' or
             'hCoV-19' in seqs[seq][0]['strain']) and
            seq != wt_seq and seq.startswith('M'))
    ])

    null_changes = np.array([
        np.linalg.norm(seqs[seq][0]['embedding'].mean(0) - wt_embedding.mean(0))
        for seq in sorted_seqs
    ])

    for name, mutations in zip(names, mutations_list):

        mut_seq = make_mutant(wt_seq, mutations)

        mut_embedding = embed_seqs(
            args, model, { mut_seq: [ {} ] }, vocabulary, verbose=False,
        )[mut_seq][0]['embedding']

        mut_change = np.linalg.norm(mut_embedding.mean(0) - wt_embedding.mean(0))

        print('{}: Change percentile = {}%'.format(
            name, ss.percentileofscore(null_changes, mut_change)
        ))

        for idx in np.argwhere(null_changes > mut_change).ravel():
            print('\t' + sorted_seqs[idx])

        for mutation in mutations:
            mut_seq = make_mutant(wt_seq, [ mutation ])
            embedding = embed_seqs(
                args, model, { mut_seq: [ {} ] }, vocabulary, verbose=False,
            )[mut_seq][0]['embedding']
            change = np.linalg.norm(embedding.mean(0) - wt_embedding.mean(0))
            print('\tMutation {}: {}'.format(mutation, change))


if __name__ == '__main__':
    args = parse_args()

    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V', 'X', 'Z', 'J', 'U', 'B',
    ]
    vocabulary = { aa: idx + 1 for idx, aa in enumerate(sorted(AAs)) }

    model, seqs = setup(args)

    if 'esm' in args.model_name:
        args.checkpoint = args.model_name
    elif args.checkpoint is not None:
        model.model_.load_weights(args.checkpoint)
        tprint('Model summary:')
        tprint(model.model_.summary())

    if args.train:
        batch_train(args, model, seqs, vocabulary, batch_size=1000)

    if args.train_split or args.test:
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

        from escape import load_baum2020, load_greaney2020
        tprint('Baum et al. 2020...')
        seq_to_mutate, seqs_escape = load_baum2020()
        analyze_semantics(args, model, vocabulary,
                          seq_to_mutate, seqs_escape, comb_batch=10000,
                          prob_cutoff=0, beta=1., plot_acquisition=True,)
        tprint('Greaney et al. 2020...')
        seq_to_mutate, seqs_escape = load_greaney2020()
        analyze_semantics(args, model, vocabulary,
                          seq_to_mutate, seqs_escape, comb_batch=10000,
                          min_pos=318, max_pos=540, # Restrict to RBD.
                          prob_cutoff=0, beta=1., plot_acquisition=True,
                          plot_namespace='cov2rbd')

    if args.combfit:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')

        from combinatorial_fitness import load_starr2020
        tprint('Starr et al. 2020...')
        wt_seqs, seqs_fitness = load_starr2020()
        strains = sorted(wt_seqs.keys())
        for strain in strains:
            analyze_comb_fitness(args, model, vocabulary,
                                 strain, wt_seqs[strain], seqs_fitness,
                                 comb_batch=10000, prob_cutoff=0., beta=1.)

    if args.reinfection:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')

        from reinfection import load_to2020, load_ratg13, load_sarscov1
        from plot_reinfection import plot_reinfection

        tprint('To et al. 2020...')
        wt_seq, mutants = load_to2020()
        analyze_reinfection(args, model, seqs, vocabulary, wt_seq, mutants,
                            namespace='to2020')
        plot_reinfection(namespace='to2020')
        null_combinatorial_fitness(args, model, seqs, vocabulary,
                                   wt_seq, mutants, n_permutations=100000000,
                                   namespace='to2020')

        tprint('Positive controls...')
        wt_seq, mutants = load_ratg13()
        analyze_reinfection(args, model, seqs, vocabulary, wt_seq, mutants,
                            namespace='ratg13')
        plot_reinfection(namespace='ratg13')
        wt_seq, mutants = load_sarscov1()
        analyze_reinfection(args, model, seqs, vocabulary, wt_seq, mutants,
                            namespace='sarscov1')
        plot_reinfection(namespace='sarscov1')

    if args.ukmut:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')

        analyze_uk_mutation(args, model, seqs, vocabulary)
