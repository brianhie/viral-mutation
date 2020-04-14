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

    model = get_model(args, seq_len, vocab_size, batch_size=32)

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
    sorted_seqs = np.array([ str(s) for s in sorted(seqs.keys()) ])
    batch_size = 10000
    n_batches = math.ceil(len(sorted_seqs) / float(batch_size))
    for batchi in range(n_batches):
        start = batchi * batch_size
        end = (batchi + 1) * batch_size
        seqs_batch = { seq: seqs[seq] for seq in sorted_seqs[start:end] }
        seqs_batch = embed_seqs(args, model, seqs_batch, vocabulary,
                                use_cache=False)
        for seq in seqs_batch:
            for meta in seqs[seq]:
                meta['embedding'] = seqs_batch[seq][0]['embedding'].mean(0)
        del seqs_batch

    X, obs = [], {}
    obs['n_seq'] = []
    obs['seq'] = []
    for seq in seqs:
        meta = seqs[seq][0]
        X.append(meta['embedding'])
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
        tprint(model.model_.summary())

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

        from escape import load_dingens2019
        tprint('Dingens et al. 2019...')
        seq_to_mutate, escape_seqs = load_dingens2019()

        cache_fname = ('target/hiv/semantics/cache/plot_{}_{}.npz'
                       .format(args.model_name, args.dim))
        analyze_semantics(
            args, model, vocabulary, seq_to_mutate, escape_seqs,
            prob_cutoff=0., beta=1., plot_acquisition=True,
            cache_fname=cache_fname,
        )
