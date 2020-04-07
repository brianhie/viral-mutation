from mutation import *

from Bio import BiopythonWarning
from Bio import SeqIO

np.random.seed(1)
random.seed(1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Flu sequence analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., hmm, lstm)')
    parser.add_argument('--namespace', type=str, default='flu',
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
                                    if value != '-N/A-' else None
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
            if 'Reference_Perth2009_HA_coding_sequence' in record.description:
                continue
            if record.seq not in seqs:
                seqs[record.seq] = []
            accession = record.description.split('|')[0].split(':')[1]
            seqs[record.seq].append(metas[accession])
    return seqs

def split_seqs(seqs, split_method='random'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BiopythonWarning)

        train_seqs, test_seqs, val_seqs = {}, {}, {}

        old_cutoff = 1990
        new_cutoff = 2018

        tprint('Splitting seqs...')
        for seq in seqs:
            # Pick validation set based on date.
            seq_dates = [
                meta['Collection Date'] for meta in seqs[seq]
                if meta['Collection Date'] is not None
            ]
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
    fnames = [ 'data/influenza/ird_influenzaA_HA_allspecies.fa' ]
    meta_fnames = [ 'data/influenza/ird_influenzaA_HA_allspecies_meta.tsv' ]

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

def plot_umap_time(adata):
    sc.pp.neighbors(adata, n_neighbors=100, use_rep='X')
    sc.tl.umap(adata, min_dist=0.1, n_components=1)
    adata.obsm['X_umap'] = np.hstack([
        np.array(adata.obs['Collection Date']).reshape(-1, 1),
        adata.obsm['X_umap']
    ])
    sc.pl.umap(adata, color='Host Species', save='_time_species.png')
    sc.pl.umap(adata, color='Subtype', save='_time_subtype.png')
    sc.pl.umap(adata, color='louvain', save='_time_louvain.png')

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
    plot_umap_time(adata_human)

    interpret_clusters(adata)
    #seq_clusters(adata)

if __name__ == '__main__':
    args = parse_args()

    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V', 'X', 'Z', 'J', 'U', 'B', 'Z'
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

        from escape import load_lee2018, load_lee2019

        #tprint('Lee et al. 2018...')
        #seq_to_mutate, escape_seqs = load_lee2018()
        #analyze_semantics(args, model, vocabulary, seq_to_mutate, escape_seqs,
        #                  prob_cutoff=1e-6, beta=0.,)
        #tprint('')

        tprint('Lee et al. 2019...')
        seq_to_mutate, escape_seqs = load_lee2019()
        analyze_semantics(args, model, vocabulary, seq_to_mutate, escape_seqs,
                          prob_cutoff=0., beta=1., plot_acquisition=True,)
