from utils import *

from intervaltree import IntervalTree

def load(virus):
    if virus == 'h1':
        escape_fname = ('results/flu/semantics/'
                        'analyze_semantics_flu_h1_bilstm_512.txt')
        region_fname = 'data/influenza/h1_regions.txt'
    elif virus == 'h3':
        escape_fname = ('results/flu/semantics/'
                        'analyze_semantics_flu_h3_bilstm_512.txt')
        region_fname = 'data/influenza/h3_regions.txt'
    elif virus == 'hiv':
        escape_fname = ('results/hiv/semantics/'
                        'analyze_semantics_hiv_bilstm_512.txt')
        region_fname = 'data/hiv/bg505_regions.txt'
    elif virus == 'sarscov2':
        escape_fname = ('results/cov/semantics/'
                        'analyze_semantics_cov_bilstm_512.txt')
        region_fname = 'data/cov/sarscov2_regions.txt'
    else:
        raise ValueError('Virus {} not supported'.format(virus))
    return escape_fname, region_fname

def regional_escape(virus, beta=1., n_permutations=100000):
    escape_fname, region_fname = load(virus)

    # Parse protein regions, keep track of intervals,
    # sizes and scores.

    regions = IntervalTree()
    name2size, name2escape = {}, {}
    with open(region_fname) as f:
        f.readline()
        for line in f:
            [ start, end, name ] = line.rstrip().split()
            start, end = int(start) - 1, int(end) - 1
            regions[start:(end + 1)] = name
            if name not in name2escape:
                name2escape[name] = []
                name2size[name] = 0
            name2size[name] += end - start + 1

    # Load semantic data into memory.

    data = []
    with open(escape_fname) as f:
        columns = f.readline().rstrip().split()
        for line in f:
            if line.rstrip().split()[2] in { 'U', 'B', 'J', 'X', 'Z' }:
                continue
            data.append(line.rstrip().split('\t'))
    df_all = pd.DataFrame(data, columns=columns)
    df_all['pos'] = pd.to_numeric(df_all['pos'])
    df_all['prob'] = pd.to_numeric(df_all['prob'])
    df_all['change'] = pd.to_numeric(df_all['change'])
    df_all['acquisition'] = ss.rankdata(df_all.change) + \
                            (beta * ss.rankdata(df_all.prob))

    # Reformat data for easy plotting and P-value computation.

    plot_data = []
    pos2scores = {}
    for i in range(len(df_all)):
        pos = df_all['pos'][i]
        acquisition = df_all['acquisition'][i]
        names = regions[pos]
        for name in names:
            name2escape[name.data].append(acquisition)
            plot_data.append([ name.data, acquisition ])
        if pos not in pos2scores:
            pos2scores[pos] = []
        pos2scores[pos].append(acquisition)

    # Compute permutation-based P-value for each region.

    seq_start = min(df_all['pos'])
    seq_end = max(df_all['pos'])
    all_pos = list(range(seq_start, seq_end + 1))
    plot_data = []

    for name in name2escape:
        real_score = np.mean(name2escape[name])
        size = name2size[name]
        null_distribution = []
        for perm in range(n_permutations):
            rand_positions = np.random.choice(all_pos, size=size,
                                              replace=False)
            null_score = np.concatenate([
                np.array(pos2scores[pos]) for pos in rand_positions
            ]).mean()
            null_distribution.append(null_score)
        null_distribution = np.array(null_distribution)

        tprint('Enriched for escapes:')
        p_val = (sum(null_distribution >= real_score)) / \
                (n_permutations)
        if p_val == 0:
            p_val = 1. / n_permutations
            tprint('{}, P < {}'.format(name, p_val))
        else:
            tprint('{}, P = {}'.format(name, p_val))
        plot_data.append([ name, -np.log10(p_val), 'enriched' ])

        tprint('Depleted for escapes:')
        p_val = (sum(null_distribution <= real_score)) / \
                (n_permutations)
        if p_val == 0:
            p_val = 1. / n_permutations
            tprint('{}, P < {}'.format(name, p_val))
        else:
            tprint('{}, P = {}'.format(name, p_val))
        plot_data.append([ name, -np.log10(p_val), 'depleted' ])

        tprint('')

    # Plot each region in bar plot.

    plot_data = pd.DataFrame(plot_data,
                             columns=[ 'region', 'score', 'direction' ])
    plt.figure()
    sns.barplot(data=plot_data, x='region', y='score', hue='direction',
                order=sorted(set(plot_data['region'])))
    fdr = 0.05 / len(sorted(set(plot_data['region'])))
    plt.axhline(y=-np.log10(fdr), color='gray', linestyle='--')
    plt.xticks(rotation=60)
    plt.savefig('figures/regional_escape_{}.svg'.format(virus))

if __name__ == '__main__':
    virus = sys.argv[1]

    regional_escape(virus)
