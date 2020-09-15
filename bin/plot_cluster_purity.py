from utils import *

def load_phylo(namespace=None):
    if namespace is None:
        name = 'phylo'
        namespace = ''
    else:
        name = namespace
        namespace = '_' + namespace

    data = []
    with open('target/log/cluster_purity{}.log'.format(namespace)) as f:
        for line in f:
            line = ' | '.join(line.split(' | ')[1:]).rstrip()
            if line.startswith('Flu HA'):
                virus = 'flu'
                continue
            if line.startswith('HIV Env'):
                virus = 'hiv'
                continue
            if line.startswith('Calculating purity of '):
                entry = line[len('Calculating purity of '):]
                entry = entry.lower()
                if entry == 'host species':
                    entry = 'species'
                continue
            if line.startswith('Cluster'):
                fields = line.split()
                cluster = fields[1].rstrip(',')
                value = float(fields[-1])
                data.append([ virus, entry, name, cluster, value ])
    return data

def load_louvain(virus):
    data = []
    fname = '{}_embed.log'.format(virus)
    with open(fname) as f:
        for line in f:
            line = ' | '.join(line.split(' | ')[1:]).rstrip()
            if line.startswith('\tCluster '):
                fields = line.lstrip().split()
                cluster = fields[1].rstrip(',')
                entry = fields[3]
                value = float(fields[-1])
                data.append([ virus, entry, 'louvain', cluster, value ])
    return data

if __name__ == '__main__':
    data = load_phylo('mafft') + \
           load_phylo('mafft_sl') + \
           load_phylo('clustalomega') + \
           load_phylo('clustalomega_sl') + \
           load_phylo('mrbayes') + \
           load_phylo('mrbayes_sl') + \
           load_phylo('raxml') + \
           load_phylo('raxml_sl') + \
           load_phylo('fasttree') + \
           load_phylo('fasttree_sl') + \
           load_louvain('flu') + \
           load_louvain('hiv')
    df = pd.DataFrame(data, columns=[
        'virus', 'entry', 'cluster_type', 'cluster', 'value'
    ])

    viruses = sorted(set(df['virus']))
    entries = sorted(set(df['entry']))

    for virus in viruses:
        for entry in entries:
            df_subset = df[(df['virus'] == virus) & (df['entry'] == entry)]
            if len(df_subset) == 0:
                continue
            plt.figure()
            sns.barplot(data=df_subset, x='cluster_type', y='value',
                        capsize=0.5)
            plt.title('{} {}'.format(virus, entry))
            plt.ylim([ 0.65, 1.05 ])
            plt.savefig('figures/cluster_purity_{}_{}.svg'
                        .format(virus, entry))
