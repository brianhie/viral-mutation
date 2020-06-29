from utils import *

def load_phylo():
    data = []
    with open('cluster_purity.log') as f:
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
                data.append([ virus, entry, 'phylo', cluster, value ])
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
    data = load_phylo() + load_louvain('flu') + load_louvain('hiv')
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
            plt.ylim([ 0.75, 1.05 ])
            plt.savefig('figures/cluster_purity_{}_{}.svg'
                        .format(virus, entry))

            phylo = df_subset[df_subset['cluster_type'] == 'phylo'].value
            louvain = df_subset[df_subset['cluster_type'] == 'louvain'].value

            print('{} {}'.format(virus, entry))
            print('t-test t = {}, P = {}'
                  .format(*ss.ttest_ind(phylo, louvain)))
            print('')
