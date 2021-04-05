from utils import *

if __name__ == '__main__':
    fname = sys.argv[1]
    namespace = fname.split('.')[0].split('_')[-1]

    method_dict = {
        'Bepler': 'Bepler and Berger',
        'epistatic': 'EVcouplings (epi)',
        'independent': 'EVcouplings (ind)',
        'Mutation frequency': 'MAFFT MSA',
        'tape_transformer': 'TAPE',
        'unirep': 'UniRep',
    }

    data = []
    with open(fname) as f:
        for line in f:
            if line.startswith('cutoff = '):
                cutoff = float(line.rstrip().split()[-1])
                continue

            if ' | Results for ' in line:
                method = method_dict[line.rstrip().split('(')[-1].rstrip('):')]
                continue
            elif line.startswith('AUC ('):
                method = line.rstrip().split('(')[-1].split(')')[0].rstrip(')')
                if 'CSCS' in line:
                    auc = float(line.rstrip().split(',')[0].split()[-1])
                    sub = False
                else:
                    auc = float(line.split()[-1])
                    sub = True
                data.append([ cutoff, method, auc, sub ])
                continue

            if ' | AUC = ' in line:
                auc = float(line.split()[-1])
                data.append([ cutoff, method, auc, False ])
                continue

    df = pd.DataFrame(data, columns=[ 'cutoff', 'method', 'auc', 'sub' ])

    plt.figure()
    sns.lineplot(data=df[df['sub'] == False], x='cutoff', y='auc', hue='method',)
    plt.savefig(f'figures/sweep_{namespace}.svg')
    plt.close()

    plt.figure()
    sns.lineplot(data=df[df['sub'] | (df['method'] == 'CSCS')], x='cutoff', y='auc', hue='method',)
    plt.savefig(f'figures/sweep_decomposed_{namespace}.svg')
    plt.close()

    for method in set(df['method']):
        print('{}: Mean AUC = {}'.format(
            method, np.mean(df[df['method'] == method]['auc'])
        ))
