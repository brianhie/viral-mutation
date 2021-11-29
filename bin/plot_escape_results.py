from utils import *

def load_data():
    data = []
    fname = 'data/escape_results.txt'
    with open(fname) as f:
        header = f.readline().rstrip().split('\t')
        strains = [ field.split()[-1].rstrip(')') for field in header[1:] ]
        for line in f:
            fields = line.rstrip().split('\t')
            model = fields[0]
            values = [ float(field) for field in fields[1:] ]
            assert(len(strains) == len(values))
            for strain, value in zip(strains, values):
                data.append([ model, strain, value ])
    df = pd.DataFrame(data, columns=[ 'model', 'strain', 'value' ])
    return df

def plot_escape_benchmark(df):
    model_order = [
        # Fitness/"grammaticality" first.
        'mafft', 'EVcouplings (indep)', 'EVcouplings (epist)',
        # Then semantic change.
        'Bepler', 'TAPE transformer', 'UniRep',
        # Then CSCS.
        'CSCS',
    ]
    colors = [
        '#dbe3e2', '#dbe3e2', '#dbe3e2',
        '#b7c8c4', '#b7c8c4', '#b7c8c4',
        '#6f8a91',
    ]

    strains = sorted(set(df['strain']))
    for strain in strains:
        df_subset = df[(df['strain'] == strain)]

        plt.figure()
        sns.barplot(data=df_subset, x='model', y='value',
                    order=model_order, palette=colors)
        plt.ylim([ 0.4, 0.9 ])
        plt.axhline(y=0.5, color='gray', linestyle='--')
        plt.savefig('figures/escape_barplot_benchmark_{}.svg'
                    .format(strain))

if __name__ == '__main__':
    df = load_data()

    plot_escape_benchmark(df)
