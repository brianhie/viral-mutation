from utils import *

def load_data():
    data = []
    fname = 'data/fitness_results.txt'
    with open(fname) as f:
        header = f.readline().rstrip().split('\t')
        proteins = [ field.split(', ')[0] for field in header[1:] ]
        strains = [ field.split(', ')[1] for field in header[1:] ]
        for line in f:
            fields = line.rstrip().split('\t')
            model = fields[0]
            values = [ float(field) for field in fields[1:] ]
            assert(len(proteins) == len(strains) == len(values))
            for prot, strain, value in zip(proteins, strains, values):
                if np.isnan(value):
                    continue
                data.append([ model, prot, strain, value ])
    df = pd.DataFrame(data, columns=[ 'model', 'prot', 'strain', 'value' ])
    return df

def plot_cscs_fitness(df):
    df_subset = df[(df['model'] == 'Semantic change') |
                   (df['model'] == 'Grammaticality')]

    plt.figure()
    sns.barplot(data=df_subset, x='strain', y='value', hue='model',
                palette=[ '#ADD8E6', '#FFAAAA' ])
    plt.savefig('figures/fitness_barplot_cscs.svg')

def plot_fitness_benchmark(df):
    model_order = [ 'mafft', 'EVcouplings (indep)', 'EVcouplings (epist)',
                    'Grammaticality' ]
    for strain in [ 'WSN33', 'BF520', 'BG505' ]:
        df_subset = df[(df['strain'] == strain) &
                       (df['model'] != 'Semantic change')]
        plt.figure()
        sns.barplot(data=df_subset, x='model', y='value',
                    order=model_order)
        plt.savefig('figures/fitness_barplot_benchmark_{}.svg'
                    .format(strain))

if __name__ == '__main__':
    df = load_data()

    plot_cscs_fitness(df)

    plot_fitness_benchmark(df)
