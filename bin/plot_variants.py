import matplotlib.pyplot as plt
import pandas as pd
import sys

if __name__ == '__main__':
    fname = sys.argv[1]

    df = pd.read_csv(fname, sep='\t')

    plt.figure(figsize=(4, 6))
    plt.scatter(
        df['Grammaticality'],
        df['Semantic change']
    )
    plt.xlabel('Grammaticality')
    plt.ylabel('Semantic change')
    plt.savefig('figures/cov_fasta.png')
    plt.close()
