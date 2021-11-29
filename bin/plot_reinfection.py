from utils import *

def plot_reinfection(namespace='reinfection'):
    with open('target/cov/reinfection/cache/cov_mut_{}.txt'
              .format(namespace)) as f:
        fields = f.readline().rstrip().split()
        base_prob, base_change = float(fields[2]), float(fields[3])

        n_hprob_hchange = 0
        n_hprob_lchange = 0
        n_lprob_hchange = 0
        n_lprob_lchange = 0
        probs, changes, higher = [], [], []
        for line in f:
            fields = line.rstrip().split()
            prob, change = float(fields[2]), float(fields[3])
            if prob > base_prob and change > base_change:
                n_hprob_hchange += 1
                higher.append(fields[0])
            elif prob > base_prob and change <= base_change:
                n_hprob_lchange += 1
            elif prob <= base_prob and change > base_change:
                n_lprob_hchange += 1
            else:
                n_lprob_lchange += 1
            probs.append(prob)
            changes.append(np.log10(change))

        print('Higher prob, higher change: {}'.format(n_hprob_hchange))
        print('Higher prob, lower change: {}'.format(n_hprob_lchange))
        print('Lower prob, higher change: {}'.format(n_lprob_hchange))
        print('Lower prob, lower change: {}'.format(n_lprob_lchange))

        p = ss.fisher_exact([[ n_hprob_hchange, n_hprob_lchange ],
                             [ n_lprob_hchange, n_lprob_lchange ]])[1]
        print('Fisher\'s exact P = {}'.format(p))

        print('Higher:')
        print('\t' + '\n\t'.join(higher))

        plt.figure(figsize=(4, 4))
        plt.scatter(probs, changes, c='#aaaaaa', s=10)
        plt.scatter([ base_prob ], [ np.log10(base_change) ], c='k')
        plt.axhline(y=np.log10(base_change), color='k', linestyle='--')
        plt.axvline(x=base_prob, color='k', linestyle='--')
        plt.savefig('figures/cov_reinfection_{}.png'
                    .format(namespace), dpi=300)
        plt.close()

if __name__ == '__main__':
    plot_reinfection(sys.argv[1])
