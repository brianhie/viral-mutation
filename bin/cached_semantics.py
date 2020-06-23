from utils import *

from sklearn.metrics import auc

def cached_escape_semantics(cache_fname, beta, plot=True,
                            namespace='semantics'):
    data = np.load(cache_fname)
    prob = data['prob']
    change = data['change']
    escape_idx = data['escape_idx']
    viable_idx = data['viable_idx']

    acquisition = ss.rankdata(change) + (beta * ss.rankdata(prob))

    pos_change_idx = change > 0

    pos_change_escape_idx = np.logical_and(pos_change_idx, escape_idx)
    escape_prob = prob[pos_change_escape_idx]
    escape_change = change[pos_change_escape_idx]
    prob = prob[pos_change_idx]
    change = change[pos_change_idx]

    log_prob, log_change = np.log10(prob), np.log10(change)
    log_escape_prob, log_escape_change = (np.log10(escape_prob),
                                          np.log10(escape_change))

    if plot:
        plt.figure()
        plt.scatter(log_prob, log_change, c=acquisition[pos_change_idx],
                    cmap='viridis', alpha=0.3)
        plt.scatter(log_escape_prob, log_escape_change, c='red',
                    alpha=0.5, marker='x')
        plt.xlabel(r'$ \log_{10}(\hat{p}(x_i | \mathbf{x}_{[N] ∖ \{i\} })) $')
        plt.ylabel(r'$ \log_{10}(\Delta \mathbf{\hat{z}}) $')
        plt.savefig('figures/{}_acquisition.png'
                    .format(namespace), dpi=300)
        plt.close()

        rand_idx = np.random.choice(len(prob), len(escape_prob))
        plt.figure()
        plt.scatter(log_prob, log_change, c=acquisition[pos_change_idx],
                    cmap='viridis', alpha=0.3)
        plt.scatter(log_prob[rand_idx], log_change[rand_idx], c='red',
                    alpha=0.5, marker='x')
        plt.xlabel(r'$ \log_{10}(\hat{p}(x_i | \mathbf{x}_{[N] ∖ \{i\} })) $')
        plt.ylabel(r'$ \log_{10}(\Delta \mathbf{\hat{z}}) $')
        plt.savefig('figures/{}_acquisition_rand.png'
                    .format(namespace), dpi=300)
        plt.close()

    acq_argsort = ss.rankdata(-acquisition)
    escape_rank_dist = acq_argsort[escape_idx]

    size = len(prob)
    print('Number of escape seqs: {} / {}'
          .format(len(escape_rank_dist), sum(escape_idx)))
    print('Mean rank: {} / {}'.format(np.mean(escape_rank_dist), size))
    print('Median rank: {} / {}'.format(np.median(escape_rank_dist), size))
    print('Min rank: {} / {}'.format(np.min(escape_rank_dist), size))
    print('Max rank: {} / {}'.format(np.max(escape_rank_dist), size))
    print('Rank stdev: {} / {}'.format(np.std(escape_rank_dist), size))

    max_consider = len(prob)
    n_consider = np.array([ i + 1 for i in range(max_consider) ])

    n_escape = np.array([ sum(escape_rank_dist <= i + 1)
                          for i in range(max_consider) ])
    norm = max(n_consider) * max(n_escape)
    norm_auc = auc(n_consider, n_escape) / norm

    escape_rank_prob = ss.rankdata(-data['prob'])[escape_idx]
    n_escape_prob = np.array([ sum(escape_rank_prob <= i + 1)
                               for i in range(max_consider) ])
    norm_auc_prob = auc(n_consider, n_escape_prob) / norm

    escape_rank_change = ss.rankdata(-data['change'])[escape_idx]
    n_escape_change = np.array([ sum(escape_rank_change <= i + 1)
                                 for i in range(max_consider) ])
    norm_auc_change = auc(n_consider, n_escape_change) / norm

    if plot:
        plt.figure()
        plt.plot(n_consider, n_escape)
        plt.plot(n_consider, n_escape_change, c='C0', linestyle='-.')
        plt.plot(n_consider, n_escape_prob, c='C0', linestyle=':')
        plt.plot(n_consider, n_consider * (len(escape_prob) / len(prob)),
                 c='gray', linestyle='--')

        plt.xlabel(r'$ \log_{10}() $')
        plt.ylabel(r'$ \log_{10}(\Delta \mathbf{\hat{z}}) $')

        plt.legend([
            r'$ \Delta \mathbf{\hat{z}} + ' +
            r'\beta \hat{p}(x_i | \mathbf{x}_{[N] ∖ \{i\} }) $,' +
            (' AUC = {:.3f}'.format(norm_auc)),
            r'$  \Delta \mathbf{\hat{z}} $ only,' +
            (' AUC = {:.3f}'.format(norm_auc_change)),
            r'$ \hat{p}(x_i | \mathbf{x}_{[N] ∖ \{i\} }) $ only,' +
            (' AUC = {:.3f}'.format(norm_auc_prob)),
            'Random guessing, AUC = 0.500'
        ])
        plt.xlabel('Top N')
        plt.ylabel('Number of escape mutations in top N')
        plt.savefig('figures/{}_consider_escape.png'
                    .format(namespace), dpi=300)
        plt.close()


    print('Escape semantics, beta = {} [{}]'
          .format(beta, namespace))

    print('AUC (CSCS): {}'.format(norm_auc))
    print('AUC (semantic change only): {}'.format(norm_auc_change))
    print('AUC (grammaticality only): {}'.format(norm_auc_prob))

    print('{:.4g} (mean log prob), {:.4g} (mean log prob escape), '
          '{:.4g} (p-value)'
          .format(log_prob.mean(), log_escape_prob.mean(),
                  ss.mannwhitneyu(log_prob, log_escape_prob,
                                  alternative='two-sided')[1]))
    print('{:.4g} (mean log change), {:.4g} (mean log change escape), '
          '{:.4g} (p-value)'
          .format(change.mean(), escape_change.mean(),
                  ss.mannwhitneyu(change, escape_change,
                                  alternative='two-sided')[1]))

if __name__ == '__main__':
    cached_escape_semantics(sys.argv[1], 1.)
