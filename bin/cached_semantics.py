from utils import *

from sklearn.metrics import auc

def compute_p(true_val, n_interest, n_total, n_permutations=10000):
    null_distribution = []
    norm = n_interest * n_total
    for _ in range(n_permutations):
        interest = set(np.random.choice(n_total, size=n_interest,
                                        replace=False))
        n_acquired = 0
        acquired, total = [], []
        for i in range(n_total):
            if i in interest:
                n_acquired += 1
            acquired.append(n_acquired)
            total.append(i + 1)
        null_distribution.append(auc(total, acquired) / norm)
    null_distribution = np.array(null_distribution)
    return sum(null_distribution >= true_val) / n_permutations

def cached_escape(cache_fname, beta,
                  cutoff=None, expr_cutoff=None, bind_cutoff=None,
                  plot=True, namespace='semantics'):
    if 'flu_h1' in cache_fname:
        from escape import load_doud2018
        if cutoff is None:
            wt_seq, seqs_escape = load_doud2018()
        else:
            wt_seq, seqs_escape = load_doud2018(survival_cutoff=cutoff)
    elif 'flu_h3' in cache_fname:
        from escape import load_lee2019
        if cutoff is None:
            wt_seq, seqs_escape = load_lee2019()
        else:
            wt_seq, seqs_escape = load_lee2019(survival_cutoff=cutoff)
    elif 'hiv' in cache_fname:
        from escape import load_dingens2019
        if cutoff is None:
            wt_seq, seqs_escape = load_dingens2019()
        else:
            wt_seq, seqs_escape = load_dingens2019(survival_cutoff=cutoff)
    elif '_cov_' in cache_fname:
        from escape import load_baum2020
        wt_seq, seqs_escape = load_baum2020()
    elif 'cov2rbd' in cache_fname:
        from escape import load_greaney2020
        if cutoff is None:
            wt_seq, seqs_escape = load_greaney2020()
        elif expr_cutoff is not None:
            wt_seq, seqs_escape = load_greaney2020(expr_cutoff=expr_cutoff)
        else:
            wt_seq, seqs_escape = load_greaney2020(survival_cutoff=cutoff)
    else:
        raise ValueError('invalid option {}'.format(cache_fname))

    prob, change, escape_idx, viable_idx = [], [], [], []
    with open(cache_fname) as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split('\t')
            pos = int(fields[0])
            if 'rbd' in cache_fname:
                if pos < 330 or pos > 530:
                    continue
            if fields[2] in { 'U', 'B', 'J', 'X', 'Z' }:
                continue
            aa_wt = fields[1]
            aa_mut = fields[2]
            assert(wt_seq[pos] == aa_wt)
            mut_seq = wt_seq[:pos] + aa_mut + wt_seq[pos+1:]
            if mut_seq not in seqs_escape:
                continue
            prob.append(float(fields[3]))
            change.append(float(fields[4]))
            viable_idx.append(fields[5] == 'True')
            escape_idx.append(
                (mut_seq in seqs_escape) and
                (sum([ m['significant']
                       for m in seqs_escape[mut_seq] ]) > 0)
            )

    prob, orig_prob = np.array(prob), np.array(prob)
    change, orig_change  = np.array(change), np.array(change)
    escape_idx = np.array(escape_idx)
    viable_idx = np.array(viable_idx)

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
        mkdir_p('figures')

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

    if len(escape_prob) == 0:
        print('No escape mutations found.')
        return

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

    escape_rank_prob = ss.rankdata(-orig_prob)[escape_idx]
    n_escape_prob = np.array([ sum(escape_rank_prob <= i + 1)
                               for i in range(max_consider) ])
    norm_auc_prob = auc(n_consider, n_escape_prob) / norm

    escape_rank_change = ss.rankdata(-orig_change)[escape_idx]
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

    norm_auc_p = compute_p(norm_auc, sum(escape_idx), len(escape_idx))

    print('AUC (CSCS): {}, P = {}'.format(norm_auc, norm_auc_p))
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
    cutoff, expr_cutoff, bind_cutoff = None, None, None
    if len(sys.argv) > 2:
        cutoff = float(sys.argv[2])
    if len(sys.argv) > 3:
        expr_cutoff = float(sys.argv[3])

    cached_escape(sys.argv[1], 1., cutoff, expr_cutoff)
