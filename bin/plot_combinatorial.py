from utils import *

from combinatorial import *

def plot_combinatorial(fname, n_mutations, mutants):
    data, targets = [], {}
    with open(fname) as f:
        for line in f:
            fields = line.rstrip().split()
            mut_str = fields[0]
            xy = [ float(fields[1]), np.log10(float(fields[2])) ]
            if mut_str in mutants:
                targets[mut_str] = xy
            else:
                data.append(xy)
    df = pd.DataFrame(data, columns=[ 'prob', 'change' ])

    for mut_str in targets:
        hprob_hchange = [
            (x > targets[mut_str][0] and y > targets[mut_str][1])
            for x, y in data
        ]
        hprob_lchange = [
            (x > targets[mut_str][0] and y <= targets[mut_str][1])
            for x, y in data
        ]
        lprob_hchange = [
            (x <= targets[mut_str][0] and y > targets[mut_str][1])
            for x, y in data
        ]
        lprob_lchange = [
            (x <= targets[mut_str][0] and y <= targets[mut_str][1])
            for x, y in data
        ]

        tprint('{} mutations:'.format(n_mutations))
        tprint('Higher prob, higher change: {}'.format(sum(hprob_hchange)))
        tprint('Higher prob, lower change: {}'.format(sum(hprob_lchange)))
        tprint('Lower prob, higher change: {}'.format(sum(lprob_hchange)))
        tprint('Lower prob, lower change: {}'.format(sum(lprob_lchange)))
        tprint('Fisher\'s exact P = {}'.format(ss.fisher_exact(
            [ [ sum(hprob_hchange), sum(hprob_lchange), ],
              [ sum(lprob_hchange), sum(lprob_lchange), ] ])[1]
        ))
        tprint('')

        plt.figure()
        plt.scatter(df.prob, df.change, c=hprob_hchange, cmap='viridis')
        plt.scatter([ targets[mut_str][0] ], [ targets[mut_str][1] ],
                    c='k')
        plt.savefig('figures/combinatorial_mut_{}_{}.png'
                    .format(n_mutations, mut_str), dpi=300)
        plt.close()

if __name__ == '__main__':
    for n_mutations in [ 4, 8, 12 ]:
        fname = ('target/cov/combinatorial/cache/cov_mut_{}.txt.1'
                 .format(n_mutations))

        if n_mutations == 2:
            mutants = [ 'K417R,Q498H', 'I472V,D614G' ]
        elif n_mutations == 4:
            mutants = [ 'L18F,A222V,D614G,Q780E' ]
        elif n_mutations == 8:
            mutants = [ 'N439K,Y449F,F486L,Q493Y,'
                        'S494R,Q498Y,N501D,Y505H' ]
        elif n_mutations == 12:
            mutants = [ 'K417V,N439R,G446T,L455Y,F456L,A475P,'
                        'F486L,Q493N,S494D,Q498Y,N501T,V503I' ]
        else:
            raise NotImplementedError()

        plot_combinatorial(fname, n_mutations, mutants)
