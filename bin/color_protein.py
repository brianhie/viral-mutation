from utils import *

import matplotlib

def write_color(chain, resi, cmap, acq, of):
    of.write('select toColor, resi {} and chain {}\n'
             .format(resi, chain))
    rgb = cmap(acq)
    of.write('color {}, toColor\n'
             .format(matplotlib.colors.rgb2hex(rgb))
             .replace('#', '0x'))

def color_lee2019():
    acq_fname = 'target/flu/semantics/cache/plot_h3_bilstm_512.npz'
    mut_fname = 'target/flu/semantics/cache/flu_mutations_h3.txt'
    beta = 1.

    data = np.load(acq_fname)
    prob = data['prob']
    change = data['change']
    acquisition = ss.rankdata(change) + (beta * ss.rankdata(prob))

    idxs = []
    with open(mut_fname) as f:
        f.readline()
        for line in f:
            if line.rstrip() == 'NA':
                idxs.append(-1)
            else:
                idxs.append(int(line.split('\t')[0]))
    idxs = np.array(idxs)
    assert(len(acquisition) == len(idxs))

    idx_pos = {}
    with open('data/influenza/escape_lee2019/avg_sel_tidy.csv') as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(',')
            idx_pos[int(fields[13])] = fields[4]

    pos_pdb = {}
    with open('data/influenza/escape_lee2019/'
              'H3_site_to_PDB_4o5n.csv') as f:
        f.readline()
        for line in f:
            pos, chain, resi = line.rstrip().split(',')
            pos_pdb[pos] = (chain, resi)

    cmap = matplotlib.cm.get_cmap('viridis')

    idxs_uniq = sorted(set(idxs))

    max_acq_max = max([ acquisition[idxs == idx].max()
                        for idx in idxs_uniq ])
    max_acq_mean = max([ acquisition[idxs == idx].mean()
                         for idx in idxs_uniq ])

    dirname = 'target/flu/structure'
    mkdir_p(dirname)
    of_max = open(dirname + '/pdb_color_h3_max.pml', 'w')
    of_mean = open(dirname + '/pdb_color_h3_mean.pml', 'w')

    for idx in idxs_uniq:
        if idx < 0:
            continue
        chain, resi = pos_pdb[idx_pos[idx]]
        idx_acq = acquisition[idxs == idx]
        acq_max = idx_acq.max() / max_acq_max
        acq_mean = idx_acq.mean() / max_acq_mean

        write_color(chain, resi, cmap, acq_max, of_max)
        write_color(chain, resi, cmap, acq_mean, of_mean)

    of_max.close()
    of_mean.close()

def color_lee2018():
    acq_fname = 'target/flu/semantics/cache/plot_h1_bilstm_512.npz'
    mut_fname = 'target/flu/semantics/cache/flu_mutations_h1.txt'
    beta = 3.

    data = np.load(acq_fname)
    prob = data['prob']
    change = data['change']
    acquisition = ss.rankdata(change) + (beta * ss.rankdata(prob))

    idxs = []
    with open(mut_fname) as f:
        f.readline()
        for line in f:
            if line.rstrip() == 'NA':
                idxs.append(-1)
            else:
                idxs.append(int(line.split('\t')[0]))
    idxs = np.array(idxs)
    assert(len(acquisition) == len(idxs))

    idx_pdb = {}
    with open('data/influenza/escape_lee2018/H1toH3_renumber.csv') as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(',')
            if '(HA2)' in fields[1]:
                chain = 'B'
                resi = int(fields[1].split(')')[-1])
            else:
                chain = 'A'
                resi = fields[1]
            idx_pdb[int(fields[0]) - 1] = (chain, resi)

    cmap = matplotlib.cm.get_cmap('viridis')

    idxs_uniq = sorted(set(idxs))

    max_acq_max = max([ acquisition[idxs == idx].max()
                        for idx in idxs_uniq ])
    max_acq_mean = max([ acquisition[idxs == idx].mean()
                         for idx in idxs_uniq ])

    dirname = 'target/flu/structure'
    mkdir_p(dirname)
    of_max = open(dirname + '/pdb_color_h1_max.pml', 'w')
    of_mean = open(dirname + '/pdb_color_h1_mean.pml', 'w')

    for idx in idxs_uniq:
        if idx < 0:
            continue
        chain, resi = idx_pdb[idx]
        idx_acq = acquisition[idxs == idx]
        acq_max = idx_acq.max() / max_acq_max
        acq_mean = idx_acq.mean() / max_acq_mean

        write_color(chain, resi, cmap, acq_max, of_max)
        write_color(chain, resi, cmap, acq_mean, of_mean)

    of_max.close()
    of_mean.close()

def color_dingens2018():
    acq_fname = 'target/hiv/semantics/cache/plot_bilstm_512.npz'
    mut_fname = 'target/hiv/semantics/cache/hiv_mutations.txt'
    beta = 10.

    data = np.load(acq_fname)
    prob = data['prob']
    change = data['change']
    acquisition = ss.rankdata(change) + (beta * ss.rankdata(prob))

    idxs = []
    with open(mut_fname) as f:
        f.readline()
        for line in f:
            if line.rstrip() == 'NA':
                idxs.append(-1)
            else:
                idxs.append(int(line.split('\t')[0]))
    idxs = np.array(idxs)
    assert(len(acquisition) == len(idxs))

    idx_pdb = {}
    for idx in idxs:
        if idx < 320:
            pos = str(idx + 2)
        elif idx == 320:
            pos = '321A'
        else:
            pos = str(idx + 1)
        idx_pdb[idx] = ('G', pos)

    cmap = matplotlib.cm.get_cmap('viridis')

    idxs_uniq = sorted(set(idxs))

    max_acq_max = max([ acquisition[idxs == idx].max()
                        for idx in idxs_uniq ])
    max_acq_mean = max([ acquisition[idxs == idx].mean()
                         for idx in idxs_uniq ])

    dirname = 'target/hiv/structure'
    mkdir_p(dirname)
    of_max = open(dirname + '/pdb_color_gp120_max.pml', 'w')
    of_mean = open(dirname + '/pdb_color_gp120_mean.pml', 'w')

    for idx in idxs_uniq:
        if idx < 0:
            continue
        chain, resi = idx_pdb[idx]
        idx_acq = acquisition[idxs == idx]
        acq_max = idx_acq.max() / max_acq_max
        acq_mean = idx_acq.mean() / max_acq_mean

        write_color(chain, resi, cmap, acq_max, of_max)
        write_color(chain, resi, cmap, acq_mean, of_mean)

    of_max.close()
    of_mean.close()

if __name__ == '__main__':
    color_dingens2018()
    exit()
    color_lee2018()
    color_lee2019()
