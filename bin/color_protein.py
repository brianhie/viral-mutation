from utils import *

import matplotlib

def write_color(chain, resi, acq, of):
    cmap = matplotlib.cm.get_cmap('viridis')

    of.write('select toColor, resi {} and chain {}\n'
             .format(resi, chain))
    rgb = cmap(acq)
    of.write('color {}, toColor\n'
             .format(matplotlib.colors.rgb2hex(rgb))
             .replace('#', '0x'))

def scale(point, data_min, data_max):
    return (point - data_min) / (data_max - data_min)

def generate_pymol_colors(ofname, df, idx_pdb):
    color_data = []
    for idx in sorted(set(df['pos'])):
        assert(idx >= 0)
        acq_mean = np.mean(df[df['pos'] == idx]['acquisition'])
        for chain, resi in idx_pdb[idx]:
            color_data.append([ chain, resi, acq_mean ])

    acq_min = min([ acq for _, _, acq in color_data ])
    acq_max = max([ acq for _, _, acq in color_data ])

    with open(ofname, 'w') as of_mean:
        for chain, resi, acq_mean in color_data:
            acq_scaled = scale(acq_mean, acq_min, acq_max)
            write_color(chain, resi, acq_scaled, of_mean)

def load_data(virus, beta=1.):
    from regional_escape import load
    escape_fname, region_fname = load(virus)

    data = []
    with open(escape_fname) as f:
        columns = f.readline().rstrip().split()
        for line in f:
            data.append(line.rstrip().split('\t'))
    df_all = pd.DataFrame(data, columns=columns)
    df_all['pos'] = pd.to_numeric(df_all['pos'])
    df_all['prob'] = pd.to_numeric(df_all['prob'])
    df_all['change'] = pd.to_numeric(df_all['change'])
    df_all['acquisition'] = ss.rankdata(df_all.change) + \
                            (beta * ss.rankdata(df_all.prob))

    return df_all

def color_lee2019():
    df = load_data('h3')

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
            pos_pdb[pos] = [ (chain, resi) ]

    dirname = 'target/flu/structure'
    mkdir_p(dirname)
    ofname = dirname + '/pdb_color_h3_mean.pml'

    idx_pdb = { idx: pos_pdb[idx_pos[idx]]
                for idx in sorted(set(df['pos'])) }

    generate_pymol_colors(ofname, df, idx_pdb)

def color_doud2018():
    df = load_data('h1')

    idx_pdb = {
        resi: [ ('A', resi), ('B', resi), ('C', resi) ]
        for resi in range(575)
    }

    dirname = 'target/flu/structure'
    mkdir_p(dirname)
    ofname = dirname + '/pdb_color_h1_mean.pml'

    generate_pymol_colors(ofname, df, idx_pdb)

def color_dingens2018():
    df = load_data('hiv')
    idxs = sorted(set(df['pos']))

    idx_pdb = {}
    for idx in idxs:
        if idx < 320:
            pos = str(idx + 2)
            idx_pdb[idx] = [ ('G', pos) ]
        elif idx == 320:
            pos = '321A'
            idx_pdb[idx] = [ ('G', pos) ]
        elif idx < 514:
            pos = str(idx + 1)
            idx_pdb[idx] = [ ('G', pos) ]
        else:
            pos = str(idx + 4)
            idx_pdb[idx] = [ ('B', pos) ]

    dirname = 'target/hiv/structure'
    mkdir_p(dirname)
    ofname = dirname + '/pdb_color_gp120_mean.pml'

    generate_pymol_colors(ofname, df, idx_pdb)

def color_starr2020():
    df = load_data('sarscov2')

    idx_pdb = { idx: [ ('A', str(idx + 1)),
                       ('B', str(idx + 1)),
                       ('C', str(idx + 1)) ]
                for idx in sorted(set(df['pos'])) }

    dirname = 'target/cov/structure'
    mkdir_p(dirname)
    ofname = dirname + '/pdb_color_sarscov2_mean.pml'

    generate_pymol_colors(ofname, df, idx_pdb)

if __name__ == '__main__':
    color_doud2018()
    color_lee2019()
    color_dingens2018()
    color_starr2020()
