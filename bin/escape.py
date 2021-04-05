from utils import Seq, SeqIO

def load_doud2018(survival_cutoff=0.05):
    pos_map = {}
    with open('data/influenza/escape_doud2018/pos_map.csv') as f:
        f.readline() # Consume header.
        for line in f:
           fields = line.rstrip().split(',')
           pos_map[fields[1]] = int(fields[0]) - 1

    fname = 'data/influenza/escape_doud2018/WSN1933_H1_HA.fa'
    seqs = []
    for record in SeqIO.parse(fname, 'fasta'):
        seq = record.seq
        seqs.append(seq)

    seqs_escape = {}
    antibodies = [
        'C179', 'FI6v3', 'H17L10', 'H17L19', 'H17L7', 'S139',
    ]
    for antibody in antibodies:
        fname = ('data/influenza/escape_doud2018/' +
                 'medianfracsurvivefiles/' +
                 'antibody_{}_median.csv'.format(antibody))
        with open(fname) as f:
            f.readline() # Consume header.
            for line in f:
                fields = line.rstrip().split(',')
                frac_survived = float(fields[3])
                pos = pos_map[fields[0]]
                if seq[pos] != fields[1]:
                    print((seq[pos], fields[1], pos))
                assert(seq[pos] == fields[1])
                escaped = seq[:pos] + fields[2] + seq[pos + 1:]
                assert(len(seq) == len(escaped))
                if escaped not in seqs_escape:
                    seqs_escape[escaped] = []
                seqs_escape[escaped].append({
                    'frac_survived': frac_survived,
                    'antibody': antibody,
                    'significant': frac_survived >= survival_cutoff,
                })

    return seq, seqs_escape

def load_lee2019(survival_cutoff=None):
    fname = 'data/influenza/escape_lee2019/Perth2009_H3_HA.fa'
    for record in SeqIO.parse(fname, 'fasta'):
        seq = record.seq
        break

    seqs_escape = {}
    fname = 'data/influenza/escape_lee2019/avg_sel_tidy.csv'
    with open(fname) as f:
        f.readline() # Consume header.
        for line in f:
            fields = line.rstrip().split(',')
            if survival_cutoff is None:
                significant = fields[14] == 'True'
            else:
                if fields[7].strip():
                    significant = float(fields[7]) >= survival_cutoff
                else:
                    significant = False
            pos = int(fields[13])
            assert(seq[pos] == fields[5])
            escaped = seq[:pos] + fields[6] + seq[pos + 1:]
            assert(len(seq) == len(escaped))
            if escaped not in seqs_escape:
                seqs_escape[escaped] = []

            if '-age-' in fields[0]:
                species = 'human'
            elif 'ferret-' in fields[0]:
                species = 'ferret'
            else:
                species = 'antibody'

            seqs_escape[escaped].append({
                'abs_diff_selection': float(fields[11]),
                'antibody': fields[1],
                'species': species,
                'significant': significant,
            })

    return seq, seqs_escape

def load_dingens2019(survival_cutoff=0.11):
    pos_map = {}
    with open('data/hiv/escape_dingens2019/BG505_to_HXB2.csv') as f:
        f.readline() # Consume header.
        for line in f:
           fields = line.rstrip().split(',')
           pos_map[fields[1]] = int(fields[0]) - 1

    fname = 'data/hiv/escape_dingens2019/Env_protalign_manualeditAD.fasta'
    for record in SeqIO.parse(fname, 'fasta'):
        if record.description == 'BG505':
            seq = record.seq
            break

    seqs_escape = {}
    antibodies = [
        '101074', '10E8', '3BNC117-101074-pool', '3BNC117', 'PG9',
        'PGT121', 'PGT145', 'PGT151', 'VRC01', 'VRC34',
    ]
    for antibody in antibodies:
        fname = ('data/hiv/escape_dingens2019/FileS4/'
                 'fracsurviveaboveavg/{}.csv'.format(antibody))
        with open(fname) as f:
            f.readline() # Consume header.
            for line in f:
                fields = line.rstrip().split(',')
                frac_survived = float(fields[3])
                pos = pos_map[fields[0]]
                assert(seq[pos] == fields[1])
                escaped = seq[:pos] + fields[2] + seq[pos + 1:]
                assert(len(seq) == len(escaped))
                if escaped not in seqs_escape:
                    seqs_escape[escaped] = []
                seqs_escape[escaped].append({
                    'pos': pos,
                    'frac_survived': frac_survived,
                    'antibody': antibody,
                    'significant': frac_survived >= survival_cutoff,
                })

    return seq, seqs_escape

def load_baum2020():
    seq = SeqIO.read('data/cov/cov2_spike_wt.fasta', 'fasta').seq

    muts = set([
        'K417E', 'K444Q', 'V445A', 'N450D', 'Y453F', 'L455F',
        'E484K', 'G485D', 'F486V', 'F490L', 'F490S', 'Q493K',
        'H655Y', 'R682Q', 'R685S', 'V687G', 'G769E', 'Q779K',
        'V1128A',
    ])

    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V',
    ]

    seqs_escape = {}
    for idx in range(len(seq)):
        for aa in AAs:
            if aa == seq[idx]:
                continue
            mut_seq = seq[:idx] + aa + seq[idx+1:]
            mut_str = '{}{}{}'.format(seq[idx], idx + 1, aa)
            if mut_seq not in seqs_escape:
                seqs_escape[mut_seq] = []
            seqs_escape[mut_seq].append({
                'mutation': mut_str,
                'significant': mut_str in muts,
            })

    return seq, seqs_escape

def load_greaney2020(survival_cutoff=0.3,
                     binding_cutoff=-2.35, expr_cutoff=-1.5):
    seq = SeqIO.read('data/cov/cov2_spike_wt.fasta', 'fasta').seq

    sig_sites = set()
    with open('data/cov/greaney2020cov2/significant_escape_sites.csv') as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(',')
            sig_sites.add(int(fields[1]) - 1)

    binding = {}
    with open('data/cov/starr2020cov2/single_mut_effects.csv') as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(',')
            pos = float(fields[1]) - 1
            aa_orig = fields[2].strip('"')
            aa_mut = fields[3].strip('"')
            if aa_mut == '*':
                continue
            if fields[8] == 'NA':
                score = float('-inf')
            else:
                score = float(fields[8])
            if fields[11] == 'NA':
                expr = float('-inf')
            else:
                expr = float(fields[11])
            binding[(pos, aa_orig, aa_mut)] = score, expr

    seqs_escape = {}
    with open('data/cov/greaney2020cov2/escape_fracs.csv') as f:
        f.readline() # Consume header.
        for line in f:
            fields = line.rstrip().split(',')
            antibody = fields[2]
            escape_frac = float(fields[10])
            aa_orig = fields[5]
            aa_mut = fields[6]
            pos = int(fields[4]) - 1
            assert(seq[pos] == aa_orig)
            escaped = seq[:pos] + aa_mut + seq[pos + 1:]
            assert(len(seq) == len(escaped))
            if escaped not in seqs_escape:
                seqs_escape[escaped] = []
            significant = (
                escape_frac >= survival_cutoff and
                # Statements below should always be true with defaults.
                binding[(pos, aa_orig, aa_mut)][0] >= binding_cutoff and
                binding[(pos, aa_orig, aa_mut)][1] >= expr_cutoff
            )
            seqs_escape[escaped].append({
                'pos': pos,
                'frac_survived': escape_frac,
                'antibody': antibody,
                'significant': significant,
            })

    return seq, seqs_escape

if __name__ == '__main__':
    load_doud2018()
    load_lee2019()
    load_dingens2019()
    load_baum2020()
    load_greaney2020()
