from Bio import SeqIO

def load_lee2018(survival_cutoff=0.05):
    pos_map = {}
    with open('data/influenza/escape_lee2018/pos_map.csv') as f:
        f.readline() # Consume header.
        for line in f:
           fields = line.rstrip().split(',')
           pos_map[fields[1]] = int(fields[0]) - 1

    fname = 'data/influenza/escape_lee2018/WSN1933_H1_HA.fa'
    seqs = []
    for record in SeqIO.parse(fname, 'fasta'):
        seq = record.seq
        seqs.append(seq)

    seqs_escape = {}
    antibodies = [
        'C179', 'FI6v3', 'H17L10', 'H17L19', 'H17L7', 'S139',
    ]
    for antibody in antibodies:
        fname = ('data/influenza/escape_lee2018/' +
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
                    'significant': frac_survived > survival_cutoff,
                })

    return seq, seqs_escape

def load_lee2019():
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
            significant = fields[14] == 'True'
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

def load_dingens2019(survival_cutoff=0.05):
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
                    'frac_survived': frac_survived,
                    'antibody': antibody,
                    'significant': frac_survived > survival_cutoff,
                })

    return seq, seqs_escape

def load_korber2020():
    fname = 'data/cov/viprbrc_db.fasta'
    for record in SeqIO.parse(fname, 'fasta'):
        if 'SARS_CoV_2' in record.description:
            seq = record.seq
            break

    muts = [
        'D614G', 'S943P', # The big ones.
        'L5F', 'L8V', 'V367F', 'G476S', 'G476S',
        'H49Y', 'Y145H', 'Q239K', 'A831V',
        'D839Y', 'D839N', 'D839E', 'P1263L'
    ]

    seqs_escape = {}
    for mut in muts:
        aa_orig = mut[0]
        aa_mut = mut[-1]
        pos = int(mut[1:-1]) - 1
        assert(seq[pos] == aa_orig)
        escaped = seq[:pos] + aa_mut + seq[pos + 1:]
        assert(len(seq) == len(escaped))
        if escaped not in seqs_escape:
            seqs_escape[escaped] = []
        seqs_escape[escaped].append({
            'mutation': mut,
            'significant': mut in { 'D614G', 'S943P' },
        })

    return seq, seqs_escape


if __name__ == '__main__':
    load_korber2020()
    exit()
    load_lee2018()
    load_lee2019()
    load_dingens2019()
