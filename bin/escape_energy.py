from utils import *

from Bio import SeqIO
from sklearn.metrics import auc

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark methods')
    parser.add_argument('method', type=str,
                        help='Benchmarking method to use')
    parser.add_argument('virus', type=str,
                        help='Viral type to test')
    args = parser.parse_args()
    return args

def escape_energy(virus):
    if virus == 'h1':
        from escape import load_lee2018
        seq, seqs_escape = load_lee2018()
        train_fname = 'target/flu/clusters/all_h1.fasta'
        mut_fname = 'target/flu/mutation/mutations_h1.fa'
        energy_fname = 'target/flu/clusters/all_h1.fasta.E.txt'
    elif virus == 'h3':
        from escape import load_lee2019
        seq, seqs_escape = load_lee2019()
        train_fname = 'target/flu/clusters/all_h3.fasta'
        mut_fname = 'target/flu/mutation/mutations_h3.fa'
        energy_fname = 'target/flu/clusters/all_h3.fasta.E.txt'
    elif virus == 'hiv':
        from escape import load_dingens2019
        seq, seqs_escape = load_dingens2019()
        train_fname = 'target/hiv/clusters/all_BG505.fasta'
        mut_fname = 'target/hiv/mutation/mutations_hiv.fa'
        energy_fname = 'target/hiv/clusters/all_BG505.fasta.E.txt'
    else:
        raise ValueError('invalid option {}'.format(virus))

    train_seqs = list(SeqIO.parse(train_fname, 'fasta'))
    mut_seqs = list(SeqIO.parse(mut_fname, 'fasta'))
    energies = np.loadtxt(energy_fname)
    assert(len(energies) == len(train_seqs) + len(mut_seqs))

    escape_idx = [
        idx for idx, mut_seq in enumerate(mut_seqs)
        if str(mut_seq.seq).replace('-', '') in seqs_escape
    ]
    assert(len(escape_idx) == len(seqs_escape) - 1)

    mut_energies = energies[len(train_seqs):]
    acq_argsort = ss.rankdata(-mut_energies)
    escape_rank_dist = acq_argsort[escape_idx]

    max_consider = len(mut_seqs)
    n_consider = np.array([ i + 1 for i in range(max_consider) ])

    n_escape = np.array([ sum(escape_rank_dist <= i + 1)
                          for i in range(max_consider) ])
    norm = max(n_consider) * max(n_escape)
    norm_auc = auc(n_consider, n_escape) / norm

    escape_frac = len(seqs_escape) / len(mut_seqs)

    plt.figure()
    plt.plot(n_consider, n_escape)
    plt.plot(n_consider, n_consider * escape_frac,
             c='gray', linestyle='--')
    plt.legend([
        r'Potts model energy, AUC = {:.3f}'.format(norm_auc),
        'Random guessing, AUC = 0.500'
    ])
    plt.xlabel('Top N')
    plt.ylabel('Number of escape mutations in top N')
    plt.savefig('figures/{}_energy_escape.png'.format(virus), dpi=300)
    plt.close()

def escape_freq(virus):
    if virus == 'h1':
        from escape import load_lee2018
        seq, seqs_escape = load_lee2018()
        train_fname = 'target/flu/clusters/all.fasta'
        mut_fname = 'target/flu/mutation/mutations_h1.fa'
        anchor_id = ('gb:LC333185|ncbiId:BBB04702.1|UniProtKB:-N/A-|'
                     'Organism:Influenza')
    elif virus == 'h3':
        from escape import load_lee2019
        seq, seqs_escape = load_lee2019()
        train_fname = 'target/flu/clusters/all.fasta'
        mut_fname = 'target/flu/mutation/mutations_h3.fa'
        anchor_id = 'Reference_Perth2009_HA_coding_sequence'
    elif virus == 'hiv':
        from escape import load_dingens2019
        seq, seqs_escape = load_dingens2019()
        train_fname = 'target/hiv/clusters/all_BG505.fasta'
        mut_fname = 'target/hiv/mutation/mutations_hiv.fa'
        anchor_id = 'A1.KE.-.BG505_W6M_ENV_C2.DQ208458'
    else:
        raise ValueError('invalid option {}'.format(virus))

    anchor = None
    pos_aa_freq = {}
    for idx, record in enumerate(SeqIO.parse(train_fname, 'fasta')):
        mutation = str(record.seq)
        if record.id == anchor_id:
            anchor = mutation
        else:
            for pos, c in enumerate(mutation):
                if (pos, c) not in pos_aa_freq:
                    pos_aa_freq[(pos, c)] = 0.
                pos_aa_freq[(pos, c)] += 1.
    assert(anchor is not None)

    mutations = [
        str(record.seq)
        for record in SeqIO.parse(mut_fname, 'fasta')
    ]

    escape_idx = []
    mut_freqs = []
    for mut_idx, mutation in enumerate(mutations):
        if mutation.replace('-', '') in seqs_escape:
            escape_idx.append(mut_idx)
        didx = [ c1 != c2
                 for c1, c2 in zip(anchor, mutation) ].index(True)
        if (didx, mutation[didx]) in pos_aa_freq:
            mut_freqs.append(pos_aa_freq[(didx, mutation[didx])])
        else:
            mut_freqs.append(0)
    mut_freqs = np.array(mut_freqs)
    assert(len(escape_idx) == len(seqs_escape) - 1)

    acq_argsort = ss.rankdata(-mut_freqs)
    escape_rank_dist = acq_argsort[escape_idx]

    max_consider = len(mut_freqs)
    n_consider = np.array([ i + 1 for i in range(max_consider) ])

    n_escape = np.array([ sum(escape_rank_dist <= i + 1)
                          for i in range(max_consider) ])
    norm = max(n_consider) * max(n_escape)
    norm_auc = auc(n_consider, n_escape) / norm

    escape_frac = len(seqs_escape) / len(mut_freqs)

    plt.figure()
    plt.plot(n_consider, n_escape)
    plt.plot(n_consider, n_consider * escape_frac,
             c='gray', linestyle='--')
    plt.legend([
        r'Mutation frequency, AUC = {:.3f}'.format(norm_auc),
        'Random guessing, AUC = 0.500'
    ])
    plt.xlabel('Top N')
    plt.ylabel('Number of escape mutations in top N')
    plt.savefig('figures/{}_mutfreq_escape.png'.format(virus), dpi=300)
    plt.close()

def tape_embed(sequence, model, tokenizer):
    import torch
    token_ids = torch.tensor([tokenizer.encode(sequence)])
    output = model(token_ids)
    return output[0].detach().numpy().mean(1).ravel()

def escape_tape(virus, pretrained='transformer'):
    if pretrained == 'transformer':
        from tape import ProteinBertModel
        model_class = ProteinBertModel
        model_name = 'bert-base'
        fname_prefix = 'tape_transformer'
        vocab = 'iupac'
    elif pretrained == 'unirep':
        from tape import UniRepModel
        model_class = UniRepModel
        model_name = 'babbler-1900'
        fname_prefix = 'unirep'
        vocab = 'unirep'

    from tape import TAPETokenizer
    model = model_class.from_pretrained(model_name)
    tokenizer = TAPETokenizer(vocab=vocab)

    if virus == 'h1':
        from escape import load_lee2018
        seq, seqs_escape = load_lee2018()
        train_fname = 'target/flu/clusters/all.fasta'
        mut_fname = 'target/flu/mutation/mutations_h1.fa'
        embed_fname = ('target/flu/embedding/{}_h1.npz'
                       .format(fname_prefix))
        anchor_id = ('gb:LC333185|ncbiId:BBB04702.1|UniProtKB:-N/A-|'
                     'Organism:Influenza')
    elif virus == 'h3':
        from escape import load_lee2019
        seq, seqs_escape = load_lee2019()
        train_fname = 'target/flu/clusters/all.fasta'
        mut_fname = 'target/flu/mutation/mutations_h3.fa'
        embed_fname = ('target/flu/embedding/{}_h3.npz'
                       .format(fname_prefix))
        anchor_id = 'Reference_Perth2009_HA_coding_sequence'
    elif virus == 'hiv':
        from escape import load_dingens2019
        seq, seqs_escape = load_dingens2019()
        train_fname = 'target/hiv/clusters/all_BG505.fasta'
        mut_fname = 'target/hiv/mutation/mutations_hiv.fa'
        embed_fname = ('target/hiv/embedding/{}_hiv.npz'
                       .format(fname_prefix))
        anchor_id = 'A1.KE.-.BG505_W6M_ENV_C2.DQ208458'
    else:
        raise ValueError('invalid option {}'.format(virus))

    anchor = None
    for idx, record in enumerate(SeqIO.parse(train_fname, 'fasta')):
        if record.id == anchor_id:
            anchor = str(record.seq)
    assert(anchor is not None)

    base_embedding = tape_embed(anchor.replace('-', ''),
                                model, tokenizer)

    with np.load(embed_fname, allow_pickle=True) as data:
        embeddings = { name: data[name][()]['avg']
                       for name in data.files }

    mutations = [
        str(record.seq)
        for record in SeqIO.parse(mut_fname, 'fasta')
    ]

    escape_idx = []
    changes = []
    for mut_idx, mutation in enumerate(mutations):
        if mutation.replace('-', '') in seqs_escape:
            escape_idx.append(mut_idx)
        didx = [ c1 != c2
                 for c1, c2 in zip(anchor, mutation) ].index(True)
        embedding = embeddings['mut_{}_{}'.format(didx, mutation[didx])]
        changes.append(abs(base_embedding - embedding).sum())
    changes = np.array(changes)
    assert(len(escape_idx) == len(seqs_escape) - 1)

    acq_argsort = ss.rankdata(-changes)
    escape_rank_dist = acq_argsort[escape_idx]

    max_consider = len(changes)
    n_consider = np.array([ i + 1 for i in range(max_consider) ])

    n_escape = np.array([ sum(escape_rank_dist <= i + 1)
                          for i in range(max_consider) ])
    norm = max(n_consider) * max(n_escape)
    norm_auc = auc(n_consider, n_escape) / norm

    escape_frac = len(seqs_escape) / len(changes)

    plt.figure()
    plt.plot(n_consider, n_escape)
    plt.plot(n_consider, n_consider * escape_frac,
             c='gray', linestyle='--')
    plt.legend([
        r'TAPE transformer, AUC = {:.3f}'.format(norm_auc),
        'Random guessing, AUC = 0.500'
    ])
    plt.xlabel('Top N')
    plt.ylabel('Number of escape mutations in top N')
    plt.savefig('figures/{}_{}_escape.png'
                .format(virus, fname_prefix), dpi=300)
    plt.close()

def escape_bepler(virus):
    if virus == 'h1':
        from escape import load_lee2018
        seq, seqs_escape = load_lee2018()
        train_fname = 'target/flu/clusters/all.fasta'
        mut_fname = 'target/flu/mutation/mutations_h1.fa'
        embed_fname = 'target/flu/embedding/bepler_ssa_h1.txt'
        anchor_id = ('gb:LC333185|ncbiId:BBB04702.1|UniProtKB:-N/A-|'
                     'Organism:Influenza')
    elif virus == 'h3':
        from escape import load_lee2019
        seq, seqs_escape = load_lee2019()
        train_fname = 'target/flu/clusters/all.fasta'
        mut_fname = 'target/flu/mutation/mutations_h3.fa'
        embed_fname = 'target/flu/embedding/bepler_ssa_h3.txt'
        anchor_id = 'Reference_Perth2009_HA_coding_sequence'
    elif virus == 'hiv':
        from escape import load_dingens2019
        seq, seqs_escape = load_dingens2019()
        train_fname = 'target/hiv/clusters/all_BG505.fasta'
        mut_fname = 'target/hiv/mutation/mutations_hiv.fa'
        embed_fname = 'target/hiv/embedding/bepler_ssa_hiv.txt'
        anchor_id = 'A1.KE.-.BG505_W6M_ENV_C2.DQ208458'
    else:
        raise ValueError('invalid option {}'.format(virus))

    anchor = None
    for idx, record in enumerate(SeqIO.parse(train_fname, 'fasta')):
        if record.id == anchor_id:
            anchor = str(record.seq)
    assert(anchor is not None)

    embeddings = {}
    with open(embed_fname) as f:
        for line in f:
            if line.startswith('>'):
                name = line.rstrip()[1:]
            embedding = np.array(
                [ float(field)
                  for field in f.readline().rstrip().split() ]
            )
            embeddings[name] = embedding
    base_embedding = embeddings['base']

    mutations = [
        str(record.seq)
        for record in SeqIO.parse(mut_fname, 'fasta')
    ]

    escape_idx = []
    changes = []
    for mut_idx, mutation in enumerate(mutations):
        if mutation.replace('-', '') in seqs_escape:
            escape_idx.append(mut_idx)
        didx = [ c1 != c2
                 for c1, c2 in zip(anchor, mutation) ].index(True)
        embedding = embeddings['mut_{}_{}'.format(didx, mutation[didx])]
        changes.append(abs(base_embedding - embedding).sum())
    changes = np.array(changes)
    assert(len(escape_idx) == len(seqs_escape) - 1)

    acq_argsort = ss.rankdata(-changes)
    escape_rank_dist = acq_argsort[escape_idx]

    max_consider = len(changes)
    n_consider = np.array([ i + 1 for i in range(max_consider) ])

    n_escape = np.array([ sum(escape_rank_dist <= i + 1)
                          for i in range(max_consider) ])
    norm = max(n_consider) * max(n_escape)
    norm_auc = auc(n_consider, n_escape) / norm

    escape_frac = len(seqs_escape) / len(changes)

    plt.figure()
    plt.plot(n_consider, n_escape)
    plt.plot(n_consider, n_consider * escape_frac,
             c='gray', linestyle='--')
    plt.legend([
        r'Bepler, AUC = {:.3f}'.format(norm_auc),
        'Random guessing, AUC = 0.500'
    ])
    plt.xlabel('Top N')
    plt.ylabel('Number of escape mutations in top N')
    plt.savefig('figures/{}_bepler_escape.png'.format(virus), dpi=300)
    plt.close()

if __name__ == '__main__':
    args = parse_args()

    if args.method == 'bepler':
        escape_bepler(args.virus)

    elif args.method == 'energy_louie':
        escape_energy(args.virus)

    elif args.method == 'energy_hopf':
        pass

    elif args.method == 'freq':
        escape_freq(args.virus)

    elif args.method == 'tape':
        escape_tape(args.virus)

    elif args.method == 'unirep':
        escape_tape(args.virus, 'unirep')
