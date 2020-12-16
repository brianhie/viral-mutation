from utils import *

import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained

def predict_sequence_prob_fb(
        seq, alphabet, model, repr_layers,
        batch_size=4096, verbose=False
):
    seqs = [ seq ]
    labels = [ 'seq0' ]

    dataset = FastaBatchedDataset(labels, seqs)
    batches = dataset.get_batch_indices(batch_size, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(),
        batch_sampler=batches
    )

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)
            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            logits = out["logits"].to(device="cpu")

    return logits.numpy()[0]

def embed_seqs_fb(
        model, seqs, repr_layers, alphabet,
        batch_size=4096, use_cache=False, verbose=True
):
    labels = [ 'seq' + str(i) for i in range(len(seqs)) ]

    dataset = FastaBatchedDataset(labels, seqs)
    batches = dataset.get_batch_indices(batch_size, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(),
        batch_sampler=batches
    )

    embedded_seqs = {}
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)
            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {
                layer: t.to(device="cpu")
                for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                seq_idx = int(label[3:])
                seq = seqs[seq_idx]
                assert(len(representations.items()) == 1)
                for _, t in representations.items():
                    representation = t[i, 1 : len(strs[i]) + 1]
                if seq not in embedded_seqs:
                    embedded_seqs[seq] = []
                embedded_seqs[seq].append({
                    'embedding': representation.numpy()
                })

    return embedded_seqs

def fb_semantics(model, repr_layers, alphabet, seq_to_mutate, escape_seqs,
                 min_pos=None, max_pos=None, prob_cutoff=0., beta=1.,
                 comb_batch=None, plot_acquisition=True,
                 namespace='fb', plot_namespace=None, verbose=True):

    if plot_acquisition:
        dirname = ('target/{}/semantics/cache'.format(namespace))
        mkdir_p(dirname)
        if plot_namespace is None:
            plot_namespace = namespace

    y_pred = predict_sequence_prob_fb(
        seq_to_mutate, alphabet, model, repr_layers,
        verbose=verbose
    )

    if min_pos is None:
        min_pos = 0
    if max_pos is None:
        max_pos = len(seq_to_mutate) - 1

    word2idx = { word: alphabet.all_toks.index(word)
                 for word in alphabet.all_toks }

    word_pos_prob = {}
    for i in range(min_pos, max_pos + 1):
        for word in alphabet.all_toks:
            if '<' in word:
                continue
            if seq_to_mutate[i] == word:
                continue
            prob = y_pred[i + 1, word2idx[word]]
            word_pos_prob[(word, i)] = 10 ** prob

    prob_seqs = { seq_to_mutate: [ { 'word': None, 'pos': None } ] }
    seq_prob = {}
    for (word, pos), prob in word_pos_prob.items():
        mutable = seq_to_mutate[:pos] + word + seq_to_mutate[pos + 1:]
        seq_prob[mutable] = prob
        if prob >= prob_cutoff:
            prob_seqs[mutable] = [ { 'word': word, 'pos': pos } ]

    seqs = np.array([ str(seq) for seq in sorted(seq_prob.keys()) ])

    base_embedding = embed_seqs_fb(
        model, [ seq_to_mutate ], repr_layers, alphabet,
        use_cache=False, verbose=False
    )[seq_to_mutate][0]['embedding']

    if comb_batch is None:
        comb_batch = len(seqs)
    n_batches = math.ceil(float(len(seqs)) / comb_batch)

    seq_change = {}
    for batchi in range(n_batches):
        start = batchi * comb_batch
        end = (batchi + 1) * comb_batch
        tprint('Analyzing sequences {} to {}...'.format(start, end))

        prob_seqs_batch = [
            seq for seq in seqs[start:end] if seq != seq_to_mutate
        ]
        prob_seqs_batch = embed_seqs_fb(
            model, prob_seqs_batch, repr_layers, alphabet,
            use_cache=False, verbose=False
        )
        for mut_seq in prob_seqs_batch:
            meta = prob_seqs_batch[mut_seq][0]
            sem_change = abs(base_embedding - meta['embedding']).sum()
            seq_change[mut_seq] = sem_change

    cache_fname = dirname + (
        '/analyze_semantics_{}_{}.txt'
        .format(plot_namespace, model.model_version)
    )
    probs, changes = [], []
    with open(cache_fname, 'w') as of:
        fields = [ 'pos', 'wt', 'mut', 'prob', 'change',
                   'is_viable', 'is_escape' ]
        of.write('\t'.join(fields) + '\n')
        for seq in seqs:
            prob = seq_prob[seq]
            change = seq_change[seq]
            mut = prob_seqs[seq][0]['word']
            pos = prob_seqs[seq][0]['pos']
            orig = seq_to_mutate[pos]
            is_viable = seq in escape_seqs
            is_escape = ((seq in escape_seqs) and
                         (sum([ m['significant']
                                for m in escape_seqs[seq] ]) > 0))
            fields = [ pos, orig, mut, prob, change, is_viable, is_escape ]
            of.write('\t'.join([ str(field) for field in fields ]) + '\n')
            probs.append(prob)
            changes.append(change)

    if plot_acquisition:
        from cached_semantics import cached_escape
        cached_escape(cache_fname, beta,
                      plot=plot_acquisition,
                      namespace=plot_namespace)

    return seqs, np.array(probs), np.array(changes)

if __name__ == '__main__':
    name = 'esm1_t34_670M_UR50S'

    model, alphabet = pretrained.load_model_and_alphabet(name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    assert(all(
        -(model.num_layers + 1) <= i <= model.num_layers
        for i in [ -1 ]
    ))
    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1)
        for i in [ -1 ]
    ]

    from escape import *

    tprint('Lee et al. 2018...')
    seq_to_mutate, escape_seqs = load_doud2018()
    fb_semantics(
        model, repr_layers, alphabet, seq_to_mutate, escape_seqs,
        comb_batch=2000, plot_namespace='flu_h1'
    )

    tprint('')
    tprint('Lee et al. 2019...')
    seq_to_mutate, escape_seqs = load_lee2019()
    fb_semantics(
        model, repr_layers, alphabet, seq_to_mutate, escape_seqs,
        comb_batch=2000, plot_namespace='flu_h3'
    )

    tprint('')
    tprint('Dingens et al. 2019...')
    seq_to_mutate, escape_seqs = load_dingens2019()
    positions = [ escape_seqs[seq][0]['pos'] for seq in escape_seqs ]
    min_pos, max_pos = min(positions), max(positions)
    fb_semantics(
        model, repr_layers, alphabet, seq_to_mutate, escape_seqs,
        min_pos=min_pos, max_pos=max_pos, plot_acquisition=True,
        comb_batch=2000, plot_namespace='hiv',
    )

    tprint('')
    tprint('Baum et al. 2020...')
    seq_to_mutate, escape_seqs = load_baum2020()
    fb_semantics(
        model, repr_layers, alphabet, seq_to_mutate, escape_seqs,
        comb_batch=2000, plot_acquisition=True,
        plot_namespace='cov2',
    )

    tprint('')
    tprint('Greaney et al. 2020...')
    seq_to_mutate, escape_seqs = load_greaney2020()
    fb_semantics(
        model, repr_layers, alphabet, seq_to_mutate, escape_seqs,
        comb_batch=2000, min_pos=318, max_pos=540, # Restrict to RBD.
        plot_acquisition=True, plot_namespace='cov2rbd'
    )
