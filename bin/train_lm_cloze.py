from __future__ import print_function,division

import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

import src.fasta as fasta
from src.alphabets import Uniprot21
import src.models.sequence_v2
import src.models.dense_rnn
from src.utils import pack_sequences

parser = argparse.ArgumentParser('Train protein language model given fasta sequences using cloze approach')
parser.add_argument('path', help='path to file containing sequences')

parser.add_argument('-b', '--minibatch-size', type=int, default=32, help='minibatch size (default: 32)')
parser.add_argument('-n', '--num-steps', type=int, default=1000000, help='number of training steps (default: 1,000,000)')

parser.add_argument('--arch', choices=['skip', 'dense'], default='skip', help='model architecture (default: skip)')
parser.add_argument('--hidden-dim', type=int, default=512, help='hidden dimension of RNN (default: 512)')
parser.add_argument('--num-layers', type=int, default=3, help='number of RNN layers (default: 3)')
parser.add_argument('--dropout', type=float, default=0, help='dropout (default: 0)')
parser.add_argument('--l2', type=float, default=0, help='l2 regularizer (default: 0)')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--clip', type=float, default=np.inf, help='gradient clipping max norm (default: inf)')

parser.add_argument('-p', type=float, default=0.1, help='cloze residue masking rate (default: 0.1)')

parser.add_argument('-d', '--device', type=int, default=-2, help='device to use, -1: cpu, 0+: gpu (default: gpu if available, else cpu)')

parser.add_argument('-o', '--output', help='where to write training curve (default: stdout)')
parser.add_argument('--save-prefix', help='path prefix for saving models (default: no saving)')

parser.add_argument('--debug', action='store_true')


def preprocess_sequence(s, alphabet):
    x = alphabet.encode(s)
    return z

def load_fasta(path, alphabet, debug=False):
    # load path sequences and families
    with open(path, 'rb') as f:
        sequences = []
        for name,sequence in fasta.parse_stream(f):
            x = alphabet.encode(sequence.upper())
            sequences.append(x)
            if debug and len(sequences) >= 100000:
                break
    sequences = np.array(sequences)
    return sequences

class ClozeDataset:
    def __init__(self, x, p, noise, length=None):
        self.x = x
        self.p = p
        self.noise = noise
        self.length = length

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        if self.length is not None and len(x) > self.length:
            length = self.length
            # randomly sample subsequence of length from x
            j = np.random.randint(len(x) - length + 1)
            x = x[j:j+length]

        p = self.p

        mask = np.random.binomial(1, p, size=len(x))
        y = mask*x + 20*(1-mask)

        n = np.sum(mask)
        x = x.copy()
        # sample the masked elements from the noise distribution
        x[mask==1] = np.random.choice(21, size=n, p=self.noise)

        return x, y

def main():
    args = parser.parse_args()

    alph = Uniprot21()
    ntokens = len(alph)

    ## load the training sequences
    path = args.path
    print('# loading sequences:', path, file=sys.stderr)
    debug = args.debug
    X_train = load_fasta(path, alph, debug=debug)
    print('# loaded', len(X_train), 'sequences', file=sys.stderr)

    # calculate the distribution over the amino acids
    # to use as the noise distribution
    counts = np.zeros(21)
    for x in X_train:
        v,c = np.unique(x, return_counts=True)
        counts[v] = counts[v] + c
    noise = counts/counts.sum()
    print('# amino acid marginal distribution:', noise, file=sys.stderr)
    # based on this distribution, calculate what the
    # NLL and perplexity would be if we guessed the marginal
    nll = -np.sum(np.log(noise)*noise)
    perplex = np.exp(nll)
    print('# NLL =', str(nll), ', perplexity =', str(perplex), file=sys.stderr)

    # make the noised dataset and minibatch iterator

    # to train more efficiently with long sequences
    # chop sequences into fragments of max length
    max_length = 500
    p = args.p # fraction of residues to mask during training

    # during sampling, weight each sequence by the number of chunks of
    # max length it contains (to try to represent each chunk equally)

    dataset = ClozeDataset(X_train, p, noise, length=max_length)
    
    # do this by duplicating each sequence
    #X_duped = []
    #for x in X_train:
    #    repeat = int(np.ceil(len(x)/max_length))
    #    for _ in range(repeat):
    #        X_duped.append(x)
    #dataset = ClozeDataset(X_duped, p, noise, length=max_length)

    def collate(args):
        x,y = zip(*args)
        x = [torch.from_numpy(x_).long() for x_ in x] 
        y = [torch.from_numpy(y_).long() for y_ in y]

        x,order = pack_sequences(x)
        y,_ = pack_sequences(y, order=order)

        return x,y

    mb = args.minibatch_size
    num_steps = args.num_steps # number of training steps to run
    """
    train_iterator = torch.utils.data.DataLoader(dataset, batch_size=mb,
                                                 shuffle=True,
                                                 collate_fn=collate)
    def make_iterator(iterator):
        while True:
            for x in iterator:
                yield x

    train_iterator = make_iterator(train_iterator)
    """

    L = np.array([len(x) for x in X_train])
    weight = np.maximum(L/max_length, 1)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, mb*num_steps)
    train_iterator = torch.utils.data.DataLoader(dataset, batch_size=mb,
                                                 sampler=sampler,
                                                 collate_fn=collate)
    train_iterator = iter(train_iterator)

    ## initialize the model
    nin = ntokens
    nout = ntokens
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    dropout = args.dropout

    arch = args.arch
    if arch == 'skip':
        model = src.models.sequence_v2.SkipLSTM(nin, nout, hidden_dim, num_layers
                                               , dropout=dropout)
    elif arch == 'dense':
        model = src.models.dense_rnn.DenseRNN(nin, nout, hidden_dim, num_layers
                                             , dropout=dropout)
    else:
        raise Exception('Unknown architecture: ' + arch)

    print('# initialized model', file=sys.stderr)

    device = args.device
    use_cuda = torch.cuda.is_available() and (device == -2 or device >= 0)
    if device >= 0:
        torch.cuda.set_device(device)
    if use_cuda:
        model = model.cuda()

    ## form the data iterators and optimizer
    lr = args.lr
    l2 = args.l2
    solver = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    ## fit the model!

    print('# training model', file=sys.stderr)

    output = sys.stdout
    if args.output is not None:
        output = open(args.output, 'w')

    clip = args.clip

    save_prefix = args.save_prefix
    digits = int(np.floor(np.log10(num_steps))) + 1

    print('epoch\tsplit\tlog_p\tperplexity\taccuracy', file=output)
    output.flush()

    n = 0
    accuracy = 0
    loss_accum = 0

    save_iter = 10
    for i in range(num_steps):
        x,y = next(train_iterator)
        y = y.data
        if use_cuda:
            x = PackedSequence(x.data.cuda(), x.batch_sizes)
            y = y.cuda()

        logits = model(x).data

        # only calculate loss for noised positions
        mask = (y < 20)
        d = mask.float().sum().item()
        # make sure we have masked positions
        loss = 0
        correct = 0
        if d > 0:
            logits = logits[mask]
            y = y[mask]

            loss = F.cross_entropy(logits, y)
        
            loss.backward()

            # clip the gradient
            if not np.isinf(clip):
                nn.utils.clip_grad_norm_(model.parameters(), clip)

            solver.step()
            solver.zero_grad()

            loss = loss.item()

            _,y_hat = torch.max(logits, 1)
            correct = torch.sum((y == y_hat).float()).item()

        n += d
        delta = d*(loss - loss_accum)
        loss_accum += delta/n
        delta = correct - d*accuracy
        accuracy += delta/n

        if (i+1)%10 == 0:
            print('# [{}/{}] loss={:.5f}, perlexity={:.5f}'.format(i+1,num_steps,loss_accum,np.exp(loss_accum)), end='\r', file=sys.stderr)

        if i+1 == save_iter:
            #print(' '*100, end='\r', file=sys.stderr)
            perplex = np.exp(loss_accum)
            string = str(i+1).zfill(digits) + '\t' + 'train' + '\t' + str(loss_accum) \
                     + '\t' + str(perplex) + '\t' + str(accuracy)
            print(string, file=output)
            output.flush()
            n = 0
            accuracy = 0
            loss_accum = 0

            ## save the model
            if save_prefix is not None:
                model.eval()
                save_path = save_prefix + '_iter' + str(i+1).zfill(digits) + '.sav'
                model = model.cpu()
                torch.save(model, save_path)
                if use_cuda:
                    model = model.cuda()
                model.train()

            save_iter = min(save_iter*10, save_iter+100000)




if __name__ == '__main__':
    main()


