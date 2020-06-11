from utils import *

from transformer_layers import Encoder

import tensorflow as tf
from tf.keras.callbacks.callbacks import ModelCheckpoint
from tf.keras.layers import concatenate, Dense, Input
from tf.keras.models import Model
from tf.keras.optimizers import Adam, SGD
from tf.keras.preprocessing.sequence import pad_sequences

def _split_and_pad(X_cat, lengths, seq_len, vocab_size, verbose):
    if X_cat.shape[0] != sum(lengths):
        raise ValueError('Length dimension mismatch: {} and {}'
                         .format(X_cat.shape[0], sum(lengths)))

    if verbose > 1:
        tprint('Splitting {} seqs...'.format(len(lengths)))
    X_seqs = [
        X_cat[start:end].flatten()
        for start, end in iterate_lengths(lengths, seq_len)
    ]

    y = [
        X_seq[i] for X_seq in X_seqs for i in range(len(X_seq))
    ]
    X_seqs = [
        np.concatenate([ X_seq[:i], X_seq[i + 1:] ]).ravel()
        for X_seq in X_seqs for i in range(len(X_seq))
    ]

    if verbose > 1:
        tprint('Padding {} splitted...'.format(len(X_seqs)))
    X = pad_sequences(
        X_seqs, maxlen=seq_len - 1,
        dtype='int32', padding='pre', truncating='pre', value=0
    )
    y = np.array(y).reshape(-1, 1)

    if verbose > 1:
        tprint('Done splitting and padding.')
    return X, y

class TransfomerLanguageModel(object):
    def __init__(
            self,
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=256,
            n_hidden=2,
            n_heads=8,
            dff=2048,
            dropout_rate=0.1,
            n_epochs=1,
            batch_size=1000,
            cache_dir='.',
            verbose=False
    ):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        input_ = Input(shape=(seq_len - 1,))

        self.encoder_ = Encoder(
            n_hidden, hidden_dim, n_heads, dff,
            vocab_size, seq_len, dropout_rate
        )
        x = self.encoder_(input_, None)

        output = Dense(vocab_size + 1, activation='softmax')(x)

        self.model_ = Model(inputs=input_, outputs=output)

        self.seq_len_ = seq_len
        self.vocab_size_ = vocab_size
        self.embedding_dim_ = embedding_dim
        self.hidden_dim_ = hidden_dim
        self.n_hidden_ = n_hidden
        self.n_heads_ = n_heads
        self.dff_ = dff
        self.dropout_rate_ = dropout_rate
        self.n_epochs_ = n_epochs
        self.batch_size_ = batch_size
        self.cache_dir_ = cache_dir
        self.verbose_ = verbose

    def fit(self, X_cat, lengths):
        X = _split_and_pad(
            X_cat, lengths, self.seq_len_, self.vocab_size_, self.verbose_
        )[0]

        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                   amsgrad=False)
        self.model_.compile(
            loss='sparse_categorical_crossentropy', optimizer=opt,
            metrics=[ 'accuracy' ]
        )

        mkdir_p('{}/checkpoints.old/attention'.format(self.cache_dir_))
        model_name = 'attention'
        checkpoint = ModelCheckpoint(
            '{}/checkpoints.old/attention/{}_{}'
            .format(self.cache_dir_, model_name, self.hidden_dim_) +
            '-{epoch:02d}.hdf5',
            save_best_only=False, save_weights_only=False,
            mode='auto', period=1
        )

        self.model_.fit(
            X, y, epochs=self.n_epochs_, batch_size=self.batch_size_,
            shuffle=True, verbose=self.verbose_ > 0,
            callbacks=[ checkpoint ],
        )

    def score(self, X_cat, lengths):
        X, y_true = _split_and_pad(
            X_cat, lengths, self.seq_len_, self.vocab_size_, self.verbose_
        )

        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                   amsgrad=False)
        self.model_.compile(
            loss='sparse_categorical_crossentropy', optimizer=opt,
            metrics=[ 'accuracy' ]
        )

        metrics = self.model_.evaluate(X, y_true, verbose=self.verbose_ > 0,
                                       batch_size=self.batch_size_)

        for val, metric in zip(metrics, self.model_.metrics_names):
            if self.verbose_:
                tprint('Metric {}: {}'.format(metric, val))

        return metrics[self.model_.metrics_names.index('loss')] * -len(lengths)
