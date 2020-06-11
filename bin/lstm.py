from utils import *

import tensorflow as tf
tf.set_random_seed(1)

from tf.keras.callbacks.callbacks import ModelCheckpoint
from tf.keras.layers import Dense, Embedding, LSTM
from tf.keras.models import Sequential
from tf.keras.optimizers import Adam, SGD
from tf.keras.preprocessing.sequence import pad_sequences

def _split_and_pad(X_cat, lengths, seq_len, vocab_size, verbose):
    if X_cat.shape[0] != sum(lengths):
        raise ValueError('Length dimension mismatch: {} and {}'
                         .format(X_cat.shape[0], sum(lengths)))

    if verbose > 1:
        tprint('Splitting...')
    X_seqs = [
        X_cat[start:end].flatten()[:i + 1]
        for start, end in iterate_lengths(lengths, seq_len)
        for i in range(end - start)
    ]

    if verbose > 1:
        tprint('Padding...')
    padded = pad_sequences(
        X_seqs, maxlen=seq_len,
        dtype='int32', padding='pre', truncating='pre', value=0.
    )
    X, y = padded[:, :-1], padded[:, -1]

    if verbose > 1:
        tprint('Done splitting and padding.')
    return X, y

class LSTMLanguageModel(object):
    def __init__(
            self,
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=256,
            n_hidden=2,
            dff=512,
            n_epochs=1,
            batch_size=1000,
            cache_dir='.',
            fp_precision='float32',
            verbose=False
    ):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        model = Sequential()
        model.add(Embedding(vocab_size + 1, embedding_dim,
                            input_length=seq_len - 1))

        for _ in range(n_hidden - 1):
            model.add(LSTM(hidden_dim, return_sequences=True))
        model.add(LSTM(hidden_dim))

        model.add(Dense(dff, activation='relu'))
        model.add(Dense(vocab_size + 1, activation='softmax'))

        self.model_ = model

        self.seq_len_ = seq_len
        self.vocab_size_ = vocab_size
        self.embedding_dim_ = embedding_dim
        self.hidden_dim_ = hidden_dim
        self.dff_ = dff
        self.n_hidden_ = n_hidden
        self.n_epochs_ = n_epochs
        self.batch_size_ = batch_size
        self.cache_dir_ = cache_dir
        self.verbose_ = verbose

    def fit(self, X_cat, lengths):
        X, y = _split_and_pad(
            X_cat, lengths, self.seq_len_, self.vocab_size_, self.verbose_
        )

        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                   amsgrad=False)
        #opt = SGD(learning_rate=0.001, momentum=0.2, nesterov=True)
        self.model_.compile(
            loss='sparse_categorical_crossentropy', optimizer=opt,
            metrics=[ 'accuracy' ]
        )

        mkdir_p('{}/checkpoints.old/lstm'.format(self.cache_dir_))
        checkpoint = ModelCheckpoint(
            '{}/checkpoints.old/lstm/lstm_{}'
            .format(self.cache_dir_, self.hidden_dim_) +
            '-{epoch:02d}.hdf5',
            save_best_only=False, save_weights_only=False,
            mode='auto', period=1
        )

        self.model_.fit(
            X, y, epochs=self.n_epochs_, batch_size=self.batch_size_,
            shuffle=True, verbose=self.verbose_ > 0, callbacks=[ checkpoint ],
        )

    def transform(self, X_cat, lengths, embed_fname=None):
        X = _split_and_pad(
            X_cat, lengths,
            self.seq_len_, self.vocab_size_, self.verbose_,
        )[0]

        hidden = Model(
            inputs=model.model_.input,
            outputs=model.model_.get_layer(
                'lstm_{}'.format(self.n_hidden_)).output
        )
        hidden.compile('adam', 'mean_squared_error')

        if self.verbose_:
            tprint('Embedding...')
        X_embed_cat = hidden.predict(X, batch_size=self.batch_size_,
                                     verbose=self.verbose_)
        if self.verbose_:
            tprint('Done embedding.')

        X_embed = np.array([
            X_embed_cat[start:end]
            for start, end in
            iterate_lengths(lengths, self.seq_len_)
        ])

        return X_embed

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
