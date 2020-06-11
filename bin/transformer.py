from utils import *

from transformer_layers import Encoder
from lstm import _iterate_lengths

import tensorflow as tf

def _split_and_pad(X_cat, lengths, seq_len, vocab_size, verbose):
    if X_cat.shape[0] != sum(lengths):
        raise ValueError('Length dimension mismatch: {} and {}'
                         .format(X_cat.shape[0], sum(lengths)))

    if verbose > 1:
        tprint('Splitting {} seqs...'.format(len(lengths)))
    X_seqs = [
        X_cat[start:end].flatten()
        for start, end in _iterate_lengths(lengths, seq_len)
    ]

    if verbose > 1:
        tprint('Padding {} splitted...'.format(len(X_seqs)))
    padded = pad_sequences(
        X_seqs, maxlen=seq_len,
        dtype='int32', padding='pre', truncating='pre', value=0
    )
    X_pre = padded[:, :-2]
    X_post = padded[:, 2:]
    y = padded[:, 1:-1, None]

    X = [ X_pre, X_post ]

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

        input_ = tf.keras.Input(shape=(seq_len,))

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

        opt = tf.keras.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
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
        X, _ = _split_and_pad(
            X_cat, lengths, self.seq_len_, self.vocab_size_, self.verbose_
        )

        X_

        (enc_padding_mask,
         combined_mask,
         dec_padding_mask) = create_masks(X, )

        predictions, _ = self.model_(
            X, X[:, :-1], True,
                enc_padding_mask,
                combined_mask,
                dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        train_loss(loss)
        train_accuracy(tar_real, predictions)
