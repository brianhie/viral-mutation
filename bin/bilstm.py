from utils import *
from lstm import *

from lstm import _iterate_lengths

from keras.models import Model
from keras.layers import Concatenate, Input, Lambda

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

class BiLSTMLanguageModel(object):
    def __init__(
            self,
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=256,
            n_hidden=2,
            n_epochs=1,
            batch_size=1000,
            cache_dir='.',
            verbose=False
    ):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))

        input_pre = Input(shape=(seq_len - 2,))
        input_post = Input(shape=(seq_len - 2,))

        embed = Embedding(vocab_size + 1, embedding_dim,
                          input_length=seq_len - 2)
        x_pre = embed(input_pre)
        x_post = embed(input_post)

        for _ in range(n_hidden):
            lstm = LSTM(hidden_dim, return_sequences=True)
            x_pre = lstm(x_pre)
            x_post = lstm(x_post)

        x_post = Lambda(
            lambda x: K.reverse(x, axes=1),
            output_shape=(seq_len - 2, hidden_dim),
        )(x_post)

        x = Concatenate(axis=2)([ x_pre, x_post ])

        output = TimeDistributed(
            Dense(vocab_size + 1, activation='softmax'),
            input_shape=(seq_len - 2, 2 * hidden_dim),
        )(x)

        self.model_ = Model(inputs=[ input_pre, input_post ],
                            outputs=output)

        self.seq_len_ = seq_len
        self.vocab_size_ = vocab_size
        self.embedding_dim_ = embedding_dim
        self.hidden_dim_ = hidden_dim
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
        self.model_.compile(
            loss='sparse_categorical_crossentropy', optimizer=opt,
            metrics=[ 'accuracy' ]
        )

        mkdir_p('{}/checkpoints.old/bilstm'.format(self.cache_dir_))
        model_name = 'bilstm'
        checkpoint = ModelCheckpoint(
            '{}/checkpoints.old/bilstm/{}_{}'
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
