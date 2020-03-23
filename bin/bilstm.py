from utils import *
from lstm import *

from lstm import _iterate_lengths

from keras.models import Model
from keras.layers import concatenate, Input

def _split_and_pad(X_cat, lengths, seq_len, vocab_size, verbose):
    if X_cat.shape[0] != sum(lengths):
        raise ValueError('Length dimension mismatch: {} and {}'
                         .format(X_cat.shape[0], sum(lengths)))

    if verbose > 1:
        tprint('Splitting...')
    X_seqs = [
        X_cat[start:end].flatten()
        for start, end in _iterate_lengths(lengths, seq_len)
    ]
    X_pre = [
        X_seq[:i] for X_seq in X_seqs for i in range(len(X_seq))
    ]
    X_post = [
        X_seq[i + 1:] for X_seq in X_seqs for i in range(len(X_seq))
    ]
    y = np.array([
        X_seq[i] for X_seq in X_seqs for i in range(len(X_seq))
    ])

    if verbose > 1:
        tprint('Padding...')
    X_pre = pad_sequences(
        X_pre, maxlen=seq_len - 1,
        dtype='int32', padding='pre', truncating='pre', value=0.
    )
    X_post = pad_sequences(
        X_post, maxlen=seq_len - 1,
        dtype='int32', padding='post', truncating='post', value=0.
    )
    X_post = np.flip(X_post, 1)
    X = [ X_pre, X_post ]

    if verbose > 1:
        tprint('Done splitting and padding.')
    return X, y

class BiLSTMLanguageModel(object):
    def __init__(
            self,
            seq_len,
            vocab_size,
            attention=False,
            embedding_dim=20,
            hidden_dim=256,
            n_hidden=2,
            n_epochs=1,
            batch_size=1000,
            cache_dir='.',
            verbose=False
    ):
        input_pre = Input(shape=(seq_len - 1,))
        input_post = Input(shape=(seq_len - 1,))

        embed = Embedding(vocab_size + 1, embedding_dim,
                          input_length=seq_len - 1)
        x_pre = embed(input_pre)
        x_post = embed(input_post)

        for _ in range(n_hidden - 1):
            lstm = LSTM(hidden_dim, return_sequences=True)
            x_pre = lstm(x_pre)
            x_post = lstm(x_post)
        lstm = LSTM(hidden_dim, return_sequences=attention)
        x_pre = lstm(x_pre)
        x_post = lstm(x_post)

        x = concatenate([ x_pre, x_post ])

        if attention:
            from seq_self_attention import SelfAttention
            x = SelfAttention()(x)

        output = Dense(vocab_size + 1, activation='softmax')(x)

        self.model_ = Model(inputs=[ input_pre, input_post ],
                            outputs=output)

        self.seq_len_ = seq_len
        self.vocab_size_ = vocab_size
        self.attention_ = attention
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

        mkdir_p('{}/checkpoints/bilstm'.format(self.cache_dir_))
        model_name = 'bilstm{}'.format('-a' if self.attention_ else '')
        checkpoint = ModelCheckpoint(
            '{}/checkpoints/bilstm/{}_{}'
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
        y_pred = self.model_.predict(X, verbose=self.verbose_ > 0,
                                     batch_size=self.batch_size_)

        prob = y_pred[y_true == 1.].flatten()
        with np.errstate(divide='ignore'):
            logprob = np.log(prob)
        return np.sum(logprob)
