from utils import *
from lstm import *

from lstm import  _iterate_lengths, _split_and_pad

from keras.layers import Bidirectional

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
    X = [
        np.concatenate([ X_seq[:i], X_seq[i + 1:] ])
        for X_seq in X_seqs for i in range(len(X_seq))
    ]
    y = np.array([
        X_seq[i]
        for X_seq in X_seqs for i in range(len(X_seq))
    ])

    if verbose > 1:
        tprint('Padding...')
    X = pad_sequences(
        X, maxlen=seq_len,
        dtype='int32', padding='pre', truncating='pre', value=0.
    )
    y = to_categorical(y, num_classes=vocab_size + 1)

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
            verbose=False
    ):
        model = Sequential()
        model.add(Embedding(vocab_size + 1, embedding_dim,
                            input_length=seq_len))
        for _ in range(n_hidden - 1):
            model.add(Bidirectional(
                LSTM(hidden_dim, return_sequences=True)
            ))
        model.add(Bidirectional(LSTM(hidden_dim)))
        model.add(Dense(vocab_size + 1, activation='softmax'))
        self.model_ = model

        self.seq_len_ = seq_len
        self.vocab_size_ = vocab_size
        self.embedding_dim_ = embedding_dim
        self.hidden_dim_ = hidden_dim
        self.n_hidden_ = n_hidden
        self.n_epochs_ = n_epochs
        self.batch_size_ = batch_size
        self.verbose_ = verbose

    def fit(self, X_cat, lengths):
        X, y = _split_and_pad(
            X_cat, lengths, self.seq_len_, self.vocab_size_, self.verbose_
        )

        opt = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                   amsgrad=False)
        self.model_.compile(
            loss='categorical_crossentropy', optimizer=opt,
            metrics=[ 'accuracy' ]
        )

        checkpoint = ModelCheckpoint(
            'target/checkpoints/lstm/bilstm_allcond_256-{epoch:02d}.hdf5',
            save_best_only=False, save_weights_only=False,
            mode='auto', period=1
        )

        self.model_.fit(
            X, y, epochs=self.n_epochs_, batch_size=self.batch_size_,
            shuffle=True, verbose=self.verbose_ > 0, callbacks=[ checkpoint ],
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
