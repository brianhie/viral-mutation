from utils import *
from lstm import *
from bilstm import _split_and_pad

from keras.models import Model
from keras.layers import concatenate, Input, Reshape

class DNNLanguageModel(object):
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
            fp_precision='float32',
            verbose=False
    ):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))
        if fp_precision != K.floatx():
            K.set_floatx(fp_precision)

        input_pre = Input(shape=(seq_len - 1,))
        input_post = Input(shape=(seq_len - 1,))

        embed = Embedding(vocab_size + 1, embedding_dim,
                          input_length=seq_len - 1)
        x_pre = Reshape((embedding_dim * (seq_len - 1),))(embed(input_pre))
        x_post = Reshape((embedding_dim * (seq_len - 1),))(embed(input_post))

        for _ in range(n_hidden):
            dense = Dense(hidden_dim, activation='relu')
            x_pre = dense(x_pre)
            x_post = dense(x_post)

        x = concatenate([ x_pre, x_post ])

        output = Dense(vocab_size + 1, activation='softmax')(x)

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

        mkdir_p('{}/checkpoints.1/dnn'.format(self.cache_dir_))
        model_name = 'dnn'
        checkpoint = ModelCheckpoint(
            '{}/checkpoints.1/dnn/{}_{}'
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
