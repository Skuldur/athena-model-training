from os import path

import numpy
from keras.layers import Dense, CuDNNLSTM, LSTM, concatenate, SpatialDropout1D, Bidirectional, Embedding, Input, Dropout, TimeDistributed, GlobalAveragePooling1D
from keras.layers.merge import Concatenate
from keras.models import Model
from intent_trainer.layers import CRF
from keras.callbacks import ModelCheckpoint


class KerasModel():
    def train(self, train_seq, test_seq, epochs):
        self.model.fit_generator(
            generator=train_seq,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            validation_data=test_seq,
        )

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def save_weights(self, name):
        self.model.save_weights(name)


class BiLSTMCRF(KerasModel):
    """An implementation of a Bidirectional LSTM with CRF model.
    """
    def __init__(self, labels, n_words, n_chars):
        self.n_labels = len(labels)
        self.n_words = n_words
        self.n_chars = n_chars

        print(n_chars)

        #Word embedding
        word_in = Input(shape=(None,))
        word_emb = Embedding(input_dim=self.n_words+1, output_dim=128)(word_in)
        
        #Character embedding
        char_in = Input(shape=(None, None,))
        char_emb = TimeDistributed(Embedding(input_dim=self.n_chars + 2, output_dim=16,
                         mask_zero=True))(char_in)

        # character LSTM to get word encodings by characters
        char_enc = TimeDistributed(LSTM(units=28, return_sequences=False,
                                        recurrent_dropout=0.5))(char_emb)

        concat = concatenate([word_emb, char_enc])
        bi_lstm = SpatialDropout1D(0.3)(concat)

        for i in range(2):
            bi_lstm = Bidirectional(
                LSTM(
                    units=256, 
                    return_sequences=True,
                    recurrent_dropout=0.3
                )
            )(bi_lstm)

        linear = TimeDistributed(Dense(self.n_labels, activation='relu'))(bi_lstm)  # softmax output layer

        crf = CRF(self.n_labels, sparse_target=False)
        pred = crf(linear)

        self.model = Model(inputs=[word_in, char_in], outputs=pred)
        self.loss = crf.loss_function
        self.accuracy = crf.accuracy
        self.model.compile(loss=self.loss, optimizer='adam', metrics=[self.accuracy])

    def predict(self, input):
        p = self.model.predict(input)
        p = numpy.argmax(p, axis=-1)

        return p[0]


class TextClassification(KerasModel):

    def __init__(self, labels, n_words, dropout=0.3):
        self.n_labels = len(labels)
        self.n_words = n_words
        self.dropout = dropout

        # build word embedding
        input = Input(shape=(None,))
        model = Embedding(input_dim=self.n_words+1, output_dim=50)(input)
        #model = Dropout(self.dropout)(model)
        model = GlobalAveragePooling1D()(model)
        out = Dense(256, activation="relu")(model)  # softmax output layer
        out = Dense(self.n_labels, activation="softmax")(model)  # softmax output layer

        self.model = Model(inputs=input, outputs=out)
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def predict(self, input):
        p = self.model.predict(input)
        p = numpy.argmax(p, axis=-1)

        return p