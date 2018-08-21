import os
import argparse
import collections
import datetime as dt

import numpy as np
import tensorflow as tf

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint


def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").replace("\n", "<eos>").split()


def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def load_data(data_path):
    # get the data paths
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    print(train_data[:5])
    print(word_to_id)
    print(vocabulary)
    print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary


class KerasBatchGenerator(object):
    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y


class KerasLSTM(object):
    def __init__(self, vocabulary, num_layers, hidden_size, num_epochs, batch_size, num_steps, model_save_path, use_dropout=True):
        self.vocabulary = vocabulary
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.use_dropout = use_dropout
        self.model = None
        self.checkpointer = ModelCheckpoint(filepath=model_save_path + '/model-{epoch:02d}.hdf5', verbose=1)

    def build_model(self):
        m = Sequential()
        m.add(Embedding(self.vocabulary, self.hidden_size, input_length=self.num_steps))
        for i in range(self.num_layers):
            m.add(LSTM(self.hidden_size, return_sequences=True))
        if self.use_dropout:
            m.add(Dropout(0.5))
        m.add(TimeDistributed(Dense(vocabulary)))
        m.add(Activation('softmax'))
        m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        self.model = m
        print(self.model.summary())

    def train(self, train_data, valid_data):
        train_data_generator = KerasBatchGenerator(train_data, self.num_steps, self.batch_size, self.vocabulary,
                                                   skip_step=self.num_steps)
        valid_data_generator = KerasBatchGenerator(valid_data, self.num_steps, self.batch_size, self.vocabulary,
                                                   skip_step=self.num_steps)

        if self.model is not None:
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        else:
            print("Please build model firstly")

        self.model.fit_generator(train_data_generator.generate(),
                                 len(train_data) // (self.batch_size * self.num_steps),
                                 self.num_epochs,
                                 validation_data=valid_data_generator.generate(),
                                 validation_steps=len(valid_data) // (self.batch_size * self.num_steps),
                                 callbacks=[self.checkpointer])
        # model.fit_generator(train_data_generator.generate(), 2000, num_epochs,
        #                     validation_data=valid_data_generator.generate(),
        #                     validation_steps=10)
        self.model.save(data_path + "final_model.hdf5")


if __name__ == '__main__':
    data_path = '/Users/wind/WORK/public_data_set/ptb/simple-examples/data'
    train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data(data_path)

    print('train data size: ' + str(len(train_data)))

    lstm_model = KerasLSTM(vocabulary, num_layers=2, hidden_size=500, num_epochs=50, batch_size=20, num_steps=30, model_save_path='./', use_dropout=True)
    lstm_model.build_model()
    lstm_model.train(train_data, valid_data)
