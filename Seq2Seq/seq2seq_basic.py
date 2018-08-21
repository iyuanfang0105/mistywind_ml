# coding=utf-8
import os
import collections
import logging
from io import open
import string
import re
import random
import codecs

from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf
import numpy as np

import jieba

logging.basicConfig(level=logging.INFO)

UNK = "<UNK>"
SOS = "<WD_SOS>"
EOS = "<WD_EOS>"

def replace_puncuation(s):
    '''
    replace puncuation in a string
    :param s:
    :return:
    '''
    en_puncuation = string.punctuation
    ch_puncuation = """！@#￥%……&*（）——+{}【】：；'"《》，。？、~`、|""".decode('utf-8')
    result = re.sub(ur"[%s]+" % en_puncuation, " ", s)
    result = re.sub(ur"[%s]+" % ch_puncuation, " ", result)

    return result


def read_translation_corpus(filename):
    with tf.gfile.GFile(filename, "r") as f:
        data = f.read().decode("utf-8").split('\n')
    inputs = []
    targets = []
    for line in data:
        if len(line) > 1:
            temp = line.strip().split('\t')
            inputs.append(replace_puncuation(temp[0]).strip())
            targets.append(replace_puncuation(temp[1]).strip())
    assert len(inputs) == len(targets)
    return inputs, targets


class Languages(object):
    def __init__(self, sentences):
        self.sentences = sentences
        self.sentences_segmented = None
        self.max_sentence_len = None
        self.vocb_size = None
        self.word_to_id = None
        self.id_to_word = None

    def build_vocab(self, sep='space', max_vocab_size=None, vocab_file='', sos_eos=False):
        '''
        build vocabulary
        :param data: a list of words
        :return:
        '''
        self.sentences_segmented, self.max_sentence_len = self.words_segment(self.sentences, sep=sep, sos_eos=sos_eos)

        if vocab_file != '':
            self.word_to_id = np.load(vocab_file).item()
            self.id_to_word = {y: x for x, y in self.word_to_id.items()}
            assert len(self.word_to_id) == len(self.id_to_word)
            self.vocb_size = len(self.word_to_id)
        else:
            word_list = []
            for sentence in self.sentences_segmented:
                word_list.extend(sentence)

            self.word_to_id, self.id_to_word, self.vocb_size = self.parse_word_list(word_list, max_vocab_size=max_vocab_size)

    @staticmethod
    def words_segment(sentents, sep='space', sos_eos=False):
        '''
        segment word from corpus
        :return:
        :corpus: list of sentents
        :split: space or jieba or char
        '''

        sentences = []
        sentence_max_len = 0
        for sentence in sentents:
            words = None
            if sep == 'space':
                temp = [word.strip() for word in sentence.split(' ') if word.strip() != '']
                if len(temp) > sentence_max_len:
                    sentence_max_len = len(temp)
                if sos_eos:
                    temp = [SOS] + temp + [EOS]
                sentences.append(temp)
            elif sep == 'jieba':
                temp = [word.strip() for word in jieba.cut(sentence, cut_all=False) if word.strip() != '']
                if len(temp) > sentence_max_len:
                    sentence_max_len = len(temp)
                if sos_eos:
                    temp = [SOS] + temp + [EOS]
                sentences.append(temp)
            elif sep == 'char':
                temp = [word.strip() for word in sentence if word.strip() != '']
                if len(temp) > sentence_max_len:
                    sentence_max_len = len(temp)
                sentences.append(temp)
            else:
                print("====>>>> separate symbol is not defined")
        return sentences, sentence_max_len

    @staticmethod
    def parse_word_list(word_list, max_vocab_size=2000):
        '''
        convert words to {word: id}
        :param data: list of word
        :return:
        '''
        counter = collections.Counter(word_list)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        if max_vocab_size is not None:
            count_pairs = count_pairs[:max_vocab_size - 3]

        words, _ = list(zip(*count_pairs))

        word_to_id = {}
        id_to_word = {}
        for index, word in enumerate(words):
            idx = index
            word_to_id[word] = idx
            id_to_word[idx] = word

        assert len(word_to_id) == len(id_to_word)

        # for i in random.sample(range(len(id_to_word)), 2):
        #     print("id: " + str(i) + " - " + id_to_word[i])
        #     print("word: " + id_to_word[i] + " - " + str(word_to_id[id_to_word[i]]))

        return word_to_id, id_to_word, len(word_to_id)


class Seq2SeqBasic(object):
    def __init__(self, input_dim, output_dim, hidden_num, model_save_path):
        self.input_dim = input_dim
        self.hidden_num = hidden_num
        self.output_dim = output_dim
        self.encoder = None
        self.decoder = None
        self.seq2seq_model = None
        self.model_save_name = 'weights/Seq2Seq-weights-improvement-{epoch:04d}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.hdf5'
        self.checkpoint = ModelCheckpoint(os.path.join(model_save_path, self.model_save_name),
                                          monitor='val_loss',
                                          verbose=1,
                                          save_best_only=True)
        self.early_stop = EarlyStopping(monitor='val_loss', patience=5)

    def build_model(self, is_train=True, pretrained_model_path='', embedding=True):
        if is_train:
            # enc_input = Input(shape=(None, self.input_dim))
            enc_input = Input(shape=(None,))
            # encoder
            enc = LSTM(self.hidden_num, return_state=True)

            if embedding:
                embed_enc_input = Embedding(input_dim=self.input_dim, output_dim=self.hidden_num)(enc_input)
                enc_output, state_h, state_c = enc(embed_enc_input)
            else:
                enc_output, state_h, state_c = enc(enc_input)
            enc_states = [state_h, state_c]

            # decoder
            # Set up the decoder, using `encoder_states` as initial state.
            # dec_input = Input(shape=(None, self.output_dim))
            dec_input = Input(shape=(None,))
            # We set up our decoder to return full output sequences,
            # and to return internal states as well. We don't use the
            # return states in the training model, but we will use them in inference.
            dec = LSTM(self.hidden_num, return_sequences=True, return_state=True)

            if embedding:
                embed_dec_input = Embedding(input_dim=self.output_dim, output_dim=self.hidden_num)(dec_input)
                dec_output, _, _ = dec(embed_dec_input, initial_state=enc_states)
            else:
                dec_output, _, _ = dec(dec_input, initial_state=enc_states)

            dec_dense = Dense(self.output_dim, activation='softmax')
            dec_output = dec_dense(dec_output)

            # Define the model that will turn
            # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
            self.model = Model([enc_input, dec_input], dec_output)
            print("====>>>> Seq2Seq Model: ")
            print(self.model.summary())
        else:
            m = load_model(pretrained_model_path)
            enc_input = m.input[0]
            # embed_enc_input = m.layers[2].output
            enc_outputs, state_h_enc, state_c_enc = m.layers[4].output  # lstm_1
            enc_states = [state_h_enc, state_c_enc]
            self.encoder = Model(enc_input, enc_states)
            print("====>>>> Encoder: ")
            logging.info(self.encoder.summary())

            dec_inputs = Input(shape=(None,))
            embed_dec_input = m.layers[3](dec_inputs)
            dec_state_input_h = Input(shape=(self.hidden_num,))
            dec_state_input_c = Input(shape=(self.hidden_num,))
            dec_states_inputs = [dec_state_input_h, dec_state_input_c]
            dec_lstm = m.layers[5]
            dec_outputs, state_h_dec, state_c_dec = dec_lstm(embed_dec_input, initial_state=dec_states_inputs)
            dec_states = [state_h_dec, state_c_dec]
            dec_dense = m.layers[6]
            dec_outputs = dec_dense(dec_outputs)
            self.decoder = Model([dec_inputs] + dec_states_inputs, [dec_outputs] + dec_states)
            print("====>>>> Decoder: ")
            logging.info(self.decoder.summary())

    def trian(self, encoder_input_data, decoder_input_data, decoder_target_data, batch_size, epochs):
        if self.model is not None:
            self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
            self.history = self.model.fit([encoder_input_data, decoder_input_data],
                                          decoder_target_data,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          validation_split=0.2,
                                          callbacks=[self.checkpoint, self.early_stop])

    def encode(self, sentence):
        states_value = None
        if self.encoder is not None:
            # Encode the input as state vectors.
            states_value = self.encoder.predict(sentence)
        else:
            print("====>>>> Please build model firstly")
        return states_value

    def decode(self, init_states, vocab, reverse_vocab, max_sentence_len=100):
        decoded_sentence = []

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = vocab[SOS]

        if self.decoder is not None:
            stop_condition = False
            while not stop_condition:
                output, h, c = self.decoder.predict([target_seq] + init_states)

                # Sample a token
                sampled_token_index = np.argmax(output[0, -1, :])
                sampled_char = reverse_vocab[sampled_token_index]
                decoded_sentence.append(sampled_char)

                # Exit condition: either hit max length
                # or find stop character.
                if sampled_char == EOS or (len(decoded_sentence) > max_sentence_len):
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1))
                target_seq[0, 0] = sampled_token_index

                # Update states
                init_states = [h, c]

        else:
            print("====>>>> Please build model firstly")
        return decoded_sentence


if __name__ == '__main__':
    data_path = 'en_ch.txt'
    hidden_num = 300
    epochs = 200
    batch_size = 128
    model_save_path = ''
    is_train = False
    embedding = True

    input_sentences, target_sentences = read_translation_corpus(data_path)
    lang_input = Languages(input_sentences)
    lang_target = Languages(target_sentences)

    if is_train:
        lang_input.build_vocab(sep='space', max_vocab_size=3500)
        lang_target.build_vocab(sep='jieba', max_vocab_size=3000)
    else:
        lang_input.build_vocab(vocab_file='lang_input.w2id.npy', sep='space', max_vocab_size=3500)
        lang_target.build_vocab(vocab_file='lang_target.w2id.npy', sep='jieba', max_vocab_size=3000)


    # dataset
    if embedding:
        enc_input_data = np.zeros((len(lang_input.sentences), lang_input.max_sentence_len), dtype=np.float32)
        dec_input_data = np.zeros((len(lang_target.sentences), lang_target.max_sentence_len), dtype=np.float32)
        dec_ouput_data = np.zeros((len(lang_target.sentences), lang_target.max_sentence_len, lang_target.vocb_size),
                                  dtype=np.float32)

        for i, (input_text, target_text) in enumerate(
                zip(lang_input.sentences_segmented, lang_target.sentences_segmented)):
            for t, word in enumerate(input_text):
                if lang_input.word_to_id.has_key(word):
                    enc_input_data[i, t] = lang_input.word_to_id[word]
                else:
                    enc_input_data[i, t] = lang_input.word_to_id[UNK]

            for t, word in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                if lang_target.word_to_id.has_key(word):
                    dec_input_data[i, t] = lang_target.word_to_id[word]
                    if t > 0:
                        # decoder_target_data will be ahead by one timestep
                        # and will not include the start character.
                        dec_ouput_data[i, t - 1, lang_target.word_to_id[word]] = 1.
                else:
                    dec_input_data[i, t] = lang_target.word_to_id[UNK]
                    if t > 0:
                        # decoder_target_data will be ahead by one timestep
                        # and will not include the start character.
                        dec_ouput_data[i, t - 1, lang_target.word_to_id[UNK]] = 1.

        print('====>>>> encoder input: ' + str(enc_input_data.shape))
        print('====>>>> decoder input: ' + str(dec_input_data.shape))
        print('====>>>> decoder output: ' + str(dec_ouput_data.shape))
    else:
        enc_input_data = np.zeros((len(lang_input.sentences), lang_input.max_sentence_len, lang_input.vocb_size),
                                  dtype=np.float32)
        dec_input_data = np.zeros((len(lang_target.sentences), lang_target.max_sentence_len, lang_target.vocb_size),
                                  dtype=np.float32)
        dec_ouput_data = np.zeros((len(lang_target.sentences), lang_target.max_sentence_len, lang_target.vocb_size),
                                  dtype=np.float32)

        for i, (input_text, target_text) in enumerate(
                zip(lang_input.sentences_segmented, lang_target.sentences_segmented)):
            for t, word in enumerate(input_text):
                enc_input_data[i, t, lang_input.word_to_id[word]] = 1.
            for t, word in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                dec_input_data[i, t, lang_target.word_to_id[word]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    dec_ouput_data[i, t - 1, lang_target.word_to_id[word]] = 1.

    # train and infer
    if is_train:
        # model
        m = Seq2SeqBasic(lang_input.vocb_size, lang_target.vocb_size, hidden_num, model_save_path)
        m.build_model(is_train=is_train, embedding=embedding)
        m.trian(enc_input_data, dec_input_data, dec_ouput_data, batch_size, epochs)
        # with open('log/trainning_history.plk', 'w') as fo:
        #     pickle.dump(m.history.history, fo, pickle.HIGHEST_PROTOCOL)
        np.save('log/trainning_history.npy', m.history.history)
        np.save('lang_input.w2id.npy', lang_input.word_to_id)
        np.save('lang_target.w2id.npy', lang_target.word_to_id)
    else:
        # pretrainned_model_path = 'weights-improvement-18-0.10.hdf5'
        pretrainned_model_path = 'weights/Seq2Seq-weights-improvement-0015-1.25215-0.08681.hdf5'

        m = Seq2SeqBasic(lang_input.vocb_size, lang_target.vocb_size, hidden_num, model_save_path)
        m.build_model(is_train=is_train, pretrained_model_path=pretrainned_model_path)
        # m.trian(enc_input_data, dec_input_data, dec_ouput_data, batch_size, epochs)
        for i in range(100):
            sentence = enc_input_data[i: i + 1]

            enc_states = m.encode(sentence)
            dec_sentence = m.decode(enc_states, lang_target.word_to_id, lang_target.id_to_word,
                                    max_sentence_len=lang_target.max_sentence_len)
            print('-')
            print('Input: ' + lang_input.sentences[i])
            print('Decoded: ' + ' '.join(dec_sentence))
