#coding=utf-8
import re
import collections
import jieba
import string
import numpy as np
import tensorflow as tf

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
                word_list += sentence

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
                if sos_eos:
                    temp = [SOS] + temp + [EOS]
                if len(temp) > sentence_max_len:
                    sentence_max_len = len(temp)
                sentences.append(temp)
            elif sep == 'jieba':
                temp = [word.strip() for word in jieba.cut(sentence, cut_all=False) if word.strip() != '']
                if sos_eos:
                    temp = [SOS] + temp + [EOS]
                if len(temp) > sentence_max_len:
                    sentence_max_len = len(temp)
                sentences.append(temp)
            elif sep == 'ch_word':
                temp = [word.strip() for word in sentence if word.strip() != '']
                if sos_eos:
                    temp = [SOS] + temp + [EOS]
                if len(temp) > sentence_max_len:
                    sentence_max_len = len(temp)
                sentences.append(temp)
            else:
                print("====>>>> separate symbol is not defined")
        return sentences, sentence_max_len

    @staticmethod
    def parse_word_list(word_list, max_vocab_size=None):
        '''
        convert words to {word: id}
        :param data: list of word
        :return:
        '''
        counter = collections.Counter(word_list)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        if max_vocab_size is not None:
            count_pairs = count_pairs[:max_vocab_size]

        words, _ = list(zip(*count_pairs))

        word_to_id = {}
        id_to_word = {}
        for index, word in enumerate(words):
            idx = index
            word_to_id[word] = idx
            id_to_word[idx] = word
        word_to_id[UNK] = len(word_to_id)
        id_to_word[len(id_to_word)] = UNK

        assert len(word_to_id) == len(id_to_word)

        # for i in random.sample(range(len(id_to_word)), 2):
        #     print("id: " + str(i) + " - " + id_to_word[i])
        #     print("word: " + id_to_word[i] + " - " + str(word_to_id[id_to_word[i]]))

        return word_to_id, id_to_word, len(word_to_id)


if __name__ == '__main__':
    data_path = 'ch_ch.txt'
    inputs, targets = read_translation_corpus(data_path)

    lang_input = Languages(inputs)
    lang_target = Languages(targets)

    lang_input.build_vocab(sep='ch_word')
    lang_target.build_vocab(sep='jieba', sos_eos=True)

    print('input vocab: ' + str(len(lang_input.word_to_id)))
    print('target vocab: ' + str(len(lang_target.word_to_id)))

