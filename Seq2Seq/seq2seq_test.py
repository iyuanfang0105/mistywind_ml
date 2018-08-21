from language import Languages, read_translation_corpus
from dataset import Dataset
from model import Seq2SeqBasic
import numpy as np

# read dataset file and build language
data_path = 'ch_ch.txt'
# data_path = 'en_ch.txt'
inputs, targets = read_translation_corpus(data_path)

lang_input = Languages(inputs)
lang_target = Languages(targets)

input_max_vocab_size = None
target_max_vocab_size = None

# lang_input.build_vocab(sep='space', max_vocab_size=input_max_vocab_size)
lang_input.build_vocab(sep='jieba', max_vocab_size=input_max_vocab_size)
lang_target.build_vocab(sep='jieba', sos_eos=True, max_vocab_size=target_max_vocab_size)

print('input vocab: ' + str(len(lang_input.word_to_id)))
print('input_max_sentence_len: ' + str(lang_input.max_sentence_len))
print('target vocab: ' + str(len(lang_target.word_to_id)))
print('target_max_sentence_len: ' + str(lang_target.max_sentence_len))


# build dataset
dataset = Dataset(lang_input, lang_target)
dataset.build_dataset()

print('encoder input: ' + str(dataset.enc_input_data.shape))
print('decoder input: ' + str(dataset.dec_input_data.shape))
print('decoder output: ' + str(dataset.dec_ouput_data.shape))

# build model
hidden_num = 300
model_save_path = 'weights'
is_train = False
batch_size = 64
epochs = 40
seq2seq_model = Seq2SeqBasic(lang_input.vocb_size, lang_target.vocb_size, hidden_num, model_save_path)

# trainning
if is_train:
    seq2seq_model.build_model(is_train)
    seq2seq_model.trian(dataset.enc_input_data, dataset.dec_input_data, dataset.dec_ouput_data, batch_size, epochs)
    np.save('log/train_history.npy', seq2seq_model.history.history)
else:
    # weights = 'weights/seq2seq-weights-improvement-0009-1.83286-0.77164.hdf5'
    weights = 'weights/Seq2Seq-weights-improvement-0016-1.39359-0.82885-word-ch-self-self.hdf5'
    seq2seq_model.build_model(is_train, pretrained_model_path=weights)
    for i in range(100):
        sentence = dataset.enc_input_data[i: i + 1]

        enc_states = seq2seq_model.encode(sentence)
        dec_sentence = seq2seq_model.decode(enc_states, lang_target.word_to_id, lang_target.id_to_word,
                                            max_sentence_len=lang_target.max_sentence_len)
        print('-')
        print('Input: ' + lang_input.sentences[i])
        print('Decoded: ' + ' '.join(dec_sentence))



