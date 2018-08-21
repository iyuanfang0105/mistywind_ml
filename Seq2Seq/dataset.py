import numpy as np

import language


class Dataset(object):
    def __init__(self, lang_input, lang_target):
        self.lang_input = lang_input
        self.lang_target = lang_target
        self.enc_input_data = np.zeros((len(lang_input.sentences), lang_input.max_sentence_len), dtype=np.float32)
        self.dec_input_data = np.zeros((len(lang_target.sentences), lang_target.max_sentence_len), dtype=np.float32)
        self.dec_ouput_data = np.zeros((len(lang_target.sentences), lang_target.max_sentence_len, lang_target.vocb_size), dtype=np.float32)

    def build_dataset(self):
        for i, (input_text, target_text) in enumerate(zip(self.lang_input.sentences_segmented, self.lang_target.sentences_segmented)):
            for t, word in enumerate(input_text):
                if word in self.lang_input.word_to_id:
                    self.enc_input_data[i, t] = self.lang_input.word_to_id[word]
                else:
                    self.enc_input_data[i, t] = self.lang_input.word_to_id[language.UNK]

            for t, word in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                if word in self.lang_target.word_to_id:
                    temp = self.lang_target.word_to_id[word]
                else:
                    temp = self.lang_target.word_to_id[language.UNK]
                self.dec_input_data[i, t] = temp
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    self.dec_ouput_data[i, t - 1, temp] = 1.

