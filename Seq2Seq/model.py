import os

from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

from language import SOS, EOS


class Seq2SeqBasic(object):
    def __init__(self, input_dim, output_dim, hidden_num, model_save_path):
        self.input_dim = input_dim
        self.hidden_num = hidden_num
        self.output_dim = output_dim
        self.encoder = None
        self.decoder = None
        self.seq2seq_model = None
        self.model_save_name = 'Seq2Seq-weights-improvement-{epoch:04d}-{val_loss:.5f}-{val_acc:.5f}.hdf5'
        self.checkpoint = ModelCheckpoint(os.path.join(model_save_path, self.model_save_name),
                                          monitor='val_acc',
                                          verbose=1,
                                          save_best_only=True)
        self.early_stop = EarlyStopping(monitor='val_acc', patience=5)

    def build_model(self, is_train=True, pretrained_model_path=''):
        if is_train:
            enc_input = Input(shape=(None,))

            # emdedding
            embed_enc_input = Embedding(input_dim=self.input_dim, output_dim=self.hidden_num)(enc_input)

            # encoder
            enc = LSTM(self.hidden_num, return_state=True)
            enc_output, state_h, state_c = enc(embed_enc_input)

            enc_states = [state_h, state_c]

            # decoder
            # Set up the decoder, using `encoder_states` as initial state.
            # dec_input = Input(shape=(None, self.output_dim))
            dec_input = Input(shape=(None,))
            embed_dec_input = Embedding(input_dim=self.output_dim, output_dim=self.hidden_num)(dec_input)
            # We set up our decoder to return full output sequences,
            # and to return internal states as well. We don't use the
            # return states in the training model, but we will use them in inference.
            dec = LSTM(self.hidden_num, return_sequences=True, return_state=True)
            dec_output, _, _ = dec(embed_dec_input, initial_state=enc_states)

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
            print(self.encoder.summary())

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
            print(self.decoder.summary())

    def trian(self, encoder_input_data, decoder_input_data, decoder_target_data, batch_size, epochs):
        if self.model is not None:
            self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
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
        decoded_sentence = [SOS]

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
