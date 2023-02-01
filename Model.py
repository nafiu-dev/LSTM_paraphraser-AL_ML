from keras.models import Model
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed
from keras.callbacks import EarlyStopping



class LSTM_Model:
    def __init__(self,max_len_, x_max_words, y_max_words, x_data, y_data):
        self.max_len_ = max_len_
        self.x_max_words = x_max_words
        self.y_max_words = y_max_words

        self.x_data = x_data
        self.y_data = y_data

        self.embedding_dim = 300
        self.latent_dim = 200

    def print_func(self,msg):
        print(f"[+] {msg} | {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")


    def main_model(self):
        inputs = Input(shape=[self.max_len_],)
        enc_emb = Embedding(self.x_max_words, self.embedding_dim,trainable=True)(inputs)

        # Encoder LSTM 1
        encoder_lstm1 = LSTM(self.latent_dim, return_sequences=True,return_state=True, dropout=0.4,recurrent_dropout=0.4)
        (encoder_output1, state_h1, state_c1) = encoder_lstm1(enc_emb)

        # Encoder LSTM 2
        encoder_lstm2 = LSTM(self.latent_dim, return_sequences=True,return_state=True, dropout=0.4,recurrent_dropout=0.4)
        (encoder_output2, state_h2, state_c2) = encoder_lstm2(encoder_output1)

        # Encoder LSTM 3
        encoder_lstm3 = LSTM(self.latent_dim, return_state=True,return_sequences=True, dropout=0.4,recurrent_dropout=0.4)
        (encoder_outputs3, state_h3, state_c3) = encoder_lstm3(encoder_output2)

        # Encoder LSTM 4
        encoder_lstm4 = LSTM(self.latent_dim, return_state=True,return_sequences=True, dropout=0.4,recurrent_dropout=0.4)
        (encoder_output4, state_h4, state_c4) = encoder_lstm4(encoder_outputs3)

        # Encoder LSTM 5
        encoder_lstm5 = LSTM(self.latent_dim, return_state=True,return_sequences=True, dropout=0.4,recurrent_dropout=0.4)
        (encoder_output5, state_h5, state_c5) = encoder_lstm5(encoder_output4)

        # Encoder LSTM 6
        encoder_lstm6 = LSTM(self.latent_dim, return_state=True,return_sequences=True, dropout=0.4,recurrent_dropout=0.4)
        (encoder_output6, state_h6, state_c6) = encoder_lstm6(encoder_output5)

        # Encoder LSTM 7
        encoder_lstm7 = LSTM(self.latent_dim, return_state=True,return_sequences=True, dropout=0.4,recurrent_dropout=0.4)
        (encoder_outputs, state_h, state_c) = encoder_lstm7(encoder_output6)

        # DECODER
        decoder_inputs = Input(shape=(None, ))

        # Embedding layer
        dec_emb_layer = Embedding(self.y_max_words, self.embedding_dim, trainable=True)
        dec_emb = dec_emb_layer(decoder_inputs)

        # Decoder LSTM
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=0.4,recurrent_dropout=0.2)
        (decoder_outputs, decoder_fwd_state, decoder_back_state) = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

        # Dense layer
        decoder_dense = TimeDistributed(Dense(self.y_max_words, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model
        model = Model([inputs, decoder_inputs], decoder_outputs)

        # =============================================
        # saving model.summry
        self.print_func('*  saving model summary...')
        with open('./report/modelsummary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        self.print_func('*  saving model summary...')

        self.print_func('*    saving model before training...')
        model.save('./report/model_before_training_v1.h5')
        self.print_func('*    model saved.')

        self.model = model
        self.Training()

        # ENCODER MODEL
        self.encoder_model(inputs, encoder_outputs, state_h, state_c)
        
        # DECODER MODEL
        self.decoder_model(dec_emb_layer, decoder_inputs, decoder_lstm, decoder_dense)
        

    def Training(self):
        """
            compailing and training the model
        """

        # compiling the model
        self.model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
        
        # ================================
        self.print_func('STARTING TO TRAIN')


        self.model.fit(
            [self.x_data['x_tr'], self.x_data['y_tr'][:, :-1]],
            self.x_data['y_tr'].reshape(self.x_data['y_tr'].shape[0], self.x_data['y_tr'].shape[1], 1)[:, 1:],
            epochs=100,
            callbacks=[es],
            batch_size=128,
            validation_data=([self.y_data['x_val'], self.y_data['y_val'][:, :-1]],
                            self.y_data['y_val'].reshape(self.y_data['y_val'].shape[0], self.y_data['y_val'].shape[1], 1)[:,1:]),
        )
        
        # ================================
        self.print_func('DONE TRAINING')


        self.print_func('*  saving model after training... ')
        self.model.save('./report/model_after_training_v1.h5')
        self.print_func('*  model saved.')

        self.print_func('*  starting to evaluate')
        model_evaluation =  self.model.evaluate([self.y_data['x_val'], self.y_data['y_val'][:, :-1]],self.y_data['y_val'].reshape(self.y_data['y_val'].shape[0], self.y_data['y_val'].shape[1], 1)[:,1:])
        print(f'model Test accuracy: {model_evaluation * 100} \n')

        with open('./report/model_evaluation.txt', 'w+') as f:
            f.write('================================================')
            f.write(f'model Test accuracy: {model_evaluation * 100} \n')
            f.write('================================================')
        f.close()

    # encoder model function
    def encoder_model(self, inputs, encoder_outputs, state_h, state_c):
        self.print_func('CREATING ENCODER MODEL...')
        encoder_model = Model(inputs=inputs, outputs=[encoder_outputs,state_h, state_c])
        self.print_func('ENCODER MODEL CREATED.')

        self.print_func('saving encoder model...')
        encoder_model.save('./report/encoder_model_v1.h5')
        self.print_func('encoder model saved.')

        return encoder_model

    # decoder model function
    def decoder_model(self, dec_emb_layer, decoder_inputs, decoder_lstm, decoder_dense):
        self.print_func('CREATING DECODER MODEL...')
        # Below tensors will hold the states of the previous time step
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_hidden_state_input = Input(shape=(self.max_len_, self.latent_dim))

        # Get the embeddings of the decoder sequence
        dec_emb2 = dec_emb_layer(decoder_inputs)

        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        (decoder_outputs2, state_h2, state_c2) = decoder_lstm(dec_emb2,
                initial_state=[decoder_state_input_h, decoder_state_input_c])

        # A dense softmax layer to generate prob dist. over the target vocabulary
        decoder_outputs2 = decoder_dense(decoder_outputs2)

        # Final decoder model
        decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input,
                            decoder_state_input_h, decoder_state_input_c],
                            [decoder_outputs2] + [state_h2, state_c2])

        self.print_func('DECODER MODEL CREATED. ')


        self.print_func('saving decoder model...')
        decoder_model.save('./report/decoder_model_v1.h5')
        self.print_func('decoder model saved.')

        return decoder_model

    def run(self):
        self.print_func('starting to create models and training progress saving models as well')
        self.main_model()
        self.print_func('model trainging and saving is done')










