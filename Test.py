from keras.models import  load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# LOADING TOKNIZERS
# X
x_tokenizer_opendd = open("./report/x_tokenizer_v1", "rb")
loaded_x_tokenizer = pickle.load(x_tokenizer_opendd)
x_tokenizer_opendd.close()

# Y
y_tokenizer_opendd = open("./report/y_tokenizer_v1", "rb")
loaded_y_tokenizer = pickle.load(y_tokenizer_opendd)
y_tokenizer_opendd.close()


# LOADING WORD INDEX
# reverse_target_word_index = loaded_y_tokenizer.index_word
# reverse_source_word_index = loaded_x_tokenizer.index_word
# target_word_index = loaded_y_tokenizer.word_index
# print(reverse_target_word_index)
# LOADING WORD INDEX
word_index_opened = open("./report/word_list_object_v1", "rb")
word_index_object = pickle.load(word_index_opened)
word_index_opened.close()
# EXTRACTING WORD INDEX
reverse_target_word_index = word_index_object['reverse_target_word_index']
reverse_source_word_index = word_index_object['reverse_source_word_index']
target_word_index = word_index_object['target_word_index']

# loading max length max_len_
f = open('./report/max_len_.txt', 'r')
max_lenth_file = f.readline()
f.close()
max_len_ = int(max_lenth_file)


# LOADING ENCODER AND DECODER MODEL
loaded_encoder_model = load_model('./report/encoder_model_v1.h5', compile=False)
loaded_decoder_model = load_model('./report/decoder_model_v1.h5', compile=False)

def decode_sentence(input_seq):

    # Encode the input as state vectors.
    (e_out, e_h, e_c) = loaded_encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['start']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        (output_tokens, h, c) = loaded_decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if sampled_token != 'end':
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find the stop word.
        if sampled_token == 'end' or len(decoded_sentence.split()) \
            >= max_len_ - 1:
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        (e_h, e_c) = (h, c)

    return decoded_sentence


def generate_sentance(sentance):
    txts = loaded_x_tokenizer.texts_to_sequences(list([str(sentance)]))
    txts = pad_sequences(txts,maxlen=max_len_, padding='post')
    return decode_sentence(txts)
    # return txts


print(generate_sentance("What should"))