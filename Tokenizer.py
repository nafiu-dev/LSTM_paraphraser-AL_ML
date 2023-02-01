import pickle
import numpy as np
from pyparsing import col
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences


import pandas as pd

# testing dataframe
# data = [
#     ["hello a man", "hi man"], 
#     ["i'm the killer", "i am a killer"], 
#     ["what the hell are you cooking", "you cooking right now?"],
#     ["my sallery is $1508 USD per month", "i get $1508 USD per month"]

#     ]
# df = pd.DataFrame(data, columns=['question1', 'question2'])



class __Tokenizer:

    def __init__(self,data,cols, max_len_):
        self.data = data
        self.cols = cols
        self.max_len_ = max_len_
        

    def saving_tokenizer(self, toknizer, name):
        # tokenizer saving functions

        print(f'saving {name} tokenizer.... \n')

        tokenizer_opened = open(f"./report/{name}", "wb")
        pickle.dump(toknizer, tokenizer_opened)
        tokenizer_opened.close()

        print(f'{name} tokenizer saved. \n')


    def shape_checker(self):
        shape ={
            'x': {'x_tr': self.x_tr.shape, 'x_val': self.x_val.shape}, 
            'y': {'y_tr': self.y_tr.shape, 'y_val': self.y_val.shape}
        }
        print(shape)
        with open('./report/shapes.txt', 'w+') as f:
            f.write('================================================')
            f.write('\n')
            f.write(str(shape))
            f.write('\n')
            f.write('================================================')
            f.write('\n')
            f.write('\n')
        f.close()

    
    
    def tt_split(self):
        # adgust data from here
        self.x_tr,self.x_val, self.y_tr, self.y_val = train_test_split(
            np.array(self.data[self.cols[0]]),
            np.array(self.data[self.cols[0]]),
            test_size=0.1,
            random_state=0,
            shuffle=True
        )
        print('data spliting success \n')


    def Tokenizer_func(self, tr,val, max_words_length=0):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(tr)
        
        max_words = 0
        if max_words_length > 0:
            max_words = max_words_length
        else:
            max_words = len(tokenizer.word_counts.items())

        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(tr)

        tr_sequences = tokenizer.texts_to_sequences(tr)
        val_sequences = tokenizer.texts_to_sequences(val)

        tr = pad_sequences(tr_sequences,maxlen=self.max_len_, padding='post')
        val = pad_sequences(val_sequences,maxlen=self.max_len_, padding='post')

        voc = tokenizer.num_words +1

        return {'tr': tr, 'val': val, 'voc': voc, 'max_words':max_words, 'tokenizer': tokenizer}

    

    def Tokenize(self):
        # running and asving the tokenizer
        self.tt_split()

        # X processing 
        x_processed = self.Tokenizer_func(self.x_tr,self.x_val)
        self.x_tr, self.x_val, x_voc,x_max_words,x_tok = x_processed['tr'], x_processed['val'],x_processed['voc'],x_processed['max_words'],x_processed['tokenizer']

        # Y processing
        y_processed = self.Tokenizer_func(self.y_tr,self.y_val)
        self.y_tr, self.y_val, y_voc,y_max_words,y_tok = y_processed['tr'], y_processed['val'],y_processed['voc'],y_processed['max_words'],y_processed['tokenizer']
        

        # saving tokenizer
        self.saving_tokenizer(x_tok, 'x_tokenizer_v1') # X
        self.saving_tokenizer(y_tok, 'y_tokenizer_v1') # Y


        self.shape_checker()

        result = {
            'x_processed': {
                'x_tr': self.x_tr, 
                'x_val': self.x_val, 
                'x_voc': x_voc, 
                'x_max_words':x_max_words,
                'x_tokenizer': x_tok
            },
            'y_processed': {
                'y_tr': self.y_tr, 
                'y_val': self.y_val, 
                'y_voc': y_voc, 
                'y_max_words': y_max_words,
                'y_tokenizer': y_tok
            },
        }
        return result
        




# tok = __Tokenizer(df,['question1', 'question2'],5)
# print(tok.Tokenize()) 