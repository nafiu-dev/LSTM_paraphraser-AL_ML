import pandas as pd
from datetime import datetime

from Datacleaner import datacleaner
from DataProcessing import DataPorcesser
from Tokenizer import __Tokenizer
from GenerateWordIndex import Generater
from Model import LSTM_Model


def print_func(msg):
    print(f'[+] {msg} | {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n')


print_func('**STARTING**')

question_pairs = pd.read_csv('./dataset/quora_duplicate_questions.tsv', sep='\t')
question_pairs = question_pairs.drop(['qid1', 'qid2'], axis = 1)
question_pairs_correct_paraphrased = question_pairs[question_pairs['is_duplicate'] == 1]
question_pairs_correct_paraphrased = question_pairs_correct_paraphrased.drop(['id', 'is_duplicate'], axis=1)


# REMOVE THIS
# question_pairs_correct_paraphrased = question_pairs_correct_paraphrased.sample(n=10000)

print_func('CALLING CLEANING FUNCTION')
data = datacleaner(question_pairs_correct_paraphrased,['question1', 'question2'])
print_func('DONE CLEANING BY FUNCITON')

# DATA PROCESSING 
print_func('CALLING THE DATA PROCESSER')
D_processer = DataPorcesser(data, ['question1', 'question2'])
Result_D_processer = D_processer.proccessed()
data, max_len_ = Result_D_processer['data'], Result_D_processer['max_len_']
print_func('DONE DATA_PROCESSING')

# ADDING START AND END TO DATA (TARGET)
print_func('Adding START_ and _END to target')
data['question2'] = data['question2'].apply(lambda x : 'START_ '+ x + ' _END')

# TOKENIZING
print_func('CALLING THE TOKENIZER')
tok = __Tokenizer(data,['question1', 'question2'], max_len_)
result_tok = tok.Tokenize()
print_func('TOKNIZING COMPLETED')


# EXTRACTING INFORMATIONS FROM TOKNIZER FUNCTION
print_func('EXTRACTING INFORMATIONS FROM TOKNIZER FUNCTION')
X_returns = result_tok['x_processed']
Y_returns = result_tok['y_processed']






print_func('ORGINIZING VARIABLES X')
x_max_words = X_returns['x_max_words']
x_data = {
    'x_tr': X_returns['x_tr'],
    'y_tr': Y_returns['y_tr']
    
}

print_func('ORGINIZING VARIABLES Y')
y_max_words = Y_returns['y_max_words']
y_data = {
    'y_val': Y_returns['y_val'],
    'x_val': X_returns['x_val']
}

print_func('DONE ORGINIZING.')


# CALLING THE MODEL TO TRAIN DATA
lstm_model = LSTM_Model(max_len_, x_max_words,y_max_words,x_data,y_data)

lstm_model.run()


# CREATING WORD_INDEX
print_func('GENERATING WORD INDEX')
Generater(X_returns['x_tokenizer'], Y_returns['y_tokenizer'])
print_func('WORD INDEX GENERATED')


print_func('DONE...................................')
