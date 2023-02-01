import pickle
def Generater(x_tok,y_tok):
    reverse_target_word_index = y_tok.index_word
    reverse_source_word_index = x_tok.index_word
    target_word_index = y_tok.word_index

    # target_word_index
    word_list_object = {
        'reverse_target_word_index': reverse_target_word_index, 
        'reverse_source_word_index':  reverse_source_word_index, 
        'target_word_index': target_word_index
    }


    word_list_object_opened = open("./report/word_list_object_v1", "wb")
    pickle.dump(word_list_object, word_list_object_opened)
    word_list_object_opened.close()