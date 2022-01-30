from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.models import Sequential,load_model
from keras.layers import Embedding,Dense,Flatten
from keras.models import load_model
import numpy as np
import os,sys
import math
from keras.layers import Bidirectional
from keras_self_attention import SeqWeightedAttention
import tensorflow as tf
import sys


def get_dynamic_test_data(file_names, dynamic_test, test_size, dynamic_tokenizer):
    c = 0
    temp_array = []
    seq_len_dynamic = 10000
    while True:
        if c >= len(file_names):
            c = 0
        for i, j in enumerate(file_names[c:c + test_size]):
            l = open(os.path.join(dynamic_test, j), 'r',encoding="utf-8").read().strip()
            padded_sequence = sequence.pad_sequences(dynamic_tokenizer.texts_to_sequences([l]), maxlen=seq_len_dynamic, padding='post',
                                                     truncating='post')
            temp_array.append(padded_sequence[0])

        temp_array = np.array(temp_array)
        # label_array = np.array(label_array)
        # temp_array = temp_array[..., np.newaxis]
        # print("\nYIELDING FROM c = ",c," c+validation_size = ",c+validation_size," and length of temp_array = ",len(temp_array))
        yield (temp_array)

        temp_array = []
        c += test_size


def get_static_test_data(file_names, static_test, test_size, static_tokenizer):
    c = 0
    temp_array = []
    seq_len_static = 7000
    while True:
        if c >= len(file_names):
            c = 0
        for i, j in enumerate(file_names[c:c + test_size]):
            l = open("static_test/" + j, 'r',encoding="utf-8").read().strip()
            padded_sequence = sequence.pad_sequences(static_tokenizer.texts_to_sequences([l]), maxlen=seq_len_static, padding='post',
                                                     truncating='post')
            temp_array.append(padded_sequence[0])

        temp_array = np.array(temp_array)
        # label_array = np.array(label_array)
        # temp_array = temp_array[..., np.newaxis]
        # print("\nYIELDING FROM c = ",c," c+validation_size = ",c+validation_size," and length of temp_array = ",len(temp_array))
        yield (temp_array)

        temp_array = []
        c += test_size


if __name__ == '__main__':
    static_word2vec = sys.argv[1]
    dynamic_word2vec = sys.argv[2]
    static_model = sys.argv[3]
    dynamic_model = sys.argv[4]
    static_test = sys.argv[5]
    dynamic_test = sys.argv[6]
    test_size = 50

    word2vec_model_static = KeyedVectors.load(static_word2vec, mmap='r')
    word2vec_model_dynamic = KeyedVectors.load(dynamic_word2vec, mmap='r')
    # 动态部分
    dynamic_tokenizer = Tokenizer(filters='#\n')
    dynamic_tokenizer.fit_on_texts(word2vec_model_dynamic.wv.vocab.keys())
    dynamic_word_index = dynamic_tokenizer.word_index

    model = load_model(dynamic_model, custom_objects=SeqWeightedAttention.get_custom_objects())  # 载入模型
    list_of_test_files = os.listdir(dynamic_test)
    test_generator = get_dynamic_test_data(list_of_test_files, dynamic_test, test_size, dynamic_tokenizer)
    prediction_dynamic = model.predict_generator(test_generator, math.ceil(len(list_of_test_files) / test_size))

    # 静态部分
    static_tokenizer = Tokenizer(filters='#\n')
    static_tokenizer.fit_on_texts(word2vec_model_static.wv.vocab.keys())
    static_word_index = static_tokenizer.word_index

    model = load_model(static_model, custom_objects=SeqWeightedAttention.get_custom_objects())  # 载入模型
    list_of_test_files = os.listdir(static_test)
    test_generator = get_static_test_data(list_of_test_files, static_test, test_size, static_tokenizer)
    prediction_static = model.predict_generator(test_generator, math.ceil(len(list_of_test_files) / test_size))

    prediction = (prediction_dynamic + prediction_static) / 2
    # print(prediction[:3])

    pred = []
    for i in prediction:
        pred.append(i[1])

    np.savetxt('lstm_pos_prob.txt', np.array(pred))

    predict_labels = np.argmax(prediction, axis=1)
    np.savetxt('lstm_predict_labels.txt', predict_labels)

