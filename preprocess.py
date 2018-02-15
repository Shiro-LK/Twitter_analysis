# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 17:03:59 2017

@author: shiro

Function to preprocess the train/test data so as to feed the neural network

"""

import numpy as np
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.stem import WordNetLemmatizer

def convert_label(labels):
    '''
        convert label to int
    '''
    label_ = []
    for i in range(len(labels)):
        if labels[i].find('0') != -1:
            label_.append(0)
        elif labels[i].find('1') != -1:
            label_.append(1)
        else:
            print('Error label, check!')
    return np.asarray(label_)
        
def loadText(filename, lemmatisation=False):
    '''
        load text already preprocess
    '''
    if lemmatisation == True:
        lemmatizer = WordNetLemmatizer()
        
    with open(filename, 'r', encoding='utf8') as f:
        dataset =[line.split('\t') for line in f]
        Data = []
        Label = []
        for (label, data, _) in dataset:
            Label.append(label)
            data2 = data.split()
            if lemmatisation == True:
                #print(data2)
                for i, wrd in enumerate(data2):
                    data2[i] = lemmatizer.lemmatize(data2[i])
                #print(data2)
                
            Data.append(data2)
    del dataset
    return Data, Label
def word_to_number(lword, threshold):
    '''
        lword : dictionnary, key = word, value = count
        return a dictionnary which associate each word to a number (value 1 = UNKNOWN)
    '''
    word_number = {}
    if threshold <= 0:
        ite = 2
        for word, count in lword.items():
            word_number[word] = ite
            ite += 1
        
    else:
        ite = 2
        for word, count in lword.items():
            if count > threshold:
                word_number[word] = ite
                ite += 1
            else:
                word_number[word] = 1

    return word_number
    
def transform_word_to_number(data, dict_word_to_number):
    data_ = []
    for sentence in data:
        temp = []
        for word in sentence:
            if word not in dict_word_to_number:
                temp.append(1)
            else:
                temp.append(dict_word_to_number[word])
        data_.append(temp)
    return data_
        
def load_data(filetrain, filetest, threshold = 5, seq=-1, coeff=1.5, lemmatisation=False):
    '''
        preprocess data : tokenize and tranform word into number (1-N)
        if threshold <= 0, no threshold
        
    '''
    data_tr, label_tr = loadText(filetrain, lemmatisation=lemmatisation)
    data_test, label_test = loadText(filetest, lemmatisation=lemmatisation)
    
    # convert label
    label_tr = convert_label(label_tr)
    label_test = convert_label(label_test)
    
    # count occurence of each word in the training set
    dic_count = {}
    seq_max = 0
    
    
        
        
    for sentence in data_tr:
        seq_max = max(len(sentence), seq_max)
        for word in sentence:
            
            if word in dic_count:
                dic_count[word] +=1
            else:
                dic_count[word] =1
    if seq != -1:
        seq_max=seq
    print('sequence max ', seq_max)
    word_number = word_to_number(dic_count, threshold)
    
    # convert word to number
    data_tr_ = transform_word_to_number(data_tr, word_number)
    data_test_ =  transform_word_to_number(data_test, word_number)

        
    data_tr =  pad_sequences(data_tr_, maxlen=int(seq_max*coeff), padding='post', truncating = 'post')
    data_test = pad_sequences(data_test_, maxlen=int(seq_max*coeff), padding='post', truncating = 'post')
    del data_tr_
    del data_test_
    return data_tr, label_tr, data_test, label_test, seq_max, word_number


    