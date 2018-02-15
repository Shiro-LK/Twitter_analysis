# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:22:27 2017

@author: shiro
"""
import numpy as np
### Prepare embedding vectors ###
import nltk
from nltk.stem import WordNetLemmatizer

def load_GloveEmbedding(embedding_dim, dic_word, lemmatisation=False):
    files = ['glove.6B.50d.txt', 'glove.6B.100d.txt', 'glove.6B.200d.txt', 'glove.6B.300d.txt']
    if embedding_dim == 50:
        file = files[0]
    elif embedding_dim == 100:
        file = files[1]
    elif embedding_dim == 200:
        file = files[2]
    elif embedding_dim ==300:
        file = files[3]
    else:
        return None
        
    if lemmatisation==True:
        lemmatizer = WordNetLemmatizer()
    embeddings_index = {}
    # load all coeff depending of the words
    f = open('glove.6B/'+ file ,'r', encoding='utf8')
    for line in f:
        values = line.split()
        if lemmatisation==True:
            word = lemmatizer.lemmatize(values[0])
        else:
            word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    print('Found %s word vectors.' % len(embeddings_index))
    
    
    number_word = max(list(dic_word.values()))
    embedding_matrix = np.zeros((number_word + 1, embedding_dim))
    ite = 0
    for word, i in dic_word.items():
        if i != 1:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
                ite += 1
    print(' %s Nombre de mot trouvé sur %s' % (ite, number_word))
    return embedding_matrix
