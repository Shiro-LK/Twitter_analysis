# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 14:04:02 2017

@author: shiro
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Lambda, Reshape
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.layers import Conv1D, Dropout, LSTM, MaxPooling1D, BatchNormalization,GlobalMaxPool1D

def create_CBOW(input_dim, max_length, embedding, output_dim=50, n_class=3):
    print('\n\n ## CBOW MODEL ## \n')
    model = Sequential()
    if embedding == 'None':
        model.add(Embedding(input_dim+1, output_dim , input_length=max_length))
    else:
        print('loading embedding in CBOW')
        model.add(Embedding(input_dim+1, output_dim ,weights=[embedding], input_length=max_length))
    model.add(Lambda(lambda x : K.sum(x,axis=1), output_shape=(output_dim,)))
    model.add(Dense(n_class, activation='softmax', name='softmax'))
    print(model.summary())
    return model

def create_CBOWNN(input_dim, max_length, embedding, output_dim=50, n_class=3):
    model = Sequential()
    if embedding == 'None':
        model.add(Embedding(input_dim+1, output_dim , input_length=max_length))
    else:
        print('loading embedding in CBOW')
        model.add(Embedding(input_dim+1, output_dim ,weights=[embedding], input_length=max_length))
    model.add(Lambda(lambda x : K.sum(x,axis=1), output_shape=(output_dim,)))

    model.add(Dense(1024,  activation='relu', name='fc1'))#, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1024,  activation='relu', name='fc2'))#, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax', name='softmax'))
    # compile the model
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # summarize the model
    print(model.summary())
    return model

def create_simpleModel(input_dim, max_length, embedding, output_dim=50, n_class=3):
    model = Sequential()
    if embedding == None:
        model.add(Embedding(input_dim+1, output_dim , input_length=max_length))
    else:
        print('loading embedding in simpleModel')
        model.add(Embedding(input_dim+1, output_dim ,weights=[embedding], input_length=max_length))
    
    
    model.add(Conv1D(128, 5, activation='relu')) # (length, input_dim = output dim of the embedding)
    model.add(BatchNormalization())     
    model.add(MaxPooling1D(2))
    #model.add(Conv1D(128, 5, activation='relu'))
    #model.add(BatchNormalization())     
    #model.add(MaxPooling1D(2))
    #model.add(Conv1D(128, 5, activation='relu'))
    #model.add(BatchNormalization())     
    #model.add(MaxPooling1D(5))

    model.add(Flatten())
    #model.add(Dropout(0.2))
    #model.add(Dense(512,activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1024,activation='relu'))
    #↕model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    # compile the model
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # summarize the model
    print(model.summary())
    return model

def create_convModel(input_dim, max_length, embedding, output_dim=50, n_class=3):
    model = Sequential()
    if embedding == 'None':
        model.add(Embedding(input_dim+1, output_dim , input_length=max_length))
    else:
        print('loading embedding in convModel')
        model.add(Embedding(input_dim+1, output_dim ,weights=[embedding], input_length=max_length))
    
    
    model.add(Conv1D(512, 3, activation='elu')) # (length, input_dim = output dim of the embedding)
    model.add(BatchNormalization())
    model.add(GlobalMaxPool1D())
    model.add(Dense(1024,activation='relu'))#, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # summarize the model
    print(model.summary())
    return model
  
def create_lstm(input_dim, max_length, embedding, output_dim=10, n_class=3):
    model = Sequential()
    if embedding == None:
        model.add(Embedding(input_dim+1, output_dim , input_length=max_length))
    else:
        print('loading embedding in LSTM')
        model.add(Embedding(input_dim+1, output_dim ,weights=[embedding], input_length=max_length))
    #model.add(Flatten())
    #model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    #model.add(BatchNormalization())    
    #model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    #model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    #model.add(Conv1D(64, 3, border_mode='same')) # (length, input_dim = output dim of the embedding)
    #model.add(Conv1D(128, 3, border_mode='same'))
    #model.add(Conv1D(256, 3, border_mode='same'))
    #model.add(Conv1D(512, 3, border_mode='same'))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Flatten())
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))#, activation = 'relu', recurrent_activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    #model.add(Dense(n_class, activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # summarize the model
    print(model.summary())
    return model