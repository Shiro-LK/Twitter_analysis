# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 18:16:35 2017

@author: shiro
"""
import pickle
from preprocess import load_data
from model import create_CBOW, create_CBOWNN, create_simpleModel, create_lstm, create_convModel,  create_convModel2
from metrics import f1, accuracy_per_class0, accuracy_per_class1
import keras
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
import random
from sklearn.cross_validation import train_test_split
from embedding_word import load_GloveEmbedding
import os 
from sklearn.metrics import confusion_matrix
keras.metrics.accuracy_per_class0 = accuracy_per_class0
keras.metrics.accuracy_per_class1 = accuracy_per_class1

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def eval_model(model, gen_x_val, y_val, step, get_seq=False, filename=None, dic=None):
    '''
        evaluate a model on a validation generator
        gen_x_val : generator of data
        y_val : label of the data
        step : number of step for the gen_x_val
        get_seq : save in file wrong prediction
        filename : name of the file where to save wrong pred
        dic : dic of word to number
    '''
    preds = model.predict_generator(gen_x_val, step)
    preds = np.argmax(preds, axis=1)
    
    # Save data in file
    if get_seq == True:
        with open(filename, 'r', encoding='utf-8') as f :
            data = [line.split('\t') for line in f]
        wrong = (preds == y_val).astype(np.int)
        print(preds.shape, y_val.shape, wrong.shape)
        with open(filename.replace('.txt', '_wrong.txt'), 'w') as f2:
            for i, res in enumerate(wrong):
                if res == 0:
                    f2.write(data[i][0]+'\t'+data[i][1]+'\n')
                    if dic != None:
                        s = data[i][1].split()
                        for word in s:
                            if word in dic:
                                f2.write(str(dic[word])+' ')
                            else:
                                f2.write('1 ')
                        f2.write('\n')
                        
    conf = confusion_matrix(y_val, preds)
    print(conf)
    print(np.diag(conf)/np.sum(conf, axis=1))
    
    
"""
    get a generator which choose twetts randomly
"""
def generator_shuffle(features, labels, batch_size, dtype=np.int32, input_shape=68):
     # Create empty arrays to contain batch of features and labels#
    '''
        features : ndarray
        labels : ndarray
        num_classes : int
        batch size : int
    '''
    num_classes = len(set(labels))
    while True:
        batch_features = np.ndarray(shape=(batch_size, input_shape), dtype=dtype)
        batch_labels =  np.ndarray(shape=(batch_size,  num_classes), dtype=dtype)

        index= np.random.randint(features.shape[0]-1, size=batch_size)
        #print(index)
        batch_features[:] = features[index]
        batch_labels[:] = np_utils.to_categorical(labels[index], num_classes)
        yield batch_features, batch_labels

"""
    Create simple generator
"""
def generator(features, labels, batch_size, dtype=np.int32, input_shape=68):
     # Create empty arrays to contain batch of features and labels#
    num_classes = len(set(labels))
    while True:
          for cpt in range(0, int(len(features)/batch_size)):
            #print('gen')
            batch_features = np.ndarray(shape=(batch_size, input_shape), dtype=dtype)
            batch_labels =  np.ndarray(shape=(batch_size, num_classes), dtype=dtype)
            for i in range(0, batch_size):
                index = cpt*batch_size + i
                batch_features[i] = features[index]
                batch_labels[i] = np_utils.to_categorical(labels[index], num_classes)
                #batch_labels[i] = labels[index].reshape(-1,1)
            yield batch_features, batch_labels


def generator_test(features, batch_size, dtype=np.int, input_shape=68):
    '''    
        Create generator which return only the feature without label by batch
    '''
    while True:
          for cpt in range(0, int(len(features)/batch_size)):
            #print('gen')
            batch_features = np.ndarray(shape=(batch_size, input_shape), dtype=dtype)
            for i in range(0, batch_size):
                index = cpt*batch_size + i
                batch_features[i] = features[index]
            yield batch_features
 
##
def copy_weight(newmodel, oldmodel):
    dic_w = {}
    for layer in oldmodel.layers:
        print(layer.name)
        dic_w[layer.name] = layer.get_weights()
    
    for layer in newmodel.layers:
        if layer.name in dic_w and layer.name.find('embedding') == -1:
            layer.set_weights(dic_w[layer.name])
            print(layer.name)
    return newmodel            

def train_model(x_tr, y_tr, x_val, y_val, max_length, size_voc, dic_word, output_dim=100, batch_size = 50, embedding_glove = False, lemmatisation=False, output_name = 'temp'):
    '''
        train model from train and validatoin data
        max length = size max of a sequence (tweet)
        size_voc : size of the vocabulary in the train data
        dic_word : dictionary which contain the conversion word to number
        output_name: name we give to the model when we save it
        output_dim: output dimension of the embedding layers
        batch_size : 
        embedding glove : choose to load or not GloVe Embedding weight
        func : function which return the function of the model used for the training (see model.py for the model)
    '''    
    if embedding_glove == False:
        emb = None
    else: 
        print('loading glove ....')
        emb = load_GloveEmbedding(output_dim, dic_word, lemmatisation=lemmatisation)
    # create model

    model = create_lstm(input_dim=size_voc, max_length=max_length, output_dim=output_dim, n_class= 2, embedding = emb)
    
    old = load_model('lstm_glove_bigDataset.hdf5')
    # load weight for transfer learning
    model = copy_weight(model, old)
    del old
    
    
    sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-5, nesterov=True)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    adagrad = keras.optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    
    # create generator 
    generator_train = generator_shuffle(x_tr, y_tr, batch_size=batch_size, input_shape=(max_length))
    generator_valid = generator(x_val, y_val, batch_size=batch_size, input_shape=(max_length))
    
    step_train = int(len(x_tr)/batch_size)-1
    step_val = int(len(x_val)/batch_size)-1
    print('shape:', len(x_tr))
    print('shape:', len(x_val))
    print('step train :' , step_train)
    print('step test :' , step_val)
    
     # callback
    callback_tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/'+output_name, histogram_freq=0, 
                                                       batch_size=16, write_graph=True, write_grads=False, 
                                                       write_images=False, embeddings_freq=0, embeddings_layer_names=None, 
                                                       embeddings_metadata=None)
    lr_decay = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.0)    
    checkpoints = ModelCheckpoint(output_name+'.hdf5', verbose=1, save_best_only=True, period=1) # -{epoch:02d}
    callbacks_list = [callback_tensorboard, checkpoints,lr_decay]
    
    # train 
    model.fit_generator(generator_train,
          steps_per_epoch=step_train,
          epochs=15,
          verbose=1,
          validation_data=generator_valid,
          validation_steps=step_val,
          callbacks=callbacks_list)
    
    
    ## test
    
    return model

def main_SemEval():
    
    #-- Load data --#
    '''
        word_number : dictionnary which associates word to number depending of the training set
        length : maximum length of a sequence
    '''
    coeff=1.0
    lemmatisation=False
    output_name = 'lstmTF_semeval'
    X_train, Y_train, X_test, Y_test, length, word_number = load_data('data/twb_cleanv5.txt', 'data/twb-dev_cleanv5.txt', threshold = 5, seq = 40, coeff= coeff, lemmatisation=lemmatisation)
    #print(X_test)lstm
    word_pkl = 'word_number_lstmTF_semeval.pkl' #'word_number_cbowdataset2_noglove.pkl'
    # list vocabulaire
    size_vocab = max(list(word_number.values()))
    print('size vocabulary :', size_vocab)
    print('negative :' ,list(Y_train).count(0))
    #print('neutral :' ,list(Y_train).count(1))
    print('postive :' ,list(Y_train).count(1))
    
    print('negative :' ,list(Y_test).count(0))
    #print('neutral :' ,list(Y_test).count(1))
    print('postive :' ,list(Y_test).count(1))
    
    ## enrengistre le dictionnaire qui contient la conversion entre mot -> nombre. 
    ##  Attention : le dictionnaire change à chaque nouveau entrainement, si on veut evaluer le modele, penser à charger le dictionnaire et donc utiliser la fonction load3
    pickle.dump(word_number, open(word_pkl, 'wb'))
    
    model = train_model(X_train, Y_train, X_test, Y_test, int(coeff*length), size_vocab, output_dim=300, batch_size = 32, dic_word = word_number, embedding_glove = True, lemmatisation=lemmatisation, output_name = output_name)
    #print(model.evaluate(data_test, label_rt))
    
def main_big():
    
    #-- Load data --#
    '''
        word_number : dictionnary which associates word to number depending of the training set
        length : maximum length of a sequence
    '''
    coeff=1.1
    lemmatisation=True
    output_name = 'conv2_bigDataset_lem'
    #X_train, Y_train, X_test, Y_test, length, word_number = load_data('data/twb_clean.txt', 'data/twb-dev_clean.txt', threshold = 10, seq = 60)
    X_train, Y_train, X_test, Y_test, length, word_number = load_data('Sentiment_Analysis_Dataset_correct_train_preproces.txt', 'Sentiment_Analysis_Dataset_correct_test_preproces.txt', threshold = 5, seq = -1, coeff= coeff, lemmatisation=lemmatisation)
    #print(X_test)
    
    # list vocabulaire
    size_vocab = max(list(word_number.values()))
    print(' ### Parameters ###')
    print('lemmatisation : ', lemmatisation)
    print('name saved : ', output_name)
    print('size vocabulary :', size_vocab)
    print('negative :' ,list(Y_train).count(0))
    #print('neutral :' ,list(Y_train).count(1))
    print('postive :' ,list(Y_train).count(1))
    
    print('negative :' ,list(Y_test).count(0))
    #print('neutral :' ,list(Y_test).count(1))
    print('postive :' ,list(Y_test).count(1))
    # split train into train and valid data
    #X_train, X_val, y_train, y_val = train_test_split(data_train, etq_train, test_size=0.2)
    
    ## enrengistre le dictionnaire qui contient la conversion entre mot -> nombre. 
    ##  Attention : le dictionnaire change à chaque nouveau entrainement, si on veut evaluer le modele, penser à charger le dictionnaire et donc utiliser la fonction load3
    pickle.dump(word_number, open('word_number_conv2_glove_lem.pkl', 'wb'))
    model = train_model(X_train, Y_train, X_test, Y_test, int(coeff*length), size_vocab, output_dim=300, batch_size = 128, dic_word = word_number, embedding_glove = True, lemmatisation=lemmatisation, output_name = output_name)
    #print(model.evaluate(data_test, label_rt))

def main_eval():
    coeff= 1.0
    output = 'lstmTF_semeval' #'conv2_bigDataset_lem' #'lstm_bigDataset'
    word_pkl = 'word_number_lstmTF_semeval.pkl' #'word_number_conv2_glove_lem.pkl' #'word_number_lstm_glove.pkl'
    word_number = pickle.load(open(word_pkl, 'rb'))
    lemmatisation=False
    X_train, Y_train, x_val, y_val, length, word_number2 = load_data('data/twb_cleanv5.txt', 'data/twb-dev_cleanv5.txt', threshold = 5, seq = 40,coeff=coeff, lemmatisation=lemmatisation, word_number=word_number)
    #X_train, Y_train, x_val, y_val, length, word_number2 = load_data('Sentiment_Analysis_Dataset_correct_test_preproces.txt', 'Sentiment_Analysis_Dataset_correct_test_preproces.txt', threshold = 5, seq = 117, coeff=coeff, lemmatisation=lemmatisation,word_number=word_number)
    
    size_vocab = max(list(word_number.values()))    
    model = load_model(output+'.hdf5')
    #print(model.evaluate_generator(generator_valid, step_val))
    #print(model.evaluate_generator(generator_valid, step_val))
    generator_valid = generator(x_val, y_val, batch_size=1, input_shape=int(length*coeff))
    print(model.evaluate_generator(generator_valid, int(len(x_val)/1)))
    generator_valid2 = generator_test(x_val, batch_size=1, input_shape=int(length*coeff))
    eval_model(model, generator_valid2, y_val, len(x_val)/1)
    
if __name__ == "__main__":
    main_SemEval()
    #main_big()
    main_eval()
