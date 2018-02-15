# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 13:39:39 2017

@author: shiro
"""
import re
import numpy as np
import csv

'''
    functions where we preprocess the data so as to save them in a file corrected
'''
def clean_str(string, tolower=True):
    """
    Tokenization/string cleaning.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # Different regex parts for smiley faces
    happy =	[':-\)', ':\)', '\(:', '\(-:' ] 
    laugh =	[':-d', ':d', 'x-d', 'xd', 'xd' ]
    love=	['<3', ':\*' ]
    wink = [';-\)', ';\)', ';-d', ';d', '\(;', '\(-;', ] 
    sad  = [':-\(', ':\(' , '\):', '\)-:', ':,\(', ':\'\(', ':/',  ':"\(', ':\(\(', 't.t', ':-\|'] 
    surprise = [':o']

    special = [",", ":", "\"", "=", "&", ";", "%", "$","@", "%", "^", "*",
               "(", ")", "{", "}", "[", "]", "|", "/", "\\", ">", "<", "-",
               "!", "?", ".", "'", "--", "---", "#"]
    if tolower:
        string = string.lower()
    # contraction
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\n", "", string)
    string = re.sub(' +',' ', string)
    # tweet
    string = re.sub(r'\@[^\s]+[\s]?', "<USER> ", string) # replace user @ by <user>
    string = re.sub(r'\#[^\s]+[\s]?', "<HASHTAG>", string) # remove hashtag
    string = re.sub(r"\$[\d]+(\.)?[\d]*", "<PRICE>", string) # float and int prince
    string = re.sub(r"http.?://[^\s]+[\s]?", "<WEBSITE>", string) # detect url
    string = re.sub(r"[\d]+:[\d]+", "<TIME> ", string) # detect TIME
    string = re.sub(r"[\d]+:[\d]+:[\d]+", "<TIME> ", string) # detect TIME
    string = re.sub(r"[\d]+h[\d]+", "<TIME> ", string) # detect TIME
    string = re.sub(r"[\d]+", "<NUMBER> ", string) # detect number
     # detect emoticonne
    for emo in happy:
        string = re.sub(emo, "<HAPPY> ", string) # happy
    for emo in laugh:
        string = re.sub(emo, "<LAUGH> ", string) # laugh
    for emo in love:
        string = re.sub(emo, "<LOVE> ", string) # love
    for emo in wink:
        string = re.sub(emo, "<LAUGH> ", string) # wink
    for emo in sad:
        #print(emo)
        string = re.sub(emo, "<SAD> ", string) # sad
    for emo in surprise:
        string = re.sub(emo, "<SURPRISE> ", string) # sad
#        
#    # remove special characters
    string = re.sub(r"[^A-Za-z0-9,!?.\'\`]<>", " ", string) #()
    #string = re.sub(r"\.{1,1}", ".", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", " ", string)
    string.replace('...', ' ... ')
    string = re.sub(' +',' ', string)
    
#    
#    string = re.sub(r"\(", " \( ", string)
#    string = re.sub(r"\)", " \) ", string)
    #for s in special:
    #    string.replace(s, ' ')
    
    return string.strip()
    
def clean_str2(string, tolower=True):
    """
    Tokenization/string cleaning.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # Different regex parts for smiley faces
    happy =	[':-\)', ':\)', '\(:', '\(-:' ] 
    laugh =	[':-d', ':d', 'x-d', 'xd', 'xd' ]
    love=	['<3', ':\*' ]
    wink = [';-\)', ';\)', ';-d', ';d', '\(;', '\(-;', ] 
    sad  = [':-\(', ':\(' , '\):', '\)-:', ':,\(', ':\'\(', ':/',  ':"\(', ':\(\(', 't.t', ':-\|'] 
    surprise = [':o']

    special = [",", ":", "\"", "=", "&", ";", "%", "$","@", "%", "^", "*",
               "(", ")", "{", "}", "[", "]", "|", "/", "\\", ">", "<", "-",
               "!", "?", ".", "'", "--", "---", "#"]
    
    # contraction
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\n", "", string)
    string = re.sub('\.{2,100}',' ...', string) # take in account triple points
    string = re.sub(' +',' ', string)
    
    
    return string.strip()
    
def loadTextsAndDelete(filename, limit=-1):
    '''
        load dataset and remove feature useless and remove useless 
        characters like triple space etc
    '''
    f = open(filename, encoding="utf8")
    dataset=[line.split('\t') for line in f]
    dataset.pop(0)
    for i, data in enumerate(dataset):
        
        if i%100000==0:
            print(i)
        try:
            data.remove('\n')
        except:
            pass
        data.pop(0)
        data.pop(1)
        data[1] = clean_str2(data[1])
    return dataset
    
def loadTexts(filename, limit=-1):
    '''
        load dataset and preprocess it replace # by hashtag etc without save it
    '''
    f = open(filename, encoding="utf8")
    dataset=[line.split('\t') for line in f]
    for i, data in enumerate(dataset[0:10]):
        
        if i%100000==0:
            print(i)
        try:
            data.remove('\n')
        except:
            pass
        data[1] = clean_str(data[1])
        print(data)
    return dataset
    
def loadTextsAndPreprocess(filename, limit=-1):
    '''
        load dataset and preprocess it replace # by hashtag etc and save it
    '''
    f = open(filename, encoding="utf8")
    dataset=[line.split('\t') for line in f]
    for i, data in enumerate(dataset):
        
        if i%100000==0:
            print(i)
        try:
            data.remove('\n')
        except:
            pass
        data[1] = clean_str(data[1])
        #print(data)
    
    with open(filename.replace('.txt','_preproces.txt'), 'w', encoding='utf8') as f:
        for d in dataset:
            for wr in d:
                f.write(wr+'\t')
            f.write('\n')
    return dataset
    
def equilibrate_data(filename):
    y = []
    x = []
    f = open(filename, 'r', encoding="utf8")
    dataset=[line.split('\t') for line in f]
    for data in dataset:
        y.append(data[0])
        x.append(data[1])
    cpt = []
    cpt.append(y.count('0'))
    cpt.append(y.count('1'))
    print(cpt)
    minVal = min(cpt)
    
    f = open(filename.replace('.txt', '_eq.txt'),  'w', encoding="utf8")
    f2 = open(filename.replace('.txt', '_bonus.txt'), 'w', encoding="utf8")
    dic = {}
    dic['0']=0
    dic['1']=0
    for i, label in enumerate(y):
        if dic[label] >= minVal:
            f2.write(label+'\t' + x[i] + '\t' + '\n')
        else:
            f.write(label+'\t' + x[i] + '\t' + '\n')
            dic[label] += 1
    f.close()
    f2.close()
    
def split_data(filename, ratio=0.8):
    y = []
    x = []
    f = open(filename, 'r', encoding="utf8")
    dataset=[line.split('\t') for line in f]
    for data in dataset:
        y.append(data[0])
        x.append(data[1])
    
    f = open(filename.replace('.txt', '_train.txt'),  'w', encoding="utf8")
    f2 = open(filename.replace('.txt', '_test.txt'), 'w', encoding="utf8")

    for i, label in enumerate(y):
        nb = np.random.uniform(0,1)
        if nb <= ratio:
            f.write(label+'\t' + x[i] + '\t' + '\n')
        else:
            f2.write(label+'\t' + x[i] + '\t' + '\n')
            
    f.close()
    f2.close()
    
def clean_save(dataset, name):
    with open(name, 'w', encoding='utf8') as f:
        for data in dataset:
            f.write(data[0]+'\t'+data[1]+'\t\n')

def load_csv(namefile):
    data = []
    with open(namefile, 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data
def main():
#    data = loadTextsAndDelete('data/Sentiment_Analysis_Dataset.txt')
#    print(data)
    data = loadTextsAndPreprocess('Sentiment_Analysis_Dataset_correct_train.txt')
#    equilibrate_data('Sentiment_Analysis_Dataset_correct.txt')
#    split_data('Sentiment_Analysis_Dataset_correct.txt')
#    with open('Sentiment_Analysis_Dataset_correct.txt', 'w', encoding='utf8') as f:
#        for d in data:
#            for wr in d:
#                f.write(wr+'\t')
#            f.write('\n')
#    data = loadTexts('twa.txt')
#    clean_save(data, 'twa_clean.txt')
#    
#    data2 = loadTexts('twa-dev.txt')
#    clean_save(data2, 'twa-dev_clean.txt')
    #data = loadTextsAndPreprocess('data/Sentiment_Analysis_Dataset.txt')
    #clean_save(data, 'Sentiment_Analysis_Datasetv2.txt')
main()
#test = 'Bonjour !!!!! ! @moi je #connard suis $3 et $3.56 ou $345.6 Ou $354, http://zablo.net/blog/post/twitter-sentiment-analysis-python-scikit-word2vec-nltk-xgboost .'
#print(clean_str(test))