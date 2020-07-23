"""Importing libraries 
    pandas,numpy,matplotlib,gensim,nltk,keras
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import timeit
import io
import json

from gensim.models import Word2Vec

import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

import keras 
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, Dropout, Activation, Flatten,Conv1D,MaxPooling1D
from keras import regularizers
from keras.preprocessing.text import tokenizer_from_json

# from train_NLP import random_testing,texts_to_sequences
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def sorting_alpha(path):
    """
    Used for get the file names with the same order as in the file system. simple sorted() function
    which is the built function returns [1, 13, 4]. This function returns [1,4,13] 
    Args:
        path(string)
    
    Attributes:
        path(string): path to ['pos', 'neg'] in test and train.

    Returns: 
        list of sorted file names with the same order as in the filesystem.

    """
    conversion_lambda = lambda text: int(text) if text.isdigit() else text.lower()
    alphanumeric_keys = lambda keys: [ conversion_lambda(char) for char in re.split('([0-9]+)', keys) ]
    sort_list=sorted(path, key=alphanumeric_keys)
    return sort_list

def load_data(data_path):
    """
    calls sorting_alpha to get the serialized file names for each type of review files in test and train.
    Reads all the text files in each set [test,train].
    Appends the file content of each text file in a list and the corresponding labels in a list.
    
    Args:
        data_path(string)
    
    Attributes:
        path(string): path to test and train.

    Returns: 
        X: A list having all the reviews from pos and neg for train/test.
        y: A list with corresponding labels. 

    """
    X=[]
    y=[]
    for sentiment in ['pos', 'neg']:
        for file_name in sorting_alpha(os.listdir(os.path.join(data_path,sentiment))):
            if file_name.endswith('.txt'):
                with open(os.path.join(data_path,sentiment,file_name)) as f:
                    X.append(f.read())
                y.append(0 if sentiment == 'neg' else 1)
    return X,y

def random_testing(*vars):
    """
    A function with flags to check the returned values by different functions.
    
    Arguments:
        Variable number of arguments having
            - dt1
            - dt2
            - ..
            - flag

    """
    print("Random Testing..")
    args=[]
    for var in vars:
        args.append(var)
    
    if args[-1]==1:
        print("Randomly checking train features: ",args[0][0:1])
        print("Length of train features: ",len(args[0]))
        print("Randomly checking train labels: ",args[1][0:1])
        print("Length of train labels: ",len(args[1]))

    if args[-1]==2:
        print(args[0].head(2))
        print(args[0].tail(2))


def base_df(fea,labels):
    """converts features and labels lists into DataFrame
    """
    df=pd.concat([pd.DataFrame(fea,columns=['Text']),pd.DataFrame(labels,columns=['Label'])],axis=1)
    
    assert df.shape[0]==25000

    return df


def punctuation_remover(series,punctuations):
    """
    Removes the punctuations from the series using sub.

    Args:
        series(dataframe)
        punctuation(string) 

    Attributes: 
        series: A series variable with Text column as reviews
        punctuations(string): A string with specific punctuations to remove
    
    Return: 
        A list of reviews with removed punctuations.
    """
    series_list=[re.sub(punctuations," ",row) for row in series]
    return series_list

def punctuation_remover_basic(list_text):
    """
    Removes the punctuations from the list of reviews. 
    Removes the extra spaces. 
    Removes the patterns with html tags.

    Args:
        list_text(list)

    Attributes: 
        list_text(list): A list of reviews
    
    Return: 
        A clean list with removed punctuations and tags.        
    """    

    list_text=[re.sub(r'[^a-zA-Z0-9]+'," ",review) for review in list_text]
    list_text=[re.sub(r'\s+'," ",review) for review in list_text]
    list_text=[re.sub(r'(<br\s*/><br\s*/>)|(\-)|(\/)'," ",review) for review in list_text]
    return list_text



def lower_text(list_text):
    """
    Lowers the text in the list 
    
    Args:
        list_text(list)
    
    Attributes: 
        list_text(list): list of reviews.

    Return: 
        list of lowercase reviews

    """
    list_text=[review.lower() for review in list_text]
    return list_text



def stopwords_removal(list_text,stopword_list):
    """Removes the stopwords from the list
    Args: 
        list_text(list)
        stopword_list(list)
    
    Attributes:
        list_text(list): 
        stopword_list(list):

    Return:
        returns a list with no stopwords.
    """
    list_text_nsw=[]
    for review in list_text:
        list_text_nsw.append(' '.join([word for word in review.split() if word not in stopword_list]))
    return list_text_nsw



def tokenize_word_sentences(list_text):       
    """
    Tokenizes the words in the reviews.
    Args: 
        list_text(list)
    
    Attributes:
        list_text(list): list of reviews to be tokenized 

    Returns: 
        A list of list of strings as tokens. 
    """
    tokenized_list = []
    for review in list_text:
        tokenized_list.append(word_tokenize(review))  # using nltk's word tokenize to tokenize the reviews by words.
          
    return tokenized_list  


def get_vars():
    stopword_list_1=[ 'we', 'our', 'ours', 'ourselves', 'you', "you're",'i', 'me', 'my', 'myself', "you've", "you'll", "you'd", 'your',
                        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself']
    stopword_list_2=["a", "about", 'other',  'such',  'own', 'same', 'so', 'than','s',
                            't', 'can', 'will', 'don', 'now', 'd','s']
    stopword_list=list(set(stopword_list_1).union(set(stopword_list_2)))

    return stopword_list


if __name__ == "__main__":

    X_test,y_test=load_data(os.path.join("data",'aclImdb','test'))  # Loading the dataset
    random_testing(X_test,y_test,1)                                 # Randomly chceking the shape and values of X_train,y_train

    test_df=base_df(X_test,y_test)                                  # converting into dataframe
    random_testing(test_df,2)

    print("Removing punctuatons..")
    punctuations="[\s.;:!\'?,\"()\[\]/_ÉºèíÁ{()ïá`&₤%äÜô“êðßÈ³¾âø;Ø\®»!-=ñÀöã?!#$%&'()*+,-./:;<=>?@[\]^_`{|}~å°@0¨ë:¢û*$´ùóüçúî~½<’æ‘§}£.«ÕÊì¤ÃÄ·,éý|-Åō#ò¿–à><]"
    
    X_test=punctuation_remover(test_df['Text'],punctuations)        # A list of reviews with removed punctuations (Manually found punctuations)
    X_test=punctuation_remover_basic(X_test)                        # A list of reviews with removed punctuations (html tags and patterns) 


    X_test=lower_text(X_test)                                       # lowercases the test data
    print("Removing stopwords..")
    stopword_list=get_vars()                                        # Stopwords list
    X_test=stopwords_removal(X_test,stopword_list)                  #Removes the stopwords from test set
    print("Tokenizing..")
    X_test=tokenize_word_sentences(X_test)                          #A list of list of strings as tokens. 
    
    max_length=700                                                 # same as trainset
    
    with open(os.path.join("models",'tokenizer.json')) as f:        # Loading the saved tokenizer from json
        data_json = json.load(f)

    token= tokenizer_from_json(data_json)                           # using keras function to load the object

    tst=[' '.join(seq[:]) for seq in X_test]                        #['This product is very good','']

    seq_test_data= token.texts_to_sequences(tst)                    # converting the test data into sequences.
    print("Pad sequencing..")
    pad_test_data=pad_sequences(seq_test_data, maxlen=max_length, padding='post', truncating='post') #Padding them, converting into np array.

    X_test=pd.DataFrame(pad_test_data)  

    y_test=np.asarray(test_df['Label'])
    y_test=np_utils.to_categorical(y_test)
    y_test=pd.DataFrame(y_test)

    nlp_model=keras.models.load_model(os.path.join('models/',"nlp_model.h5"))

    print("Predicting..")
    y_pred=nlp_model.predict(X_test)
    print(y_pred)
    y_pred=np.argmax(y_pred, axis=-1)
    y_pred=y_pred.tolist()

    print(y_pred[0:10])
    y_pred=pd.DataFrame(y_pred)
    acc=nlp_model.evaluate(X_test,y_test)
    print(acc)
	# 1. Load your saved model

	# 2. Load your testing data

	# 3. Run prediction on the test data and print the test accuracy