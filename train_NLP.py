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

def random_testing(dt1,dt2,flag):
    """
    A function with flags to check the returned values by different functions.
    """
    if flag==1:
        print("Randomly checking train features: ",dt1[0:1])
        print("Length of train features: ",len(dt1))
        print("Randomly checking train labels: ",dt2[0:1])
        print("Length of train labels: ",len(dt2))
    








if __name__ == "__main__": 

    X_train,y_train=load_data(os.path.join("data",'aclImdb','train')) #Loading the dataset
    random_testing(X_train,y_train,flag=1)                         # Randomly chceking the shape and values of X_train,y_train

	# 2. Train your network
	# 		Make sure to print your training loss and accuracy within training to show progress
	# 		Make sure you print the final training accuracy

	# 3. Save your model