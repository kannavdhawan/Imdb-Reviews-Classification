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
    print("Loading dataset")
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
    print("Dataframe conversions..")
    df=pd.concat([pd.DataFrame(fea,columns=['Text']),pd.DataFrame(labels,columns=['Label'])],axis=1)
    assert df.shape[0]==25000

    return df

def get_punctuations(train_df):
    """
    To get the punctuations from the training dataframe.
    """
    print("Getting all punctuations in the dataset..")
    # uncomment for rerun 

    # set_punc=set()
    
    # for reviews in train_df['Text']:
        
    #     for chr in reviews:
        
    #         set_punc=set_punc.union(set(re.findall(r"[^a-zA-Z0-9\s]",chr)))

    # print("Punctuations in our dataset: ",set_punc)

    # relevant punctuations updated from above list
    punctuations="[\s.;:!\'?,\"()\[\]/_ÉºèíÁ{()ïá`&₤%äÜô“êðßÈ³¾âø;Ø\®»!-=ñÀöã?!#$%&'()*+,-./:;<=>?@[\]^_`{|}~å°@0¨ë:¢û*$´ùóüçúî~½<’æ‘§}£.«ÕÊì¤ÃÄ·,éý|-Åō#ò¿–à><]"
    return punctuations

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
    print("Removing all punctuations found..")
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
    print("Removing special patterns.. ")

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
    print("Lowercasing the text..")
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
    print("Removing stopwords..")

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
    print("Tokenizing reviews..")
    tokenized_list = []
    for review in list_text:
        tokenized_list.append(word_tokenize(review))  # using nltk's word tokenize to tokenize the reviews by words.
          
    return tokenized_list      


def get_vars():
    print("Getting stopwords..")
    stopword_list_1=[ 'we', 'our', 'ours', 'ourselves', 'you', "you're",'i', 'me', 'my', 'myself', "you've", "you'll", "you'd", 'your',
                        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself']
    stopword_list_2=["a", "about", 'other',  'such',  'own', 'same', 'so', 'than','s',
                            't', 'can', 'will', 'don', 'now', 'd','s']
    stopword_list=list(set(stopword_list_1).union(set(stopword_list_2)))

    return stopword_list

def w2v_embedding(formatted_dataset):
    """
    Generates the word embeddings using word2vec.
    
    Args:
        formatted_dataset(list)
    
    Attributes: 
        formatted_dataset(list): A list of list of tokens
    
    Return: 
        returns the w2v objects with embeddings. 
    """

    print("Generating embeddings using w2vec...")
    start=timeit.default_timer()
    w2v=Word2Vec(sentences=formatted_dataset,min_count=5, size=350,window=3,workers=4,iter=8) #sample=e-5, alpha=0.01,min_alpha=0.0001
    stop=timeit.default_timer()
    print("Time taken: ",stop-start)
    w2v.save(os.path.join("models/","word2vec.model"))

    return w2v

def most_sim(model,word,n):
    """
    To get the most similar words for a given word 

    Args:
        model(ob)
        word(str)
        n(int)
    
    Attributes:
        model(ob): w2v model
        word(str): A string to check the most similar words to 
        n(int): Number of similar words 

    """
    print("\n\n---------------",word,": Most similar words---------------")
    try:
        alltups=[]
        
        for i in range(n):
            tup=model.most_similar(positive=[word], topn=n)[i]
            alltups.append(tup)
        for k, v in dict(alltups).items():
            print (k, '-->', v)
    except:
        print("word : ",word," not in vocab")


def fit_on_text(data):
    """
    1. Updates internal vocabulary based on a list of texts according to given params.
    2. Setting max length for pad_sequences and embedding layer.
            Embedding layer is capable of processing sequence of heterogenous length, if you don't pass an explicit input_length 
                argument to the layer).
            If max is big, reveiws will have too many padded values left for short reviews and decreases the accuracy in turn. 4
            so setting a maximum length by (max-avg).
    3. This method creates the vocabulary index based on word frequency.

        So if you give it something like, "the boy drove on the road." It will create a dictionary 
                            s.t. 
                                 word_index["the"] = 1;
                                 word_index["boy"] = 2;
                                 0 is reserved for padding. 
                                 
        so this way, each word gets a unique integer value.
        So lower integer means more frequent word.


    Args:
        data(list)

    Arguments:
        data(list): list of list of tokens 
    
    Returns: 
        Max_length for padding.
        token object.=> A dictionary word_index with values as indexes . lowest one is most frequent.
    """
    
    
    print("Inside fit on text..")

    length_list=[len(seq) for seq in data]
    avg=sum(length_list)/len(length_list)
    max_length= int(max(length_list)-avg)                                 #max-average number of words in each sentence.   
    # max_length=500 
    print("Max length for pad sequences: ",max_length)
    

    token=Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')       #Defining the Tokenizer object 
    
    list_of_strings_full_data=[' '.join(seq[:]) for seq in data]          # Making list of strings for token ob
    
    token.fit_on_texts(list_of_strings_full_data)   
    
    tokenizer_json = token.to_json()
    with io.open(os.path.join("models",'tokenizer.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    return max_length,token

def texts_to_sequences(token,max_length,X_train):

    tr=[' '.join(seq[:max_length]) for seq in X_train]  #['This product is very good','']
   
    print(tr[0:1])

    X_train = token.texts_to_sequences(tr)

    print(tr[0:1])
   

    X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')

    print("shape train:",X_train.shape)
    print(X_train[0])

    return X_train




if __name__ == "__main__": 

    
    X_train,y_train=load_data(os.path.join("data",'aclImdb','train')) #Loading the dataset
    random_testing(X_train,y_train,1)                                 #Randomly chceking the shape and values of X_train,y_train

    train_df=base_df(X_train,y_train)                                 #Making dataframe with labels 
    random_testing(train_df,2)                                        #Randomlychecking the df 

    punctuations=get_punctuations(train_df)                           #Checking punctuations in dataset

    
    X_train=punctuation_remover(train_df['Text'],punctuations)        #A list of reviews with removed punctuations (Manually found punctuations)
    X_train=punctuation_remover_basic(X_train)                        #A list of reviews with removed punctuations (html tags and patterns) 

    X_train=lower_text(X_train)                                       #lowercasing the train dataset

    
    
    
    stopword_list=get_vars()                                          #Stopwords list
    X_train=stopwords_removal(X_train,stopword_list)                  #Removes the stopwords from train set
    
    X_train=tokenize_word_sentences(X_train)                          #Returns list of list of strings as tokens. 
    
    w2v=w2v_embedding(X_train)                                        #Returns the w2v embeddings 

    most_sim(w2v,"good",10)                                           #Prints 10 words similar to "good"

    max_length,token=fit_on_text(X_train)

    X_train=texts_to_sequences(token,max_length,X_train)

	# 2. Train your network
	# 		Make sure to print your training loss and accuracy within training to show progress
	# 		Make sure you print the final training accuracy

	# 3. Save your model