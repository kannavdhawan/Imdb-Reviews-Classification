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
    stopword_list_1=[ 'we', 'our', 'ours', 'ourselves', 'you', "you're",'i', 'me', 'he',
                     'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself']
    stopword_list_2=["a",'s','t', 'can', 'will', 'don', 'now', 'd','s']
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
    w2v=Word2Vec(sentences=formatted_dataset,min_count=8, size=350,window=3,workers=4,iter=5) #sample=e-5, alpha=0.01,min_alpha=0.0001
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

    # length_list=[len(seq) for seq in data]
    # avg=sum(length_list)/len(length_list)
    # max_length= int(max(length_list)-avg)/5                                 #max-average number of words in each sentence.   
    max_length=450      #defining after finding optimal value
    # print("Max length for pad sequences: ",max_length)
    

    token=Tokenizer()       #Defining the Tokenizer object 
    
    list_of_strings_full_data=[' '.join(seq[:]) for seq in data]          # Making list of strings for token ob
    
    token.fit_on_texts(list_of_strings_full_data)   
    
    tokenizer_json = token.to_json()
    with io.open(os.path.join("models",'tokenizer.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    return max_length,token

def texts_to_sequences(token,max_length,list_of_list_tokens):
    """
    Transforms each text in list to a sequence of integers.
    Takes each word in the list and replaces it with its corresponding integer value from the word_index dictionary created.
    Only words known by the tokenizer will be taken into account.
    
    text_to_sequence: List of list where each element is a number/int taken from word_index dict.
    
    pad_sequence:   A numpy array with padded values as per max_length defined.
    
    Attributes:
        token(ob): saved token of w2vec embeddings.
        max_length(int): for padding
        list_of_list_of_tokens(list):list of tokenized reviews
    
    Returns:
        padded_np_array: Numpy array for reviews with values for each token taken from word_index dict.

    """

    tr=[' '.join(seq[:max_length]) for seq in list_of_list_tokens]  #['This product is very good','']
   
    # print(tr[0:1])

    list_of_seq_strings = token.texts_to_sequences(tr)

    # print(tr[0:1])
   

    padded_np_array = pad_sequences(list_of_seq_strings, maxlen=max_length, padding='post', truncating='post')

    # print("shape: ",padded_np_array.shape)
    # print(padded_np_array[0])

    return padded_np_array

def embedding_matrix(token,w2v_embeddings):
    """
    Loading the embeddings created by w2v for feeding the corresponding vectors in embedding matrix.

    Arguments: 
        token(ob): keras Tokenizer object having word_index dict. 
        w2v_embeddings(ob): w2v embedding object 
    Returns: 
        e_dim: vector size taken during embeddings
        v_size: total number of words in the dictionary.
        embed_matrix: Embedding matrix with array from word2vec embeddings. shape: (v_size,e_dim) 
    """
    e_dim=w2v_embeddings.vector_size                # vector size taken during embeddings. i.e. 350

    # print("vector size embedding",e_dim)            

    v_size=len(token.word_index)+1                  # total number of words in the dictionary.

    # print("vocabulary_size: ",v_size)
    
    # making the embedding matrtix and feeding with array from word2vec embeddings.

    embed_matrix=np.random.randn(v_size,e_dim) 
    for word,index in token.word_index.items():
        if word in w2v_embeddings.wv.vocab:
            embed_matrix[index]=w2v_embeddings[word]        #feeding the embedding matrix with array from word2vec embeddings.
        else:
            embed_matrix[index]=np.random.randn(1,e_dim)    #If word from word index is not there in word2vec embeddings, input randomly.
    

    return e_dim,v_size,embed_matrix


def model_nlp(v_size, e_dim, embed_matrix, max_length, X_train, y_train):
    """
    Attributes:
        v_size: vocab size of word_index dictionary.
        e_dim:  embedding dimensions is the vector size taken at word2vec. 
        embed_matrix: Matrix with shape(v_size,emb_dim) which is fed at the embedding layer. 
        max_length: input length to the embediing layer. It is same for all the reviews because of padding. 
        X_train: padded sequence dataframe
        y_train: categorical labels dataframe

    Trains the model and saves the model. 
    """

    model = Sequential()

    # Embedding layer with Trainable=True so that the backpropagation can be done and the weights can be updated in accordance with the loss.
    model.add(Embedding(input_dim = v_size, output_dim = e_dim, weights = [embed_matrix], input_length = max_length,trainable=True))

    #Con1D layer is taken with 32 filters for feature extraction. 
    model.add(Conv1D(32, 3, padding='same', activation='relu'))

    model.add(Flatten())        #Flattened for Dense layer

    model.add(Dense(512, activation = 'relu'))  #A dense layer with 512 units and relu as activation. 

    model.add(Dense(2,activation ='softmax'))   #A f inal fully connected layer. 

    model.compile(loss ='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) 

    model.fit(X_train, y_train, batch_size = 128, epochs = 5)  # Training the model 

    acc_score = model.evaluate(X_train,y_train)         #evaluating the model 

    print("Training Accuracy is {}% ".format(acc_score[1] * 100))

    model.save(os.path.join("models/","20831774_NLP.h5"))



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

    max_length,token=fit_on_text(X_train)                             # creating the dictionary word index 

    X_train=texts_to_sequences(token,max_length,X_train)              #converting the text to sequences adn padding the same with max_length  

    e_dim,v_size,embed_matrix=embedding_matrix(token,w2v)             # Making the Embedding matrix for embedding layer.

    # print(X_train.shape)                                              #np array

    X_train=pd.DataFrame(X_train)                                       

                                            # Converting into categorical data because softmax and categorical_crossentropy loss is taken .
    y_train=np.asarray(train_df['Label'])
    y_train=np_utils.to_categorical(y_train)
    y_train=pd.DataFrame(y_train)
    # print(y_train)
    # print(X_train.shape)
    # print(y_train.shape)
    
    # calling the model 
    model_nlp(v_size, e_dim, embed_matrix, max_length, X_train, y_train)