import numpy as np
from preprocessing import preprocessData
from keras.preprocessing.text import Tokenizer
import nltk
from nltk import word_tokenize
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gensim
import gensim.downloader as api

# Read in the data
x_train, x_test, y_train, y_test = preprocessData()
x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1, random_state=329)

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(x_train)

def sparseVector(): 
    #Sparse Vector - Count vectorizer
    
    x_sparse_train = tokenizer.texts_to_matrix(x_train, mode='count')
    y_sparse_train = np.asarray(y_train).astype('float32')
    x_sparse_dev = tokenizer.texts_to_matrix(x_dev, mode='count')
    y_sparse_dev = np.asarray(y_dev).astype('float32')
    x_sparse_test = tokenizer.texts_to_matrix(x_test, mode='count')
    y_sparse_test = np.asarray(y_test).astype('float32')
    
    model=LogisticRegression(multi_class="auto",random_state=329).fit(x_sparse_train,y_sparse_train)
    
    dev_pred = model.predict(x_sparse_dev)
    dev_results=accuracy_score(y_sparse_dev,dev_pred)
    
    test_pred = model.predict(x_sparse_test)
    test_results= accuracy_score(y_sparse_test, test_pred)
    
    print(dev_results)
    print(test_results)
    #Dev Accuracy: 0.5557899671823723
    #Test Accuracy: 0.5618612283169245
    

def denseVector(): 
    #Dense Vector - word2vec
    x_dense_train = [word_tokenize(i) for i in x_train]
    y_dense_train = np.asarray(y_train).astype('float32')
    x_dense_dev = [word_tokenize(i) for i in x_dev]
    y_dense_dev = np.asarray(y_dev).astype('float32')
    x_dense_test = [word_tokenize(i) for i in x_test]
    y_dense_test = np.asarray(y_test).astype('float32')
    
    model = gensim.models.Word2Vec(sentences=x_dense_train)
    
    words = set(model.wv.index_to_key )
    X_train_vect = np.array([np.array([model.wv[i] for i in ls if i in words])for ls in x_dense_train])
    X_dev_vect=np.array([np.array([model.wv[i] for i in ls if i in words])for ls in x_dense_dev])
    X_test_vect = np.array([np.array([model.wv[i] for i in ls if i in words])for ls in x_dense_test])
    
    X_train_vect_avg = []
    for v in X_train_vect:
        if v.size:
            X_train_vect_avg.append(v.mean(axis=0))
        else:
            X_train_vect_avg.append(np.zeros(100, dtype=float))
            
    X_dev_vect_avg = []
    for v in X_dev_vect:
        if v.size:
            X_dev_vect_avg.append(v.mean(axis=0))
        else:
            X_dev_vect_avg.append(np.zeros(100, dtype=float))
            
    X_test_vect_avg = []
    for v in X_test_vect:
        if v.size:
            X_test_vect_avg.append(v.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(100, dtype=float))

    
            
    logModel=LogisticRegression(multi_class="auto",random_state=329).fit(X_train_vect_avg,y_dense_train)
    
    dev_pred = logModel.predict(X_dev_vect_avg)
    dev_results=accuracy_score(y_dense_test,dev_pred)
    
    test_pred = logModel.predict(X_test_vect_avg)
    test_results= accuracy_score(y_dense_test, test_pred)
    
    print(dev_results)
    print(test_results)
    
           
    
denseVector()
