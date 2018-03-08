# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 17:51:34 2018

@author: ashwa
"""

from readInput import read_train_struct,read_test_struct,read_word_indexes
from sklearn.svm import LinearSVC
import matplotlib.pyplot as mp
import numpy as np

def train(xtrain,y_train,c): 
    clf = LinearSVC(random_state=0,C=c)
    clf.fit(xtrain,y_train)
    return clf

def test(model,X_test,y_test):
    y_pred=model.predict(X_test)
    score=(model.score(X_test,y_test))
    return y_pred,score

def form_words(y1,y2):
    word_ends=[]
    word_idx=read_word_indexes()
    word_ends=np.insert(np.where(word_idx==-1),0,0)
    given_words=[]
    pred_words=[] 
    start=0
    for i in range(len(word_ends)-1):   
        end=word_ends[i+1]+1
        g_word=y1[start:end]
        p_word=y2[start:end]
        start=end
        given_words.append(g_word)
        pred_words.append(p_word)
    return given_words, pred_words    
    
def word_accuracy(words1,words2):
    count=0
    for i,j in zip(words1,words2):
        if(np.array_equal(i,j)):
            count+=1
    return count/len(words1)
    
def plot():
    C=[1,10,100,1000]
    test_accuracy =[]
    word_acr =[]
    X_train ,y_train=read_train_struct();
    X_test,y_test=read_test_struct()

    for i in C : 
        y_train=y_train.ravel()
        clf=train(X_train,y_train,i/len(y_train))
        y_pred,score=test(clf,X_test,y_test)
        test_accuracy.append(score*100)
        y_train=y_train.reshape(len(y_train,))
        given_words, pred_words=form_words(y_test,y_pred)
        w_acc=word_accuracy(given_words,pred_words)
        word_acr.append(w_acc*100)
        
    mp.figure(1)
    mp.title('Letter wise Accuracy vs C - SVM-MC ')
    mp.plot(C,test_accuracy)
    mp.ylabel('Accuracy')
    mp.xlabel('C')
    mp.figure(2)
    mp.plot(C,word_acr)
    mp.title('Word wise Accuracy vs C - SVM-MC ')
    mp.ylabel('Accuracy')
    mp.xlabel('C')

    
#plot()
    

