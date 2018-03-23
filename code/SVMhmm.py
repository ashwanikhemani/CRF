# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 23:00:04 2018

@author: ashwa
"""

import subprocess
from readInput import read_train_struct,read_test_struct,read_word_indexes
import numpy as np
import os.path
import matplotlib.pyplot as mp

def train(c):    
#   output = subprocess.getoutput('svm_hmm_learn -c 1 train_struct.txt declaration.model')
    subprocess.call(["svm_hmm_learn", "-c", str(c), "train_struct.txt", "declaration.model"])

def test():
#    output = subprocess.getoutput('svm_hmm_classify test_struct.txt declaration.model test.outtags ')
    subprocess.call(["svm_hmm_classify", "test_struct.txt", "declaration.model","test.outtags"])

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
    
def get_test_accuracy(y1,y2):
    count=0
    for i,j in zip(y1,y2):
        if(np.array_equal(i,j)):
            count+=1
    return count/len(y1)

    
def plot():
   C=[1,10]
   test_accuracy =[]
   word_acr =[]
   X_train ,y_train=read_train_struct();
   X_test,y_test=read_test_struct()
   for i in C : 
       train(i)
       test()
       my_path = os.path.abspath(os.path.dirname(__file__))
       path = os.path.join(my_path, "test.outtags")
       y_pred=np.loadtxt(path,usecols=(0,))
       y_test=y_test.reshape(len(y_test),)
       test_acc=get_test_accuracy(y_test,y_pred)
       test_accuracy.append(test_acc*100)
       given_words, pred_words=form_words(y_test,y_pred)
       w_acc=word_accuracy(given_words,pred_words)
       word_acr.append(w_acc*100)
       
   mp.figure(1)
   mp.plot(C,test_accuracy)
   mp.ylabel('Accuracy')
   mp.title('Letter wise Accuracy vs C - SVM-HMM ')
   mp.xlabel('C')
   mp.figure(2)
   mp.plot(C,word_acr)
   mp.ylabel('Accuracy')
   mp.xlabel('C')
   mp.title('Word wise Accuracy vs C - SVM-HMM ')
 
plot()

