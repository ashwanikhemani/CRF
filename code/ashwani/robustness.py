# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 10:55:32 2018

@author: ashwa
"""

import translation as t 
import rotation as r
import readInput as ri
import SVM_MC as svm
import os.path
import numpy as np
import matplotlib.pyplot as mp

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "../../data/transform.txt")


f = open(path, 'r')
#raw_data = file.read()
x=[0,50]
X_train ,y_train=ri.read_train_struct();
X_test,y_test=ri.read_test_struct()
test_accuracy =[]
word_acr =[]
train_data={}

for i in range(len(X_train)):
    train_data[i+1]=y_train[i]


for num in x:
    x_trans=[]
    y_trans=[]
    for i in range(num):
        line=next(f)
        line=line[0:len(line)-1]
        X,y=ri.read_train_struct()
        line=line.split(' ')
        offset=[]
        c=X[int(line[1]),:]
        example=np.array(c).reshape(8,16)    
        if(line[0]=='r'):
            alpha=int(line[2])
            x_result=r.rotate(example,alpha)
            x_result=x_result.reshape(128,)
        elif (line[0]=='t'):
            offset.append(int(line[2]))
            offset.append(int(line[3]))
            x_result=t.translate(example,offset)
            x_result=x_result.reshape(128,)
        
        x_trans.append(x_result)
        y_trans.append(train_data[int(line[1])])
    if(num==0):
        clf=svm.train(X_train,y_train,i/len(y_train))
    else:        
        clf=svm.train(x_trans,y_trans,i/len(y_trans))
    y_test,y_pred,score=svm.test(clf,X_test,y_test)
    test_accuracy.append(score*100)
    given_words, pred_words=svm.form_words(y_test,y_pred)
    w_acc=svm.word_accuracy(given_words,pred_words)
    word_acr.append(w_acc*100)

mp.figure(1)
mp.plot(x,test_accuracy)
mp.ylabel('Accuracy')
mp.xlabel('X')
mp.figure(2)
mp.plot(x,word_acr)
mp.ylabel('Accuracy')
mp.xlabel('X')
        
        

