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
import copy,decode,max_sum_decode
import optimize 

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "../data/transform.txt")


f = open(path, 'r')
x=[0,500,1000,1500]

train_data={}

def restore_range(OldValue,OldMax,OldMin,NewMin,NewMax):
    OldRange = (OldMax - OldMin)
    if (OldRange == 0):
        NewValue = NewMin
    else:
        NewRange = (NewMax - NewMin)  
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return np.round(NewValue)


#method to peform transformation on the data and check robustness
def tamper(model):
    test_acc =[]
    wrd_acr =[]
    y_pred=[]

    X_train,y_train=ri.read_train_struct()
    X_test,y_test=ri.read_test_struct()
    for num in x:
        print(num)
        X_trans = copy.deepcopy(X_train)
        for i in range(num):
            line=next(f)
            line=line.split(' ')
            offset=[]
            c=X_trans[int(line[1]),:]
            example=np.array(c).reshape(16,8)    
            if(line[0]=='r'):
                alpha=float(line[2])
                x_result=r.rotate(example,alpha)
                x_result=x_result.reshape(128,)
            elif (line[0]=='t'):
                offset.append(int(line[2]))
                offset.append(int(line[3]))
                x_result=t.translate(example,offset)
                x_result=x_result.reshape(128,)  
            x_max_old=0
            x_min_old=255
            x_min_new=0
            x_max_new=1
            new_value=restore_range(x_result,x_max_old,x_min_old,x_min_new,x_max_new)
            X_trans[int(line[1])]=new_value
        #training 
        if(num==0):
            if(model=='svm'):
                clf=svm.train(X_train,y_train,1000/len(y_train))
            else:      
                x_y=X_train,y_train
                optimize.get_params(x_y)
                a=np.loadtxt("best_Weights_tampered",usecols=(0,))
                W=np.array(a[:26*128].reshape(26,128))
                T=np.array(a[26*128:26*128+26*26].reshape(26,26))   
        #training with more than 1 transformation 
        else:
            if(model=='svm'):
                clf=svm.train(X_trans,y_train,1000/len(y_train))
            else:      
                x_y=X_trans,y_train
                print(type(x_y))
                optimize.get_params(x_y)
                a=np.loadtxt("best_Weights_tampered",usecols=(0,))
                W=np.array(a[:26*128].reshape(26,128))
                T=np.array(a[26*128:26*128+26*26].reshape(26,26))   

        #testing 
        if(model=='svm'):
            y_pred,score=svm.test(clf,X_test,y_test)
        else:
            y_pred = decode.max_sum(X_test, W, T)
            y_pred=[y+1 for y in y_pred]
            y_test=y_test.reshape(26198,)
            y_pred=np.array(y_pred).reshape(len(y_pred,))               
            print((y_test))
            print((y_pred))
            score=max_sum_decode.get_test_accuracy(y_test,y_pred)
        test_acc.append(score*100)
        y_test=y_test.reshape(len(y_test,))
        given_words, pred_words=svm.form_words(y_test,y_pred)
        w_acc=svm.word_accuracy(given_words,pred_words)
        wrd_acr.append(w_acc*100)
    return test_acc,wrd_acr



def test_tamper_svm():
    test_accuracy,word_acr=tamper('svm')
    mp.figure(70)
    mp.title('Letter Wise Accuracy vs C - SVM-MC')
    mp.plot(x,test_accuracy)
    mp.ylabel('Letter Wise Accuracy')
    mp.xlabel('X')
    mp.figure(77)
    mp.plot(x,word_acr)
    mp.ylabel('Word Wise Accuracy')
    mp.xlabel('X')
    mp.title('Word Wise Accuracy vs C - SVM-MC')

def test_tamper_crf():
    test_accuracy,word_acr=tamper('crf')
    mp.figure(108)
    mp.title('Letter Wise Accuracy vs C - CRF')
    mp.plot(x,test_accuracy)
    mp.ylabel('Letter Wise Accuracy')
    mp.xlabel('X')
    mp.figure(107)
    mp.plot(x,word_acr)
    mp.ylabel('Word Wise Accuracy')
    mp.xlabel('X')
    mp.title('Word Wise Accuracy vs C - CRF')
    
#test_tamper_svm()
#test_tamper_crf()    
def plot():
    x=[0,500,1000,1500]
    test_accuracy=[80.23,79.12,77.78,76.12]
    word_acr=[14.12,3.89,2.78,1.18]
    mp.figure(1)
    mp.plot(x,test_accuracy)
    mp.title('Letter wise Accuracy vs C - CRF ')
    mp.ylabel('Accuracy')
    mp.xlabel('C')
    mp.figure(2)
    mp.plot(x,word_acr)
    mp.ylabel('Accuracy')
    mp.xlabel('C')  
    mp.title('Word wise Accuracy vs C - CRF ')    

plot()
        

