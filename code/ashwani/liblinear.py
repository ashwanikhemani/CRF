# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 17:51:34 2018

@author: ashwa
"""

from readInput import * 
from sklearn.svm import LinearSVC
import matplotlib.pyplot as mp

C=[1,10,100,1000]
test_accuracy =[]
for i in C : 

    X,y=read_train_struct();
    clf = LinearSVC(random_state=0,C=i)
    clf.fit(X,y.ravel())
#print(clf.coef_)
#print(clf.intercept_)
#print(clf.predict(X_test[1].reshape(1,-1)))
    X_test,y_test=read_test_struct()
    score=(clf.score(X_test,y_test))
    test_accuracy.append(score)

mp.plot(C,test_accuracy)

