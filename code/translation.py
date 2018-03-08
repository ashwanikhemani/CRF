# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 10:52:13 2018

@author: ashwa
"""

#Translate X by [offset(1) offset(2)] (horizontally and vertically, resp), 
#and return the resulting image Y
#offsets can be negative

import matplotlib.pyplot as mp
import readInput as ri
import numpy as np

def translate(X, offset):
    ox = offset[0]; 
    oy = offset[1];
    lenx, leny = X.shape;
  #  Special case where ox and oy are both positive (used in this project)
    Y = np.zeros((lenx, leny),float)
#    Y[oy:leny,ox:lenx]=X[0:leny-1-oy+1,0:lenx-1-ox+1]
    if(ox>0 and oy>0):
        Y[1+ox:lenx, 1+oy:leny] = X[1:lenx-ox, 1:leny-oy];
    else:
        #  General case where ox and oy can be negative 
        Y[max(1,1+ox):min(lenx, lenx+ox), max(1,1+oy):min(leny, leny+oy)] = X[max(1,1-ox):min(lenx, lenx-ox), max(1,1-oy):min(leny, leny-oy)];

    return Y

#X_train,y_train=ri.read_train_struct()
#example=np.array(X_train[0]).reshape(16,8)    
#mp.figure(1)
#mp.imshow(example)
#trans=translate(example,[1,1])
#trans=np.array(trans).reshape(16,8)    
#mp.figure(2)
#mp.imshow(trans)
    
