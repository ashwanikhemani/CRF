# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 10:52:13 2018

@author: ashwa
"""

#Translate X by [offset(1) offset(2)] (horizontally and vertically, resp), 
#and return the resulting image Y
#offsets can be negative
def translate(X, offset):
    Y = X;  
    ox = offset[0]; 
    oy = offset[1];
    lenx, leny = X.shape;
  #  Special case where ox and oy are both positive (used in this project)
#  
    if(ox>0 and oy>0):
        Y[1+ox:lenx, 1+oy:leny] = X[1:lenx-ox, 1:leny-oy];
    else:
        #  General case where ox and oy can be negative 
        Y[max(1,1+ox):min(lenx, lenx+ox), max(1,1+oy):min(leny, leny+oy)] = X[max(1,1-ox):min(lenx, lenx-ox), max(1,1-oy):min(leny, leny-oy)];

    return Y
    
