# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 10:48:02 2018

@author: ashwa
"""
from scipy.misc import imrotate
import math
import numpy as np
#Rotate X by alpha degrees (angle) in a counterclockwise direction around its center point.
#This may enlarge the image.
#So trim the result back to the original size, around its center point.
def rotate(X, alpha):
#  Y = imrotate(X, alpha); % Python counterpart: scipy.misc.imrotate
    Y=imrotate(X,alpha)
    lenx1, lenx2 = X.shape;
    leny1, leny2 = Y.shape;

#  Trim the result back to the original size, around its center point.
    fromx = math.floor((leny1 + 1 - lenx1)/2);
    fromy = math.floor((leny2 + 1 - lenx2)/2);
  
    Y = Y[fromx:fromx+lenx1, fromy:fromy+lenx2];
  
#  imrotate could pad 0 at some pixels.
#  At those pixels, fill in the original values
    rows,col=np.where(Y==0)
    for i,j in zip(rows,col):
        Y[i][j] = X[i][j];
    return Y

#x=np.arange(10).reshape(2,5)
#print (x)
#print (rotate(x,30))
    

    