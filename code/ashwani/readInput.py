# -*- coding: utf-8 -*-

import os.path
import numpy as np

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "../../data/decode_input.txt")
path1 = os.path.join(my_path, "../../data/train_struct.txt")


def read_parameters():
    
    input= np.loadtxt(path,ndmin=1)
    x=np.array(input[0:100*128]).reshape(100,128);
    w=np.array(input[100*128:100*128+26*128]).reshape(26,128);
    t=np.array(input[100*128+26*128:]).reshape(26,26);
    return x,w,t;

#def read_train_data():
#    with open(path1) as f:
#        lines=f.readlines();
#    for line in lines:
#        myarray = np.fromstring(line, dtype=float, sep=' ')
#        print(myarray)
    
def read_train_struct():
#function to read data into list structure
#dataX number of examples by 128 currently a numpy array
#dataY number of examples by 2 "array" (each example has a label and a qid)

	with open("../../data/train_struct.txt", "r") as f:
		raw_data = f.read()
	raw_data = raw_data.split("\n")

	dataX, dataY = [], []
	for line in raw_data[:-2]: #-2 because last element is empty
		line = line.split(" ")
		dataY.append([int(line[0])-1])
		datax = [0]*128
		for f1 in line[2:]:
			end = f1.find(":")
			datax[int(f1[:end])-1] = 1
		dataX.append(datax)
    
	return np.array(dataX, dtype=float), np.array(dataY,dtype=int)

def read_test_struct():
#function to read data into list structure
#dataX number of examples by 128 currently a numpy array
#dataY number of examples by 2 "array" (each example has a label and a qid)

	with open("../../data/test_struct.txt", "r") as f:
		raw_data = f.read()
	raw_data = raw_data.split("\n")

	dataX, dataY = [], []
	for line in raw_data[:-2]: #-2 because last element is empty
		line = line.split(" ")
		dataY.append([int(line[0])-1])
		datax = [0]*128
		for f1 in line[2:]:
			end = f1.find(":")
			datax[int(f1[:end])-1] = 1
		dataX.append(datax)
    
	return np.array(dataX, dtype=float), np.array(dataY,dtype=int)