import numpy

def read_decode_input():
#function to read the data from decode_input
#100 letters each 128 elements
#26 weight vectors each 128 elements
#T which is a 26 x 26 weight matrix
#T is row major T_11, T_21, T_31 ..

	with open("../data/decode_input.txt", "r") as f:
		raw_data = f.read().split("\n")

	X = numpy.array(raw_data[:100*128], dtype=float).reshape(100,128)
	W = numpy.array(raw_data[100*128:100*128+26*128]\
		, dtype=float).reshape(26,128)
	T = numpy.array(raw_data[100*128+26*128:-1], dtype=float).reshape(26,26)
	T = numpy.swapaxes(T, 0, 1)

	return X, W, T

def read_input():
#function to read data into list structure
#dataX number of examples by 128 "array"
#dataY number of examples by 2 "array" (each example has a label and a qid)

	with open("../data/train_struct.txt", "r") as f:
		raw_data = f.read()
	raw_data = raw_data.split("\n")

	dataX, dataY = [], []
	for line in raw_data[:-2]: #-2 because last element is empty
		line = line.split(" ")
		dataY.append([ int(line[0]), int(line[1][4:]) ])
		datax = [0]*128
		for f1 in line[2:]:
			end = f1.find(":")
			datax[int(f1[:end])-1] = 1
		dataX.append(datax)

	return dataX, dataY


X, W, T = read_decode_input()
dataX, dataY = read_input()
exampleX = numpy.array(dataX[:3], dtype=float)

alphabet = [ i for i in range(1,27) ]
m = 3

def max_sum(x, alphabet, W, T):
#max sum function will return the best set of letters
#runs in O(mY^2) time

	trellis = []
	for i in alphabet:
		trellis.append([[i], numpy.dot(x[0,:], W[i-1,:]), [numpy.dot(x[0,:], W[i-1,:])]])
	for i in range(1,len(x)):
		for node_1 in trellis:
			best_sum, best_node2 = 0,0
			for j in alphabet:
				temp_sum = node_1[1] +\
					numpy.dot(x[i,:], W[j-1,:])+\
					T[node_1[0][-1]-1,j-1]
				if best_sum < temp_sum:
					best_sum = temp_sum
					best_node2 = j
			node_1[0] += [best_node2]
			node_1[1] = best_sum
			node_1[2] += [best_sum]
	best_node = trellis[0]
	for node in trellis:
		if best_node[1] < node[1]:
			best_node = node

	return best_node
