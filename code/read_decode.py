#script to read the data from decode_input
#100 letters each 128 elements
#26 weight vectors each 128 elements
#T which is a 26 x 26 weight matrix
#T is row major T_11, T_21, T_31 ..

with open("../data/decode_input.txt", "r") as f:
	raw_data = f.read().split("\n")

import numpy

X = numpy.array(raw_data[:100*128]).reshape(100,128)
W = numpy.array(raw_data[100*128:100*128+26*128]).reshape(26,128)
T = numpy.array(raw_data[100*128+26*128:-1]).reshape(26,26)
T = numpy.swapaxes(T, 0, 1)
