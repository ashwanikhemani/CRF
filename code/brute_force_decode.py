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

X, W, T = read_decode_input()

alphabet = [ i for i in range(1,27) ]
m = 3

def generate_mcombs(alphabet, m):
#generate all possible combinations, word size is specified here as m
#itertools.product is much better than this

	combinations = [[]]
	for i in range(m): 
		combinations = [j + [k] for k in alphabet for j in combinations]

	return combinations

combinations = generate_mcombs(alphabet, m)

def compute_prob(x, y, W, T):
#decode a single example, the decoder is specified in project pdf (3)
#the decoder computes sum of <W_y, Xi> + sum of T_ij

	x_sum, t_sum = 0,0
	for i in range(len(x)-1):
		x_sum += numpy.dot(x[i,:],W[y[i]-1,:])
		t_sum += T[y[i]-1, y[i+1]-1]
	x_sum += numpy.dot(x[len(x)-1,:],W[y[len(x)-1]-1,:])

	return x_sum + t_sum

def find_max(x, combinations, W, T):
#now find the max decoder value and best y for a given word x

	max_val, likely_y = 0, 0
	for y in combinations:
		val = compute_prob(x, y, W, T)
		if max_val < val:
			max_val = val
			likely_y = y

	return max_val, likely_y


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

dataX, dataY = read_input()
exampleX = numpy.array(dataX[:3], dtype=float)

