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

def compute_prob(x, y, W, T):
#decode a single example, the decoder is specified in project pdf (3)
#the decoder computes sum of <W_y, Xi> + sum of T_ij

	x_sum, t_sum = 0,0
	for i in range(len(x)-1):
		x_sum += numpy.dot(x[i,:],W[y[i]-1,:])
		t_sum += T[y[i]-1, y[i+1]-1]
	x_sum += numpy.dot(x[len(x)-1,:],W[y[len(x)-1]-1,:])

	return x_sum + t_sum

def compute_prob_letter(x_1, x_2, alphabet, W, T):
#compute highest likely letter label given letter inputs 1 and 2

	best_x1, best_x2, best_sum = 0,0,0
	for i in alphabet:
		for j in alphabet:
			temp_sum = numpy.dot(x_1, W[i-1,:]) + T[i-1,j-1]
			if best_sum < temp_sum:
				best_x1, best_x2, best_sum = i, j, temp_sum

	return best_x1, best_x2

def max_sum(x, alphabet, m, W, T):
#max sum function will return the best set of letters

	trellis = []
	for i in alphabet:
		trellis.append([[i], numpy.dot(x[0,:], W[i-1,:])])
	for i in range(1,m):
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
			node_1[1] += best_sum
	best_sum, best_list = 0, None
	for node in trellis:
		if best_sum < node[1]:
			best_sum = node[1]
			best_list = node[0]

	return best_sum, best_list
