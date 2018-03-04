import math, numpy
numpy.set_printoptions(threshold=numpy.nan)

def read_model():
#function to read model for 2a
	with open("../data/model.txt", "r") as f:
		raw_data = f.read()
	raw_data = raw_data.split("\n")

	W = numpy.array(raw_data[:26*128], dtype=float).reshape(26, 128)
	T = numpy.array(raw_data[26*128:-1], dtype=float).reshape(26, 26)
	T = numpy.swapaxes(T, 0, 1)
	return W, T

W, T = read_model()

def read_train():
#function to read train data
	from string import ascii_lowercase
	mapping = list(enumerate(ascii_lowercase))
	mapping = { i[1]:i[0] for i in mapping }

	with open("../data/train.txt", "r") as f:
		raw_data = f.read()
	raw_data = raw_data.split("\n")

	dataX, dataY = [], []
	tempX, tempY = [], []
	for row in raw_data[:-1]:
		row = row.split(" ")
		tempY.append( mapping[row[1]])
		tempX.append( numpy.array(row[5:], dtype=float) )
		if int(row[2]) < 0:
			dataX.append(numpy.array(tempX))
			dataY.append(numpy.array(tempY, dtype=int))
			tempX, tempY = [], []

	ret = zip(dataX, dataY)
	return list(ret)

data = read_train()

def log_p_w(W, X, y, T):
#this is computes the log prob of an example 
	alpha_len = 26 #I would like to make this a parameter for generality
			
	sum_num = numpy.dot(W[y[0]], X[0])
	for i in range(1, X.shape[0]):
		sum_num += numpy.dot(W[y[i]], X[i]) + T[y[i-1], y[i]]
	
	trellis = numpy.zeros((X.shape[0], alpha_len))
	interior = numpy.zeros(alpha_len)
	for i in range(1, X.shape[0]):
		for j in range(alpha_len):
			for k in range(alpha_len):
				interior[k] = numpy.dot(W[k], X[i-1]) +\
					T[k,j] + trellis[i-1, k]
			M = numpy.max(interior)
			numpy.add(interior, -1*M, out=interior)
			numpy.exp(interior, out=interior)
			trellis[i, j] = M + math.log(numpy.sum(interior))

	for i in range(alpha_len):
		interior[k] = numpy.dot(W[k], X[-1]) + trellis[-1, k]
	M = numpy.max(interior)
	numpy.add(interior, -1*M, out=interior)
	numpy.exp(interior, out=interior)
	
	log_z = M + math.log(numpy.sum(interior))

	return sum_num - log_z

def fb_prob(X, y_i, y_i_pos, W, T):
#computes the marginal prob of y_i by incorporating backward messages
#y_i is the label and y_i_pos is the letter position
	alpha_len = 26
	trellis = numpy.zeros((X.shape[0], alpha_len))
	interior = numpy.zeros(alpha_len)

	#forward part
	for i in range(1, y_i_pos):
		for j in range(alpha_len):
			for k in range(alpha_len):
				interior[k] = numpy.dot(W[k], X[i-1]) +\
					T[k,j] + trellis[i-1, k]
			M = numpy.max(interior)
			numpy.add(interior, -1*M, out=interior)
			numpy.exp(interior, out=interior)
			trellis[i, j] = M + math.log(numpy.sum(interior))

	forward_part = 0
	if y_i_pos > 0:
		for k in range(alpha_len):
			interior[k] = numpy.dot(W[k], X[y_i_pos-1]) +\
				T[k, y_i] + trellis[y_i_pos-1, k]
		M = numpy.max(interior)
		numpy.add(interior, -1*M, out=interior)
		numpy.exp(interior, out=interior)
		forward_part = M + math.log(numpy.sum(interior))

	#backward part
	for i in range(X.shape[0]-2, y_i_pos, -1):
		for j in range(alpha_len):
			for k in range(alpha_len):
				interior[k] = numpy.dot(W[k], X[i+1]) +\
					T[j, k] + trellis[i+1, k]
			M = numpy.max(interior)
			numpy.add(interior, -1*M, out=interior)
			numpy.exp(interior, out=interior)
	
	
	backward_part = 0
	if y_i_pos < X.shape[0] - 1:
		for k in range(alpha_len):
			interior[k] = numpy.dot(W[k], X[y_i_pos+1]) +\
				T[y_i, k] + trellis[y_i_pos+1, k]
		M = numpy.max(interior)
		numpy.add(interior, -1*M, out=interior)
		numpy.exp(interior, out=interior)
		backward_part = M + math.log(numpy.sum(interior))

	#final result
	prob = forward_part + backward_part + numpy.dot(W[y_i], X[y_i_pos])
#	print(prob)
	#i think that there needs to be some normalization here
	return prob

def log_p_wgrad(W, X, y, T):
#will compute the gradient of an example
	grad = numpy.zeros((26, 128)) #size of the alphabet by 128 elems
	expect = numpy.zeros(26)
	letter_grad = numpy.zeros((26, 128))
	for i in range(X.shape[0]):
		expect[y[i]] = 1
		prob = fb_prob(X, y[i], i, W, T)
		for j in range(26):
			expect[j] -= prob
			numpy.multiply(expect[j], X[i], out=letter_grad[j])
		numpy.add(grad, letter_grad, out=grad)
		expect[:] = 0
	print(grad)
	return grad

log_p_wgrad(W, data[0][0], data[0][1], T)
