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

def fb_prob(X, W, T):
#returns forward trellis and backward trellis and normalizer
#y_i is the label and y_i_pos is the letter position
	alpha_len = 26
	trellisfw = numpy.zeros((X.shape[0], alpha_len))
	trellisbw = numpy.zeros((X.shape[0], alpha_len))
	interior = numpy.zeros(alpha_len)

	#forward part
	for i in range(1, X.shape[0]):
		for j in range(alpha_len):
			for k in range(alpha_len):
				interior[k] = numpy.dot(W[k], X[i-1]) +\
					T[k,j] + trellisfw[i-1, k]
			M = numpy.max(interior)
			numpy.add(interior, -1*M, out=interior)
			numpy.exp(interior, out=interior)
			trellisfw[i, j] = M + math.log(numpy.sum(interior))

	for i in range(alpha_len):
		interior[i] = numpy.dot(W[i], X[-1]) + trellisfw[-1, i]
	M = numpy.max(interior)
	numpy.add(interior, -1*M, out=interior)
	numpy.exp(interior, out=interior)
	
	log_z = M + math.log(numpy.sum(interior))


	#backward part
	for i in range(X.shape[0]-2, -1, -1):
		for j in range(alpha_len):
			for k in range(alpha_len):
				interior[k] = numpy.dot(W[k], X[i+1]) +\
					T[j, k] + trellisbw[i+1, k]
			M = numpy.max(interior)
			numpy.add(interior, -1*M, out=interior)
			numpy.exp(interior, out=interior)
			trellisbw[i, j] = M + math.log(numpy.sum(interior))
	
	
	return trellisfw, trellisbw, log_z

def log_p_tgrad(T, X, y, W):
#will compute the gradient of an example
	grad = numpy.zeros((26, 26)) #size of the alphabet by 128 elems
	expect = numpy.zeros((26, 26))
	letter_grad = numpy.zeros((26, 26))
	trellisfw, trellisbw, log_z = fb_prob(X, W, T)
	for i in range(X.shape[0]-1):
		prob = (trellisfw[i, y[i]] + trellisbw[i+1, y[i+1]] +\
			numpy.dot(W[y[i]], X[i]) + numpy.dot(W[y[i+1]], X[i+1]) +\
			T[y[i], y[i+1]]) - log_z
		prob = math.exp(prob)
		expect[y[i], y[i+1]] = 1
		numpy.add(expect, -1*prob, out=expect)
		numpy.add(grad, expect, out=grad)
		expect[:, :] = 0
	return grad

grad = numpy.zeros((26, 26))
import time

t0 = time.time()
for i in range(len(data)):
	numpy.add(log_p_tgrad(T, data[i][0], data[i][1], W), grad, out=grad)
t1 = time.time()

print(f"{numpy.divide(grad, len(data))}")

print(f"Time: {t1-t0}")
