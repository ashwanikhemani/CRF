import math, numpy

def compute_log_p(X, y, W, T):
#returns the log probability of a set of labels given X
#the parameters should all be numpy arrays of some kind
#I assume the labels are all shifted to the left by one
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
		interior[i] = numpy.dot(W[i], X[-1]) + trellis[-1, i]
	M = numpy.max(interior)
	numpy.add(interior, -1*M, out=interior)
	numpy.exp(interior, out=interior)
	
	log_z = M + math.log(numpy.sum(interior))

	return sum_num - log_z

def fb_prob(X, W, T):
#returns forward trellis and backward trellis and normalizer
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

def log_p_wgrad(W, X, y, T):
#will compute the gradient of an example
	grad = numpy.zeros((26, 128)) #size of the alphabet by 128 elems
	expect = numpy.zeros(26)
	trellisfw, trellisbw, log_z = fb_prob(X, W, T)
	prob = numpy.zeros(26)
	for i in range(X.shape[0]):
		for j in range(26):
			prob[j] = (trellisfw[i, j] + trellisbw[i, j] +\
				numpy.dot(W[j], X[i])) - log_z
			#prob that the sth character is j
		numpy.exp(prob, out=prob)
		expect[y[i]] = 1
		numpy.add(expect, -1*prob, out=expect)
		#duplicate X
		letter_grad = numpy.tile(X[i], (26, 1))
		#multiply by transpose expect
		numpy.multiply(expect[:, numpy.newaxis], letter_grad,\
			out=letter_grad)
		numpy.add(grad, letter_grad, out=grad)
		expect[:] = 0
	return grad

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
