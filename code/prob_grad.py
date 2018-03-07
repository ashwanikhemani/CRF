import math, numpy

def compute_log_p(X, y, W, T):
#returns the log probability of a set of labels given X
#the parameters should all be numpy arrays of some kind
#I assume the labels are all shifted to the left by one
	alpha_len = 26 #I would like to make this a parameter for generality
			
	sum_num = numpy.dot(W[y[0]], X[0])
	for i in range(1, X.shape[0]):
		sum_num += numpy.dot(W[y[i]], X[i]) + T[y[i-1], y[i]]
	
	trellisfw = numpy.zeros((X.shape[0], alpha_len))
	interior = numpy.zeros(alpha_len)
	for i in range(1, X.shape[0]):
		for j in range(alpha_len):
			dots = numpy.matmul(W, X[i-1])
			numpy.add(dots, T[:,j], out=interior)
			numpy.add(interior, trellisfw[i-1], out=interior)

			M = numpy.max(interior)
			numpy.add(interior, -1*M, out=interior)
			numpy.exp(interior, out=interior)
			trellisfw[i, j] = M + math.log(numpy.sum(interior))

	dots = numpy.matmul(W, X[-1])
	numpy.add(dots, trellisfw[-1], out=interior)

	M = numpy.max(interior)
	numpy.add(interior, -1*M, out=interior)
	numpy.exp(interior, out=interior)
	
	log_z = M + math.log(numpy.sum(interior))

	return sum_num - log_z

def fb_prob(X, W, T):
	alpha_len = 26

	trellisfw = numpy.zeros((X.shape[0], alpha_len))
	trellisbw = numpy.zeros((X.shape[0], alpha_len))

	interior = numpy.zeros(alpha_len)
	messages = numpy.zeros((26, 26))

	#forward part
	for i in range(1, X.shape[0]):
		numpy.matmul(W, X[i-1], out=interior)
		numpy.add(interior, trellisfw[i-1], out=interior)
		numpy.add(T, interior[:, numpy.newaxis], out=messages)
		maxes = messages.max(axis=0)
		numpy.add(messages, -1*maxes, out=messages)
		numpy.exp(messages, out=messages)
		numpy.sum(messages, axis=0, out=interior)
		numpy.log(interior, out=interior)
		numpy.add(maxes, interior, out=trellisfw[i])
		
	dots = numpy.matmul(W, X[-1])
	numpy.add(dots, trellisfw[-1], out=interior)
	M = numpy.max(interior)
	numpy.add(interior, -1*M, out=interior)
	numpy.exp(interior, out=interior)
	
	log_z = M + math.log(numpy.sum(interior))

	#backward part
	for i in range(X.shape[0]-2, -1, -1):
		numpy.matmul(W, X[i+1], out=interior)
		numpy.add(interior, trellisbw[i+1], out=interior)
		numpy.add(T, interior, out=messages)
		numpy.swapaxes(messages, 0, 1)
		maxes = messages.max(axis=1)
		numpy.add(messages, -1*maxes[:, numpy.newaxis], out=messages)
		numpy.exp(messages, out=messages)
		numpy.sum(messages, axis=1, out=interior)
		numpy.log(interior, out=interior)
		numpy.add(maxes, interior, out=trellisbw[i])

	return trellisfw, trellisbw, log_z

def log_p_wgrad(W, X, y, T):
#will compute the gradient of an example
	grad = numpy.zeros((26, 128)) #size of the alphabet by 128 elems
	expect = numpy.zeros(26)
	trellisfw, trellisbw, log_z = fb_prob(X, W, T)
	prob = numpy.zeros(26)
	for i in range(X.shape[0]):
		#compute the marginal probability for all nodes in this column
		numpy.add(trellisfw[i, :], trellisbw[i, :], out=prob)
		numpy.add(numpy.matmul(W, X[i]), prob, out=prob)
		numpy.add(-1*log_z, prob, out=prob)
		numpy.exp(prob, out=prob)

		#compute the expectation?
		expect[y[i]] = 1
		numpy.add(expect, -1*prob, out=expect)
		letter_grad = numpy.tile(X[i], (26, 1))
		#compute letter wise probabililty
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
	prob = numpy.zeros((26, 26))
	trellisfw, trellisbw, log_z = fb_prob(X, W, T)
	for i in range(X.shape[0]-1):
		for j in range(26):
			for k in range(26):
				prob[j, k] = (trellisfw[i, j] + trellisbw[i+1, k] +\
					numpy.dot(W[j], X[i]) + numpy.dot(W[k], X[i+1]) +\
					T[j, k]) - log_z
		numpy.exp(prob, out=prob)
		expect[y[i], y[i+1]] = 1
		numpy.add(expect, -1*prob, out=expect)
		numpy.add(grad, expect, out=grad)
		expect[:, :] = 0
	return grad
