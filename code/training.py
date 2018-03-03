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
	interior = numpy.zeros((1,alpha_len))
	for i in range(1, X.shape[0]):
		for j in range(alpha_len):
			interior[0, :], M, sum_ = 0, 0, 0
			for k in range(alpha_len):
				interior[0, k] = numpy.dot(W[k], X[i-1]) +\
					T[k,j] + trellis[i-1, k]
				if M < interior[0, k]:
					M = interior[0, k]
			for k in range(alpha_len):
				sum_ += math.exp(interior[0, k] - M)
			trellis[i, j] = M + math.log(sum_)

	interior[0, :], M, sum_ = 0, 0, 0
	for i in range(alpha_len):
		interior[0, i] = numpy.dot(W[i], X[-1]) + trellis[-1, i]
		if M < interior[0, i]:
			M = interior[0, i]
	for i in range(alpha_len):
		sum_ += math.exp(interior[0, i] - M)
	
	log_z = M + math.log(sum_)

	return sum_num - log_z
