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
		interior[k] = numpy.dot(W[k], X[-1]) + trellis[-1, k]
	M = numpy.max(interior)
	numpy.add(interior, -1*M, out=interior)
	numpy.exp(interior, out=interior)
	
	log_z = M + math.log(numpy.sum(interior))
#	print(math.exp(sum_num - log_z))

	return sum_num - log_z

def fb_prob(X, y_i, y_i_pos, W, T):
#computes the marginal prob of y_i by incorporating backward messages
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

	#backward part
	for i in range(X.shape[0]-2, y_i_pos, -1):
		for j in range(alpha_len):
			for k in range(alpha_len):
				interior[k] = numpy.dot(W[k], X[i+1]) +\
					T[j, k] + trellis[i+1, k]
			M = numpy.max(interior)
			numpy.add(interior, -1*M, out=interior)
			numpy.exp(interior, out=interior)
	
	prob = math.log(trellis[y_i_pos 
			trellis[i, j] = M + math.log(numpy.sum(interior))


def compute_gradw(X, y, W, T):
#returns the gradient of the W vector for a given X, y pair
#the parameters should all be numpy arrays of some kind
#I assume the labels are all shifted to the left by one
#the gradient will be the collapsed trellis (prob) * X
	alpha_len = 26 #I would like to make this a parameter for generality
	grad = numpy.zeros((W.shape))
	trellis = numpy.zeros((X.shape[0], alpha_len))
	interior = numpy.zeros((1,alpha_len))
	for i in range(y.shape[0]):
		trellis[:, :] = 0
		for j in range(1, X.shape[0]):
			#determine if how many nodes for this character
			if j == i:
				nodes_cur = [ y[j] ]
			else:
				nodes_cur = range(26)
			for k in nodes_cur:
				interior[0, :], M, sum_ = 0, 0, 0
				if j - 1 == i:
					nodes_prev = [ y[j-1] ]
				else:
					nodes_prev = range(26)
				for l in nodes_prev:
					interior[0, l] = numpy.dot(W[l], X[j-1])+\
						T[l, k] + trellis[j-1, l]
					if M < interior[0, l]:
						M = interior[0, l]
#				print(f"Max for ({j}, {k}: {M}")
				for l in nodes_prev:
					sum_ += math.exp(interior[0, l] - M)
				trellis[j, k] = M + math.log(sum_)
		interior[0, :], M, sum_ = 0, 0, 0
		if i == y.shape[0] - 1:
			nodes_fin = [ y[-1] ]
		else:
			nodes_fin = range(26)
		for j in nodes_fin:
			interior[0, j] = numpy.dot(W[j], X[-1]) + trellis[-1, j]
			if M < interior[0, j]:
				M = interior[0, j]
		for j in nodes_fin:
			sum_ += math.exp(interior[0, j] - M)
		prob = M + math.log(sum_)
		print(prob)
