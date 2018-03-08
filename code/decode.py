import numpy

def generate_mcombs(alphabet, m):
#generate all possible combinations, word size is specified here as m
#itertools.product is much better than this

	combinations = [[]]
	for i in range(m): 
		combinations = [j + [k] for k in alphabet for j in combinations]

	return combinations

def compute_prob(x, y, W, T):
#decode a single example, the decoder is specified in project pdf (3)
#the decoder computes sum of <W_y, Xi> + sum of T_ij

	x_sum, t_sum = 0,0
	for i in range(len(x)-1):
		x_sum += numpy.dot(x[i,:],W[y[i],:])
		t_sum += T[y[i], y[i+1]]
	x_sum += numpy.dot(x[len(x)-1,:],W[y[len(x)-1],:])

	return x_sum + t_sum

def find_max(x, combinations, W, T):
#now find the max decoder value and best y for a given word x and combinations

	max_val, likely_y = 0, 0
	for y in combinations:
		val = compute_prob(x, y, W, T)
		if max_val < val:
			max_val = val
			likely_y = y

	return  likely_y, max_val

def max_sum(X, W, T):
#decodes by running the max sum algorithm
#X, W, T are numpy arrays (X is the input)
	alpha_len = 26
	trellis = numpy.zeros((X.shape[0],alpha_len))
	interior = numpy.zeros(alpha_len)
	y_star = numpy.zeros(X.shape[0], dtype=int)

	for i in range(1, X.shape[0]):
		for j in range(alpha_len):
			for k in range(alpha_len):
				interior[k] = numpy.dot(W[k], X[i-1]) +\
					T[k, j] + trellis[i-1, k]
			trellis[i, j] = numpy.max(interior)
	
	for i in range(alpha_len):
		interior[i] = numpy.dot(W[i], X[-1]) + trellis[-1, k]
	y_star[-1] = numpy.argmax(interior)
	print(interior[y_star[-1]])

	for i in range(X.shape[0]-1, 0, -1):
		for j in range(alpha_len):
			interior[j] = numpy.dot(W[j], X[i-1]) +\
				T[j, y_star[i]] + trellis[i-1, j]
		y_star[i-1] = numpy.argmax(interior)

	return y_star
