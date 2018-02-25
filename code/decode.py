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

def max_sum(x, alphabet, W, T):
#max sum function will return the best set of letters
#runs in O(mY^2) time

	trellis = []
	for i in alphabet:
		trellis.append([[i], numpy.dot(x[0,:], W[i-1,:]), [numpy.dot(x[0,:], W[i-1,:])]])
	for i in range(1,len(x)):
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
			node_1[1] = best_sum
			node_1[2] += [best_sum]
	best_node = trellis[0]
	for node in trellis:
		if best_node[1] < node[1]:
			best_node = node

	return best_node
