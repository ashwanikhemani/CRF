import numpy, data_read, prob_grad
from scipy.optimize import check_grad

X_y = data_read.read_train()
W, T = data_read.read_model()

def func(T, *args):
	t = T.reshape((26, 26))
	dataX = args[0]
	dataY = args[1]
	W = args[2]

	return prob_grad.compute_log_p(dataX, dataY, W, t)

def func_prime(T, *args):
	t = T.reshape((26, 26))
	dataX = args[0]
	dataY = args[1]
	W = args[2]
	
	return prob_grad.log_p_tgrad(t, dataX, dataY, W).reshape(26*26)

x0 = numpy.random.rand(26*26)

print(check_grad(func, func_prime, x0, X_y[0][0], X_y[0][1], W))
