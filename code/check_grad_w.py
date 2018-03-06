import numpy, data_read, prob_grad
from scipy.optimize import check_grad

X_y = data_read.read_train()
W, T = data_read.read_model()

def func(W, *args):
	w = W.reshape(26, 128)
	dataX = args[0]
	dataY = args[1]
	T = args[2]

	return prob_grad.compute_log_p(dataX, dataY, w, T)

def func_prime(W, *args):
	w = W.reshape(26, 128)
	dataX = args[0]
	dataY = args[1]
	T = args[2]
	
	return prob_grad.log_p_wgrad(w, dataX, dataY, T).reshape(26*128)

x0 = numpy.random.rand(26*128)

print(check_grad(func, func_prime, x0, X_y[0][0], X_y[0][1], T))
