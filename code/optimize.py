import numpy, data_read, prob_grad
from scipy.optimize import fmin_bfgs
from numpy.linalg import norm

X_y = data_read.read_train()
W, T = data_read.read_model()

def func(params, *args):
#objective function specified in the handout
	W, T = params[:26*128].reshape((26, 128)), params[26*128:].reshape((26, 26))
	data = args[0]
	C = args[1]

	log_sum = 0
	for example in data:
		log_sum += prob_grad.compute_log_p(example[0], example[1], W, T)
	
	return -1*(C/len(data))*log_sum + 0.5*norm(W)**2 + 0.5*numpy.sum(numpy.square(T))

def func_prime(params, *args):
#derivative of objective function specified in the handout
	W, T = params[:26*128].reshape((26, 128)), params[26*128:].reshape((26, 26))
	data = args[0]
	C = args[1]

	log_grad_w = numpy.zeros((26, 128))
	log_grad_t = numpy.zeros((26, 26))

	#gradient of logP w/ W
	for example in data:
		numpy.add(log_grad_w, prob_grad.log_p_wgrad(W,\
			example[0], example[1], T), out=log_grad_w)
		numpy.add(log_grad_t, prob_grad.log_p_tgrad(T,\
			example[0], example[1], W), out=log_grad_t)

	#multiply C/N
	numpy.multiply(log_grad_w, -1*C/len(data), out=log_grad_w)
	numpy.multiply(log_grad_t, -1*C/len(data), out=log_grad_t)

	#add gradient of norm
	numpy.add(log_grad_w, W, out=log_grad_w)

	#add normalizing factor
	numpy.add(log_grad_t, T, out=log_grad_t)

	return numpy.concatenate([log_grad_w.reshape(26*128),\
		log_grad_t.reshape(26*26)])

#on = numpy.concatenate([W.reshape(26*128), T.reshape(26*26)])

#res = func(on, X_y[:9], 1000)
#result = func_prime(on, X_y[:9], 1000)

#need to flatten for the optimizer
initial_guess = numpy.zeros((26*128+26*26))

#bounds = [(-10000000, 10000000)]*(28*128+26*26)


ret = fmin_bfgs(func, initial_guess, fprime=func_prime, args=(X_y[:5], 1000),\
	maxiter=10, retall=True, full_output=True)
