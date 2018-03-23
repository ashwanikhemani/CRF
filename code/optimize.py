import numpy, data_read, prob_grad
from scipy.optimize import fmin_bfgs
import time
import pickle
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

	norm = numpy.zeros(26)
	for i in range(26):
		norm[i] = numpy.linalg.norm(W[i])

	numpy.square(norm, out=norm)
	
	return -1*(C/len(data))*log_sum + 0.5*numpy.sum(norm) + 0.5*numpy.sum(numpy.square(T))

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


#ret = fmin_bfgs(func, initial_guess, fprime=func_prime, args=(X_y[:5], 1000),\
#	maxiter=2, retall=True, full_output=True)

def get_params(x_y):
    t0 = time.time()
    ret=fmin_bfgs(func, initial_guess, fprime=func_prime, args=(x_y,1000),\
    	 maxiter=1,retall=True, full_output=True)
    t1 = time.time()
    with open("best_Weights_tampered","+bw") as f :
        pickle.dump(ret,f)
    numpy.savetxt("best_Weights_tampered",ret[0])
    #numpy.savetxt("best_func_c_10",ret[1])
    
    print(f"Time: {t1-t0}")
    
#get_params(X_y)