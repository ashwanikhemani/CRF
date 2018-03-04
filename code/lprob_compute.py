import numpy, data_read, training

W, T = data_read.read_model()

X, Y = data_read.read_train()

import time

t0 = time.time()

a = numpy.zeros(len(X))
for i in range(len(X)):
	a[i] = training.compute_log_p(X[i], Y[i], W, T)
print(f"AVG: {numpy.sum(a)/len(X)}")
t1 = time.time()

print(f"Time: {t1-t0}")
