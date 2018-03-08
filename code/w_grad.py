import math, numpy, data_read, prob_grad
numpy.set_printoptions(threshold=numpy.nan)

W, T = data_read.read_model()
data = data_read.read_train()

grad = numpy.zeros((26, 128))
import time

t0 = time.time()
for i in range(len(data)):
	numpy.add(prob_grad.log_p_wgrad(W, data[i][0], data[i][1], T),\
		grad, out=grad)
t1 = time.time()

print(f"Time: {t1-t0}")

avg = numpy.divide(grad, len(data))

for i in range(26):
	for j in range(128):
		print(avg[i, j])
