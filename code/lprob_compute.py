import numpy, data_read, prob_grad

W, T = data_read.read_model()

data = data_read.read_train()

import time

t0 = time.time()

a = []
for example in data:
	a.append(prob_grad.compute_log_p(example[0], example[1], W, T))
print(f"AVG: {sum(a)/len(data)}")
t1 = time.time()

print(f"Time: {t1-t0}")
