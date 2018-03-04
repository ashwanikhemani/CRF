import numpy, data_read, training

W, T = data_read.read_model()

X, Y = data_read.read_train()

import time

t0 = time.time()

for i in range(len(X)):
	for j in range(Y[i].shape[0]):
#		print(f"Computing Marginal Prob on Letter: {Y[i][j]} Position {j}")
		training.fb_prob(X[i], Y[i][j], j, W, T)

t1 = time.time()

print(f"Time: {t1-t0}")
