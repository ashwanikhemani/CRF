#script to generate solution.txt from optimizer output
import numpy

f = numpy.loadtxt("best_params")

W, T = f[:26*128].reshape((26, 128)), f[26*128:].reshape((26, 26))


for i in range(26):
	for j in range(128):
		print(W[i, j])

for i in range(26):
	for j in range(26):
		print(T[j, i])
