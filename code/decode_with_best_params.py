import numpy, data_read, decode

f = numpy.loadtxt("../data/best_params")

W, T = f[:26*128].reshape((26, 128)), f[26*128:].reshape((26, 26))

X = data_read.read_test_decoder()

y_star = decode.max_sum(X, W, T)

from string import ascii_lowercase
mapping = dict(enumerate(ascii_lowercase))

for i in range(y_star.shape[0]):
	print(y_star[i]+1)
