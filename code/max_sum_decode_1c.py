import numpy, data_read, decode

X, W, T = data_read.read_decode_input()

y_star = decode.max_sum(X, W, T)

from string import ascii_lowercase
mapping = dict(enumerate(ascii_lowercase))

for i in range(y_star.shape[0]):
	print(y_star[i]+1)
