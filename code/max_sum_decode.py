import numpy, data_read, decode

X, W, T = data_read.read_decode_input()
dataX, dataY = data_read.read_train_struct()
exampleX = numpy.array(dataX[:3], dtype=float)

alphabet = [ i for i in range(26) ]
m = 100

print(decode.max_sum(X, alphabet, W, T))
#remember to add one to all of these to get a valid letter sequence
