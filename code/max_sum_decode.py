import numpy, data_read, decode

X, W, T = data_read.read_decode_input()
dataX, dataY = data_read.read_train_struct()
exampleX = numpy.array(dataX[:3], dtype=float)

alphabet = [ i for i in range(1,27) ]
m = 3

