import numpy, decode, data_read

X, W, T = data_read.read_decode_input()

alphabet = [ i for i in range(1,27) ]
m = 3

combinations = decode.generate_mcombs(alphabet, m)

dataX, dataY = data_read.read_train_struct()
exampleX = numpy.array(dataX[:3], dtype=float)

