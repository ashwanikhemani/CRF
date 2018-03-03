import numpy, data_read, training

dataX, dataY = data_read.read_train_struct()
X, W, T = data_read.read_decode_input()

exampleX = numpy.array(dataX[3:11], dtype=float)
exampleY = numpy.array([ dataY[i][0] for i in range(3,11) ], dtype=int)

print(training.compute_gradw(exampleX, exampleY, W, T))
