import numpy, decode, data_read

X, W, T = data_read.read_decode_input()
dataX, dataY = data_read.read_train_struct()

alphabet = [ i for i in range(26) ]
m = 3

combinations = decode.generate_mcombs(alphabet, m)

print(decode.find_max(dataX[:3], combinations, W, T))
#remember to add one to all of these to get a valid letter sequence
