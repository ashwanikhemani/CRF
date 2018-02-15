#script to read input data into list structure

with open("../data/train_struct.txt", "r") as f:
	raw_data = f.read()
raw_data = raw_data.split("\n")

dataX, dataY = [], []
for line in raw_data[:-2]: #-2 because last element is empty
	line = line.split(" ")
	dataY.append([ int(line[0]), int(line[1][4:]) ]) 
	datax = [0]*128
	for f1 in line[2:]:
		end = f1.find(":")
		datax[int(f1[:end])-1] = 1
	dataX.append(datax)

def print_image(X):
	for i in range(16):
		print(X[8*i:8*i+8])
