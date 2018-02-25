#the orphaned functions that don't belong anywhere

def print_image(X):
#prints an image, if it is in a list
	for i in range(16):
		print(X[8*i:8*i+8])
