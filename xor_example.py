import neural as ai
import numpy as np

# XOR test data set
i = np.matrix([[0, 0],
	       [1, 0],
	       [0, 1],
	       [1, 1]])

t = np.matrix([[1],
 	       [0],
	       [0],
 	       [1]])

# Run
nn = ai.NeuralNetwork(2, 3, 1)
nn.train(i, t)

while True:
	# Get input and return evaluation
	sample = input("Enter an array: ")
	sample = np.matrix(sample)
	print(np.round(nn.eval(sample)))
