import numpy as np

# Lambda functions #

# Sigmoid activation and its derivative
SIG = lambda x : 1 / (1 + np.exp(-x))
dSIG = lambda o : np.multiply(o, (1 - o)) 

# Error function and its derivative
ERROR = lambda t, o : np.multiply(0.5, np.multiply((t - o), (t - o)))
dERROR = lambda t, o : o - t


class Layer(object):

	def __init__ (self, inNodes, outNodes, alpha):
		# Numer of input and output nodes
		self.inNodes = inNodes + 1
		self.outNodes = outNodes
		self.alpha = alpha

		# Matrix of weights init + bias weight
		self.weights = np.round(np.random.rand(self.inNodes, self.outNodes) - 0.5, 2)


	def fwd (self, input):
		# Add the bias value to the input
		self.input = np.concatenate((input, [[1]]), axis=1)

		# Sum the inputs and normalize for each output
		sum = np.dot(self.input, self.weights)
		self.output = SIG(sum)

		return self.output

	def bck (self, dL1):
		# Derivative of output and L1
		dOUT = np.multiply(dL1, dSIG(self.output))

		# Derivative of the respective weights
		input_T = np.transpose(self.input)
		dW = np.dot(input_T, dOUT)
		self.weights -= self.alpha * dW

		# Derivative that is passed to L0
		transferVector = np.ones_like(self.output)
		dW_T = np.transpose(dW)
		dL0 = np.dot(transferVector, dW_T)
		dL0 = np.delete(dL0, -1, 1)
		return dL0


class NeuralNetwork(object):

	def __init__ (self, features, hiddenNeurons, classes, hiddenLayers = 1, alpha = 0.5):
		# Create the first layer
		self.layerStack = np.array([Layer(features, hiddenNeurons, alpha)])

		# Create the hidden layers
		for x in range(hiddenLayers - 1):
			self.layerStack = np.append(self.layerStack, [Layer(hiddenNeurons, hiddenNeurons, alpha)])

		# Create the output layer
		self.layerStack = np.append(self.layerStack, [Layer(hiddenNeurons, classes, alpha)])

	def layers(self):
		for layer in self.layerStack:
			print(layer.weights, "\n")
	
	def eval(self, input):
		lastInput = input
		for layer in self.layerStack:
			lastInput = layer.fwd(lastInput)

		return lastInput

	def train(self, input, target, iterations = 10000):
		for i in range(iterations):
			for j in range(input.shape[0]):
				# For each input vector in the data get the output
				inputVector = input[j]
				out = self.eval(inputVector)

				# Get target value for training set and calc error
				t = target[j]
				errorVector = ERROR(t, out)

				print(i, "\t", errorVector)

				# Backpropagate the error
				errorDerivative = dERROR(t, out)
				for l in range(len(self.layerStack) -1, -1, -1):
					errorDerivative = self.layerStack[l].bck(errorDerivative)
				
				


# XOR Test
i = np.matrix([[0, 0],
			   [1, 0],
			   [0, 1],
			   [1, 1]
			   ])

t = np.matrix([[1],
 			   [0], 
 			   [0], 
 			   [1]])

# Run
nn = NeuralNetwork(2, 3, 1)
nn.train(i, t)

while True:
	sample = input("Enter an array: ")
	sample = np.matrix(sample)
	print(nn.eval(sample))
