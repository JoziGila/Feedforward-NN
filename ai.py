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
		# Derivative of L1 /w respect to the sum
		dSIG_OUT = dSIG(self.output)
		dL1_SUM = np.multiply(dL1, dSIG_OUT)

		# Derivative of L1 /w respect to input (to be passed to L0)
		W_T = np.transpose(self.weights)
		dL1_L0 = np.dot(dL1_SUM, W_T)
		dL1_L0 = np.delete(dL1_L0, -1, 1)


		# Change the weights using derivative of L1 /w respect to W
		input_T = np.transpose(self.input)
		dL1_W = np.dot(input_T, dL1_SUM)
		self.weights -= self.alpha * dL1_W

		return dL1_L0


class NeuralNetwork(object):

	def __init__ (self, features, hiddenNeurons, classes, hiddenLayers = 1, alpha = 0.5):
		# Create the first layer
		self.layerStack = np.array([Layer(features, hiddenNeurons, alpha)])

		# Create the hidden layers
		for x in range(hiddenLayers - 1):
			self.layerStack = np.append(self.layerStack, [Layer(hiddenNeurons, hiddenNeurons, alpha)])

		# Create the output layer
		self.layerStack = np.append(self.layerStack, [Layer(hiddenNeurons, classes, alpha)])

	def eval(self, input):
        # Forward the signal through the layers
		lastInput = input
		for l in self.layerStack:
			lastInput = l.fwd(lastInput)

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

				# Logging the error in the output
				print(i, "\t", errorVector)

				# Backpropagate the error though the layers
				errorDerivative = dERROR(t, out)
				for l in range(len(self.layerStack) -1, -1, -1):
					errorDerivative = self.layerStack[l].bck(errorDerivative)


# XOR Test
i = np.matrix([[0, 0],
	       [1, 0],
	       [0, 1],
	       [1, 1]])

t = np.matrix([[1],
 	       [0],
 	       [0],
 	       [1]])

# Run
nn = NeuralNetwork(2, 3, 1, 1)
nn.train(i, t)

while True:
	# Get input and return evaluation
	sample = input("Enter an array: ")
	sample = np.matrix(sample)
	print(np.round(nn.eval(sample)))
