# Feedforward Neural Network

To learn the basics of neural networks I decided to implement one in python.
In my script layers are the processing unit and they function using simply matrix operation such as 
Hadammard or Dot product. Basic features such as learning modifier (alpha) and bias units are implemented as well.

The NeuralNetwork class simply stacks the layers and feeds the signal forward or backwards depending
on the state of the network (processing input or backpropagating)

The network is dynamic meaning that it can be created with an arbitrary number of features and classes
as well as an abitraty number of hidden layers (default is one). The hidden layers currently are all initialised
with the same neuron count but in future commit I may as well implement custom functions as an argument, that map the
depth of the layer to the neuron count.
The code comes hardcoded with a XOR example in which the structure of the input should become apparent.
The function train() has two required parameters, an input matrix that in each row has a feature vector,
and a target matrix in which each row represent the desired output. They are connected by their row index so its
important for them to be arranged correctly. Eval() simple takes a one row feature matrix and returns the evaluation.
