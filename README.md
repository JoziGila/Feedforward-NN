# Feedforward Neural Network

To learn the basics of neural networks I decided to implement one in python.
The simplest solution was to build Neuron classes which would be the signal processors.
In my script layers are the processing unit and they function using simply matrix operation such as 
Hadammard or Dot product. Basic improvements such as learning modifier (alpha) and bias units are implemented as well.

The NeuralNetwork class simply stacks the layers and feed the signal forward or backwards depending
on the state of the network (processing input or backpropagating)

The network is dynamic meaning that it can be created with an unlimited number of features and classifications
as well as an abitraty number of hidden layers (default is one). The hidden layers currently are all initialised
with the same neuron count but in future commit I may as well implement accepting custom functions that considering
the position in the layerStack decide the count of the neurons.

The code comes hardcoded with a XOR example in which the structure of the input should become apparent.
The function train() has two required parametres, an input matrix that in each row has a feature vector,
and a target matrix in which each row represent the desired output. They are connected by their row index so its
important for them to be arranged correctly. Eval() simple takes a one row feature matrix and returns the evaluation.

Future implementation should include data sanitising and auto formatting for a more fluid experience.
