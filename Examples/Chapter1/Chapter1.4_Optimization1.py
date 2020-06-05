# Chapter1.4_Optimization1.py

import numpy as np

def loss(W):
	return 2*W[0]**2 + W[1]**2

def grad(W):
	return np.array([4*W[0],2*W[1]])

# initialization
W0 = np.array([2,2])
alpha = 0.1

# iteration 1
gradW0 = grad(W0)
print("gradW0: {}".format(gradW0))
W1 = W0 - alpha*gradW0
print("W1: {}".format(W1))
print("Loss(W1): {}".format(loss(W1)))

# iteration 2
gradW1 = grad(W1)
print("gradW1: {}".format(gradW1))
W2 = W1 - alpha*gradW1
print("W2: {}".format(W2))
print("Loss(W2): {}".format(loss(W2)))