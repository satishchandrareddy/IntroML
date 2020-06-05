# Chapter3_optimization.py
#
import numpy as np

def loss(W):
	return 2*W[0]*W[0] + W[1]*W[1]

def gradient(W):
	return np.array([4*W[0],2*W[1]])

# RmsProp
print("RmsProp")
W0 = np.array([2,2])
alpha = 0.1
beta = 0.9
print("W0: {}".format(W0))
print("Loss W0: {}".format(loss(W0)))
v0 = 0
# Loop 1
print("Gradient W0: {}".format(gradient(W0)))
print("Gradient2 W0: {}".format(np.square(gradient(W0))))
v1 = beta*v0 + (1-beta)*np.square(gradient(W0))
W1 = W0 - alpha*gradient(W0)/np.sqrt(v1)
print("v1: {}".format(v1))
print("W1: {}".format(W1))
print("Loss W1: {}".format(loss(W1)))
# Loop 2
print("Gradient W1: {}".format(gradient(W1)))
print("Gradient2 W1: {}".format(np.square(gradient(W1))))
v2 = beta*v1 + (1-beta)*np.square(gradient(W1))
W2 = W1 - alpha*gradient(W1)/np.sqrt(v2)
print("v2: {}".format(v2))
print("W2: {}".format(W2))
print("Loss W2: {}".format(loss(W2)))
