# functions_activation.py

import numpy as np

def activation(activation_fun,Z):
	if activation_fun == "linear":
		return Z
	elif activation_fun == "sigmoid":
		return 1/(1+np.exp(-Z))
	elif activation_fun == "relu":
		return np.maximum(Z,0.0)
	elif activation_fun == "tanh":
		return np.tanh(Z)
	elif activation_fun == "softplus":
		return np.log(1+np.exp(Z))

def activation_der(activation_fun,A):
	if activation_fun == "linear":
		return np.ones(A.shape)
	elif activation_fun == "sigmoid":
		return A - np.square(A)
	elif activation_fun == "relu":
		return np.piecewise(A,[A<0,A>=0],[0,1])
	elif activation_fun == "tanh":
		return 1 - np.square(A)
	elif activation_fun == "softplus":
		return 1 - np.exp(-A)