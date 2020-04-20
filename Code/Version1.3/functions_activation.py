# functions_activation.py

import numpy as np

def activation(activation_fun,Z):
	if activation_fun=="linear":
		return Z
	if activation_fun=="sigmoid":
		return 1/(1+np.exp(-Z))

def activation_der(activation_fun,A):
	if activation_fun == "linear":
		return np.ones(A.shape)
	elif activation_fun == "sigmoid":
		return A - np.square(A)